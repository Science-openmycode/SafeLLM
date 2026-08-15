from __future__ import annotations

import time
import uuid
from collections.abc import Iterator
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from aloepri.models.modeling_aloepri_deepseek_v3 import register_aloepri_deepseek_v3
from aloepri.models.modeling_aloepri_glm4_moe import register_aloepri_glm4_moe
from aloepri.models.modeling_aloepri_qwen2 import register_aloepri_qwen2
from aloepri.serving.protocol import GenerateRequest, GenerateResponse, Usage


class PrivateHFRuntime:
    def __init__(
        self,
        model_dir: Path,
        *,
        device: str = "auto",
        dtype: str = "auto",
        max_input_tokens: int = 2048,
        max_output_tokens: int = 512,
        gpu_memory_fraction: float = 0.70,
    ) -> None:
        register_aloepri_qwen2()
        register_aloepri_deepseek_v3()
        register_aloepri_glm4_moe()
        if device not in {"auto", "cpu", "cuda", "cuda-auto"}:
            raise ValueError(f"unsupported device: {device}")
        actual_device = "cuda" if device == "auto" and torch.cuda.is_available() else device
        self.device = "cpu" if actual_device == "auto" else actual_device
        if max_input_tokens < 1 or max_output_tokens < 1:
            raise ValueError("token limits must be positive")
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens
        if not 0.1 <= gpu_memory_fraction <= 0.9:
            raise ValueError("gpu_memory_fraction must be between 0.1 and 0.9")
        if dtype not in {"auto", "float32", "bfloat16"}:
            raise ValueError(f"unsupported dtype: {dtype}")
        load_options: dict[str, object] = {
            "local_files_only": True,
            "attn_implementation": "eager",
        }
        if dtype != "auto":
            load_options["dtype"] = torch.float32 if dtype == "float32" else torch.bfloat16
        if self.device in {"cuda", "cuda-auto"}:
            for index in range(torch.cuda.device_count()):
                torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction, device=index)
        if self.device == "cuda-auto":
            max_memory = {
                index: int(
                    torch.cuda.get_device_properties(index).total_memory
                    * gpu_memory_fraction
                )
                for index in range(torch.cuda.device_count())
            }
            load_options["device_map"] = "auto"
            load_options["max_memory"] = max_memory
            self.model = AutoModelForCausalLM.from_pretrained(model_dir, **load_options)
            self.input_device = str(self.model.get_input_embeddings().weight.device)
            device_map = getattr(self.model, "hf_device_map", {})
            offloaded = [
                name
                for name, placement in device_map.items()
                if str(placement) in {"cpu", "disk"}
            ]
            if offloaded:
                raise RuntimeError(
                    f"model was offloaded outside GPUs: {offloaded[:10]}"
                )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_dir, **load_options).to(
                self.device
            )
            self.input_device = self.device
        self.model.eval()
        metadata = getattr(self.model.config, "aloepri", None)
        if not isinstance(metadata, dict):
            raise ValueError("checkpoint has no AloePri metadata")
        self.model_id = str(metadata["model_id"])
        self.key_id = str(metadata["key_id"])
        eos = self.model.generation_config.eos_token_id
        self.eos_ids = {eos} if isinstance(eos, int) else set(eos or [])

    def validate(self, request: GenerateRequest) -> None:
        if request.model_id != self.model_id:
            raise ValueError("model_id does not match loaded checkpoint")
        if request.key_id != self.key_id:
            raise ValueError("key_id does not match loaded checkpoint")
        vocab_size = self.model.config.vocab_size
        if min(request.input_ids) < 0 or max(request.input_ids) >= vocab_size:
            raise ValueError("input_ids contains an out-of-vocabulary id")
        if len(request.input_ids) > self.max_input_tokens:
            raise ValueError("input token count exceeds configured maximum")
        if request.max_new_tokens > self.max_output_tokens:
            raise ValueError("output token count exceeds configured maximum")

    def iter_token_ids(self, request: GenerateRequest) -> Iterator[tuple[int, float]]:
        self.validate(request)
        generator: torch.Generator | None = None
        input_ids = torch.tensor([request.input_ids], dtype=torch.long, device=self.input_device)
        attention_mask = torch.ones_like(input_ids)
        past_key_values = None
        current = input_ids
        for _ in range(request.max_new_tokens):
            started = time.perf_counter()
            with torch.inference_mode():
                result = self.model(
                    input_ids=current,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            logits = result.logits[:, -1].float()
            if request.temperature == 0.0:
                next_id = logits.argmax(dim=-1, keepdim=True)
            else:
                if generator is None and request.seed is not None:
                    generator = torch.Generator(device=logits.device).manual_seed(
                        request.seed
                    )
                sampling_logits = logits / request.temperature
                if request.top_k:
                    top_k = min(request.top_k, sampling_logits.shape[-1])
                    threshold = sampling_logits.topk(top_k, dim=-1).values[:, -1:]
                    sampling_logits = sampling_logits.masked_fill(
                        sampling_logits < threshold, -torch.inf
                    )
                probabilities = torch.softmax(sampling_logits, dim=-1)
                if request.top_p < 1.0:
                    sorted_probs, sorted_ids = probabilities.sort(descending=True)
                    cumulative = sorted_probs.cumsum(dim=-1)
                    remove = cumulative - sorted_probs > request.top_p
                    sorted_probs[remove] = 0
                    sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
                    sampled = torch.multinomial(sorted_probs, 1, generator=generator)
                    next_id = sorted_ids.gather(-1, sampled)
                else:
                    next_id = torch.multinomial(probabilities, 1, generator=generator)
            if self.device in {"cuda", "cuda-auto"}:
                torch.cuda.synchronize(self.input_device)
            elapsed_ms = (time.perf_counter() - started) * 1000
            token_id = int(next_id.item())
            yield token_id, elapsed_ms
            if token_id in self.eos_ids:
                break
            past_key_values = result.past_key_values
            current = next_id.to(self.input_device)
            attention_mask = torch.cat(
                (
                    attention_mask,
                    torch.ones((1, 1), device=self.input_device, dtype=torch.long),
                ),
                dim=-1,
            )

    def generate(self, request: GenerateRequest) -> GenerateResponse:
        output_ids: list[int] = []
        durations: list[float] = []
        for token_id, elapsed_ms in self.iter_token_ids(request):
            output_ids.append(token_id)
            durations.append(elapsed_ms)
        return GenerateResponse(
            request_id=str(uuid.uuid4()),
            model_id=self.model_id,
            key_id=self.key_id,
            output_ids=output_ids,
            usage=Usage(input_tokens=len(request.input_ids), output_tokens=len(output_ids)),
            ttft_ms=durations[0],
            tpot_ms=sum(durations[1:]) / max(1, len(durations) - 1),
        )
