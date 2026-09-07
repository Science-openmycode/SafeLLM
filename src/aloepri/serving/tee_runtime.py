from __future__ import annotations

import secrets
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM

from aloepri.models.modeling_aloepri_qwen2 import register_aloepri_qwen2
from aloepri.secure_head import MaskedOutsourceHeadEngine, OneTimeMaskPool
from aloepri.serving.protocol import GenerateRequest, GenerateResponse, Usage
from aloepri.tee.boundary import GenerationParameters, SoftwareTrustedBoundary
from aloepri.tee.execution_trace import tensor_evidence, trace_event


class TeeSplitHFRuntime:
    """Qwen body runtime split at Embedding and LM Head boundaries.

    ``software_sim`` executes the trusted-boundary object in this process.  The
    same narrow calls are used by the TDX gateway, where only the body model is
    hosted by the untrusted GPU worker.
    """

    def __init__(
        self,
        model_dir: Path,
        boundary_package: Path,
        *,
        device: str = "auto",
        dtype: str = "auto",
        max_input_tokens: int = 2048,
        max_output_tokens: int = 512,
        gpu_memory_fraction: float = 0.70,
        head_mode: str = "local",
        initial_mask_count: int = 1,
    ) -> None:
        register_aloepri_qwen2()
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError(f"unsupported TEE body device: {device}")
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else device
        if self.device == "auto":
            self.device = "cpu"
        if dtype not in {"auto", "float32", "bfloat16"}:
            raise ValueError(f"unsupported dtype: {dtype}")
        if max_input_tokens < 1 or max_output_tokens < 1:
            raise ValueError("token limits must be positive")
        if not 0.1 <= gpu_memory_fraction <= 0.9:
            raise ValueError("gpu_memory_fraction must be between 0.1 and 0.9")
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens
        if self.device == "cuda":
            torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction)

        options: dict[str, object] = {
            "local_files_only": True,
            "attn_implementation": "eager",
        }
        if dtype != "auto":
            options["dtype"] = torch.float32 if dtype == "float32" else torch.bfloat16
        auto_model = cast(Any, AutoModelForCausalLM)
        self.model = auto_model.from_pretrained(model_dir, **options).to(self.device)
        self.model.eval()
        metadata = getattr(self.model.config, "aloepri", None)
        if not isinstance(metadata, dict):
            raise ValueError("checkpoint has no Yinbian metadata")
        if metadata.get("security_mode") != "tee_gm":
            raise ValueError("checkpoint is not a tee_gm body package")
        if metadata.get("boundary_mode") != "tee_split":
            raise ValueError("checkpoint does not use a split trusted boundary")
        if metadata.get("hardware_attested") is not False:
            raise ValueError("software runtime cannot load a hardware-attested package")

        self.boundary = SoftwareTrustedBoundary(boundary_package)
        if head_mode not in {"local", "masked_outsource"}:
            raise ValueError("head_mode must be local or masked_outsource")
        if initial_mask_count < 0:
            raise ValueError("initial_mask_count must be non-negative")
        self.head_mode = head_mode
        self.head_metrics = {
            "decisions": 0,
            "outsourced": 0,
            "fallback": 0,
            "candidate_total": 0,
        }
        self.mask_pool: OneTimeMaskPool | None = None
        self.mask_engine: MaskedOutsourceHeadEngine | None = None
        self.last_execution_trace: list[dict[str, object]] = []
        if head_mode == "masked_outsource":
            basis_file = load_file(
                boundary_package / "head-coordinate-key.safetensors", device="cpu"
            )
            worker_file = load_file(
                model_dir / "masked-head-worker.safetensors", device="cpu"
            )
            pool = OneTimeMaskPool()
            engine = MaskedOutsourceHeadEngine(
                exact_head=self.boundary.exact_head,
                head_basis=basis_file["head_basis"],
                outsourced_head=worker_file["outsourced_head"],
                mask_pool=pool,
            )
            for index in range(initial_mask_count):
                engine.make_mask(seed=20260829 + index)
            self.boundary.set_head_engine(engine)
            self.mask_pool = pool
            self.mask_engine = engine
        self.model_id = str(metadata["model_id"])
        self.key_id = str(metadata["key_id"])
        if (self.model_id, self.key_id) != (
            self.boundary.model_id,
            self.boundary.key_id,
        ):
            raise ValueError("TEE boundary and body model/key versions do not match")
        if int(self.model.config.vocab_size) != self.boundary.vocab_size:
            raise ValueError("TEE boundary and body vocabulary sizes do not match")
        eos = self.model.generation_config.eos_token_id
        self.eos_ids = {eos} if isinstance(eos, int) else set(eos or [])
        self._readiness_result: dict[str, object] | None = None

    def validate(self, request: GenerateRequest) -> None:
        if request.model_id != self.model_id:
            raise ValueError("model_id does not match loaded TEE deployment")
        if request.key_id != self.key_id:
            raise ValueError("key_id does not match loaded TEE deployment")
        if min(request.input_ids) < 0 or max(request.input_ids) >= self.boundary.vocab_size:
            raise ValueError("input_ids contains an out-of-vocabulary id")
        if len(request.input_ids) > self.max_input_tokens:
            raise ValueError("input token count exceeds configured maximum")
        if request.max_new_tokens > self.max_output_tokens:
            raise ValueError("output token count exceeds configured maximum")

    def readiness(self) -> dict[str, object]:
        if self._readiness_result is not None:
            return self._readiness_result
        candidate = getattr(self.model.config, "bos_token_id", None)
        if isinstance(candidate, list):
            candidate = candidate[0] if candidate else None
        input_id = int(candidate) if isinstance(candidate, int) else 0
        response = self.generate(
            GenerateRequest(
                model_id=self.model_id,
                key_id=self.key_id,
                input_ids=[input_id],
                max_new_tokens=1,
                temperature=0.0,
            )
        )
        if len(response.output_ids) != 1:
            raise RuntimeError("TEE split readiness generation returned no token")
        self._readiness_result = {
            "status": "ready",
            "security_mode": "tee_gm",
            "tee_backend": "software_sim",
            "hardware_attested": False,
            "model_id": self.model_id,
            "key_id": self.key_id,
            "generated_tokens": 1,
        }
        return self._readiness_result

    def iter_token_ids(self, request: GenerateRequest) -> Iterator[tuple[int, float]]:
        self.validate(request)
        current_ids = torch.tensor([request.input_ids], dtype=torch.int64)
        attention_mask = torch.ones(
            (1, len(request.input_ids)), dtype=torch.long, device=self.device
        )
        past_key_values = None
        self.boundary.set_trace_enabled(request.include_execution_trace)
        try:
            for step in range(request.max_new_tokens):
                step_trace: list[dict[str, object]] = []
                if (
                    request.include_execution_trace
                    and step == 0
                    and self.mask_engine is not None
                ):
                    mask_started = time.perf_counter()
                    mask_id = self.mask_engine.make_mask(seed=secrets.randbits(63))
                    step_trace.append(
                        trace_event(
                            "masked_head_mask_ready",
                            "TEE掩码池",
                            "为本次首Token生成一次性掩码",
                            elapsed_ms=(time.perf_counter() - mask_started) * 1000,
                            detail="software_sim为可观测演示即时生成；真实部署应由离线掩码工厂补充。",
                            evidence={
                                "mask_id_prefix": mask_id[:12],
                                "pool_counts": self.mask_pool.counts() if self.mask_pool else {},
                            },
                        )
                    )
                started = time.perf_counter()
                private_embedding = self.boundary.embed(current_ids, device=self.device)
                step_trace.extend(self.boundary.last_embedding_trace)
                if self.device == "cuda":
                    torch.cuda.synchronize()
                body_started = time.perf_counter()
                with torch.inference_mode():
                    body_result = self.model.model(
                        inputs_embeds=private_embedding,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                if self.device == "cuda":
                    torch.cuda.synchronize()
                body_ms = (time.perf_counter() - body_started) * 1000
                if request.include_execution_trace:
                    cache = body_result.past_key_values
                    cache_evidence: dict[str, object] = {
                        "type": type(cache).__name__,
                        "sequence_length": int(cache.get_seq_length())
                        if hasattr(cache, "get_seq_length")
                        else attention_mask.shape[-1],
                    }
                    step_trace.append(
                        trace_event(
                            "gpu_transformer_body",
                            "GPU模型主体",
                            "执行P/Q Transformer主体",
                            elapsed_ms=body_ms,
                            detail=(
                                "中间层按现有私有Attention、FFN、RMSNorm、Residual"
                                "和KV Cache真实运行。"
                            ),
                            evidence={
                                "decode_step": step,
                                "mode": "prefill" if step == 0 else "decode",
                                "model_layers": int(self.model.config.num_hidden_layers),
                                "input": tensor_evidence(private_embedding),
                                "last_hidden_state": tensor_evidence(
                                    body_result.last_hidden_state
                                ),
                                "kv_cache": cache_evidence,
                            },
                        )
                    )
                decision = self.boundary.head(
                    body_result.last_hidden_state[:, -1],
                    GenerationParameters(
                        temperature=request.temperature,
                        top_k=request.top_k,
                        top_p=request.top_p,
                        seed=request.seed + step if request.seed is not None else None,
                    ),
                )
                step_trace.extend(self.boundary.last_head_trace)
                self.head_metrics["decisions"] += 1
                self.head_metrics["candidate_total"] += decision.candidate_count
                if decision.used_fallback:
                    self.head_metrics["fallback"] += 1
                elif self.head_mode == "masked_outsource":
                    self.head_metrics["outsourced"] += 1
                if self.device == "cuda":
                    torch.cuda.synchronize()
                elapsed_ms = (time.perf_counter() - started) * 1000
                token_id = decision.token_id
                if request.include_execution_trace:
                    step_trace.append(
                        trace_event(
                            "token_step_complete",
                            "TEE生成控制器",
                            "完成一个自回归Token",
                            elapsed_ms=elapsed_ms,
                            detail="Token随后被加密返回；若不是EOS，则重新进入TEE Embedding。",
                            evidence={
                                "decode_step": step,
                                "token_id": token_id,
                                "head_mode_configured": self.head_mode,
                                "head_used_fallback": decision.used_fallback,
                                "head_verified": decision.verified,
                                "candidate_count": decision.candidate_count,
                            },
                        )
                    )
                self.last_execution_trace = step_trace
                yield token_id, elapsed_ms
                if token_id in self.eos_ids:
                    break
                past_key_values = body_result.past_key_values
                current_ids = torch.tensor([[token_id]], dtype=torch.int64)
                attention_mask = torch.cat(
                    (
                        attention_mask,
                        torch.ones((1, 1), device=self.device, dtype=torch.long),
                    ),
                    dim=-1,
                )
        finally:
            self.boundary.set_trace_enabled(False)

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
