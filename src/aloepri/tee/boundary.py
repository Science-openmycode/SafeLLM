from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import Tensor

from aloepri.tee.execution_trace import tensor_evidence, trace_event
from aloepri.tee.package import verify_manifest_files


@dataclass(frozen=True)
class GenerationParameters:
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    seed: int | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be between 0 and 2")
        if self.top_k < 0:
            raise ValueError("top_k must be non-negative")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")


@dataclass(frozen=True)
class HeadDecision:
    token_id: int
    used_fallback: bool
    candidate_count: int
    verified: bool


class HeadEngine(Protocol):
    def decide(self, plain_hidden: Tensor, parameters: GenerationParameters) -> HeadDecision: ...


class TrustedBoundary(Protocol):
    model_id: str
    key_id: str
    vocab_size: int

    def embed(self, token_ids: Tensor, *, device: torch.device | str) -> Tensor: ...
    def recover_hidden(self, private_hidden: Tensor) -> Tensor: ...
    def head(
        self, private_hidden: Tensor, parameters: GenerationParameters
    ) -> HeadDecision: ...


def sample_logits(logits: Tensor, parameters: GenerationParameters) -> int:
    if logits.ndim != 1:
        raise ValueError("sampler expects one vocabulary logit vector")
    if not torch.isfinite(logits).all():
        raise ValueError("logits contain NaN or Inf")
    if parameters.temperature == 0.0:
        return int(logits.argmax().item())
    working = logits.float() / parameters.temperature
    if parameters.top_k:
        top_k = min(parameters.top_k, working.numel())
        threshold = working.topk(top_k).values[-1]
        working = working.masked_fill(working < threshold, -torch.inf)
    probabilities = torch.softmax(working, dim=-1)
    if parameters.top_p < 1.0:
        sorted_probs, sorted_ids = probabilities.sort(descending=True)
        cumulative = sorted_probs.cumsum(dim=-1)
        remove = cumulative - sorted_probs > parameters.top_p
        sorted_probs[remove] = 0
        sorted_probs /= sorted_probs.sum()
        generator = torch.Generator(device=logits.device)
        if parameters.seed is not None:
            generator.manual_seed(parameters.seed)
        sampled = torch.multinomial(sorted_probs, 1, generator=generator)
        return int(sorted_ids[sampled].item())
    generator = torch.Generator(device=logits.device)
    if parameters.seed is not None:
        generator.manual_seed(parameters.seed)
    return int(torch.multinomial(probabilities, 1, generator=generator).item())


class LocalTeeHeadEngine:
    """Reference Head retained entirely inside the trusted boundary."""

    def __init__(self, exact_head: Tensor) -> None:
        if exact_head.ndim != 2:
            raise ValueError("exact Head must have shape [vocab, hidden]")
        self.exact_head = exact_head.detach().float().cpu().contiguous()
        self.last_trace: list[dict[str, object]] = []
        self.trace_enabled = False

    def logits(self, plain_hidden: Tensor) -> Tensor:
        hidden = plain_hidden.detach().float().cpu().reshape(-1)
        if hidden.numel() != self.exact_head.shape[1]:
            raise ValueError("plain hidden size does not match exact Head")
        return F.linear(hidden, self.exact_head)

    def decide(self, plain_hidden: Tensor, parameters: GenerationParameters) -> HeadDecision:
        started = time.perf_counter()
        logits = self.logits(plain_hidden)
        token_id = sample_logits(logits, parameters)
        if not self.trace_enabled:
            self.last_trace = []
            return HeadDecision(
                token_id=token_id,
                used_fallback=False,
                candidate_count=logits.numel(),
                verified=True,
            )
        self.last_trace = [
            trace_event(
                "tee_local_head_matmul",
                "TEE私有内存",
                "完整LM Head矩阵乘法",
                elapsed_ms=(time.perf_counter() - started) * 1000,
                detail="本次Token完全在TEE边界内计算logits，没有调用外包Head。",
                evidence={
                    "head_mode": "local",
                    "head_shape": list(self.exact_head.shape),
                    "hidden": tensor_evidence(plain_hidden),
                    "logits": tensor_evidence(logits),
                },
            ),
            trace_event(
                "tee_sampling",
                "TEE私有内存",
                "从真实Logits采样普通Token",
                detail="采样结果仍留在TEE，随后由加密传输层封装。",
                evidence={"token_id": token_id, "verified": True},
            ),
        ]
        return HeadDecision(
            token_id=token_id,
            used_fallback=False,
            candidate_count=logits.numel(),
            verified=True,
        )


class SoftwareTrustedBoundary:
    """Functional TEE simulator.  It never represents hardware attestation."""

    def __init__(self, package: Path, *, head_engine: HeadEngine | None = None) -> None:
        verify_manifest_files(package, "tee-manifest.json")
        manifest = json.loads((package / "tee-manifest.json").read_text(encoding="utf-8"))
        if manifest.get("security_mode") != "tee_gm":
            raise ValueError("boundary package is not a tee_gm package")
        if manifest.get("boundary_mode") != "tee_split":
            raise ValueError("boundary package is not split at model boundaries")
        self.model_id = str(manifest["model_id"])
        self.key_id = str(manifest["key_id"])
        embedding = load_file(package / "embedding-private.safetensors", device="cpu")
        exact = load_file(package / "exact-head-archive.safetensors", device="cpu")
        coordinate = load_file(package / "final-coordinate-key.safetensors", device="cpu")
        self.embedding = embedding["embedding_private"].contiguous()
        self.exact_head = exact["exact_head"].contiguous()
        self.final_q = coordinate["q_final"].float().contiguous()
        self.vocab_size = self.embedding.shape[0]
        if self.exact_head.shape[0] != self.vocab_size:
            raise ValueError("Embedding and Head vocab sizes differ")
        if self.embedding.shape[1] != self.final_q.shape[0]:
            raise ValueError("private Embedding and final Q dimensions differ")
        if self.exact_head.shape[1] != self.final_q.shape[1]:
            raise ValueError("exact Head and plain hidden dimensions differ")
        self._head_engine = head_engine or LocalTeeHeadEngine(self.exact_head)
        self.last_embedding_trace: list[dict[str, object]] = []
        self.last_head_trace: list[dict[str, object]] = []
        self.trace_enabled = False

    def embed(self, token_ids: Tensor, *, device: torch.device | str) -> Tensor:
        started = time.perf_counter()
        ids = token_ids.detach().to(dtype=torch.int64, device="cpu")
        if ids.numel() and (int(ids.min()) < 0 or int(ids.max()) >= self.vocab_size):
            raise ValueError("input token id is outside the TEE vocabulary")
        if not self.trace_enabled:
            self.last_embedding_trace = []
            return self.embedding[ids].to(device)
        lookup_started = time.perf_counter()
        private_cpu = self.embedding[ids]
        lookup_ms = (time.perf_counter() - lookup_started) * 1000
        transfer_started = time.perf_counter()
        private_device = private_cpu.to(device)
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        transfer_ms = (time.perf_counter() - transfer_started) * 1000
        flat_ids = ids.reshape(-1)
        token_ids_head = flat_ids[:8].tolist()
        token_ids_tail = flat_ids[-4:].tolist() if flat_ids.numel() > 8 else []
        self.last_embedding_trace = [
            trace_event(
                "tee_embedding_lookup",
                "TEE私有内存",
                "使用普通Token查询E·P",
                elapsed_ms=lookup_ms,
                detail="实际读取TEE边界包中的私有Embedding行。",
                evidence={
                    "token_count": ids.numel(),
                    "token_shape": list(ids.shape),
                    "decrypted_token_ids_head": token_ids_head,
                    "decrypted_token_ids_tail": token_ids_tail,
                    "private_embedding": tensor_evidence(private_cpu),
                },
            ),
            trace_event(
                "tee_to_gpu_transfer",
                "TEE边界 → GPU主体",
                "把z₀复制到GPU计算设备",
                elapsed_ms=transfer_ms,
                detail="software_sim实际执行CPU到模型设备的Tensor复制；真实TDX使用显式共享页。",
                evidence={
                    "transfer_kind": "cpu_to_device_software_sim",
                    "output": tensor_evidence(private_device),
                    "total_embed_call_ms": round(
                        (time.perf_counter() - started) * 1000, 4
                    ),
                },
            ),
        ]
        return private_device

    def recover_hidden(self, private_hidden: Tensor) -> Tensor:
        return private_hidden.float().cpu() @ self.final_q

    def head(
        self, private_hidden: Tensor, parameters: GenerationParameters
    ) -> HeadDecision:
        if not self.trace_enabled:
            self.last_head_trace = []
            plain = self.recover_hidden(private_hidden).reshape(-1)
            return self._head_engine.decide(plain, parameters)
        receive_event = trace_event(
            "gpu_to_tee_transfer",
            "GPU主体 → TEE边界",
            "接收最后一层私有隐藏状态zL",
            detail="software_sim实际执行设备到CPU的恢复路径；不向浏览器返回原始值。",
            evidence={"private_hidden": tensor_evidence(private_hidden)},
        )
        recover_started = time.perf_counter()
        plain = self.recover_hidden(private_hidden).reshape(-1)
        recover_event = trace_event(
            "tee_recover_hidden",
            "TEE私有内存",
            "使用QL恢复普通隐藏状态h",
            elapsed_ms=(time.perf_counter() - recover_started) * 1000,
            detail="h=zL·QL；h只传给TEE内Head控制器。",
            evidence={
                "private_hidden": tensor_evidence(private_hidden),
                "q_final_shape": list(self.final_q.shape),
                "plain_hidden": tensor_evidence(plain),
            },
        )
        decision = self._head_engine.decide(plain, parameters)
        engine_trace = list(getattr(self._head_engine, "last_trace", []))
        self.last_head_trace = [receive_event, recover_event, *engine_trace]
        return decision

    def set_head_engine(self, engine: HeadEngine) -> None:
        """Install an audited optimization while retaining exact local state."""

        self._head_engine = engine
        if hasattr(engine, "trace_enabled"):
            cast(Any, engine).trace_enabled = self.trace_enabled

    def set_trace_enabled(self, enabled: bool) -> None:
        self.trace_enabled = enabled
        if hasattr(self._head_engine, "trace_enabled"):
            cast(Any, self._head_engine).trace_enabled = enabled
