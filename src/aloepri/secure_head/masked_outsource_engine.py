"""Verified one-time-mask outsourcing for the large LM Head matrix product."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from torch import Tensor

from aloepri.secure_head.mask_pool import OneTimeMaskPool
from aloepri.tee.execution_trace import tensor_evidence, trace_event
from aloepri.tee.trusted_boundary import (
    GenerationParameters,
    HeadDecision,
    LocalTeeHeadEngine,
)

_RING_MASK = 0xFFFFFFFF
def _signed_int32(values: Tensor) -> Tensor:
    unsigned = torch.bitwise_and(values.to(torch.int64), _RING_MASK)
    return torch.where(unsigned >= 2**31, unsigned - 2**32, unsigned)


@dataclass(frozen=True)
class QuantizedHead:
    weight_q: Tensor
    scales: Tensor
    row_norms: Tensor
    error_norms: Tensor

    @classmethod
    def from_float(cls, weight: Tensor) -> QuantizedHead:
        working = weight.detach().float().cpu().contiguous()
        if working.ndim != 2 or not torch.isfinite(working).all():
            raise ValueError("outsource Head must be a finite [vocab, hidden] matrix")
        scales = working.abs().amax(dim=1).clamp_min(torch.finfo(torch.float32).tiny) / 127.0
        weight_q = torch.round(working / scales[:, None]).clamp(-127, 127).to(torch.int8)
        reconstructed = weight_q.float() * scales[:, None]
        return cls(
            weight_q=weight_q,
            scales=scales,
            row_norms=torch.linalg.vector_norm(working, dim=1),
            error_norms=torch.linalg.vector_norm(working - reconstructed, dim=1),
        )


class InProcessGpuHeadWorker:
    """Reference ring worker; a CUDA kernel can implement the same narrow contract."""

    def __init__(self, weight_q: Tensor) -> None:
        self.weight_q = weight_q.detach().to(torch.int64).cpu().contiguous()

    def compute(self, masked_input: Tensor) -> Tensor:
        values = masked_input.detach().to(torch.int64).cpu().reshape(1, -1)
        return torch.bitwise_and(values @ self.weight_q.T, _RING_MASK).reshape(-1)


class MaskedOutsourceHeadEngine:
    """Certified greedy Head outsourcing with fail-closed local fallback."""

    def __init__(
        self,
        *,
        exact_head: Tensor,
        head_basis: Tensor,
        outsourced_head: Tensor,
        mask_pool: OneTimeMaskPool,
        max_candidates: int = 256,
        verification_rounds: int = 2,
        verification_seed: int = 20260829,
    ) -> None:
        self.exact = exact_head.detach().float().cpu().contiguous()
        self.basis = head_basis.detach().float().cpu().contiguous()
        self.outsourced = outsourced_head.detach().float().cpu().contiguous()
        if self.basis.ndim != 2 or self.basis.shape[0] != self.basis.shape[1]:
            raise ValueError("Head basis must be square")
        if self.exact.shape != self.outsourced.shape:
            raise ValueError("exact and outsourced Head shapes differ")
        if self.exact.shape[1] != self.basis.shape[0]:
            raise ValueError("Head basis does not match hidden size")
        if max_candidates < 1:
            raise ValueError("max_candidates must be positive")
        if verification_rounds < 2:
            raise ValueError("at least two Freivalds rounds are required")
        self.quantized = QuantizedHead.from_float(self.outsourced)
        self.worker = InProcessGpuHeadWorker(self.quantized.weight_q)
        self.pool = mask_pool
        self.max_candidates = max_candidates
        self.local = LocalTeeHeadEngine(self.exact)
        self.trace_enabled = False
        self.last_trace: list[dict[str, object]] = []
        generator = torch.Generator().manual_seed(verification_seed)
        self._checks: list[tuple[Tensor, Tensor]] = []
        matrix = self.quantized.weight_q.to(torch.int64).T
        for _ in range(verification_rounds):
            # Binary Freivalds vectors keep W @ a safely inside int64.  The
            # equality itself is checked in the same Z/(2^32) ring as the
            # outsourced matmul; changing fields after wraparound is unsound.
            challenge = torch.randint(
                0,
                2,
                (self.exact.shape[0],),
                generator=generator,
                dtype=torch.int64,
            )
            compressed = torch.bitwise_and(matrix @ challenge, _RING_MASK)
            self._checks.append((challenge, compressed))

    def make_mask(self, *, seed: int) -> str:
        generator = torch.Generator().manual_seed(seed)
        rho = torch.randint(
            0,
            2**32,
            (self.exact.shape[1],),
            generator=generator,
            dtype=torch.int64,
        )
        correction = torch.bitwise_and(
            rho.reshape(1, -1) @ self.quantized.weight_q.to(torch.int64).T,
            _RING_MASK,
        ).reshape(-1)
        return self.pool.add(rho, correction)

    def _verify(self, masked_input: Tensor, masked_output: Tensor) -> bool:
        left_values = torch.bitwise_and(masked_output.to(torch.int64), _RING_MASK)
        right_values = torch.bitwise_and(masked_input.to(torch.int64), _RING_MASK)
        for challenge, compressed in self._checks:
            left = sum(
                int(value) * int(bit)
                for value, bit in zip(left_values.tolist(), challenge.tolist(), strict=True)
            ) & _RING_MASK
            # A single uint32 product can exceed signed int64.  Python integers
            # make this reference verifier exact; the TDX native implementation
            # uses uint64 multiply-add with reduction after every term.
            right = sum(
                int(value) * int(coefficient)
                for value, coefficient in zip(
                    right_values.tolist(), compressed.tolist(), strict=True
                )
            ) & _RING_MASK
            if left != right:
                return False
        return True

    def _fallback(
        self,
        plain_hidden: Tensor,
        parameters: GenerationParameters,
        *,
        reason: str,
        prefix: list[dict[str, object]] | None = None,
    ) -> HeadDecision:
        self.local.trace_enabled = self.trace_enabled
        result = self.local.decide(plain_hidden, parameters)
        if self.trace_enabled:
            self.last_trace = [
                *(prefix or []),
                trace_event(
                    "masked_head_fallback",
                    "TEE Head控制器",
                    "安全回退到TEE内完整Head",
                    status="fallback",
                    detail=reason,
                    evidence={"head_mode": "masked_outsource", "fallback": True},
                ),
                *self.local.last_trace,
            ]
        else:
            self.last_trace = []
        return HeadDecision(
            token_id=result.token_id,
            used_fallback=True,
            candidate_count=result.candidate_count,
            verified=True,
        )

    def decide(self, plain_hidden: Tensor, parameters: GenerationParameters) -> HeadDecision:
        trace: list[dict[str, object]] = []
        hidden = plain_hidden.detach().float().cpu().reshape(-1)
        if hidden.numel() != self.basis.shape[0] or not torch.isfinite(hidden).all():
            return self._fallback(hidden, parameters, reason="隐藏向量形状或有限性检查失败")
        # Exact certification of nucleus/random sampling requires the complete
        # probability distribution.  The first implementation therefore uses
        # the reference TEE Head for every sampling request.
        if parameters.temperature != 0.0 or parameters.top_p < 1.0:
            return self._fallback(
                hidden,
                parameters,
                reason="Top-p或随机采样首版固定使用精确TEE Head",
            )
        record = self.pool.reserve()
        if record is None:
            return self._fallback(hidden, parameters, reason="一次性掩码池暂无可用记录")
        try:
            transform_started = time.perf_counter()
            transformed = hidden @ self.basis
            activation_scale = transformed.abs().amax().clamp_min(
                torch.finfo(torch.float32).tiny
            ) / 127.0
            activation_q = torch.round(transformed / activation_scale).clamp(-127, 127).to(
                torch.int64
            )
            if record.rho.numel() != activation_q.numel():
                raise ValueError("reserved mask has a different hidden size")
            masked_input = torch.bitwise_and(activation_q + record.rho, _RING_MASK)
            if self.trace_enabled:
                trace.extend(
                    [
                        trace_event(
                            "masked_head_coordinate",
                            "TEE私有内存",
                            "恢复h后切换到独立Head坐标并量化",
                            elapsed_ms=(time.perf_counter() - transform_started) * 1000,
                            detail="实际执行x=hB与INT8激活量化。",
                            evidence={
                                "hidden": tensor_evidence(hidden),
                                "basis_shape": list(self.basis.shape),
                                "transformed": tensor_evidence(transformed),
                                "activation_q": tensor_evidence(activation_q),
                                "activation_scale": float(activation_scale.item()),
                            },
                        ),
                        trace_event(
                            "masked_head_outbound",
                            "TEE → 不可信Head Worker",
                            "发送一次性掩码输入u",
                            detail="只把u=xq+ρ写入外包Worker输入；ρ和修正向量s留在TEE。",
                            evidence={
                                "mask_id_prefix": record.mask_id[:12],
                                "mask_state": "RESERVED",
                                "u": tensor_evidence(masked_input),
                            },
                        ),
                    ]
                )
            worker_started = time.perf_counter()
            masked_output = self.worker.compute(masked_input)
            worker_ms = (time.perf_counter() - worker_started) * 1000
            verify_started = time.perf_counter()
            verified = self._verify(masked_input, masked_output)
            verify_ms = (time.perf_counter() - verify_started) * 1000
            if self.trace_enabled:
                trace.extend(
                    [
                        trace_event(
                            "masked_head_worker",
                            "不可信Head Worker",
                            "执行外包词表大矩阵乘法",
                            elapsed_ms=worker_ms,
                            detail="software_sim实际调用InProcessGpuHeadWorker参考实现。",
                            evidence={
                                "worker_backend": type(self.worker).__name__,
                                "weight_shape": list(self.quantized.weight_q.shape),
                                "y": tensor_evidence(masked_output),
                            },
                        ),
                        trace_event(
                            "masked_head_inbound",
                            "不可信Head Worker → TEE",
                            "返回掩码矩阵乘法结果y",
                            detail="y被复制回TEE控制器，但尚未用于采样。",
                            evidence={"y": tensor_evidence(masked_output)},
                        ),
                        trace_event(
                            "masked_head_freivalds",
                            "TEE私有内存",
                            "执行两轮Freivalds完整性校验",
                            elapsed_ms=verify_ms,
                            status="complete" if verified else "failed",
                            detail="校验y·a与u·(WB,q·a)是否一致。",
                            evidence={"verified": verified, "rounds": len(self._checks)},
                        ),
                    ]
                )
            if not verified:
                self.pool.burn(record.mask_id)
                return self._fallback(
                    hidden,
                    parameters,
                    reason="Freivalds完整性校验失败，掩码已烧毁",
                    prefix=trace,
                )
            recover_started = time.perf_counter()
            recovered = _signed_int32(masked_output - record.correction)
            approximate = recovered.float() * activation_scale * self.quantized.scales

            reconstructed_input = activation_q.float() * activation_scale
            delta_input_norm = torch.linalg.vector_norm(transformed - reconstructed_input)
            reconstructed_norm = torch.linalg.vector_norm(reconstructed_input)
            error = (
                delta_input_norm * self.quantized.row_norms
                + reconstructed_norm * self.quantized.error_norms
            )
            lower = approximate - error
            upper = approximate + error
            threshold = lower.max()
            candidates = torch.nonzero(upper >= threshold, as_tuple=False).reshape(-1)
            recover_ms = (time.perf_counter() - recover_started) * 1000
            if self.trace_enabled:
                trace.append(
                    trace_event(
                        "masked_head_recover",
                        "TEE私有内存",
                        "去除修正量并构造认证候选集",
                        elapsed_ms=recover_ms,
                        detail="实际执行l̂=y-s及逐词表误差区间筛选。",
                        evidence={
                            "approximate_logits": tensor_evidence(approximate),
                            "candidate_count": int(candidates.numel()),
                        },
                    )
                )
            if candidates.numel() == 0 or candidates.numel() > self.max_candidates:
                self.pool.consume(record.mask_id)
                return self._fallback(
                    hidden,
                    parameters,
                    reason=f"认证候选数为{int(candidates.numel())}，超出安全范围",
                    prefix=trace,
                )
            exact_started = time.perf_counter()
            exact_logits = self.exact.index_select(0, candidates) @ hidden
            local_index = int(exact_logits.argmax().item())
            token_id = int(candidates[local_index])
            self.pool.consume(record.mask_id)
            if self.trace_enabled:
                trace.extend(
                    [
                        trace_event(
                            "masked_head_candidate_exact",
                            "TEE私有内存",
                            "对候选列执行FP32精确重算",
                            elapsed_ms=(time.perf_counter() - exact_started) * 1000,
                            detail="只有认证候选进入精确Head，最终Token来自真实Logits。",
                            evidence={
                                "candidate_count": int(candidates.numel()),
                                "exact_candidate_logits": tensor_evidence(exact_logits),
                            },
                        ),
                        trace_event(
                            "tee_sampling",
                            "TEE私有内存",
                            "采样认证后的普通Token",
                            detail="一次性掩码状态已从RESERVED变为CONSUMED。",
                            evidence={
                                "token_id": token_id,
                                "verified": True,
                                "mask_state": "CONSUMED",
                                "used_fallback": False,
                            },
                        ),
                    ]
                )
                self.last_trace = trace
            return HeadDecision(
                token_id=token_id,
                used_fallback=False,
                candidate_count=int(candidates.numel()),
                verified=True,
            )
        except Exception:
            self.pool.burn(record.mask_id)
            return self._fallback(
                hidden,
                parameters,
                reason="外包Head执行异常，掩码已烧毁",
                prefix=trace,
            )
