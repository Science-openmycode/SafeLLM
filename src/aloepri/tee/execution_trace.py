from __future__ import annotations

import hashlib
from typing import Any

import torch
from torch import Tensor


def tensor_evidence(tensor: Tensor) -> dict[str, Any]:
    """Return compact evidence derived from the tensor that actually crossed a boundary.

    A bounded first/last sample is included only when execution tracing was
    explicitly requested.  It lets the trusted local UI explain what a tensor
    contains without returning the complete hidden state or logits.
    """

    detached = tensor.detach()
    cpu = detached.contiguous().cpu()
    raw = cpu.view(torch.uint8).numpy().tobytes()
    evidence: dict[str, Any] = {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype).removeprefix("torch."),
        "device": str(detached.device),
        "elements": detached.numel(),
        "sha256_16": hashlib.sha256(raw).hexdigest()[:16],
    }
    if cpu.numel():
        flattened = cpu.reshape(-1)
        sample_size = min(4, flattened.numel())
        head = flattened[:sample_size]
        tail = flattened[-sample_size:]
        if cpu.is_complex():
            evidence["sample_head"] = [str(value) for value in head.tolist()]
            evidence["sample_tail"] = [str(value) for value in tail.tolist()]
        else:
            evidence["sample_head"] = head.tolist()
            evidence["sample_tail"] = tail.tolist()
    if cpu.numel() and (cpu.is_floating_point() or cpu.is_complex()):
        working = cpu.float()
        evidence["finite"] = bool(torch.isfinite(working).all())
        evidence["l2"] = float(torch.linalg.vector_norm(working).item())
    return evidence


def trace_event(
    stage: str,
    actor: str,
    title: str,
    *,
    elapsed_ms: float | None = None,
    status: str = "complete",
    detail: str = "",
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    event: dict[str, Any] = {
        "stage": stage,
        "actor": actor,
        "title": title,
        "status": status,
        "detail": detail,
        "evidence": evidence or {},
    }
    if elapsed_ms is not None:
        event["elapsed_ms"] = round(float(elapsed_ms), 4)
    return event
