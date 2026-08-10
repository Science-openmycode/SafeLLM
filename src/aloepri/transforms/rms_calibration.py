from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class RmsCalibration:
    mean: float
    standard_deviation: float
    standard_error: float
    median: float
    q05: float
    q95: float
    sample_count: int


def private_to_plain_rms_ratio(hidden: Tensor, p: Tensor) -> Tensor:
    """Return rms(x @ P) / rms(x) for every token vector."""
    if hidden.shape[-1] != p.shape[0]:
        raise ValueError(f"hidden width {hidden.shape[-1]} does not match P rows {p.shape[0]}")
    working = hidden.detach().float()
    projected = working @ p.to(device=working.device, dtype=working.dtype)
    plain_rms = working.square().mean(-1).sqrt()
    private_rms = projected.square().mean(-1).sqrt()
    return private_rms / plain_rms.clamp_min(torch.finfo(working.dtype).tiny)


def summarize_rms_ratios(ratios: Tensor) -> RmsCalibration:
    values = ratios.detach().float().flatten().cpu()
    if values.numel() == 0 or not torch.isfinite(values).all():
        raise ValueError("RMS calibration requires finite, non-empty samples")
    return RmsCalibration(
        mean=float(values.mean()),
        standard_deviation=float(values.std(unbiased=True)) if values.numel() > 1 else 0.0,
        standard_error=(
            float(values.std(unbiased=True) / values.numel() ** 0.5)
            if values.numel() > 1
            else 0.0
        ),
        median=float(values.median()),
        q05=float(torch.quantile(values, 0.05)),
        q95=float(torch.quantile(values, 0.95)),
        sample_count=values.numel(),
    )


def collect_qwen2_rms_calibration(
    model: nn.Module, batches: list[dict[str, Tensor]], p: Tensor
) -> dict[str, RmsCalibration]:
    collected: dict[str, list[Tensor]] = {}
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def register(name: str, module: nn.Module) -> None:
        collected[name] = []

        def hook(_module: nn.Module, inputs: tuple[Tensor, ...]) -> None:
            collected[name].append(private_to_plain_rms_ratio(inputs[0], p).flatten().cpu())

        handles.append(module.register_forward_pre_hook(hook))

    for index, layer in enumerate(model.model.layers):
        register(f"layers.{index}.input_layernorm", layer.input_layernorm)
        register(f"layers.{index}.post_attention_layernorm", layer.post_attention_layernorm)
    register("model.norm", model.model.norm)
    try:
        with torch.inference_mode():
            for batch in batches:
                model(**batch, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    return {name: summarize_rms_ratios(torch.cat(values)) for name, values in collected.items()}
