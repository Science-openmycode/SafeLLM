from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResidualBoundary:
    name: str
    plain_dim: int
    private_dim: int
    key_name: str


def qwen2_residual_boundaries(
    *, plain_dim: int, expansion_h: int, num_layers: int, key_name: str = "p_res"
) -> tuple[ResidualBoundary, ...]:
    if num_layers <= 0:
        raise ValueError("num_layers must be positive")
    private_dim = plain_dim + 2 * expansion_h
    names = ["embedding"]
    for layer in range(num_layers):
        names.extend((f"layer.{layer}.attention_residual", f"layer.{layer}.ffn_residual"))
    names.append("final_norm")
    return tuple(ResidualBoundary(name, plain_dim, private_dim, key_name) for name in names)


def verify_global_residual_key(boundaries: tuple[ResidualBoundary, ...]) -> None:
    if not boundaries:
        raise ValueError("at least one residual boundary is required")
    reference = (boundaries[0].plain_dim, boundaries[0].private_dim, boundaries[0].key_name)
    for boundary in boundaries[1:]:
        current = (boundary.plain_dim, boundary.private_dim, boundary.key_name)
        if current != reference:
            raise ValueError(
                f"residual coordinate mismatch at {boundary.name}: {current} != {reference}"
            )
