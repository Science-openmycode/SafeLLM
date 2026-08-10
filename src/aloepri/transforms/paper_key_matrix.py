"""Paper-faithful Algorithm 1 key-matrix construction.

The report's prose swaps "rows" and "columns" for C and D.  The implementation
uses the only shape-consistent interpretation: rows(C) lie in null(F.T), and
columns(D) lie in null(E).  This makes C @ F = 0 and E @ D = 0, hence P @ Q = I.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class PaperKeyPair:
    p: Tensor
    q: Tensor
    b: Tensor
    condition_b: float
    pq_relative_error: float
    spectral_norm_p: float
    spectral_norm_q: float
    algorithm1_base: PaperAlgorithm1Base | None = None


@dataclass(frozen=True)
class PaperAlgorithm1Base:
    """The shared ``INIT`` state from Algorithm 1.

    Reusing this state is essential: independently generated key and inverse-key
    matrices cancel for every pairing only when they share ``B, E, F, Z``.
    """

    b: Tensor
    b_inverse: Tensor
    e: Tensor
    f: Tensor
    z: Tensor

    def tensors(self) -> dict[str, Tensor]:
        return {
            "algorithm1.b": self.b.contiguous(),
            "algorithm1.b_inverse": self.b_inverse.contiguous(),
            "algorithm1.e": self.e.contiguous(),
            "algorithm1.f": self.f.contiguous(),
            "algorithm1.z": self.z.contiguous(),
        }


@dataclass(frozen=True)
class CompatibleInverseFamily:
    """Independent Algorithm 1 inverse keys from one shared ``INIT`` state."""

    head: Tensor
    attention_q: Tensor
    attention_k: Tensor
    attention_v: Tensor
    ffn_gate: Tensor
    ffn_up: Tensor
    maximum_relative_error: float

    def tensors(self) -> dict[str, Tensor]:
        return {
            "q.head": self.head,
            "q.attention_q": self.attention_q,
            "q.attention_k": self.attention_k,
            "q.attention_v": self.attention_v,
            "q.ffn_gate": self.ffn_gate,
            "q.ffn_up": self.ffn_up,
        }


def _orthogonal(dim: int, generator: torch.Generator, dtype: torch.dtype) -> Tensor:
    candidate = torch.randn((dim, dim), generator=generator, dtype=dtype)
    q, r = torch.linalg.qr(candidate)
    signs = torch.where(torch.diagonal(r) < 0, -1.0, 1.0).to(dtype)
    return q * signs


def _null_space(matrix: Tensor, *, expected_dim: int, tolerance: float | None = None) -> Tensor:
    _, singular_values, vh = torch.linalg.svd(matrix, full_matrices=True)
    if tolerance is None:
        largest = float(singular_values.max()) if singular_values.numel() else 0.0
        tolerance = max(matrix.shape) * torch.finfo(matrix.dtype).eps * largest
    rank = int((singular_values > tolerance).sum())
    basis = vh[rank:].mT.contiguous()
    if basis.shape[1] < expected_dim:
        raise ValueError(
            f"null space dimension {basis.shape[1]} is smaller than required {expected_dim}"
        )
    return basis[:, :expected_dim]


def make_paper_key_pair(
    dim: int,
    expansion_h: int,
    *,
    coefficient_lambda: float,
    seed: int,
    dtype: torch.dtype = torch.float64,
    condition_b_max: float = 1.0e4,
    max_attempts: int = 32,
) -> PaperKeyPair:
    if dim <= 0:
        raise ValueError("dim must be positive")
    if expansion_h <= 1 or expansion_h % 2:
        raise ValueError("expansion_h must be a positive even integer")
    if coefficient_lambda < 0:
        raise ValueError("coefficient_lambda must be non-negative")
    if dtype not in {torch.float32, torch.float64}:
        raise ValueError("Algorithm 1 generation requires float32 or float64")

    half = expansion_h // 2
    expanded_dim = dim + 2 * expansion_h
    for attempt in range(max_attempts):
        generator = torch.Generator(device="cpu").manual_seed(seed + attempt)
        u = _orthogonal(dim, generator, dtype)
        v = torch.randn((dim, dim), generator=generator, dtype=dtype) / dim**0.5
        b = u + coefficient_lambda * v
        condition_b = float(torch.linalg.cond(b))
        if not torch.isfinite(torch.tensor(condition_b)) or condition_b > condition_b_max:
            continue

        e1 = torch.randn((dim, half), generator=generator, dtype=dtype) / dim**0.5
        e2 = torch.randn((half, expansion_h), generator=generator, dtype=dtype) / dim**0.5
        e = e1 @ e2
        f1 = torch.randn((expansion_h, half), generator=generator, dtype=dtype) / dim**0.5
        f2 = torch.randn((half, dim), generator=generator, dtype=dtype) / dim**0.5
        f = f1 @ f2

        null_ft = _null_space(f.mT, expected_dim=half)
        c_coeff = torch.randn((dim, half), generator=generator, dtype=dtype) / dim**0.5
        c = c_coeff @ null_ft.mT

        null_e = _null_space(e, expected_dim=half)
        d_coeff = torch.randn((half, dim), generator=generator, dtype=dtype) / dim**0.5
        d = null_e @ d_coeff

        z = _orthogonal(expanded_dim, generator, dtype)
        b_inverse = torch.linalg.inv(b)
        p_base = torch.cat((b, c, e), dim=1)
        q_base = torch.cat((b_inverse, f, d), dim=0)
        p = (p_base @ z).contiguous()
        q = (z.mT @ q_base).contiguous()
        identity = torch.eye(dim, dtype=dtype)
        relative_error = float(
            torch.linalg.matrix_norm(p @ q - identity, ord="fro")
            / torch.linalg.matrix_norm(identity, ord="fro")
        )
        return PaperKeyPair(
            p=p,
            q=q,
            b=b,
            condition_b=condition_b,
            pq_relative_error=relative_error,
            spectral_norm_p=float(torch.linalg.matrix_norm(p, ord=2)),
            spectral_norm_q=float(torch.linalg.matrix_norm(q, ord=2)),
            algorithm1_base=PaperAlgorithm1Base(
                b=b,
                b_inverse=b_inverse,
                e=e,
                f=f,
                z=z,
            ),
        )
    raise RuntimeError(f"failed to sample B below condition threshold in {max_attempts} attempts")


def verify_paper_key_pair(
    pair: PaperKeyPair,
    *,
    dim: int,
    expansion_h: int,
    condition_b_max: float,
    relative_error_max: float,
) -> None:
    expanded_dim = dim + 2 * expansion_h
    if pair.p.shape != (dim, expanded_dim):
        raise ValueError(f"P shape is {tuple(pair.p.shape)}, expected {(dim, expanded_dim)}")
    if pair.q.shape != (expanded_dim, dim):
        raise ValueError(f"Q shape is {tuple(pair.q.shape)}, expected {(expanded_dim, dim)}")
    if pair.condition_b > condition_b_max:
        raise ValueError(f"B condition number {pair.condition_b} exceeds {condition_b_max}")
    if pair.pq_relative_error > relative_error_max:
        raise ValueError(
            f"P@Q relative error {pair.pq_relative_error} exceeds {relative_error_max}"
        )
    if not torch.isfinite(pair.p).all() or not torch.isfinite(pair.q).all():
        raise ValueError("P or Q contains NaN/Inf")


def make_compatible_inverse_family(
    pair: PaperKeyPair,
    *,
    seed: int,
    coefficient_scale: float = 1.0,
) -> CompatibleInverseFamily:
    """Sample six distinct ``INVKEYMATGEN`` results from the pair's ``INIT``.

    The paper does not state a distribution for the columns of ``D``.  We use
    Gaussian coefficients at Algorithm 1's ``1/sqrt(d)`` scale, but preserve the
    published matrix form exactly: ``Q_j = Z.T @ [B^-1; F; D_j]`` with
    ``E @ D_j = 0``.  Hence every returned key cancels the fixed ``P``.
    """
    if coefficient_scale <= 0:
        raise ValueError("coefficient_scale must be positive")
    if pair.algorithm1_base is None:
        raise ValueError("pair does not retain the shared Algorithm 1 INIT state")
    p = pair.p.double()
    base = pair.algorithm1_base
    dim, expanded_dim = p.shape
    expansion_h = base.e.shape[1]
    expected_nullity = expansion_h // 2
    null_e = _null_space(base.e.double(), expected_dim=expected_nullity)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    identity = torch.eye(dim, dtype=torch.float64)
    inverses: list[Tensor] = []
    errors: list[float] = []
    for _ in range(6):
        d_coefficients = (
            torch.randn((expected_nullity, dim), generator=generator, dtype=torch.float64)
            * (coefficient_scale / math.sqrt(dim))
        )
        d = null_e @ d_coefficients
        q_base = torch.cat((base.b_inverse.double(), base.f.double(), d), dim=0)
        inverse = (base.z.double().mT @ q_base).contiguous()
        error = float(
            torch.linalg.matrix_norm(p @ inverse - identity, ord="fro")
            / torch.linalg.matrix_norm(identity, ord="fro")
        )
        if not torch.isfinite(inverse).all() or error > 1.0e-10:
            raise RuntimeError(f"compatible inverse verification failed: {error}")
        inverses.append(inverse)
        errors.append(error)
    return CompatibleInverseFamily(*inverses, maximum_relative_error=max(errors))
