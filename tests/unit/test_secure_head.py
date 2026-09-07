from __future__ import annotations

import hashlib
import json

import torch
from safetensors.torch import save_file

from aloepri.secure_head import MaskedOutsourceHeadEngine, MaskState, OneTimeMaskPool
from aloepri.tee.attestation import sm3
from aloepri.tee.boundary import (
    GenerationParameters,
    LocalTeeHeadEngine,
    SoftwareTrustedBoundary,
)


def _orthogonal(size: int, seed: int = 7) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    matrix = torch.randn(size, size, generator=generator)
    return torch.linalg.qr(matrix).Q


def _engine(*, masks: int = 1) -> tuple[MaskedOutsourceHeadEngine, OneTimeMaskPool]:
    generator = torch.Generator().manual_seed(11)
    exact = torch.randn(13, 6, generator=generator)
    basis = _orthogonal(6)
    # x = hB and F.linear(x, W_B) = h W_exact^T.
    outsourced = exact @ basis
    pool = OneTimeMaskPool()
    engine = MaskedOutsourceHeadEngine(
        exact_head=exact,
        head_basis=basis,
        outsourced_head=outsourced,
        mask_pool=pool,
    )
    for seed in range(masks):
        engine.make_mask(seed=100 + seed)
    return engine, pool


def test_mask_pool_never_reuses_consumed_or_burned_masks() -> None:
    pool = OneTimeMaskPool()
    first = pool.add(torch.zeros(2), torch.zeros(3))
    reserved = pool.reserve()
    assert reserved is not None and reserved.mask_id == first
    pool.consume(first)
    assert pool.state(first) == MaskState.CONSUMED
    assert pool.reserve() is None

    second = pool.add(torch.ones(2), torch.ones(3))
    assert pool.reserve() is not None
    pool.burn(second)
    assert pool.state(second) == MaskState.BURNED
    assert pool.reserve() is None


def test_masked_head_matches_exact_greedy() -> None:
    engine, pool = _engine()
    hidden = torch.tensor([0.2, -1.1, 0.7, 0.4, -0.6, 1.3])
    expected = LocalTeeHeadEngine(engine.exact).decide(hidden, GenerationParameters())
    actual = engine.decide(hidden, GenerationParameters())
    assert actual.token_id == expected.token_id
    assert actual.verified
    assert pool.counts()[MaskState.CONSUMED.value] == 1


def test_tampered_worker_is_detected_and_falls_back() -> None:
    engine, pool = _engine()
    original_compute = engine.worker.compute

    def tampered(masked_input: torch.Tensor) -> torch.Tensor:
        result = original_compute(masked_input)
        result[0] = result[0] + 1
        return result

    engine.worker.compute = tampered  # type: ignore[method-assign]
    hidden = torch.tensor([0.2, -1.1, 0.7, 0.4, -0.6, 1.3])
    expected = LocalTeeHeadEngine(engine.exact).decide(hidden, GenerationParameters())
    actual = engine.decide(hidden, GenerationParameters())
    assert actual.token_id == expected.token_id
    assert actual.used_fallback
    assert pool.counts()[MaskState.BURNED.value] == 1


def test_sampling_request_uses_full_tee_head_without_reserving_mask() -> None:
    engine, pool = _engine()
    result = engine.decide(
        torch.ones(6),
        GenerationParameters(temperature=0.7, seed=3),
    )
    assert result.used_fallback
    assert pool.counts()[MaskState.AVAILABLE.value] == 1


def test_software_boundary_loads_split_package(tmp_path) -> None:
    embedding = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    final_q = torch.randn(4, 3, generator=torch.Generator().manual_seed(2))
    exact_head = torch.randn(5, 3, generator=torch.Generator().manual_seed(3))
    save_file({"embedding_private": embedding}, tmp_path / "embedding-private.safetensors")
    save_file({"exact_head": exact_head}, tmp_path / "exact-head-archive.safetensors")
    save_file({"q_final": final_q}, tmp_path / "final-coordinate-key.safetensors")
    files = []
    for path in sorted(tmp_path.iterdir()):
        content = path.read_bytes()
        files.append(
            {
                "path": path.name,
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
                "sm3": sm3(content).hex(),
            }
        )
    (tmp_path / "tee-manifest.json").write_text(
        json.dumps(
            {
                "security_mode": "tee_gm",
                "boundary_mode": "tee_split",
                "model_id": "tiny",
                "key_id": "key-tiny",
                "files": files,
            }
        ),
        encoding="utf-8",
    )

    boundary = SoftwareTrustedBoundary(tmp_path)
    token_ids = torch.tensor([[1, 4]])
    assert torch.equal(boundary.embed(token_ids, device="cpu"), embedding[token_ids])
    private_hidden = torch.randn(1, 4, generator=torch.Generator().manual_seed(4))
    expected = LocalTeeHeadEngine(exact_head).decide(
        private_hidden @ final_q,
        GenerationParameters(),
    )
    assert boundary.head(private_hidden, GenerationParameters()).token_id == expected.token_id
