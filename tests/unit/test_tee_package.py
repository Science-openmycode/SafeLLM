from __future__ import annotations

import hashlib
import json

import torch
from safetensors.torch import save_file

from aloepri.tee.attestation import sm3
from aloepri.tee.package import inspect_server_package, verify_manifest_files


def _write_manifest(package, files) -> None:  # type: ignore[no-untyped-def]
    records = []
    for path in files:
        content = path.read_bytes()
        records.append(
            {
                "path": path.name,
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
                "sm3": sm3(content).hex(),
            }
        )
    (package / "server-manifest.json").write_text(
        json.dumps({"files": records}), encoding="utf-8"
    )


def test_valid_server_body_has_only_zero_hf_boundaries(tmp_path) -> None:
    weights = tmp_path / "model.safetensors"
    save_file(
        {
            "model.embed_tokens.weight": torch.zeros(5, 4),
            "model.layers.0.self_attn.q_proj.weight": torch.ones(4, 4),
            "lm_head.weight": torch.zeros(5, 4),
        },
        weights,
    )
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps({"security_mode": "tee_gm", "boundary_mode": "tee_split"}),
        encoding="utf-8",
    )
    _write_manifest(tmp_path, [config, weights])
    result = inspect_server_package(tmp_path)
    assert result.pass_, result.failures
    assert verify_manifest_files(tmp_path, "server-manifest.json") == 2


def test_server_scanner_rejects_nonzero_boundary_and_tau(tmp_path) -> None:
    weights = tmp_path / "model.safetensors"
    save_file(
        {
            "model.embed_tokens.weight": torch.ones(5, 4),
            "lm_head.weight": torch.zeros(5, 4),
            "tau": torch.arange(5),
        },
        weights,
    )
    _write_manifest(tmp_path, [weights])
    result = inspect_server_package(tmp_path)
    assert not result.pass_
    assert any("not zero" in failure for failure in result.failures)
    assert any("tau" in failure for failure in result.failures)


def test_manifest_tampering_is_rejected(tmp_path) -> None:
    target = tmp_path / "payload.bin"
    target.write_bytes(b"original")
    _write_manifest(tmp_path, [target])
    target.write_bytes(b"tampered")
    try:
        verify_manifest_files(tmp_path, "server-manifest.json")
    except ValueError as error:
        assert "mismatch" in str(error)
    else:
        raise AssertionError("tampered package was accepted")
