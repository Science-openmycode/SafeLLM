from __future__ import annotations

from pathlib import Path

from aloepri.evidence import tokenizer_identity, verify_tokenizer_identity


def test_tokenizer_identity_tracks_only_tokenizer_assets(tmp_path: Path) -> None:
    (tmp_path / "tokenizer.json").write_text('{"version": 1}', encoding="utf-8")
    (tmp_path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"not-a-real-model")
    identity = tokenizer_identity(tmp_path)
    assert {Path(item["path"]).name for item in identity["files"]} == {
        "tokenizer.json",
        "tokenizer_config.json",
    }
    assert verify_tokenizer_identity(identity)
    (tmp_path / "tokenizer.json").write_text('{"version": 2}', encoding="utf-8")
    assert not verify_tokenizer_identity(identity)


def test_tokenizer_identity_requires_assets(tmp_path: Path) -> None:
    try:
        tokenizer_identity(tmp_path)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("empty tokenizer directory must be rejected")
