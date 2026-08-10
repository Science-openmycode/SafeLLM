import hashlib
import json

from aloepri.conversion.verify import find_secret_metadata_fields, verify_manifest


def test_manifest_verifier_detects_change(tmp_path) -> None:
    artifact = tmp_path / "model.safetensors"
    artifact.write_bytes(b"model")
    digest = hashlib.sha256(b"model").hexdigest()
    manifest = {"files": [{"path": artifact.name, "bytes": 5, "sha256": digest}]}
    (tmp_path / "aloepri_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert verify_manifest(tmp_path).ok
    artifact.write_bytes(b"changed")
    assert not verify_manifest(tmp_path).ok


def test_manifest_verifier_rejects_unlisted_file(tmp_path) -> None:
    artifact = tmp_path / "model.safetensors"
    artifact.write_bytes(b"model")
    manifest = {
        "files": [
            {
                "path": artifact.name,
                "bytes": artifact.stat().st_size,
                "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            }
        ]
    }
    (tmp_path / "aloepri_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "stale.safetensors").write_bytes(b"stale")
    result = verify_manifest(tmp_path)
    assert not result.ok
    assert "unlisted:stale.safetensors" in result.failures


def test_manifest_verifier_rejects_reconstructable_secret_metadata(tmp_path) -> None:
    artifact = tmp_path / "model.safetensors"
    artifact.write_bytes(b"model")
    digest = hashlib.sha256(b"model").hexdigest()
    manifest = {
        "metadata": {"seed": 20260803},
        "files": [{"path": artifact.name, "bytes": 5, "sha256": digest}],
    }
    (tmp_path / "aloepri_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    result = verify_manifest(tmp_path)
    assert not result.ok
    assert "secret-metadata:$.metadata.seed" in result.failures


def test_secret_metadata_scan_is_recursive() -> None:
    assert find_secret_metadata_fields({"safe": [{"inverse_tau": [0, 1]}]}) == (
        "$.safe[0].inverse_tau",
    )


def test_manifest_verifier_rejects_seed_in_server_config(tmp_path) -> None:
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"aloepri": {"seed": 7}}), encoding="utf-8")
    digest = hashlib.sha256(config.read_bytes()).hexdigest()
    manifest = {
        "metadata": {"model_id": "safe"},
        "files": [{"path": config.name, "bytes": config.stat().st_size, "sha256": digest}],
    }
    (tmp_path / "aloepri_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    result = verify_manifest(tmp_path)
    assert not result.ok
    assert "secret-config:$.aloepri.seed" in result.failures
