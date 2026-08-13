from __future__ import annotations

import scripts.build_qwen05b_v47_acceptance as acceptance
from aloepri.evidence import file_identity, sha256_file
from scripts.build_qwen05b_v47_acceptance import binding_pass, evaluate_rule, nested_value


def test_nested_acceptance_rule_evaluation() -> None:
    payload = {"metric": {"drop": 0.02, "ci": [-0.01, 0.03]}, "ok": True}
    assert nested_value(payload, "metric.drop") == 0.02
    assert nested_value(payload, "metric.ci.0") == -0.01
    assert evaluate_rule(payload, {"path": "metric.drop", "maximum": 0.035})["pass"]
    assert evaluate_rule(payload, {"path": "ok", "equals": True})["pass"]


def test_missing_acceptance_field_fails() -> None:
    result = evaluate_rule({}, {"path": "missing", "equals": True})
    assert result == {"path": "missing", "status": "MISSING_FIELD", "pass": False}


def test_formula_binding_is_portable_by_content_hash(tmp_path) -> None:
    source = tmp_path / "relocated-source"
    private = tmp_path / "relocated-private"
    key_dir = tmp_path / "relocated-keys"
    for directory in (source, private, key_dir):
        directory.mkdir()
    source_config = source / "config.json"
    private_config = private / "config.json"
    paper_key = key_dir / "paper_key.safetensors"
    source_config.write_bytes(b"source")
    private_config.write_bytes(b"private")
    paper_key.write_bytes(b"key")
    payload = {
        "source": r"E:\\old\\source",
        "private": r"E:\\old\\private",
        "key_dir": r"E:\\old\\keys",
        "source_config_sha256": sha256_file(source_config),
        "private_config_sha256": sha256_file(private_config),
        "paper_key_sha256": sha256_file(paper_key),
    }

    assert binding_pass(
        payload,
        "formula",
        source_model=source,
        private_model=private,
        full_key_dir=key_dir,
    )
    payload["paper_key_sha256"] = "0" * 64
    assert not binding_pass(
        payload,
        "formula",
        source_model=source,
        private_model=private,
        full_key_dir=key_dir,
    )


def test_isolated_score_binding_is_portable_and_content_bound(tmp_path) -> None:
    project_root = tmp_path / "relocated-project"
    evidence_path = project_root / "artifacts" / "privacy" / "score.json"
    script = project_root / "scripts" / "score_attack.py"
    prediction = evidence_path.parent / "prediction.json"
    source = project_root / "data" / "models" / "source"
    private = project_root / "data" / "packages" / "private"
    key_dir = project_root / "data" / "keys" / "full"
    for directory in (script.parent, evidence_path.parent, source, private, key_dir):
        directory.mkdir(parents=True, exist_ok=True)
    script.write_bytes(b"scorer")
    prediction.write_bytes(b"predictions")
    (source / "model.safetensors").write_bytes(b"source-model")
    (private / "model.safetensors").write_bytes(b"private-model")
    (key_dir / "key_manifest.json").write_bytes(b"manifest")
    (key_dir / "paper_key.safetensors").write_bytes(b"key")

    def old_identity(path):
        record = file_identity(path)
        record["path"] = rf"E:\\old\\{path.name}"
        return record

    payload = {
        "schema": "aloepri-isolated-attack-score-v1",
        "scoring_only_target_key_access": True,
        "provenance": {
            "schema_version": 1,
            "formal_run_binding": True,
            "script": old_identity(script),
            "data_files": [old_identity(prediction)],
            "original_model": {"files": [old_identity(source / "model.safetensors")]},
            "private_model": {"files": [old_identity(private / "model.safetensors")]},
            "key": {
                "files": [
                    old_identity(key_dir / "key_manifest.json"),
                    old_identity(key_dir / "paper_key.safetensors"),
                ]
            },
        },
    }

    arguments = {
        "source_model": source,
        "private_model": private,
        "full_key_dir": key_dir,
        "evidence_path": evidence_path,
        "project_root": project_root,
    }
    assert binding_pass(payload, "isolated_score", **arguments)

    prediction.write_bytes(b"tampered")
    assert not binding_pass(payload, "isolated_score", **arguments)


def test_paired_comparison_binding_rejects_dtype_mismatch(tmp_path, monkeypatch) -> None:
    source = tmp_path / "source"
    private = tmp_path / "private"
    key_dir = tmp_path / "keys"
    for directory in (source, private, key_dir):
        directory.mkdir()
    key = key_dir / "paper_key.safetensors"
    key.write_bytes(b"key")
    monkeypatch.setattr(acceptance, "verify_model_identity", lambda record: True)
    monkeypatch.setattr(acceptance, "verify_file_identity", lambda record: True)
    identity = {"path": str(tmp_path / "evidence"), "size": 1, "sha256": "0" * 64}
    payload = {
        "dtype_match": False,
        "provenance": {
            "formal_run_binding": True,
            "baseline_artifact": identity,
            "candidate_artifact": identity,
            "script": identity,
            "baseline_run": {
                "model": {"path": str(source)},
                "key": None,
                "script": identity,
            },
            "candidate_run": {
                "model": {"path": str(private)},
                "key": {"path": str(key)},
                "script": identity,
            },
        },
    }

    assert not binding_pass(
        payload,
        "paired_model_comparison",
        source_model=source,
        private_model=private,
        full_key_dir=key_dir,
    )


def test_paired_comparison_binding_accepts_current_artifacts(tmp_path, monkeypatch) -> None:
    source = tmp_path / "source"
    private = tmp_path / "private"
    key_dir = tmp_path / "keys"
    for directory in (source, private, key_dir):
        directory.mkdir()
    key = key_dir / "paper_key.safetensors"
    key.write_bytes(b"key")
    monkeypatch.setattr(acceptance, "verify_model_identity", lambda record: True)
    monkeypatch.setattr(acceptance, "verify_file_identity", lambda record: True)
    identity = {"path": str(tmp_path / "evidence"), "size": 1, "sha256": "0" * 64}
    payload = {
        "dtype_match": True,
        "provenance": {
            "formal_run_binding": True,
            "baseline_artifact": identity,
            "candidate_artifact": identity,
            "script": identity,
            "baseline_run": {
                "model": {"path": str(source)},
                "key": None,
                "script": identity,
            },
            "candidate_run": {
                "model": {"path": str(private)},
                "key": {"path": str(key)},
                "script": identity,
            },
        },
    }

    assert binding_pass(
        payload,
        "paired_model_comparison",
        source_model=source,
        private_model=private,
        full_key_dir=key_dir,
    )
