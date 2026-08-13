from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from aloepri.conversion.verify import verify_manifest
from aloepri.evidence import (
    file_identity,
    sha256_file,
    verify_file_identity,
    verify_model_identity,
)
from aloepri.packaging import inspect_server_package


def nested_value(payload: dict[str, Any], dotted_path: str) -> Any:
    value: Any = payload
    for part in dotted_path.split("."):
        if isinstance(value, dict) and part in value:
            value = value[part]
        elif isinstance(value, list) and part.isdigit() and int(part) < len(value):
            value = value[int(part)]
        else:
            raise KeyError(dotted_path)
    return value


def evaluate_rule(payload: dict[str, Any], rule: dict[str, Any]) -> dict[str, Any]:
    path = str(rule["path"])
    try:
        actual = nested_value(payload, path)
    except KeyError:
        return {"path": path, "status": "MISSING_FIELD", "pass": False}
    result: dict[str, Any] = {"path": path, "actual": actual}
    if "equals" in rule:
        result.update(operator="equals", expected=rule["equals"], pass_=actual == rule["equals"])
    elif "maximum" in rule:
        result.update(
            operator="maximum", expected=rule["maximum"], pass_=float(actual) <= rule["maximum"]
        )
    elif "minimum" in rule:
        result.update(
            operator="minimum", expected=rule["minimum"], pass_=float(actual) >= rule["minimum"]
        )
    else:
        raise ValueError(f"rule has no supported operator: {rule}")
    result["pass"] = result.pop("pass_")
    return result


def portable_filename(raw_path: object) -> str:
    """Return a basename for provenance produced on Windows or POSIX."""
    return str(raw_path).replace("\\", "/").rsplit("/", 1)[-1]


def relocated_file_identity_matches(record: object, path: Path) -> bool:
    """Verify an identity by content after an evidence bundle is relocated."""
    if not isinstance(record, dict) or not path.is_file():
        return False
    try:
        return (
            path.stat().st_size == int(record["size"])
            and sha256_file(path) == str(record["sha256"])
        )
    except (KeyError, OSError, TypeError, ValueError):
        return False


def relocated_collection_matches(record: object, directory: Path) -> bool:
    """Verify every recorded file against the corresponding relocated directory."""
    if not isinstance(record, dict) or not isinstance(record.get("files"), list):
        return False
    files = record["files"]
    if not files:
        return False
    names = [portable_filename(item.get("path", "")) for item in files if isinstance(item, dict)]
    if len(names) != len(files) or len(set(names)) != len(names) or any(not name for name in names):
        return False
    return all(
        relocated_file_identity_matches(item, directory / portable_filename(item["path"]))
        for item in files
    )


def relocated_isolated_score_binding(
    payload: dict[str, Any],
    *,
    evidence_path: Path,
    project_root: Path,
    source_model: Path,
    private_model: Path,
    full_key_dir: Path,
) -> bool:
    """Bind a reusable isolated score to current code, data, models and key material.

    Absolute paths are deliberately ignored because the signed evidence bundle moves
    from Windows to Linux. Size and SHA-256 remain mandatory for every dependency.
    """
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        return False
    script = provenance.get("script")
    data_files = provenance.get("data_files")
    key = provenance.get("key")
    if (
        payload.get("schema") != "aloepri-isolated-attack-score-v1"
        or payload.get("scoring_only_target_key_access") is not True
        or provenance.get("schema_version") != 1
        or provenance.get("formal_run_binding") is not True
        or not isinstance(script, dict)
        or not isinstance(data_files, list)
        or not data_files
        or not relocated_file_identity_matches(
            script, project_root / "scripts" / portable_filename(script.get("path", ""))
        )
        or not all(
            isinstance(item, dict)
            and relocated_file_identity_matches(
                item, evidence_path.parent / portable_filename(item.get("path", ""))
            )
            for item in data_files
        )
        or not relocated_collection_matches(key, full_key_dir)
    ):
        return False

    original = provenance.get("original_model")
    private = provenance.get("private_model")
    return bool(
        (original is None or relocated_collection_matches(original, source_model))
        and (private is None or relocated_collection_matches(private, private_model))
    )


def binding_pass(
    payload: dict[str, Any],
    binding: str | None,
    *,
    source_model: Path,
    private_model: Path,
    full_key_dir: Path,
    evidence_path: Path | None = None,
    project_root: Path | None = None,
) -> bool:
    if binding is None:
        return True
    if binding == "formula":
        source_config = source_model / "config.json"
        private_config = private_model / "config.json"
        paper_key = full_key_dir / "paper_key.safetensors"
        if not all(path.is_file() for path in (source_config, private_config, paper_key)):
            return False
        return (
            payload.get("source_config_sha256") == sha256_file(source_config)
            and payload.get("private_config_sha256") == sha256_file(private_config)
            and payload.get("paper_key_sha256") == sha256_file(paper_key)
        )
    if binding == "isolated_score":
        if evidence_path is None or project_root is None:
            return False
        return relocated_isolated_score_binding(
            payload,
            evidence_path=evidence_path,
            project_root=project_root,
            source_model=source_model,
            private_model=private_model,
            full_key_dir=full_key_dir,
        )
    if binding == "paired_model_comparison":
        provenance = payload.get("provenance")
        if not isinstance(provenance, dict):
            return False
        baseline_run = provenance.get("baseline_run")
        candidate_run = provenance.get("candidate_run")
        if not isinstance(baseline_run, dict) or not isinstance(candidate_run, dict):
            return False
        baseline_model = baseline_run.get("model")
        candidate_model = candidate_run.get("model")
        candidate_key = candidate_run.get("key")
        expected_key = (full_key_dir / "paper_key.safetensors").resolve()
        identities = (
            provenance.get("baseline_artifact"),
            provenance.get("candidate_artifact"),
            provenance.get("script"),
            baseline_run.get("script"),
            candidate_run.get("script"),
        )
        return bool(
            provenance.get("formal_run_binding") is True
            and payload.get("dtype_match") is True
            and isinstance(baseline_model, dict)
            and isinstance(candidate_model, dict)
            and Path(str(baseline_model.get("path", ""))).resolve()
            == source_model.resolve()
            and Path(str(candidate_model.get("path", ""))).resolve()
            == private_model.resolve()
            and verify_model_identity(baseline_model)
            and verify_model_identity(candidate_model)
            and baseline_run.get("key") is None
            and isinstance(candidate_key, dict)
            and Path(str(candidate_key.get("path", ""))).resolve() == expected_key
            and verify_file_identity(candidate_key)
            and all(isinstance(item, dict) and verify_file_identity(item) for item in identities)
        )
    raise ValueError(f"unknown evidence binding: {binding}")


def inspect_package(private_model: Path, server_package: Path) -> dict[str, Any]:
    manifest = verify_manifest(private_model)
    forbidden_names = {
        "online_key.safetensors",
        "offline_master_key.safetensors",
        "paper_key.safetensors",
    }
    actual_names = {path.name for path in private_model.rglob("*") if path.is_file()}
    sanitized = (
        inspect_server_package(server_package)
        if server_package.is_dir()
        else {
            "pass": False,
            "findings": [f"server package does not exist: {server_package}"],
        }
    )
    checkpoint_pass = manifest.ok and not (actual_names & forbidden_names)
    return {
        "checkpoint_manifest_valid": manifest.ok,
        "checkpoint_manifest_failures": manifest.failures,
        "checkpoint_forbidden_key_files": sorted(actual_names & forbidden_names),
        "sanitized_package": sanitized,
        "pass": checkpoint_pass and bool(sanitized["pass"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the single-target v47 acceptance report")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config.get("schema") != "aloepri-qwen05b-acceptance-v2":
        raise ValueError("unsupported acceptance config schema")
    source_model = Path(config["source_model"])
    private_model = Path(config["private_model"])
    server_package = Path(config["server_package"])
    full_key_dir = Path(config["full_key_dir"])
    project_root = args.config.resolve().parents[2]

    package = inspect_package(private_model, server_package)
    checks: list[dict[str, Any]] = []
    for specification in config["evidence"]:
        path = Path(specification["path"])
        if not path.is_file():
            checks.append(
                {
                    "id": specification["id"],
                    "path": str(path),
                    "status": "NOT_TESTED",
                    "pass": False,
                }
            )
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        rules = [evaluate_rule(payload, rule) for rule in specification.get("rules", [])]
        bound = binding_pass(
            payload,
            specification.get("binding"),
            source_model=source_model,
            private_model=private_model,
            full_key_dir=full_key_dir,
            evidence_path=path,
            project_root=project_root,
        )
        passed = bound and all(rule["pass"] for rule in rules)
        checks.append(
            {
                "id": specification["id"],
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "binding_pass": bound,
                "rules": rules,
                "status": "PASS" if passed else "FAIL",
                "pass": passed,
            }
        )
    passed = package["pass"] and all(check["pass"] for check in checks)
    report = {
        "schema": "aloepri-qwen05b-acceptance-report-v2",
        "release_version": config["release_version"],
        "decision": "GO" if passed else "NO-GO",
        "decision_rule": "GO requires a valid sanitized package and every evidence check PASS",
        "target": {
            "source_model": str(source_model.resolve()),
            "private_model": str(private_model.resolve()),
            "server_package": str(server_package.resolve()),
            "full_key_dir": str(full_key_dir.resolve()),
            "product_config": file_identity(Path(config["product_config"])),
            "acceptance_config": file_identity(args.config),
        },
        "server_package": package,
        "summary": {
            "total": len(checks),
            "passed": sum(check["pass"] for check in checks),
            "failed": sum(check["status"] == "FAIL" for check in checks),
            "not_tested": sum(check["status"] == "NOT_TESTED" for check in checks),
        },
        "checks": checks,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
