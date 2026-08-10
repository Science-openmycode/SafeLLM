from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from aloepri.conversion.verify import verify_manifest
from aloepri.evidence import file_identity, sha256_file


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


def binding_pass(
    payload: dict[str, Any],
    binding: str | None,
    *,
    source_model: Path,
    private_model: Path,
    full_key_dir: Path,
) -> bool:
    if binding is None:
        return True
    if binding == "formula":
        return (
            Path(payload.get("source", "")).resolve() == source_model.resolve()
            and Path(payload.get("private", "")).resolve() == private_model.resolve()
            and Path(payload.get("key_dir", "")).resolve() == full_key_dir.resolve()
        )
    if binding == "isolated_score":
        return bool(
            payload.get("schema") == "aloepri-isolated-attack-score-v1"
            and payload.get("scoring_only_target_key_access") is True
        )
    raise ValueError(f"unknown evidence binding: {binding}")


def inspect_package(private_model: Path) -> dict[str, Any]:
    manifest = verify_manifest(private_model)
    forbidden_names = {
        "online_key.safetensors",
        "offline_master_key.safetensors",
        "paper_key.safetensors",
    }
    actual_names = {path.name for path in private_model.rglob("*") if path.is_file()}
    return {
        "manifest_valid": manifest.ok,
        "manifest_failures": manifest.failures,
        "forbidden_key_files": sorted(actual_names & forbidden_names),
        "pass": manifest.ok and not (actual_names & forbidden_names),
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
    full_key_dir = Path(config["full_key_dir"])

    package = inspect_package(private_model)
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
