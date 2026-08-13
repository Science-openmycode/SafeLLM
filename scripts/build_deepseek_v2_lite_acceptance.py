from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from aloepri.conversion.verify import verify_manifest
from aloepri.evidence import (
    file_identity,
    verify_file_identity,
    verify_model_identity,
    verify_tokenizer_identity,
)
from aloepri.packaging import inspect_server_package
from aloepri.packaging import verify_manifest as verify_key_manifest


def load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact is not an object: {path}")
    return payload


def formal_run_ok(run: object) -> bool:
    return bool(
        isinstance(run, dict)
        and run.get("formal_run_binding") is True
        and verify_model_identity(run.get("model", {}))
        and verify_tokenizer_identity(run.get("tokenizer", {}))
        and verify_file_identity(run.get("script", {}))
        and (run.get("key") is None or verify_file_identity(run.get("key", {})))
    )


def paired_runs_ok(report: dict[str, Any]) -> bool:
    provenance = report.get("provenance", {})
    return bool(
        provenance.get("formal_run_binding") is True
        and formal_run_ok(provenance.get("baseline_run"))
        and formal_run_ok(provenance.get("candidate_run"))
    )


def utility_check(report: dict[str, Any]) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if report.get("dtype_match") is not True:
        failures.append("dtype mismatch")
    if not paired_runs_ok(report):
        failures.append("formal provenance missing")
    tasks = report.get("tasks", [])
    if not tasks:
        failures.append("task-level results missing")
    for task in tasks:
        name = str(task.get("task"))
        if float(task.get("absolute_change", -1.0)) < -0.035:
            failures.append(f"{name}: point loss exceeds 3.5 percentage points")
        interval = task.get("paired_change_95_percent_ci", [])
        if len(interval) != 2 or float(interval[0]) < -0.035:
            failures.append(f"{name}: paired 95% CI lower bound is below -0.035")
    return not failures, failures


def ifeval_check(report: dict[str, Any]) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if int(report.get("sample_len", 0)) != 541:
        failures.append("IFEval sample count is not 541")
    if report.get("dtype_match") is not True:
        failures.append("IFEval dtype mismatch")
    if not paired_runs_ok(report):
        failures.append("IFEval formal provenance missing")
    metrics = report.get("metrics", {})
    if not metrics:
        failures.append("IFEval metrics missing")
    for name, metric in metrics.items():
        if float(metric.get("absolute_change", -1.0)) < -0.035:
            failures.append(f"IFEval {name}: point loss exceeds 0.035")
        interval = metric.get("paired_change_95_percent_ci", [])
        if len(interval) != 2 or float(interval[0]) < -0.035:
            failures.append(f"IFEval {name}: paired 95% CI lower bound below -0.035")
    return not failures, failures


def humaneval_check(report: dict[str, Any]) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if int(report.get("sample_len", 0)) != 164:
        failures.append("HumanEval sample count is not 164")
    if report.get("dtype_match") is not True:
        failures.append("HumanEval dtype mismatch")
    if not paired_runs_ok(report):
        failures.append("HumanEval formal provenance missing")
    if float(report.get("absolute_change", -1.0)) < -0.035:
        failures.append("HumanEval point loss exceeds 0.035")
    interval = report.get("paired_change_95_percent_ci", [])
    if len(interval) != 2 or float(interval[0]) < -0.035:
        failures.append("HumanEval paired 95% CI lower bound below -0.035")
    return not failures, failures


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build strict DeepSeek-V2-Lite architecture acceptance"
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    model = root / "data/packages/deepseek-v2-lite-chat-mla-moe"
    online_key = root / "data/keys/deepseek-v2-lite-chat-mla-moe-online"
    offline_key = root / "data/keys/deepseek-v2-lite-chat-mla-moe-offline"
    evidence = root / "artifacts/deepseek-v2-lite-chat"
    paths = {
        "preflight": root / "artifacts/cloud/deepseek-v2-lite-preflight.json",
        "source_audit": root / "artifacts/cloud/deepseek-v2-lite-source-audit.json",
        "private_audit": root / "artifacts/cloud/deepseek-v2-lite-private-audit.json",
        "package_inspection": root
        / "artifacts/cloud/deepseek-v2-lite-package-inspection.json",
        "equivalence": evidence / "equivalence.json",
        "product_smoke": evidence / "product-smoke/result.json",
        "mmlu": evidence / "accuracy/mmlu-comparison.json",
        "ceval": evidence / "accuracy/ceval-comparison.json",
        "piqa": evidence / "accuracy/piqa-comparison.json",
        "ifeval": evidence / "accuracy/ifeval-comparison.json",
        "humaneval": evidence / "accuracy/humaneval-comparison.json",
    }
    checks: list[dict[str, Any]] = []
    for name, path in paths.items():
        if not path.is_file():
            checks.append(
                {"id": name, "pass": False, "status": "NOT_TESTED", "path": str(path)}
            )
            continue
        payload = load(path)
        if name in {"mmlu", "ceval", "piqa"}:
            passed, failures = utility_check(payload)
            expected_samples = {"mmlu": 14042, "ceval": 1346, "piqa": 1838}[name]
            if int(payload.get("sample_len", 0)) != expected_samples:
                failures.append(
                    f"{name} sample count is not the locked {expected_samples}"
                )
                passed = False
        elif name == "ifeval":
            passed, failures = ifeval_check(payload)
        elif name == "humaneval":
            passed, failures = humaneval_check(payload)
        else:
            passed = payload.get("pass") is True
            failures = [] if passed else ["artifact pass field is not true"]
        checks.append(
            {
                "id": name,
                "pass": passed,
                "status": "PASS" if passed else "FAIL",
                "failures": failures,
                "artifact": file_identity(path),
            }
        )

    manifest = verify_manifest(model) if model.is_dir() else None
    package = inspect_server_package(model) if model.is_dir() else {"pass": False}
    online_failures = verify_key_manifest(online_key) if online_key.is_dir() else ["missing"]
    offline_failures = verify_key_manifest(offline_key) if offline_key.is_dir() else ["missing"]
    binding_failures: list[str] = []
    if model.is_dir() and online_key.is_dir() and offline_key.is_dir():
        model_config = load(model / "config.json").get("aloepri", {})
        model_manifest = load(model / "aloepri_manifest.json").get("metadata", {})
        online_metadata = load(online_key / "key.json")
        offline_metadata = load(offline_key / "key.json")
        expected_revision = "85864749cd611b4353ce1decdb286193298f64c7"
        expected_transform = "deepseek-v2-latent-mla-moe-v3"
        if model_config.get("transform_version") != expected_transform:
            binding_failures.append("model config transform version mismatch")
        if model_manifest.get("transform_version") != expected_transform:
            binding_failures.append("model manifest transform version mismatch")
        if offline_metadata.get("transform_version") != expected_transform:
            binding_failures.append("offline key transform version mismatch")
        if online_metadata.get("transform_version") != expected_transform:
            binding_failures.append("online key transform version mismatch")
        if model_manifest.get("source_revision") != expected_revision:
            binding_failures.append("model source revision mismatch")
        if offline_metadata.get("source_revision") != expected_revision:
            binding_failures.append("offline key source revision mismatch")
        if online_metadata.get("source_revision") != expected_revision:
            binding_failures.append("online key source revision mismatch")
        if model_config.get("model_id") != online_metadata.get("model_id"):
            binding_failures.append("model/online-key model_id mismatch")
        if model_config.get("key_id") != online_metadata.get("key_id"):
            binding_failures.append("model/online-key key_id mismatch")
        if model_manifest.get("key_id") != online_metadata.get("key_id"):
            binding_failures.append("manifest/online-key key_id mismatch")
        if offline_metadata.get("online_key_id") != online_metadata.get("key_id"):
            binding_failures.append("offline/online key binding mismatch")
    else:
        binding_failures.append("model or key package missing")
    package_pass = bool(
        manifest is not None
        and manifest.ok
        and package.get("pass") is True
        and not online_failures
        and not offline_failures
        and not binding_failures
    )
    checks.append(
        {
            "id": "package_and_key_integrity",
            "pass": package_pass,
            "status": "PASS" if package_pass else "FAIL",
            "model_manifest_failures": (
                list(manifest.failures) if manifest is not None else ["model missing"]
            ),
            "package_inspection": package,
            "online_key_manifest_failures": online_failures,
            "offline_key_manifest_failures": offline_failures,
            "binding_failures": binding_failures,
        }
    )
    architecture_pass = all(check["pass"] for check in checks)
    report = {
        "schema": "aloepri-deepseek-v2-lite-acceptance-v1",
        "target_model": "deepseek-ai/DeepSeek-V2-Lite-Chat",
        "target_revision": "85864749cd611b4353ce1decdb286193298f64c7",
        "architecture_validation_decision": "GO" if architecture_pass else "NO-GO",
        "architecture_decision_rule": (
            "GO requires hardware, source/private shape audit, package/key integrity, "
            "BF16 prefill/cache/greedy equivalence, product round trip, and all paired "
            "utility gates to pass on current artifacts"
        ),
        "deepseek_privacy_release_decision": "NO-GO",
        "deepseek_privacy_release_reason": (
            "This checkpoint validates vocabulary, full MLA, and MoE adaptation but does "
            "not yet apply hidden P/Q or embedding/head noise; direct embedding matching "
            "therefore remains outside the accepted privacy boundary."
        ),
        "implemented_scope": [
            "full vocabulary permutation and client inverse mapping",
            "MLA q/kv latent-coordinate permutation with RMSNorm synchronization",
            "MLA head/nope/decoupled-RoPE/value/output transforms",
            "MoE router/expert permutation and per-expert SwiGLU transforms",
            "streaming resumable conversion and bounded-memory repacking",
            "multi-GPU HF prefill, cached decode, generation, API and paired utility tests",
        ],
        "excluded_scope": ["MTP", "DeepSeek-V3 671B", "DeepSeek hidden P/Q and noise"],
        "summary": {
            "total": len(checks),
            "passed": sum(bool(check["pass"]) for check in checks),
            "failed": sum(not bool(check["pass"]) for check in checks),
        },
        "checks": checks,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_name(args.out.name + ".partial")
    partial.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    partial.replace(args.out)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if architecture_pass else 2)


if __name__ == "__main__":
    main()
