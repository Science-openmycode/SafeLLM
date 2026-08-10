from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml

from aloepri.conversion.verify import verify_manifest
from aloepri.evidence import (
    sha256_file,
    verify_file_identity,
    verify_model_identity,
    verify_run_provenance,
    verify_tokenizer_identity,
)


def read_optional(path: str | Path) -> dict[str, Any] | None:
    target = Path(path)
    if not target.is_file():
        return None
    return json.loads(target.read_text(encoding="utf-8"))


def add_missing(
    checks: list[dict[str, Any]], missing: list[str], check_id: str, path: str | Path
) -> None:
    path_text = str(path)
    missing.append(path_text)
    checks.append(
        {
            "id": check_id,
            "status": "NOT_TESTED",
            "artifact": path_text,
            "pass": False,
        }
    )


def same_path(left: str | Path, right: str | Path) -> bool:
    return Path(left).resolve() == Path(right).resolve()


def scoped_run_provenance_ok(
    record: dict[str, Any],
    *,
    source_model: Path | None,
    private_model: Path | None,
    key_dir: Path | None,
) -> bool:
    """Verify hashes and bind every optional model/key slot to this acceptance config."""
    if not verify_run_provenance(record):
        return False
    for field, expected in (
        ("original_model", source_model),
        ("private_model", private_model),
        ("key", key_dir),
    ):
        actual = record.get(field)
        if expected is None:
            if actual is not None:
                return False
        elif not isinstance(actual, dict) or not same_path(actual.get("path", ""), expected):
            return False
    return True


def lm_comparison_provenance_ok(
    report: dict[str, Any], *, source_model: Path, private_model: Path, key_dir: Path
) -> bool:
    provenance = report.get("provenance")
    if not isinstance(provenance, dict) or provenance.get("formal_run_binding") is not True:
        return False
    if not verify_file_identity(provenance.get("baseline_artifact", {})):
        return False
    if not verify_file_identity(provenance.get("candidate_artifact", {})):
        return False
    baseline = provenance.get("baseline_run")
    candidate = provenance.get("candidate_run")
    if not isinstance(baseline, dict) or not isinstance(candidate, dict):
        return False
    if not verify_model_identity(baseline.get("model", {})):
        return False
    if not verify_model_identity(candidate.get("model", {})):
        return False
    if not same_path(baseline["model"]["path"], source_model):
        return False
    if not same_path(candidate["model"]["path"], private_model):
        return False
    if baseline.get("key") is not None:
        return False
    candidate_key = candidate.get("key")
    return bool(
        isinstance(candidate_key, dict)
        and verify_file_identity(candidate_key)
        and Path(candidate_key["path"]).resolve().parent == key_dir.resolve()
    )


def performance_provenance_ok(
    report: dict[str, Any], *, source_model: Path, private_model: Path, key_dir: Path
) -> bool:
    protocol = report.get("protocol")
    if not isinstance(protocol, dict) or protocol.get("validated") is not True:
        return False
    if protocol.get("run_roles") != [
        "baseline",
        "candidate",
        "candidate",
        "baseline",
        "candidate",
        "baseline",
        "baseline",
        "candidate",
    ]:
        return False
    if protocol.get("dtype") != "torch.bfloat16" or protocol.get("device") != "cuda":
        return False
    if not verify_file_identity(protocol.get("script", {})):
        return False
    tokenizer = protocol.get("tokenizer")
    if not isinstance(tokenizer, dict) or not verify_tokenizer_identity(tokenizer):
        return False
    artifacts = protocol.get("source_artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 8:
        return False
    runs: list[dict[str, Any]] = []
    for record in artifacts:
        if not isinstance(record, dict) or not verify_file_identity(record):
            return False
        run = read_optional(record["path"])
        if run is None:
            return False
        run_prov = run.get("provenance")
        if not isinstance(run_prov, dict) or run_prov.get("formal_run_binding") is not True:
            return False
        if not verify_model_identity(run_prov.get("model", {})):
            return False
        if not verify_file_identity(run_prov.get("prompts", {})):
            return False
        run_tokenizer = run_prov.get("tokenizer")
        if run_tokenizer != tokenizer or not verify_tokenizer_identity(run_tokenizer):
            return False
        if not verify_file_identity(run_prov.get("script", {})):
            return False
        if run.get("dtype") != protocol["dtype"] or run.get("device") != protocol["device"]:
            return False
        if (
            run_prov.get("dtype") != protocol["dtype"]
            or run_prov.get("device") != protocol["device"]
        ):
            return False
        role = run_prov.get("model_role")
        expected_model = source_model if role == "baseline" else private_model
        if role not in {"baseline", "candidate"} or not same_path(
            run_prov["model"]["path"], expected_model
        ):
            return False
        key = run_prov.get("key")
        if role == "baseline" and key is not None:
            return False
        if role == "candidate" and not (
            isinstance(key, dict)
            and verify_file_identity(key)
            and Path(key["path"]).resolve().parent == key_dir.resolve()
        ):
            return False
        runs.append(run_prov)
    return sorted(int(item["run_order"]) for item in runs) == list(range(1, 9))


def vma_provenance_ok(
    report: dict[str, Any], *, source_model: Path, private_model: Path, key_dir: Path
) -> bool:
    provenance = report.get("provenance")
    if not isinstance(provenance, dict) or provenance.get("formal_run_binding") is not True:
        return False
    if provenance.get("cache_bound_existing_unversioned") is not False:
        return False
    inputs = provenance.get("inputs")
    if not isinstance(inputs, dict):
        return False
    if not verify_model_identity(inputs.get("original", {})):
        return False
    if not verify_model_identity(inputs.get("private", {})):
        return False
    if not same_path(inputs["original"]["path"], source_model):
        return False
    if not same_path(inputs["private"]["path"], private_model):
        return False
    if inputs.get("key_manifest_sha256") != sha256_file(key_dir / "key_manifest.json"):
        return False
    if inputs.get("paper_key_sha256") != sha256_file(key_dir / "paper_key.safetensors"):
        return False
    if inputs.get("layers") != list(range(24)):
        return False
    if set(inputs.get("combinations", [])) != {
        "We_Wh",
        "We_Wq_We_WkT",
        "We_Wgate",
        "We_Wup",
        "Wdown_Wh",
    }:
        return False
    cache_dir = Path(report.get("prediction_cache_dir", ""))
    cache_manifest_path = cache_dir / "cache_manifest.json"
    if not cache_manifest_path.is_file():
        return False
    if sha256_file(cache_manifest_path) != provenance.get("cache_manifest_sha256"):
        return False
    cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
    input_digest = hashlib.sha256(
        json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if cache_manifest.get("input_sha256") != input_digest:
        return False
    expected_files = cache_manifest.get("prediction_files", {})
    actual_files = {path.name for path in cache_dir.glob("*.pt")}
    if actual_files != set(expected_files):
        return False
    return all(
        (cache_dir / name).is_file()
        and (cache_dir / name).stat().st_size == int(record["size"])
        and sha256_file(cache_dir / name) == record["sha256"]
        for name, record in expected_files.items()
    )


def vma_protocol_complete(report: dict[str, Any]) -> bool:
    expected = {"We_Wh", "We_Wq_We_WkT", "We_Wgate", "We_Wup", "Wdown_Wh"}
    try:
        return (
            set(report["results"]) == {"16384"}
            and report.get("layers") == list(range(24))
            and set(report.get("combinations", [])) == expected
            and set(report["results"]["16384"]) == expected
            and int(report["results"]["16384"]["We_Wh"]["vote_count"]) == 1
            and all(
                int(report["results"]["16384"][name]["vote_count"]) == 24
                for name in expected - {"We_Wh"}
            )
        )
    except (KeyError, TypeError, ValueError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Qwen2.5-0.5B acceptance evidence")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    acceptance = config["acceptance"]
    evidence = config["evidence"]
    max_drop = float(acceptance["maximum_absolute_accuracy_drop"])
    max_ttrsr = float(acceptance["maximum_ttrsr"])
    max_piirsr = float(acceptance["maximum_piirsr"])
    max_bleu4 = float(acceptance["maximum_bleu4"])
    max_cossim = float(acceptance["maximum_cossim"])
    max_latency = float(acceptance["maximum_latency_degradation"])
    min_top1 = float(acceptance["minimum_prefill_top1_agreement"])
    checks: list[dict[str, Any]] = []
    missing: list[str] = []
    source_model = Path(config["source_model"])
    private_model = Path(config["output_model"])
    key_dir = Path(config["key_dir"])

    profile = [
        ("embedding_head.alpha_e", config["embedding_head"]["alpha_e"], 1.0),
        ("embedding_head.alpha_h", config["embedding_head"]["alpha_h"], 0.2),
        ("linear.expansion_h", config["linear"]["expansion_h"], 128),
        ("linear.lambda", config["linear"]["lambda"], 0.3),
        ("attention.block_beta", config["attention"]["block_beta"], 8),
        ("attention.sampling_gamma", config["attention"]["sampling_gamma"], 1000.0),
        ("attention.algorithm2", config["attention"]["algorithm2"], True),
        ("attention.blockperm_mode", config["attention"]["blockperm_mode"], "gamma-corrected"),
        (
            "attention.rope_frequency_mode",
            config["attention"]["rope_frequency_mode"],
            "qwen-actual",
        ),
        (
            "linear.inverse_key_mode",
            config["linear"]["inverse_key_mode"],
            "shared_p_independent_compatible_right_inverses",
        ),
        ("rmsnorm.kappa_mode", config["rmsnorm"]["kappa_mode"], "paper-expectation"),
        ("checkpoint.dtype", config["checkpoint"]["dtype"], "bfloat16"),
    ]
    for field, actual, expected in profile:
        matches = (
            math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12)
            if isinstance(expected, (int, float)) and not isinstance(expected, bool)
            else actual == expected
        )
        checks.append(
            {
                "id": f"paper_profile.{field}",
                "actual": actual,
                "expected": expected,
                "pass": matches,
            }
        )

    for benchmark in ("mmlu", "ceval", "piqa"):
        path = evidence[f"{benchmark}_comparison"]
        report = read_optional(path)
        if report is None:
            add_missing(checks, missing, f"accuracy.{benchmark}", path)
            continue
        change = float(report["absolute_change"])
        provenance_ok = lm_comparison_provenance_ok(
            report,
            source_model=source_model,
            private_model=private_model,
            key_dir=key_dir,
        )
        interval = report.get("paired_change_95_percent_ci")
        ci_pass = (
            isinstance(interval, list)
            and len(interval) == 2
            and float(interval[0]) >= -max_drop
        )
        checks.append(
            {
                "id": f"accuracy.{benchmark}",
                "sample_len": report["sample_len"],
                "baseline": report["baseline"],
                "candidate": report["candidate"],
                "absolute_change": change,
                "paired_change_95_percent_ci": interval,
                "dtype_match": report.get("dtype_match"),
                "provenance_bound": provenance_ok,
                "maximum_drop": max_drop,
                "pass": change >= -max_drop
                and ci_pass
                and report.get("dtype_match") is True
                and provenance_ok,
            }
        )

    for benchmark in ("ifeval", "humaneval"):
        path = evidence[f"{benchmark}_comparison"]
        report = read_optional(path)
        if report is None:
            add_missing(checks, missing, f"accuracy.{benchmark}", path)
            continue
        provenance_ok = lm_comparison_provenance_ok(
            report,
            source_model=source_model,
            private_model=private_model,
            key_dir=key_dir,
        )
        if benchmark == "ifeval":
            for metric, value in report["metrics"].items():
                change = float(value["absolute_change"])
                interval = value.get("paired_change_95_percent_ci")
                checks.append(
                    {
                        "id": f"accuracy.ifeval.{metric}",
                        "absolute_change": change,
                        "paired_change_95_percent_ci": interval,
                        "provenance_bound": provenance_ok,
                        "pass": change >= -max_drop
                        and isinstance(interval, list)
                        and len(interval) == 2
                        and float(interval[0]) >= -max_drop
                        and provenance_ok,
                    }
                )
        else:
            change = float(report["absolute_change"])
            interval = report.get("paired_change_95_percent_ci")
            checks.append(
                {
                    "id": "accuracy.humaneval",
                    "absolute_change": change,
                    "paired_change_95_percent_ci": interval,
                    "provenance_bound": provenance_ok,
                    "pass": change >= -max_drop
                    and isinstance(interval, list)
                    and len(interval) == 2
                    and float(interval[0]) >= -max_drop
                    and provenance_ok,
                }
            )

    pupa_path = evidence["pupa_conservative_candidate"]
    pupa_payload = read_optional(pupa_path)
    if pupa_payload is None:
        add_missing(checks, missing, "privacy.vma", pupa_path)
    else:
        selected_size = max(int(size) for size in pupa_payload["results"])
        vma_provenance = vma_provenance_ok(
            pupa_payload,
            source_model=source_model,
            private_model=private_model,
            key_dir=key_dir,
        )
        protocol_complete = selected_size == 16384 and vma_protocol_complete(pupa_payload)
        checks.append(
            {
                "id": "privacy.vma.candidate_coverage",
                "candidate_size": selected_size,
                "unique_query_token_ids": pupa_payload["unique_text_and_pii_token_ids"],
                "protocol_complete": protocol_complete,
                "provenance_bound": vma_provenance,
                "pass": selected_size >= pupa_payload["unique_text_and_pii_token_ids"]
                and protocol_complete
                and vma_provenance,
            }
        )
        for attack, values in pupa_payload["results"][str(selected_size)].items():
            for metric, maximum in (
                ("ttrsr", max_ttrsr),
                ("piirsr", max_piirsr),
                ("bleu4", max_bleu4),
                ("cossim", max_cossim),
            ):
                actual = float(values[metric])
                checks.append(
                    {
                        "id": f"privacy.vma.{attack}.{metric}",
                        "actual": actual,
                        "maximum": maximum,
                        "pass": actual <= maximum
                        and protocol_complete
                        and vma_provenance,
                    }
                )

    for key, label in (
        ("gate_ia", "privacy.gate_ia"),
        ("attn_ia_proxy", "privacy.attn_ia_proxy"),
        ("ima", "privacy.ima"),
        ("isa", "privacy.isa"),
        ("tfma", "privacy.tfma"),
        ("sda", "privacy.sda"),
    ):
        path = evidence[key]
        report = read_optional(path)
        if report is None:
            add_missing(checks, missing, label, path)
            continue
        attack_scope = {
            "gate_ia": (source_model, private_model, key_dir),
            "attn_ia_proxy": (source_model, private_model, key_dir),
            "ima": (source_model, private_model, key_dir),
            "isa": (source_model, private_model, key_dir),
            "tfma": (source_model, None, key_dir),
            "sda": (source_model, None, None),
        }[key]
        provenance_ok = scoped_run_provenance_ok(
            report.get("provenance", {}),
            source_model=attack_scope[0],
            private_model=attack_scope[1],
            key_dir=attack_scope[2],
        )
        try:
            if key == "gate_ia":
                actual = float(report["top1_recovery_rate"])
                passed = (
                    report["attack"] == "Gate-IA"
                    and report["candidate_scope"] == "closed_set_random_tokens"
                    and int(report["sample_size"]) >= 512
                    and int(report["layer_count"]) == 24
                    and actual <= max_ttrsr
                )
                detail = {"top1_recovery_rate": actual, "maximum": max_ttrsr}
            elif key == "attn_ia_proxy":
                actual = float(report["top1_recovery_rate"])
                passed = bool(report.get("paper_exact")) and actual <= max_ttrsr
                detail = {
                    "top1_recovery_rate": actual,
                    "maximum": max_ttrsr,
                    "paper_exact": report.get("paper_exact"),
                    "reason": "a dimensionally valid proxy cannot satisfy the paper-exact gate",
                }
            elif key == "ima":
                actual = float(report["token_recovery_rate"])
                passed = (
                    report["attack"] == "IMA"
                    and report.get("target_key_used_for_training") is False
                    and report.get("paper_exact_scale_claimed") is True
                    and int(report["steps"]) >= 2000
                    and int(report["sample_size"]) >= 8192
                    and actual <= max_ttrsr
                )
                detail = {
                    "token_recovery_rate": actual,
                    "maximum": max_ttrsr,
                    "paper_exact_scale_claimed": report.get("paper_exact_scale_claimed"),
                }
            elif key == "isa":
                actual = float(report["ttrsr"])
                optimization_succeeded = all(
                    float(row["final_loss"]) < float(row["initial_loss"])
                    for row in report["per_prompt"]
                )
                passed = (
                    report.get("paper_exact") is True
                    and report.get("attacker_uses_target_key") is False
                    and int(report["prompt_count"]) >= 20
                    and int(report["steps"]) >= 100
                    and optimization_succeeded
                    and actual <= max_ttrsr
                )
                detail = {
                    "ttrsr": actual,
                    "maximum": max_ttrsr,
                    "optimization_succeeded": optimization_succeeded,
                    "paper_exact": report.get("paper_exact"),
                    "proxy_reason": report.get("proxy_reason"),
                }
            elif key == "tfma":
                curve = report["curve"]
                max_top10 = max(float(row["top10"]) for row in curve)
                passed = (
                    report["attack"] == "TFMA"
                    and max(int(row["observed_tokens"]) for row in curve) >= 100000
                    and max_top10 <= 0.20
                )
                detail = {
                    "maximum_top10_recovery_rate": max_top10,
                    "maximum": 0.20,
                    "largest_observation": max(int(row["observed_tokens"]) for row in curve),
                    "prior": report.get("prior"),
                }
            else:
                actual = float(report["bleu4"])
                passed = (
                    report["attack"] == "SDA"
                    and report.get("paper_exact_scale_claimed") is True
                    and report.get("true_tau_used_for_training") is False
                    and actual <= max_bleu4
                )
                detail = {
                    "bleu4": actual,
                    "maximum": max_bleu4,
                    "paper_exact_scale_claimed": report.get("paper_exact_scale_claimed"),
                }
            passed = passed and provenance_ok
            status = "VALIDATED" if passed else "FAILED"
            if key == "isa" and report.get("paper_exact") is not True:
                status = "DIAGNOSTIC_PROXY"
            checks.append(
                {
                    "id": label,
                    "artifact": path,
                    "status": status,
                    "provenance_bound": provenance_ok,
                    **detail,
                    "pass": passed,
                }
            )
        except (KeyError, TypeError, ValueError) as error:
            checks.append(
                {
                    "id": label,
                    "artifact": path,
                    "status": "INVALID_SCHEMA",
                    "error": str(error),
                    "pass": False,
                }
            )

    api_path = evidence["api"]
    api = read_optional(api_path)
    if api is None:
        add_missing(checks, missing, "api.private_roundtrip", api_path)
    else:
        api_provenance = scoped_run_provenance_ok(
            api.get("provenance", {}),
            source_model=source_model,
            private_model=private_model,
            key_dir=key_dir,
        )
        api_ok = (
            api["non_stream_status"] == 200
            and api["stream_equal"] is True
            and api["wrong_key_status"] == 400
            and api["recovered_nonempty"] is True
            and api["model_id"] == Path(config["output_model"]).name
            and api["key_id"] == config["key_id"]
            and api_provenance
        )
        checks.append(
            {
                "id": "api.private_roundtrip",
                "artifact": api_path,
                "provenance_bound": api_provenance,
                "pass": api_ok,
            }
        )

    checkpoint_path = evidence["checkpoint"]
    checkpoint = read_optional(checkpoint_path)
    if checkpoint is None:
        add_missing(checks, missing, "checkpoint.forward_generate_cache", checkpoint_path)
    else:
        checkpoint_provenance = scoped_run_provenance_ok(
            checkpoint.get("provenance", {}),
            source_model=source_model,
            private_model=private_model,
            key_dir=key_dir,
        )
        cache_ok = (
            checkpoint["plain_prefill_cache_length"]
            == checkpoint["private_prefill_cache_length"]
            and checkpoint["decode_cache_length"]
            == checkpoint["private_prefill_cache_length"] + 1
        )
        top1 = float(checkpoint["prefill_top1_agreement"])
        checks.extend(
            [
                {
                    "id": "checkpoint.kv_cache",
                    "actual": cache_ok,
                    "provenance_bound": checkpoint_provenance,
                    "pass": cache_ok and checkpoint_provenance,
                },
                {
                    "id": "checkpoint.prefill_top1_agreement",
                    "actual": top1,
                    "minimum": min_top1,
                    "provenance_bound": checkpoint_provenance,
                    "pass": top1 >= min_top1 and checkpoint_provenance,
                },
            ]
        )

    manifest = verify_manifest(Path(config["output_model"]))
    checks.append(
        {
            "id": "checkpoint.manifest",
            "checked": manifest.checked,
            "failures": manifest.failures,
            "pass": manifest.ok,
        }
    )

    performance_path = evidence["performance_comparison"]
    performance = read_optional(performance_path)
    if performance is None:
        add_missing(checks, missing, "performance.hf", performance_path)
    else:
        performance_provenance = performance_provenance_ok(
            performance,
            source_model=source_model,
            private_model=private_model,
            key_dir=key_dir,
        )
        for metric in ("ttft_ms", "tpot_ms"):
            for quantile in acceptance["performance_quantiles"]:
                value = performance["metrics"][metric][quantile]
                degradation = float(value["degradation"])
                interval = value.get("degradation_95_percent_ci")
                checks.append(
                    {
                        "id": f"performance.{metric}.{quantile}",
                        "artifact": str(performance_path),
                        "degradation": degradation,
                        "degradation_95_percent_ci": interval,
                        "provenance_bound": performance_provenance,
                        "pass": degradation <= max_latency
                        and isinstance(interval, list)
                        and len(interval) == 2
                        and float(interval[1]) <= max_latency
                        and performance_provenance,
                    }
                )

    payload = {
        "scope": config["claim_scope"],
        "config": str(args.config),
        "decision": "GO" if all(check["pass"] for check in checks) else "NO-GO",
        "corrected_profile_match": all(
            check["pass"] for check in checks if check["id"].startswith("paper_profile.")
        ),
        "paper_numeric_hyperparameters_match": all(
            check["pass"]
            for check in checks
            if check["id"]
            in {
                "paper_profile.embedding_head.alpha_e",
                "paper_profile.embedding_head.alpha_h",
                "paper_profile.linear.expansion_h",
                "paper_profile.linear.lambda",
                "paper_profile.attention.block_beta",
                "paper_profile.attention.sampling_gamma",
                "paper_profile.checkpoint.dtype",
            }
        ),
        "paper_literal_algorithm_match": False,
        "checks": checks,
        "missing_evidence": missing,
        "decision_rule": "GO requires every check to pass and no NOT_TESTED evidence",
        "excluded_scope": ["7B", "14B", "DeepSeek-671B", "multi-node deployment"],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
