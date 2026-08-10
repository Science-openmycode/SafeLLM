from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"


def load(relative: str) -> dict[str, Any]:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def metric_pair(filename: str, task: str, metric: str) -> dict[str, Any]:
    original = load(f"artifacts/accuracy/{filename}-original.json")["results"][task][metric]
    private = load(f"artifacts/accuracy/{filename}-private.json")["results"][task][metric]
    drop = float(original) - float(private)
    return {"original": original, "private": private, "absolute_drop": drop, "pass": drop <= 0.035}


def main() -> None:
    exact = load("artifacts/accuracy/full-exact-fp32.json")
    privacy = load("artifacts/privacy/suite-full-noise.json")
    plain_perf = load("artifacts/performance/original-fp32.json")
    private_perf = load("artifacts/performance/private-fp32.json")
    humaneval_original = load("artifacts/accuracy/humaneval-original.json")
    humaneval_private = load("artifacts/accuracy/humaneval-private.json")
    ttft = private_perf["ttft_ms"]["p50"] / plain_perf["ttft_ms"]["p50"] - 1
    tpot = private_perf["tpot_ms"]["p50"] / plain_perf["tpot_ms"]["p50"] - 1
    checks: dict[str, Any] = {
        "fp32_generation_equivalence": {
            "value": exact["exact_generation_rate"],
            "required": 1.0,
            "pass": exact["exact_generation_rate"] == 1.0,
        },
        "accuracy": {
            "mmlu_abstract_algebra": metric_pair("mmlu", "mmlu_abstract_algebra", "acc,none"),
            "piqa": metric_pair("piqa-ceval", "piqa", "acc,none"),
            "ceval_high_school_mathematics": metric_pair(
                "piqa-ceval", "ceval-valid_high_school_mathematics", "acc,none"
            ),
            "ifeval_prompt_strict": metric_pair("ifeval", "ifeval", "prompt_level_strict_acc,none"),
        },
        "performance": {
            "ttft_p50_degradation": ttft,
            "tpot_p50_degradation": tpot,
            "required_max": 0.15,
            "pass": ttft <= 0.15 and tpot <= 0.15,
        },
        "privacy": {
            "vma_recovery_rate": privacy["VMA_NN_IMA"]["mapping_recovery_rate"],
            "tfma_recovery_rate": privacy["TFMA"]["mapping_recovery_rate"],
            "pass": False,
            "reason": (
                "VMA/TFMA recover 99.21875% of sampled mappings; the proposed transform "
                "does not meet a meaningful privacy gate."
            ),
        },
        "vllm_runtime": {
            "pass": (ARTIFACTS / "vllm-smoke.json").exists(),
            "version": "0.11.1",
            "runtime": "WSL2 Ubuntu 24.04, VLLM_USE_V1=0",
        },
        "humaneval": {
            "original": humaneval_original["results"]["humaneval"]["pass@1,create_test"],
            "private": humaneval_private["results"]["humaneval"]["pass@1,create_test"],
            "sample_len": 3,
            "pass": False,
            "reason": "Observed absolute drop is 33.33 percentage points on a 3-sample smoke run.",
        },
    }
    accuracy_pass = all(item["pass"] for item in checks["accuracy"].values())
    acceptance_pass = (
        checks["fp32_generation_equivalence"]["pass"]
        and accuracy_pass
        and checks["performance"]["pass"]
        and checks["privacy"]["pass"]
        and checks["vllm_runtime"]["pass"]
        and checks["humaneval"]["pass"]
    )
    report = {
        "model": "Qwen2.5-0.5B-Instruct",
        "implementation_complete": True,
        "acceptance_pass": acceptance_pass,
        "checks": checks,
        "decision": "NO-GO" if not acceptance_pass else "GO",
    }
    output = ARTIFACTS / "0.5b-completion-audit.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
