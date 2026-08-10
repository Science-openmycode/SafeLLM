from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def load(relative: str) -> dict[str, Any]:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def accuracy_row(relative: str) -> dict[str, Any]:
    value = load(relative)
    return {
        "artifact": relative,
        "metric": value["metric"],
        "sample_len": value["sample_len"],
        "baseline": value["baseline"],
        "candidate": value["candidate"],
        "absolute_change": value["absolute_change"],
        "paired_change_95_percent_ci": value["paired_change_95_percent_ci"],
        "confidence_interval_method": value["confidence_interval_method"],
        "baseline_dtype": value["baseline_dtype"],
        "candidate_dtype": value["candidate_dtype"],
        "dtype_match": value["dtype_match"],
        "passes_client_drop_threshold": value["absolute_change"] >= -0.035,
    }


def main() -> None:
    accuracy = {
        "v15_corrected_paper_numeric_hyperparameters": {
            "mmlu": accuracy_row(
                "artifacts/accuracy/mmlu-paper-v15-complete-bf16-comparison.json"
            ),
            "ceval": accuracy_row(
                "artifacts/accuracy/ceval-paper-v15-complete-bf16-comparison.json"
            ),
            "piqa": accuracy_row(
                "artifacts/accuracy/piqa-paper-v15-complete-bf16-comparison.json"
            ),
        },
        "piqa_ablation": {
            "v17_no_noise_bf16": accuracy_row(
                "artifacts/accuracy/piqa-paper-v17-no-noise-bf16-comparison.json"
            ),
            "v16_low_noise_fp32": accuracy_row(
                "artifacts/accuracy/"
                "piqa-paper-v16-engineering-complete-fp32-comparison.json"
            ),
        },
    }

    vma_artifact = "artifacts/privacy/vma-pupa-paper-v15-complete-bf16-c16384.json"
    vma = load(vma_artifact)
    vma_rows: dict[str, Any] = {}
    for name, value in vma["results"]["16384"].items():
        vma_rows[name] = {
            "ttrsr": value["ttrsr"],
            "ttrsr_wilson_95_percent_ci": value["ttrsr_wilson_95_percent_ci"],
            "piirsr": value["piirsr"],
            "piirsr_wilson_95_percent_ci": value["piirsr_wilson_95_percent_ci"],
            "bleu4": value["bleu4"],
            "cossim": value["cossim"],
            "unique_mapping_recovery_rate": value["unique_mapping_recovery_rate"],
            "vote_count": value["vote_count"],
            "passes_client_thresholds": (
                value["ttrsr"] <= 0.15
                and value["piirsr"] <= 0.03
                and value["bleu4"] <= 2.5
                and value["cossim"] <= 0.5
            ),
        }

    verification_path = "artifacts/verification/paper-qwen05b-v15-complete-bf16.json"
    api_path = "artifacts/api/paper-qwen05b-v15-complete-bf16.json"
    verification = load(verification_path)
    api = load(api_path)
    tfma_path = "artifacts/privacy/tfma-paper-v15-complete-bf16.json"
    direct_path = "artifacts/privacy/direct-weight-match-paper-v15-complete-bf16.json"
    gate_path = "artifacts/privacy/gate-ia-paper-v15-complete-bf16-full-vocab.json"
    attn_path = "artifacts/privacy/attn-ia-paper-v15-complete-bf16-proxy.json"
    ima_path = "artifacts/privacy/ima-paper-v15-complete-bf16.json"
    isa_path = "artifacts/privacy/isa-paper-v15-complete-bf16.json"
    sda_path = "artifacts/privacy/sda-paper-v15-complete-bf16.json"
    performance_plain_path = "artifacts/performance/hf-plaintext-bf16-20x100.json"
    performance_private_path = (
        "artifacts/performance/hf-paper-v15-complete-bf16-20x100.json"
    )
    performance_comparison_path = (
        "artifacts/performance/hf-paper-v15-complete-bf16-comparison.json"
    )
    output = {
        "generated_by": "scripts/summarize_report_evidence.py",
        "scope": "Qwen2.5-0.5B-Instruct only",
        "paper_reference": {
            "source": "AloePri arXiv v2, Table 3, Qwen3-14B",
            "accuracy_percent": {
                "mmlu": {"plaintext": 87.64, "aloepri": 84.57},
                "ceval": {"plaintext": 87.35, "aloepri": 87.12},
                "piqa": {"plaintext": 89.72, "aloepri": 90.26},
            },
            "vma": {"ttrsr_percent": 25.05, "piirsr_percent": 1.62, "bleu4": 1.72},
        },
        "client_targets": {
            "accuracy_absolute_drop_max": 0.035,
            "ttrsr_max": 0.15,
            "ttrsr_excellent_max": 0.05,
            "piirsr_max": 0.03,
            "bleu4_max": 2.5,
            "cossim_max": 0.5,
            "ttft_relative_degradation_max": 0.15,
            "tpot_relative_degradation_max": 0.15,
        },
        "accuracy": accuracy,
        "vma_v15": {
            "artifact": vma_artifact,
            "dataset": vma["dataset"],
            "layers": vma["layers"],
            "unique_text_and_pii_token_ids": vma["unique_text_and_pii_token_ids"],
            "candidate_size": 16384,
            "rows": vma_rows,
            "runtime": vma["runtime"],
        },
        "tfma_v15": {"artifact": tfma_path, **load(tfma_path)},
        "direct_weight_match_v15": {"artifact": direct_path, **load(direct_path)},
        "other_privacy_v15": {
            "gate_ia": {"artifact": gate_path, **load(gate_path)},
            "attention_ia_proxy": {"artifact": attn_path, **load(attn_path)},
            "ima": {"artifact": ima_path, **load(ima_path)},
            "isa": {"artifact": isa_path, **load(isa_path)},
            "sda_mmlu_proxy": {"artifact": sda_path, **load(sda_path)},
        },
        "performance_v15": {
            "plaintext": {"artifact": performance_plain_path, **load(performance_plain_path)},
            "private": {
                "artifact": performance_private_path,
                **load(performance_private_path),
            },
            "comparison": {
                "artifact": performance_comparison_path,
                **load(performance_comparison_path),
            },
        },
        "functional_v15": {
            "verification_artifact": verification_path,
            "prefill_cache_lengths": {
                "plaintext": verification["plain_prefill_cache_length"],
                "private": verification["private_prefill_cache_length"],
            },
            "decode_cache_length": verification["decode_cache_length"],
            "prefill_top1_agreement": verification["prefill_top1_agreement"],
            "greedy_ids_equal": verification["greedy_ids_equal"],
            "api_artifact": api_path,
            "api": {
                "non_stream_status": api["non_stream_status"],
                "stream_equal": api["stream_equal"],
                "wrong_key_status": api["wrong_key_status"],
                "recovered_nonempty": api["recovered_nonempty"],
                "pass": api["pass"],
            },
        },
        "causal_statement": {
            "supported": (
                "Within Qwen2.5-0.5B, the paper-numeric embedding/head noise is the "
                "primary measured source of PIQA degradation."
            ),
            "not_supported": (
                "Parameter count is the cause; a second model size is required to identify "
                "that causal effect."
            ),
        },
        "not_tested_on_v15": [
            "HumanEval full generation (batch=1 exceeded the 20-minute local window)",
            "IFEval full generation (1/541 samples saved; not scored)",
            "paper medical-corpus TFMA/SDA (MMLU proxy only)",
            "paper-exact Attention-IA/IMA/SDA protocol",
            "formal vLLM performance",
        ],
    }
    target = ROOT / "artifacts/report/evidence_summary_v15.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
