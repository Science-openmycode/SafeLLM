from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    payload = {
        "schema_version": 1,
        "control_checkpoint": "qwen2.5-0.5b-full-exact",
        "control_label": "square_control",
        "paper_checkpoint": "paper_expand_h128",
        "components": {
            "token_permutation": {"control": "implemented", "paper": "implemented"},
            "embedding_head_noise": {
                "control": "fixed_std_legacy",
                "paper": "implemented_alpha_times_weight_std",
            },
            "algorithm_1_pq": {"control": "fixture_only", "paper": "implemented"},
            "expanded_qwen_checkpoint": {"control": "not_implemented", "paper": "pending"},
            "attention_algorithm_2": {"control": "partial_exact_subset", "paper": "pending"},
            "ffn_scaling_and_pq": {"control": "permutation_only", "paper": "pending"},
            "expanded_rmsnorm": {"control": "not_implemented", "paper": "pending"},
            "vllm": {"control": "runtime_verified", "paper": "pending"},
        },
        "attack_labels": {
            "legacy_VMA_NN_IMA": "embedding_cosine_nearest_neighbor",
            "legacy_TFMA": "embedding_cosine_plus_frequency_penalty",
            "legacy_SDA": "known_plaintext_coverage",
            "paper_vma": "pending",
            "paper_tfma": "pending",
            "paper_sda": "pending",
            "paper_ima": "pending",
        },
    }
    output = ROOT / "artifacts" / "current-implementation-coverage.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
