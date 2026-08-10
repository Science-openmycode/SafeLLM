# The audit record deliberately keeps evidence sentences as single JSON strings.
# ruff: noqa: E501

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from aloepri.packaging import inspect_server_package


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def requirement(
    identifier: str,
    paper_location: str,
    paper_operation: str,
    implementation: str,
    status: str,
    evidence: object,
    note: str,
) -> dict[str, object]:
    return {
        "id": identifier,
        "paper_location": paper_location,
        "paper_operation": paper_operation,
        "implementation": implementation,
        "status": status,
        "evidence": evidence,
        "note": note,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit one Qwen2.5-0.5B artifact against every applicable paper step"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--key-dir", type=Path, required=True)
    parser.add_argument("--server-package", type=Path, required=True)
    parser.add_argument("--formula-evidence", type=Path, required=True)
    parser.add_argument("--layerwise-evidence", type=Path, required=True)
    parser.add_argument("--greedy-evidence", type=Path, required=True)
    parser.add_argument("--runtime-evidence", type=Path, required=True)
    parser.add_argument("--privacy-evidence", type=Path, required=True)
    parser.add_argument("--http-evidence", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    config_path = args.checkpoint / "config.json"
    key_metadata_path = args.key_dir / "key.json"
    key_tensor_path = args.key_dir / "paper_key.safetensors"
    config = load_json(config_path)
    metadata = config.get("aloepri")
    if not isinstance(metadata, dict):
        raise ValueError("checkpoint config has no AloePri metadata")
    key_metadata = load_json(key_metadata_path)
    formula = load_json(args.formula_evidence)
    layerwise = load_json(args.layerwise_evidence)
    greedy = load_json(args.greedy_evidence)
    runtime = load_json(args.runtime_evidence)
    privacy = load_json(args.privacy_evidence)
    http = load_json(args.http_evidence)
    package = inspect_server_package(args.server_package)

    model_id = str(metadata["model_id"])
    key_id = str(metadata["key_id"])
    if key_metadata.get("model_id") != model_id or key_metadata.get("key_id") != key_id:
        raise ValueError("checkpoint and key metadata identifiers do not match")
    if formula.get("private_config_sha256") != sha256_file(config_path):
        raise ValueError("formula evidence is not bound to the current checkpoint config")
    if formula.get("paper_key_sha256") != sha256_file(key_tensor_path):
        raise ValueError("formula evidence is not bound to the current offline key")

    alpha_e = float(metadata["alpha_e"])
    alpha_h = float(metadata["alpha_h"])
    block_beta = int(metadata["attention_block_beta"])
    rms_mode = str(metadata["rms_mode"])
    uvo_condition_max = metadata.get("uvo_condition_max")
    formula_pass = bool(formula.get("overall_pass"))
    requirements = [
        requirement(
            "M01",
            "PDF p.8, Sec. 5.2.2",
            "Use secret vocabulary permutation tau for input/output IDs and both vocabulary matrices.",
            "src/aloepri/transforms/vocab.py; src/aloepri/conversion/paper_qwen2.py",
            "ACTIVE_PAPER_FORMULA",
            {"vocabulary_round_trip_pass": formula.get("vocabulary_round_trip_pass")},
            "The saved embedding and LM-head rows are reconstructed in private vocabulary order.",
        ),
        requirement(
            "M02",
            "PDF pp.8-9, Sec. 5.2.2",
            "Add independent Gaussian noise scaled by each source matrix standard deviation.",
            "src/aloepri/transforms/paper_noise.py",
            (
                "ACTIVE_PAPER_FORMULA"
                if alpha_e > 0.0 or alpha_h > 0.0
                else "IMPLEMENTED_DISABLED_IN_THIS_ARTIFACT"
            ),
            {"alpha_e": alpha_e, "alpha_h": alpha_h},
            "v29 is the no-noise functional-equivalence artifact; it is not the paper-default security artifact.",
        ),
        requirement(
            "M03",
            "PDF p.8, Algorithm 1",
            "Construct d to d+2h P/Q matrices with P Q = I.",
            "src/aloepri/transforms/paper_key_matrix.py",
            "ACTIVE_PAPER_CORRECTED",
            {
                "algorithm1_pass": formula.get("algorithm1_pass"),
                "role_pq_relative_errors": formula.get("role_pq_relative_errors"),
            },
            "C rows and D columns use the only dimensionally valid null-space interpretation; see E01/E02.",
        ),
        requirement(
            "M04",
            "PDF pp.8-9, Sec. 5.2.2",
            "Transform Embedding with P, LM Head with a compatible Q, and permute rows by tau.",
            "src/aloepri/conversion/paper_qwen2.py",
            "ACTIVE_PAPER_FORMULA",
            {"all_tensor_formulas_pass": formula.get("all_tensor_formulas_pass")},
            "HF row-major storage is transposed relative to the paper notation; the saved tensors match the adapted formulas exactly.",
        ),
        requirement(
            "M05",
            "PDF p.9, Algorithm 2",
            "Apply RoPE-commuting Q/K maps, paired inverse scaling, dense Gaussian V/O maps, and inverse O map.",
            "src/aloepri/transforms/qwen_structural.py",
            "ACTIVE_NUMERICALLY_STABILIZED",
            {
                "algorithm2_pass": formula.get("algorithm2_pass"),
                "uvo_condition_max": uvo_condition_max,
                "attention_compute_dtype": metadata.get("attention_compute_dtype"),
            },
            "Uvo is sampled from the paper Gaussian then condition-number rejected; Attention is FP64. Both are disclosed numerical constraints, not literal bf16 settings.",
        ),
        requirement(
            "M06",
            "PDF p.9, Algorithm 2 BlockPerm",
            "Permute RoPE frequency blocks using beta/gamma.",
            "src/aloepri/transforms/qwen_structural.py:make_dynamic_rope_block_order",
            (
                "IMPLEMENTED_DISABLED_DUE_PAPER_ERROR"
                if block_beta == 1
                else "ACTIVE_PAPER_CORRECTED_BUT_NOT_FUNCTION_PRESERVING"
            ),
            {"beta": block_beta, "mode": metadata.get("attention_blockperm_mode")},
            "A nontrivial frequency permutation does not commute with standard Qwen RoPE; v29 fixes beta=1. See E03-E06/E10/E16.",
        ),
        requirement(
            "M07",
            "PDF p.9, Sec. 5.2.3 Inter-head Transformation",
            "Synchronize KV-head permutation with each seven-query GQA group and permute Q/O heads inside groups.",
            "src/aloepri/transforms/qwen_structural.py:make_attention_key",
            "ACTIVE_PAPER_FORMULA",
            {"algorithm2_pass": formula.get("algorithm2_pass"), "q_heads": 14, "kv_heads": 2},
            "The full formula audit checks all 24 layers for GQA group and within-group alignment.",
        ),
        requirement(
            "M08",
            "PDF p.10, Sec. 5.2.4",
            "Use one FFN permutation and reciprocal scaling across gate/up/down projections.",
            "src/aloepri/transforms/qwen_structural.py:transform_mlp_scaled",
            "ACTIVE_PAPER_FORMULA_DISTRIBUTION_UNSPECIFIED",
            {"algorithm2_pass": formula.get("algorithm2_pass")},
            "The algebra is exact; the paper does not specify the random scale distribution, so the implementation records its log-uniform range.",
        ),
        requirement(
            "M09",
            "PDF p.10, Sec. 5.2.5",
            "Compensate RMSNorm under the private residual coordinates.",
            "src/aloepri/models/modeling_aloepri_qwen2.py:AloePriMetricRMSNorm",
            (
                "ACTIVE_PAPER_CORRECTED"
                if rms_mode == "exact_metric"
                else "ACTIVE_PAPER_APPROXIMATION"
            ),
            {"rms_mode": rms_mode, "rms_formula_status": formula.get("rms_formula_status")},
            "v29 computes the original RMS exactly with G=Q Q^T; the paper's scalar kappa is not exact for a general rectangular P. The server-package disclosure reports this derived metric.",
        ),
        requirement(
            "M10",
            "PDF pp.10-11, Sec. 5.2.5 and residual composition",
            "Keep both residual branches in the same P coordinate and use compatible right inverses before transformed projections.",
            "src/aloepri/conversion/paper_qwen2.py; src/aloepri/models/modeling_aloepri_qwen2.py",
            "ACTIVE_PAPER_CORRECTED",
            {
                "formula_pass": formula_pass,
                "layerwise_operator_count": layerwise.get("operator_count"),
                "layerwise_pass_count": layerwise.get("pass_count"),
                "final_logits_nrmse": next(
                    (
                        item.get("nrmse")
                        for item in layerwise.get("records", [])
                        if item.get("operator") == "logits"
                    ),
                    None,
                ),
            },
            "All 292 saved tensors reconstruct exactly; runtime comparison has three local 1e-5 threshold misses but final generation remains identical.",
        ),
        requirement(
            "M11",
            "PDF p.10, Sec. 5.3",
            "Client tokenizes, applies tau, server infers only on the private model, client applies inverse tau.",
            "src/aloepri/client/sdk.py; src/aloepri/serving/app.py; src/aloepri/serving/hf_runtime.py",
            "ACTIVE_ENGINEERING_CORRECTION",
            {
                "privacy_boundary_all_pass": privacy.get("all_pass"),
                "http_sse_round_trip_equal": http.get("round_trip_equal"),
            },
            "The wire format is token IDs because Decode(tau(ids)) then re-tokenize is not bijective for Qwen. See E12.",
        ),
        requirement(
            "M12",
            "PDF p.12, Sec. 5.4 and Theorem 4",
            "Apply M1 and compute the RmDP privacy budget.",
            "src/aloepri/privacy/rmdp.py",
            "MIXED_PAPER_FORMULA_AND_TOKENWISE_SUBSTITUTE",
            {
                "default_privacy_mode": "permutation",
                "theorem4_uses_vocabulary_size_n": True,
            },
            "Theorem 4 uses vocabulary size n in Z_n^l and S_n. A factorial small-vocabulary exact sequence oracle verifies M1; the Qwen product composes the exact length-one mechanism token by token and labels it rmdp-tokenwise.",
        ),
        requirement(
            "M13",
            "PDF pp.26-27, Appendix D.1",
            "Evaluate VMA, Gate-IA, Attn-IA, ISA, IMA, TFMA, and SDA.",
            "src/aloepri/attacks; scripts/run_*; scripts/train_*",
            "PARTIAL_PROTOCOL_REPRODUCTION",
            {"bound_to_v29_security_artifact": False},
            "Several paper datasets/training protocols are absent and Attn-IA is dimensionally invalid; proxy code is explicitly labelled and no proxy is counted as a paper-exact result.",
        ),
        requirement(
            "M14",
            "PDF pp.9-10, MLA and MoE paragraphs",
            "Transform MLA low-rank projections and MoE router/experts when those modules exist.",
            "No Qwen2.5-0.5B module path",
            "NOT_APPLICABLE_TO_MODEL_ARCHITECTURE",
            {"model_type": config.get("model_type"), "num_experts": config.get("num_experts")},
            "Qwen2.5-0.5B is dense GQA and has neither MLA nor MoE; this is excluded from the 0.5B denominator rather than reported as implemented.",
        ),
    ]

    status_counts = Counter(str(item["status"]) for item in requirements)
    functional_runtime_pass = bool(
        formula_pass
        and greedy.get("all_exact")
        and runtime.get("greedy_ids_equal")
        and runtime.get("next_token_equal_after_inverse")
        and runtime.get("decode_input_is_tau_plain_next")
        and privacy.get("all_pass")
        and http.get("round_trip_equal")
        and package.get("pass")
    )
    literal_mismatches = [
        "Algorithm 1 uses the documented dimensionally valid C/D null-space repair",
        "the online protocol sends private token IDs instead of non-bijective private text",
    ]
    if alpha_e != 1.0 or alpha_h != 0.2:
        literal_mismatches.append("noise parameters differ from Table 10 defaults")
    if block_beta != 8:
        literal_mismatches.append("BlockPerm beta differs from Table 10 default beta=8")
    if rms_mode != "paper_kappa":
        literal_mismatches.append("RMSNorm uses the exact metric correction, not scalar kappa")
    if uvo_condition_max is not None:
        literal_mismatches.append("Uvo uses condition-number rejection")
    if config.get("dtype") != "bfloat16":
        literal_mismatches.append("checkpoint storage is not the paper experiment's bf16")
    if metadata.get("attention_compute_dtype") == "float64":
        literal_mismatches.append("Attention is evaluated in FP64")
    literal_paper_pass = bool(functional_runtime_pass and not literal_mismatches)
    payload = {
        "schema_version": 1,
        "scope": "Qwen2.5-0.5B applicable AloePri method and v29 artifact",
        "model_id": model_id,
        "key_id": key_id,
        "conclusions": {
            "saved_tensor_formula_reconstruction_pass": formula_pass,
            "functional_runtime_pass": functional_runtime_pass,
            "literal_paper_parameter_and_protocol_pass": literal_paper_pass,
            "literal_paper_mismatches": literal_mismatches,
            "security_release_pass": False,
            "security_release_reason": "v29 has alpha_e=alpha_h=0 and no v29-bound full attack/accuracy acceptance suite",
        },
        "measured": {
            "formula_tensor_checks": formula.get("tensor_check_count"),
            "formula_failed_tensors": formula.get("failed_tensors"),
            "layerwise_operator_checks": layerwise.get("operator_count"),
            "layerwise_operator_passes_at_1e-5": layerwise.get("pass_count"),
            "greedy_prompts": greedy.get("prompt_count"),
            "greedy_exact_prompts": greedy.get("exact_prompt_count"),
            "generated_tokens": greedy.get("prompt_count", 0) * greedy.get("max_new_tokens", 0),
            "prefill_top1_agreement": runtime.get("prefill_top1_agreement"),
            "private_peak_gpu_bytes": runtime.get("private_phase_peak_gpu_bytes"),
        },
        "status_counts": dict(sorted(status_counts.items())),
        "requirements": requirements,
        "server_package_inspection": package,
        "bindings": {
            "checkpoint_config": {
                "path": str(config_path.resolve()),
                "sha256": sha256_file(config_path),
            },
            "offline_key": {
                "path": str(key_tensor_path.resolve()),
                "sha256": sha256_file(key_tensor_path),
            },
            "formula_evidence": {
                "path": str(args.formula_evidence.resolve()),
                "sha256": sha256_file(args.formula_evidence),
            },
            "layerwise_evidence": {
                "path": str(args.layerwise_evidence.resolve()),
                "sha256": sha256_file(args.layerwise_evidence),
            },
            "greedy_evidence": {
                "path": str(args.greedy_evidence.resolve()),
                "sha256": sha256_file(args.greedy_evidence),
            },
            "runtime_evidence": {
                "path": str(args.runtime_evidence.resolve()),
                "sha256": sha256_file(args.runtime_evidence),
            },
            "privacy_evidence": {
                "path": str(args.privacy_evidence.resolve()),
                "sha256": sha256_file(args.privacy_evidence),
            },
            "http_evidence": {
                "path": str(args.http_evidence.resolve()),
                "sha256": sha256_file(args.http_evidence),
            },
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "conclusions": payload["conclusions"],
                "status_counts": payload["status_counts"],
                "out": str(args.out),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if not functional_runtime_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
