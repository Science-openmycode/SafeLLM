from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from aloepri.packaging import inspect_server_package, sha256_file


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def file_record(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute the v31 BlockPerm functional evidence index"
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/product/qwen05b_v31_blockperm8.yaml")
    )
    parser.add_argument(
        "--package", type=Path, default=Path("data/packages/qwen05b-product-v31-blockperm8")
    )
    parser.add_argument(
        "--online-key",
        type=Path,
        default=Path("data/keys/qwen05b-product-v31-blockperm8-online"),
    )
    parser.add_argument(
        "--verification",
        type=Path,
        default=Path("artifacts/verification/qwen05b-product-v31-blockperm8"),
    )
    parser.add_argument(
        "--vllm", type=Path, default=Path("artifacts/vllm-smoke-v31-blockperm8-32tokens.json")
    )
    parser.add_argument(
        "--hf-vllm",
        type=Path,
        default=Path("artifacts/hf-vllm-greedy-v31-blockperm8-32tokens.json"),
    )
    parser.add_argument(
        "--sglang",
        type=Path,
        default=Path("artifacts/sglang-smoke-v31-blockperm8-32tokens.json"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "artifacts/verification/qwen05b-product-v31-blockperm8/"
            "functional_evidence_index.json"
        ),
    )
    args = parser.parse_args()

    formula_path = args.verification / "formula.json"
    generation_path = args.verification / "generation.json"
    summary_path = args.verification / "verification_summary.json"
    api_path = args.verification / "private_api.json"
    formula = load_json(formula_path)
    generation = load_json(generation_path)
    summary = load_json(summary_path)
    api = load_json(api_path)
    vllm = load_json(args.vllm)
    hf_vllm = load_json(args.hf_vllm)
    sglang = load_json(args.sglang)
    package_scan = inspect_server_package(args.package)

    checks = {
        "formula_overall_pass": formula.get("overall_pass") is True,
        "formula_tensor_count_is_316": formula.get("tensor_check_count") == 316,
        "formula_has_no_failed_tensors": formula.get("failed_tensors") == [],
        "formula_has_no_missing_coverage": formula.get("missing_formula_coverage") == [],
        "product_verification_pass": summary.get("functional_pass") is True,
        "greedy_ids_equal": generation.get("greedy_ids_equal") is True,
        "cache_advances_36_to_37": (
            generation.get("plain_prefill_cache_length") == 36
            and generation.get("private_prefill_cache_length") == 36
            and generation.get("decode_cache_length") == 37
        ),
        "vllm_generated_32_tokens": len(vllm.get("private_output_tokens", [])) == 32,
        "hf_vllm_sequence_equal": hf_vllm.get("sequence_equal") is True,
        "sglang_matches_vllm": sglang.get("expected_sequence_equal") is True,
        "sglang_generated_32_tokens": len(sglang.get("private_output_tokens", [])) == 32,
        "private_api_pass": api.get("pass") is True,
        "private_api_status_is_200": api.get("non_stream_status") == 200,
        "private_api_sse_matches_non_stream": api.get("stream_equal") is True,
        "private_api_wrong_key_is_400": api.get("wrong_key_status") == 400,
        "server_package_scan_pass": package_scan.get("pass") is True,
        "server_package_has_no_findings": package_scan.get("findings") == [],
    }

    inputs = [
        args.config,
        args.package / "aloepri_manifest.json",
        args.online_key / "manifest.json",
        formula_path,
        generation_path,
        summary_path,
        api_path,
        args.vllm,
        args.hf_vllm,
        args.sglang,
        Path("src/aloepri/models/modeling_aloepri_qwen2.py"),
        Path("src/aloepri/transforms/qwen_structural.py"),
        Path("src/aloepri/evidence.py"),
        Path("src/aloepri/packaging.py"),
        Path("src/aloepri/serving/vllm_qwen2.py"),
        Path("src/aloepri/serving/sglang_models/aloepri_qwen2.py"),
        Path("scripts/verify_paper_formula_checkpoint.py"),
        Path(__file__),
    ]
    payload = {
        "schema_version": 1,
        "scope": "Qwen2.5-0.5B v31 nontrivial BlockPerm functional evidence",
        "checks": checks,
        "functional_pass": all(checks.values()),
        "package_scan": package_scan,
        "inputs": [file_record(path) for path in inputs],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not payload["functional_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
