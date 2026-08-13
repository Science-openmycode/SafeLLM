from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from aloepri.evidence import file_identity, model_identity


def _mismatch(name: str, actual: object, expected: object) -> None:
    if actual != expected:
        raise ValueError(f"{name} mismatch: actual={actual!r}, expected={expected!r}")


def validate_artifact(
    artifact: Path,
    *,
    model: Path,
    tokenizer: Path,
    key: Path | None,
    tasks: list[str],
    include_path: Path | None,
    dtype: str,
    batch_size: int,
    gpu_memory_fraction: float,
    run_script: Path,
) -> dict[str, Any]:
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    provenance = payload.get("run_provenance")
    if not isinstance(provenance, dict) or provenance.get("formal_run_binding") is not True:
        raise ValueError("artifact does not contain a formal run binding")
    _mismatch("model", provenance.get("model"), model_identity(model))
    _mismatch("tokenizer_path", provenance.get("tokenizer_path"), str(tokenizer.resolve()))
    _mismatch("key", provenance.get("key"), file_identity(key) if key else None)
    _mismatch("tasks", provenance.get("tasks"), tasks)
    _mismatch(
        "include_path",
        provenance.get("include_path"),
        str(include_path.resolve()) if include_path else None,
    )
    _mismatch("requested_dtype", provenance.get("requested_dtype"), dtype)
    _mismatch("batch_size", provenance.get("batch_size"), batch_size)
    _mismatch(
        "gpu_memory_fraction",
        provenance.get("gpu_memory_fraction"),
        gpu_memory_fraction,
    )
    _mismatch("run script", provenance.get("script"), file_identity(run_script))
    document_count = provenance.get("document_count")
    if not isinstance(document_count, int) or document_count <= 0:
        raise ValueError("artifact has no bound evaluation documents")
    if not isinstance(payload.get("results"), dict) or not payload["results"]:
        raise ValueError("artifact has no evaluation results")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--key", type=Path)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--include-path", type=Path)
    parser.add_argument("--dtype", choices=["auto", "float32", "bfloat16"], required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--gpu-memory-fraction", type=float, required=True)
    parser.add_argument("--run-script", type=Path, default=Path("scripts/run_lm_eval.py"))
    args = parser.parse_args()
    validate_artifact(
        args.artifact,
        model=args.model,
        tokenizer=args.tokenizer,
        key=args.key,
        tasks=args.tasks,
        include_path=args.include_path,
        dtype=args.dtype,
        batch_size=args.batch_size,
        gpu_memory_fraction=args.gpu_memory_fraction,
        run_script=args.run_script,
    )
    print(f"reusing validated lm-eval artifact: {args.artifact}")


if __name__ == "__main__":
    main()
