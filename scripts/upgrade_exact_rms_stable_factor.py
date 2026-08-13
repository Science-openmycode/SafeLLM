from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: object) -> None:
    partial = path.with_name(path.name + ".partial")
    if partial.exists():
        raise FileExistsError(f"stale partial file exists: {partial}")
    partial.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(partial, path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a v47-compatible checkpoint with stable exact-metric RMS"
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    partial_root = args.output.with_name(args.output.name + ".partial")
    if partial_root.exists():
        raise FileExistsError(f"stale partial directory exists: {partial_root}")

    config = json.loads((args.source / "config.json").read_text(encoding="utf-8"))
    if config.get("aloepri_rms_mode") != "exact_metric":
        raise ValueError("source checkpoint is not exact-metric RMS")
    if config.get("aloepri_rms_representation", "gram") != "gram":
        raise ValueError("source checkpoint is not the legacy Gram representation")

    index = json.loads(
        (args.source / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    weight_map = index["weight_map"]
    if "aloepri_rms_factor" in weight_map:
        raise ValueError("source checkpoint already contains aloepri_rms_factor")
    metric_shard_name = weight_map["aloepri_rms_metric"]
    metric_shard = args.source / metric_shard_name

    with safe_open(metric_shard, framework="pt", device="cpu") as handle:
        metric = handle.get_tensor("aloepri_rms_metric").to(torch.float64)
        shard_metadata = handle.metadata()
    with safe_open(args.key, framework="pt", device="cpu") as handle:
        q = handle.get_tensor("q").to(torch.float64)
    expected_shape = (int(config["hidden_size"]), int(config["plain_hidden_size"]))
    if tuple(q.shape) != expected_shape:
        raise ValueError(f"unexpected Q shape: {tuple(q.shape)} != {expected_shape}")

    exact_metric = q @ q.mT
    maximum_stored_error = float((metric - exact_metric).abs().max())
    if maximum_stored_error > 1.0e-6:
        raise ValueError(
            f"stored Gram matrix does not match offline Q: {maximum_stored_error:.3e}"
        )
    eigenvalues, eigenvectors = torch.linalg.eigh(exact_metric)
    rank = int(config["plain_hidden_size"])
    positive_values = eigenvalues[-rank:].clamp_min(0.0)
    factor = (
        eigenvectors[:, -rank:] * positive_values.sqrt().unsqueeze(0)
    ).contiguous()
    reconstruction_error = float((factor @ factor.mT - exact_metric).abs().max())
    if reconstruction_error > 1.0e-10:
        raise ValueError(f"stable factor reconstruction failed: {reconstruction_error:.3e}")

    shutil.copytree(args.source, partial_root)
    output_shard = partial_root / metric_shard_name
    tensors = load_file(output_shard, device="cpu")
    tensors["aloepri_rms_factor"] = factor
    shard_partial = output_shard.with_name(output_shard.name + ".partial")
    save_file(tensors, shard_partial, metadata=shard_metadata)
    os.replace(shard_partial, output_shard)

    config["aloepri_rms_representation"] = "stable_factor"
    aloepri_metadata = config.get("aloepri")
    if isinstance(aloepri_metadata, dict):
        aloepri_metadata["rms_representation"] = "stable-factor"
        paper_alignment = aloepri_metadata.get("paper_alignment")
        if isinstance(paper_alignment, dict):
            paper_alignment["rmsnorm"] = "corrected-exact-stable-factor"
    write_json_atomic(partial_root / "config.json", config)

    weight_map["aloepri_rms_factor"] = metric_shard_name
    index["metadata"]["total_size"] = int(index["metadata"]["total_size"]) + (
        factor.numel() * factor.element_size()
    )
    write_json_atomic(partial_root / "model.safetensors.index.json", index)

    manifest_path = partial_root / "aloepri_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["metadata"]["rms_representation"] = "stable-factor"
    manifest["metadata"]["paper_alignment"]["rmsnorm"] = (
        "corrected-exact-stable-factor"
    )
    manifest["metadata"]["rms_stable_factor"] = {
        "source": "eigendecomposition-of-Q-QT",
        "shape": list(factor.shape),
        "dtype": "float64",
        "stored_gram_maximum_absolute_error": maximum_stored_error,
        "factor_reconstruction_maximum_absolute_error": reconstruction_error,
    }
    manifest["files"] = [
        {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(partial_root.iterdir())
        if path.is_file() and path != manifest_path
    ]
    write_json_atomic(manifest_path, manifest)
    os.replace(partial_root, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "factor_shape": list(factor.shape),
                "stored_gram_maximum_absolute_error": maximum_stored_error,
                "factor_reconstruction_maximum_absolute_error": reconstruction_error,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
