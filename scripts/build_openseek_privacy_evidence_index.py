from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Bind OpenSeek privacy evidence to code and model")
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--online-key-dir", type=Path, required=True)
    parser.add_argument("--offline-key-dir", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    paths = [
        args.package / "config.json",
        args.package / "model.safetensors.index.json",
        args.package / "aloepri_manifest.json",
        args.online_key_dir / "key.json",
        args.online_key_dir / "manifest.json",
        args.offline_key_dir / "key.json",
        args.offline_key_dir / "manifest.json",
        args.evidence_dir / "privacy-completeness-audit.json",
        args.evidence_dir / "product-privacy-boundary.json",
        args.evidence_dir / "privacy-attack-smoke-64.json",
        args.evidence_dir / "real-chat-smoke.json",
        Path("configs/product/openseek_small_v1_sft_paper_complete.yaml"),
        Path("src/aloepri/conversion/deepseek_streaming.py"),
        Path("src/aloepri/transforms/deepseek.py"),
        Path("src/aloepri/models/configuration_aloepri_deepseek_v3.py"),
        Path("src/aloepri/models/modeling_aloepri_deepseek_v3.py"),
        Path("src/aloepri/serving/hf_runtime.py"),
        Path("src/aloepri/serving/app.py"),
        Path("src/aloepri/packaging.py"),
        Path("src/aloepri/privacy/rmdp.py"),
        Path("scripts/audit_openseek_paper_complete.py"),
        Path("scripts/verify_product_privacy_boundary.py"),
        Path("scripts/run_openseek_local_smoke.py"),
        Path("scripts/run_openseek_privacy_attack_smoke.py"),
        Path("scripts/build_openseek_privacy_evidence_index.py"),
        Path("docs/OPENSEEK_PRIVACY_COMPLETENESS_AUDIT.md"),
        Path("docs/OPENSEEK_SMALL_V1_SFT_LOCAL_RUNBOOK.md"),
        Path("docs/VERSION_0.4.0.md"),
    ]
    paths.extend(sorted(Path("docs/paper_errata").glob("E*.md")))
    completeness = json.loads(
        (args.evidence_dir / "privacy-completeness-audit.json").read_text(
            encoding="utf-8"
        )
    )
    boundary = json.loads(
        (args.evidence_dir / "product-privacy-boundary.json").read_text(
            encoding="utf-8"
        )
    )
    payload = {
        "schema_version": 1,
        "model_id": "openseek-small-v1-sft-paper-complete",
        "mechanism_completeness_pass": completeness.get("pass") is True,
        "online_boundary_pass": boundary.get("all_pass") is True,
        "files": [record(path) for path in paths],
    }
    payload["pass"] = bool(
        payload["mechanism_completeness_pass"] and payload["online_boundary_pass"]
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(args.out.name + ".partial")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, args.out)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not payload["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
