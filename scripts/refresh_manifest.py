from __future__ import annotations

import argparse
import json
from pathlib import Path

from aloepri.conversion.vocab_checkpoint import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh hashes after an audited metadata repair")
    parser.add_argument("model_dir", type=Path)
    args = parser.parse_args()
    manifest_path = args.model_dir / "aloepri_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"] = [
        {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(args.model_dir.iterdir())
        if path.is_file() and path != manifest_path
    ]
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
