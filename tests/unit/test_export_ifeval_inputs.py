from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


def test_reuse_existing_ifeval_manifest_without_network(tmp_path: Path) -> None:
    rows = [
        {
            "key": 1,
            "prompt": "test",
            "instruction_id_list": [],
            "kwargs": [],
        }
    ]
    canonical = json.dumps(
        rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    digest = hashlib.sha256(canonical).hexdigest()
    path = tmp_path / "ifeval.json"
    path.write_text(
        json.dumps(
            {
                "dataset": "google/IFEval",
                "content_sha256": digest,
                "rows": rows,
            }
        ),
        encoding="utf-8",
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/export_ifeval_inputs.py",
            "--out",
            str(path),
            "--expected-content-sha256",
            digest,
            "--reuse-existing",
        ],
        check=True,
    )
