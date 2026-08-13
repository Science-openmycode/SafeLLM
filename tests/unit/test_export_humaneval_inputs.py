from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from scripts.export_humaneval_inputs import FIELDS, canonical_digest


def test_reuse_existing_humaneval_export_without_network(tmp_path: Path) -> None:
    rows = [
        {field: f"{field}-{index}" for field in FIELDS}
        for index in range(164)
    ]
    output = tmp_path / "humaneval.jsonl"
    output.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest = tmp_path / "humaneval.manifest.json"
    digest = canonical_digest(rows)
    manifest.write_text(
        json.dumps(
            {
                "row_count": 164,
                "content_sha256": digest,
                "file_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/export_humaneval_inputs.py",
            "--out",
            str(output),
            "--manifest",
            str(manifest),
            "--expected-content-sha256",
            digest,
            "--reuse-existing",
        ],
        check=True,
    )
