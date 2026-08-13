from __future__ import annotations

import json

from scripts.verify_qwen05b_cloud_bundle_set import sha256, verify_extracted_manifests


def test_extracted_bundle_manifest_verifies_content_and_rejects_tampering(tmp_path) -> None:
    root = tmp_path / "AloePri"
    manifest_dir = root / "bundle_manifests"
    payload_file = root / "scripts" / "entry.py"
    manifest_dir.mkdir(parents=True)
    payload_file.parent.mkdir()
    payload_file.write_bytes(b"content")
    package_type = "aloepri-qwen05b-v47-source"
    manifest = {
        "schema_version": 1,
        "package_type": package_type,
        "files": [
            {
                "path": "scripts/entry.py",
                "bytes": payload_file.stat().st_size,
                "sha256": sha256(payload_file),
            }
        ],
    }
    (manifest_dir / f"{package_type}.json").write_text(json.dumps(manifest), encoding="utf-8")
    bundle_set = {
        "archives": [
            {
                "package_type": package_type,
                "path": "AloePri-qwen05b-v47-source.tar.gz",
                "file_count": 1,
            }
        ]
    }

    assert verify_extracted_manifests(bundle_set, root) == []
    payload_file.write_bytes(b"tampered")
    assert verify_extracted_manifests(bundle_set, root) == [
        "payload-size:scripts/entry.py"
    ]


def test_extracted_bundle_manifest_rejects_unsafe_path(tmp_path) -> None:
    root = tmp_path / "AloePri"
    manifest_dir = root / "bundle_manifests"
    manifest_dir.mkdir(parents=True)
    package_type = "aloepri-qwen05b-v47-source"
    manifest = {
        "schema_version": 1,
        "package_type": package_type,
        "files": [{"path": "../escape", "bytes": 0, "sha256": "0" * 64}],
    }
    (manifest_dir / f"{package_type}.json").write_text(json.dumps(manifest), encoding="utf-8")
    bundle_set = {
        "archives": [
            {
                "package_type": package_type,
                "path": "AloePri-qwen05b-v47-source.tar.gz",
                "file_count": 1,
            }
        ]
    }

    failures = verify_extracted_manifests(bundle_set, root)
    assert len(failures) == 1
    assert failures[0].startswith("invalid-payload-record:")
