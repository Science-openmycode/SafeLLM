from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def archive_package_type(record: dict[str, Any]) -> str:
    explicit = record.get("package_type")
    if explicit:
        return str(explicit)
    name = Path(str(record["path"])).name
    if not name.lower().endswith(".tar.gz"):
        raise ValueError(f"unsupported archive name: {name}")
    return name[:-7].lower()


def safe_payload_path(root: Path, raw_path: object) -> Path:
    relative = Path(str(raw_path))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe payload path: {raw_path}")
    target = (root / relative).resolve()
    target.relative_to(root.resolve())
    return target


def verify_extracted_manifests(payload: dict[str, Any], root: Path) -> list[str]:
    failures: list[str] = []
    expected_types: set[str] = set()
    seen_payload_paths: set[str] = set()
    manifest_dir = root / "bundle_manifests"
    for archive_record in payload["archives"]:
        try:
            package_type = archive_package_type(archive_record)
        except (KeyError, TypeError, ValueError) as error:
            failures.append(f"archive-package-type:{error}")
            continue
        if package_type in expected_types:
            failures.append(f"duplicate-package-type:{package_type}")
            continue
        expected_types.add(package_type)
        manifest_path = manifest_dir / f"{package_type}.json"
        if not manifest_path.is_file():
            failures.append(f"missing-manifest:{package_type}")
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            failures.append(f"invalid-manifest:{package_type}:{error}")
            continue
        records = manifest.get("files")
        if manifest.get("schema_version") != 1 or manifest.get("package_type") != package_type:
            failures.append(f"manifest-identity:{package_type}")
            continue
        if not isinstance(records, list) or len(records) != int(archive_record["file_count"]):
            failures.append(f"manifest-file-count:{package_type}")
            continue
        for record in records:
            try:
                raw_path = str(record["path"])
                if raw_path in seen_payload_paths:
                    failures.append(f"duplicate-payload-path:{raw_path}")
                    continue
                seen_payload_paths.add(raw_path)
                target = safe_payload_path(root, raw_path)
                if not target.is_file():
                    failures.append(f"missing-payload:{raw_path}")
                elif target.stat().st_size != int(record["bytes"]):
                    failures.append(f"payload-size:{raw_path}")
                elif sha256(target) != record["sha256"]:
                    failures.append(f"payload-sha256:{raw_path}")
            except (KeyError, OSError, TypeError, ValueError) as error:
                failures.append(f"invalid-payload-record:{package_type}:{error}")
    if manifest_dir.is_dir():
        actual_types = {path.stem for path in manifest_dir.glob("*.json")}
        for package_type in sorted(actual_types - expected_types):
            failures.append(f"unexpected-manifest:{package_type}")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Qwen0.5B cloud archives before extract")
    parser.add_argument("--bundle-set", type=Path, required=True)
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument(
        "--extracted-root",
        type=Path,
        help="also verify every extracted payload file against its embedded manifest",
    )
    args = parser.parse_args()
    payload = json.loads(args.bundle_set.read_text(encoding="utf-8"))
    failures: list[str] = []
    for record in payload["archives"]:
        archive = args.archive_dir / Path(record["path"]).name
        if not archive.is_file():
            failures.append(f"missing:{archive.name}")
            continue
        if archive.stat().st_size != int(record["bytes"]):
            failures.append(f"size:{archive.name}")
            continue
        if sha256(archive) != record["sha256"]:
            failures.append(f"sha256:{archive.name}")
    if args.extracted_root is not None:
        failures.extend(verify_extracted_manifests(payload, args.extracted_root))
    result = {"pass": not failures, "failures": failures}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
