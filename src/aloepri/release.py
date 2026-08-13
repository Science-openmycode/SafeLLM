from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import yaml

from aloepri.packaging import inspect_server_package

RELEASE_MANIFEST = Path("manifests/release_manifest.json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _link_or_copy(source: str, destination: str) -> str:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    return destination


def _runtime_config(config_path: Path) -> dict[str, Any]:
    source = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(source, dict):
        raise ValueError("release config must contain a mapping")
    server = source.get("server", {})
    if not isinstance(server, dict):
        raise ValueError("release config server section must be a mapping")
    allowed_server = {
        key: server[key]
        for key in (
            "host",
            "port",
            "device",
            "dtype",
            "max_input_tokens",
            "max_output_tokens",
            "max_request_bytes",
            "bearer_token_env",
            "tls_certfile",
            "tls_keyfile",
            "tls_terminated_by_proxy",
        )
        if key in server
    }
    return {
        "schema_version": 1,
        "model_id": source["model_id"],
        "key_id": source["key_id"],
        "output_model": "server",
        "server": allowed_server,
    }


def _records(root: Path) -> list[dict[str, object]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.relative_to(root) != RELEASE_MANIFEST
    ]


def _copy_unique(paths: list[Path], destination: Path) -> None:
    names: set[str] = set()
    for path in paths:
        if path.name in names:
            raise ValueError(f"duplicate release filename: {path.name}")
        names.add(path.name)
        shutil.copy2(path, destination / path.name)


def build_product_release(
    *,
    output: Path,
    server_package: Path,
    config_path: Path,
    wheel: Path | None = None,
    reports: list[Path] | None = None,
    evidence: list[Path] | None = None,
) -> dict[str, object]:
    inspection = inspect_server_package(server_package)
    if not inspection["pass"]:
        raise ValueError(f"server package failed inspection: {inspection['findings']}")
    if output.exists():
        raise FileExistsError(output)
    partial = output.with_name(f"{output.name}.partial")
    if partial.exists():
        raise FileExistsError(partial)
    for name in (
        "client",
        "server",
        "configs",
        "online-key-template",
        "manifests",
        "reports",
        "evidence",
    ):
        (partial / name).mkdir(parents=True, exist_ok=True)
    try:
        shutil.copytree(
            server_package,
            partial / "server",
            dirs_exist_ok=True,
            copy_function=_link_or_copy,
        )
        if wheel is not None:
            shutil.copy2(wheel, partial / "client" / wheel.name)
        runtime_config = _runtime_config(config_path)
        (partial / "configs" / "runtime.yaml").write_text(
            yaml.safe_dump(runtime_config, sort_keys=False), encoding="utf-8"
        )
        key_example = {
            "schema_version": 1,
            "model_id": runtime_config["model_id"],
            "key_id": runtime_config["key_id"],
            "vocab_size": "REPLACE_AFTER_KEY_PROVISIONING",
            "vocab_file": "online_key.safetensors",
        }
        (partial / "online-key-template" / "key.example.json").write_text(
            json.dumps(key_example, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (partial / "online-key-template" / "README.md").write_text(
            "Place the provisioned client-only online key here. Never copy it to server/.\n",
            encoding="utf-8",
        )
        _copy_unique(reports or [], partial / "reports")
        _copy_unique(evidence or [], partial / "evidence")
        (partial / "README.md").write_text(
            "# AloePri private-model product release\n\n"
            "Install the wheel in `client/`, provision the online key only on the client, "
            "then run `aloepri serve --config configs/runtime.yaml` on the server and "
            "`aloepri chat --server <HTTPS URL> --key-dir <client key dir>`.\n",
            encoding="utf-8",
        )
        (partial / "SECURITY.md").write_text(
            "The server directory must not contain tau, inverse_tau, P/Q, random seeds, "
            "plaintext prompts, or the original checkpoint. Remote clients require HTTPS "
            "and Bearer authentication.\n",
            encoding="utf-8",
        )
        records = _records(partial)
        manifest = {
            "schema_version": 1,
            "package_type": "aloepri-private-model-release",
            "release_version": "0.5.0",
            "release_status": "CODE_COMPLETE_MOCK_CLOUD_PASS",
            "environment": "mock-cloud",
            "real_cloud_validated": False,
            "real_671b_executed": False,
            "model_id": runtime_config["model_id"],
            "key_id": runtime_config["key_id"],
            "server_package_inspection": inspection,
            "files": records,
        }
        manifest_path = partial / RELEASE_MANIFEST
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(partial, output)
    except BaseException:
        if partial.exists():
            shutil.rmtree(partial)
        raise
    return inspect_product_release(output)


def inspect_product_release(release: Path) -> dict[str, object]:
    manifest_path = release / RELEASE_MANIFEST
    findings: list[str] = []
    if not manifest_path.is_file():
        return {"pass": False, "findings": ["release manifest is missing"]}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return {"pass": False, "findings": [f"invalid release manifest: {error}"]}
    records = manifest.get("files", [])
    if not isinstance(records, list):
        return {"pass": False, "findings": ["release manifest files must be a list"]}
    safe_records: list[dict[str, object]] = []
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            findings.append("release manifest contains an invalid file record")
            continue
        relative = Path(record["path"])
        if relative.is_absolute() or ".." in relative.parts or relative == RELEASE_MANIFEST:
            findings.append(f"release manifest contains an unsafe path: {record['path']}")
            continue
        safe_records.append(record)
    declared = {str(record["path"]) for record in safe_records}
    actual = {
        path.relative_to(release).as_posix()
        for path in release.rglob("*")
        if path.is_file() and path.relative_to(release) != RELEASE_MANIFEST
    }
    if declared != actual:
        findings.append("release file set differs from manifest")
    for record in safe_records:
        path = release / str(record["path"])
        if not path.is_file():
            continue
        expected_bytes = record.get("bytes")
        expected_sha256 = record.get("sha256")
        if not isinstance(expected_bytes, int) or not isinstance(expected_sha256, str):
            findings.append(f"release manifest identity is invalid: {record['path']}")
            continue
        if path.stat().st_size != expected_bytes or _sha256(path) != expected_sha256:
            findings.append(f"release file identity mismatch: {record['path']}")
    server_inspection = inspect_server_package(release / "server")
    if not server_inspection["pass"]:
        server_findings = server_inspection["findings"]
        if isinstance(server_findings, list):
            findings.extend(str(item) for item in server_findings)
    for forbidden in ("online_key.safetensors", "offline_master_key.safetensors"):
        if any(path.name == forbidden for path in release.rglob("*")):
            findings.append(f"release contains provisioned key material: {forbidden}")
    return {
        "schema_version": 1,
        "release": str(release.resolve()),
        "pass": not findings,
        "file_count": len(records),
        "findings": findings,
        "server_package_inspection": server_inspection,
    }
