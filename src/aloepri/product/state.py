from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from aloepri.product.paths import prepare_product_state
from aloepri.product.redaction import redact_secrets


class ProductJobStatus(StrEnum):
    CREATED = "CREATED"
    AWAITING_CONFIRMATION = "AWAITING_CONFIRMATION"
    PREFLIGHT = "PREFLIGHT"
    RUNNING = "RUNNING"
    PAUSING = "PAUSING"
    PAUSED = "PAUSED"
    VERIFYING = "VERIFYING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class ProductPhase(StrEnum):
    METADATA = "METADATA"
    DOWNLOADING = "DOWNLOADING"
    SOURCE_VERIFY = "SOURCE_VERIFY"
    CONVERTING = "CONVERTING"
    PRIVATE_VERIFY = "PRIVATE_VERIFY"
    UPLOADING = "UPLOADING"
    REMOTE_VERIFY = "REMOTE_VERIFY"
    FINALIZING = "FINALIZING"


class ShardStatus(StrEnum):
    PENDING = "PENDING"
    DOWNLOADING = "DOWNLOADING"
    SOURCE_VERIFIED = "SOURCE_VERIFIED"
    CONVERTING = "CONVERTING"
    PRIVATE_VERIFIED = "PRIVATE_VERIFIED"
    UPLOADING = "UPLOADING"
    REMOTE_COMMITTED = "REMOTE_COMMITTED"
    SOURCE_CLEANED = "SOURCE_CLEANED"
    FAILED = "FAILED"


class DeploymentStatus(StrEnum):
    DRAFT = "DRAFT"
    UPLOADING = "UPLOADING"
    INSTALLING = "INSTALLING"
    STARTING = "STARTING"
    HEALTHY = "HEALTHY"
    STOPPED = "STOPPED"
    DEGRADED = "DEGRADED"
    FAILED = "FAILED"
    ROLLING_BACK = "ROLLING_BACK"
    ROLLED_BACK = "ROLLED_BACK"
    DELETING = "DELETING"


_JOB_TRANSITIONS: dict[ProductJobStatus, frozenset[ProductJobStatus]] = {
    ProductJobStatus.CREATED: frozenset(
        {ProductJobStatus.AWAITING_CONFIRMATION, ProductJobStatus.CANCELLED}
    ),
    ProductJobStatus.AWAITING_CONFIRMATION: frozenset(
        {ProductJobStatus.PREFLIGHT, ProductJobStatus.CANCELLED}
    ),
    ProductJobStatus.PREFLIGHT: frozenset(
        {ProductJobStatus.RUNNING, ProductJobStatus.FAILED, ProductJobStatus.CANCELLED}
    ),
    ProductJobStatus.RUNNING: frozenset(
        {
            ProductJobStatus.PAUSING,
            ProductJobStatus.VERIFYING,
            ProductJobStatus.FAILED,
            ProductJobStatus.CANCELLED,
        }
    ),
    ProductJobStatus.PAUSING: frozenset(
        {
            ProductJobStatus.PAUSED,
            ProductJobStatus.FAILED,
            ProductJobStatus.CANCELLED,
        }
    ),
    ProductJobStatus.PAUSED: frozenset(
        {ProductJobStatus.RUNNING, ProductJobStatus.CANCELLED}
    ),
    ProductJobStatus.VERIFYING: frozenset(
        {ProductJobStatus.COMPLETED, ProductJobStatus.FAILED}
    ),
    ProductJobStatus.FAILED: frozenset(
        {ProductJobStatus.PREFLIGHT, ProductJobStatus.CANCELLED}
    ),
    ProductJobStatus.COMPLETED: frozenset(),
    ProductJobStatus.CANCELLED: frozenset(),
}


class ProductStore:
    """Versioned product state shared by the desktop applications and CLI.

    Legacy JobStore tables are intentionally left untouched so existing 0.5.0
    jobs and checkpoints remain importable during the compatibility cycle.
    """

    SCHEMA_VERSION = 2

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or prepare_product_state()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=30000")
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(
                """
                BEGIN EXCLUSIVE;
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS product_jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    plan_json TEXT NOT NULL,
                    progress_json TEXT NOT NULL,
                    error_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS model_records (
                    model_id TEXT PRIMARY KEY,
                    source_type TEXT NOT NULL,
                    source_ref TEXT NOT NULL,
                    revision TEXT,
                    adapter_id TEXT,
                    support_status TEXT NOT NULL,
                    local_path TEXT,
                    inspection_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS license_acceptances (
                    model_id TEXT NOT NULL,
                    revision TEXT NOT NULL,
                    license_sha256 TEXT NOT NULL,
                    accepted_at TEXT NOT NULL,
                    PRIMARY KEY(model_id, revision, license_sha256)
                );
                CREATE TABLE IF NOT EXISTS product_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL REFERENCES product_jobs(job_id) ON DELETE CASCADE,
                    created_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS product_shards (
                    job_id TEXT NOT NULL REFERENCES product_jobs(job_id) ON DELETE CASCADE,
                    shard_id TEXT NOT NULL,
                    source_name TEXT NOT NULL,
                    private_name TEXT,
                    status TEXT NOT NULL,
                    source_bytes INTEGER,
                    source_sha256 TEXT,
                    private_bytes INTEGER,
                    private_sha256 TEXT,
                    uploaded_bytes INTEGER NOT NULL DEFAULT 0,
                    remote_sha256 TEXT,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(job_id, shard_id)
                );
                CREATE TABLE IF NOT EXISTS server_profiles (
                    server_id TEXT PRIMARY KEY,
                    display_name TEXT NOT NULL,
                    host TEXT NOT NULL,
                    port INTEGER NOT NULL,
                    username TEXT NOT NULL,
                    auth_type TEXT NOT NULL,
                    credential_ref TEXT,
                    private_key_path TEXT,
                    host_key_fingerprint TEXT,
                    sudo_mode TEXT NOT NULL,
                    model_root TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS product_deployments (
                    deployment_id TEXT PRIMARY KEY,
                    server_id TEXT NOT NULL REFERENCES server_profiles(server_id),
                    job_id TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    key_id TEXT NOT NULL,
                    version_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    remote_port INTEGER NOT NULL,
                    previous_version_id TEXT,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS health_checks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    deployment_id TEXT NOT NULL REFERENCES product_deployments(deployment_id),
                    checked_at TEXT NOT NULL,
                    passed INTEGER NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS credential_refs (
                    credential_id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    protected_path TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            connection.execute(
                "INSERT OR IGNORE INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                (self.SCHEMA_VERSION, self._now()),
            )

    def create_job(
        self,
        job_id: str,
        plan: dict[str, Any],
        *,
        phase: ProductPhase = ProductPhase.METADATA,
    ) -> dict[str, Any]:
        now = self._now()
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO product_jobs
                   VALUES (?, ?, ?, ?, '{}', NULL, ?, ?)""",
                (
                    job_id,
                    ProductJobStatus.CREATED.value,
                    phase.value,
                    json.dumps(redact_secrets(plan)),
                    now,
                    now,
                ),
            )
            self._event(connection, job_id, ProductJobStatus.CREATED, phase, {})
        return self.get_job(job_id)

    def _event(
        self,
        connection: sqlite3.Connection,
        job_id: str,
        status: ProductJobStatus,
        phase: ProductPhase,
        payload: dict[str, Any],
    ) -> None:
        connection.execute(
            """INSERT INTO product_events
               (job_id, created_at, status, phase, payload_json) VALUES (?, ?, ?, ?, ?)""",
            (
                job_id,
                self._now(),
                status.value,
                phase.value,
                json.dumps(redact_secrets(payload)),
            ),
        )

    def get_job(self, job_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM product_jobs WHERE job_id=?", (job_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown product job: {job_id}")
        payload = dict(row)
        payload["plan"] = json.loads(payload.pop("plan_json"))
        payload["progress"] = json.loads(payload.pop("progress_json"))
        payload["error"] = (
            None if payload["error_json"] is None else json.loads(payload["error_json"])
        )
        payload.pop("error_json")
        return payload

    def list_jobs(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT job_id FROM product_jobs ORDER BY created_at DESC"
            ).fetchall()
        return [self.get_job(str(row["job_id"])) for row in rows]

    def events(self, job_id: str, *, after_id: int = 0) -> list[dict[str, Any]]:
        self.get_job(job_id)
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT * FROM product_events
                   WHERE job_id=? AND id>? ORDER BY id""",
                (job_id, after_id),
            ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            payload = dict(row)
            payload["payload"] = json.loads(payload.pop("payload_json"))
            result.append(payload)
        return result

    def transition_job(
        self,
        job_id: str,
        status: ProductJobStatus,
        *,
        phase: ProductPhase | None = None,
        progress: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        current = self.get_job(job_id)
        current_status = ProductJobStatus(current["status"])
        if status not in _JOB_TRANSITIONS[current_status]:
            raise ValueError(f"illegal product job transition: {current_status} -> {status}")
        selected_phase = phase or ProductPhase(current["phase"])
        selected_progress = progress if progress is not None else current["progress"]
        with self._connect() as connection:
            connection.execute(
                """UPDATE product_jobs SET status=?, phase=?, progress_json=?, error_json=?,
                   updated_at=? WHERE job_id=?""",
                (
                    status.value,
                    selected_phase.value,
                    json.dumps(redact_secrets(selected_progress)),
                    None if error is None else json.dumps(redact_secrets(error)),
                    self._now(),
                    job_id,
                ),
            )
            self._event(connection, job_id, status, selected_phase, selected_progress)
        return self.get_job(job_id)

    def put_shard(
        self,
        job_id: str,
        shard_id: str,
        source_name: str,
        *,
        status: ShardStatus = ShardStatus.PENDING,
        **fields: Any,
    ) -> dict[str, Any]:
        allowed = {
            "private_name",
            "source_bytes",
            "source_sha256",
            "private_bytes",
            "private_sha256",
            "uploaded_bytes",
            "remote_sha256",
            "metadata",
        }
        unknown = set(fields) - allowed
        if unknown:
            raise ValueError(f"unknown shard fields: {sorted(unknown)}")
        current = self.get_shard(job_id, shard_id, required=False) or {}
        values = {
            "private_name": None,
            "source_bytes": None,
            "source_sha256": None,
            "private_bytes": None,
            "private_sha256": None,
            "uploaded_bytes": 0,
            "remote_sha256": None,
            "metadata": {},
            **current,
            **fields,
        }
        with self._connect() as connection:
            connection.execute(
                """INSERT OR REPLACE INTO product_shards
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    job_id,
                    shard_id,
                    source_name,
                    values["private_name"],
                    status.value,
                    values["source_bytes"],
                    values["source_sha256"],
                    values["private_bytes"],
                    values["private_sha256"],
                    int(values["uploaded_bytes"]),
                    values["remote_sha256"],
                    json.dumps(values["metadata"]),
                    self._now(),
                ),
            )
        result = self.get_shard(job_id, shard_id)
        assert result is not None
        return result

    def get_shard(
        self, job_id: str, shard_id: str, *, required: bool = True
    ) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM product_shards WHERE job_id=? AND shard_id=?",
                (job_id, shard_id),
            ).fetchone()
        if row is None:
            if required:
                raise KeyError(f"unknown shard: {job_id}/{shard_id}")
            return None
        payload = dict(row)
        payload["metadata"] = json.loads(payload.pop("metadata_json"))
        return payload

    def list_shards(self, job_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT shard_id FROM product_shards WHERE job_id=? ORDER BY shard_id",
                (job_id,),
            ).fetchall()
        return [self.get_shard(job_id, str(row["shard_id"])) for row in rows]  # type: ignore[misc]

    def add_server(self, profile: dict[str, Any]) -> dict[str, Any]:
        now = self._now()
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO server_profiles VALUES
                   (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    profile["server_id"],
                    profile["display_name"],
                    profile["host"],
                    int(profile.get("port", 22)),
                    profile.get("username", "root"),
                    profile.get("auth_type", "private_key"),
                    profile.get("credential_ref"),
                    profile.get("private_key_path"),
                    profile.get("host_key_fingerprint"),
                    profile.get("sudo_mode", "root"),
                    profile.get("model_root", "/opt/yinbian"),
                    json.dumps(profile.get("metadata", {})),
                    now,
                    now,
                ),
            )
        return self.get_server(str(profile["server_id"]))

    def update_server(self, server_id: str, **fields: Any) -> dict[str, Any]:
        allowed = {
            "display_name",
            "host",
            "port",
            "username",
            "auth_type",
            "credential_ref",
            "private_key_path",
            "host_key_fingerprint",
            "sudo_mode",
            "model_root",
            "metadata",
        }
        unknown = set(fields) - allowed
        if unknown:
            raise ValueError(f"unknown server fields: {sorted(unknown)}")
        current = self.get_server(server_id)
        current.update(fields)
        with self._connect() as connection:
            connection.execute(
                """UPDATE server_profiles SET display_name=?, host=?, port=?, username=?,
                   auth_type=?, credential_ref=?, private_key_path=?, host_key_fingerprint=?,
                   sudo_mode=?, model_root=?, metadata_json=?, updated_at=? WHERE server_id=?""",
                (
                    current["display_name"],
                    current["host"],
                    int(current["port"]),
                    current["username"],
                    current["auth_type"],
                    current.get("credential_ref"),
                    current.get("private_key_path"),
                    current.get("host_key_fingerprint"),
                    current["sudo_mode"],
                    current["model_root"],
                    json.dumps(current.get("metadata", {})),
                    self._now(),
                    server_id,
                ),
            )
        return self.get_server(server_id)

    def get_server(self, server_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM server_profiles WHERE server_id=?", (server_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown server: {server_id}")
        payload = dict(row)
        payload["metadata"] = json.loads(payload.pop("metadata_json"))
        return payload

    def list_servers(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT server_id FROM server_profiles ORDER BY display_name"
            ).fetchall()
        return [self.get_server(str(row["server_id"])) for row in rows]

    def remove_server(self, server_id: str) -> None:
        with self._connect() as connection:
            linked = connection.execute(
                "SELECT COUNT(*) FROM product_deployments WHERE server_id=?", (server_id,)
            ).fetchone()[0]
            if int(linked):
                raise ValueError("server has deployment records and cannot be removed")
            cursor = connection.execute(
                "DELETE FROM server_profiles WHERE server_id=?", (server_id,)
            )
            if cursor.rowcount != 1:
                raise KeyError(f"unknown server: {server_id}")

    def put_model(self, record: dict[str, Any]) -> dict[str, Any]:
        now = self._now()
        with self._connect() as connection:
            connection.execute(
                """INSERT OR REPLACE INTO model_records VALUES
                   (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    record["model_id"],
                    record["source_type"],
                    record["source_ref"],
                    record.get("revision"),
                    record.get("adapter_id"),
                    record["support_status"],
                    record.get("local_path"),
                    json.dumps(record.get("inspection", {})),
                    record.get("created_at", now),
                    now,
                ),
            )
        return self.get_model(str(record["model_id"]))

    def accept_license(
        self, model_id: str, revision: str, license_sha256: str
    ) -> dict[str, str]:
        accepted_at = self._now()
        with self._connect() as connection:
            connection.execute(
                """INSERT OR IGNORE INTO license_acceptances
                   (model_id, revision, license_sha256, accepted_at) VALUES (?, ?, ?, ?)""",
                (model_id, revision, license_sha256, accepted_at),
            )
        return {
            "model_id": model_id,
            "revision": revision,
            "license_sha256": license_sha256,
            "accepted_at": accepted_at,
        }

    def has_license_acceptance(
        self, model_id: str, revision: str, license_sha256: str
    ) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                """SELECT 1 FROM license_acceptances
                   WHERE model_id=? AND revision=? AND license_sha256=?""",
                (model_id, revision, license_sha256),
            ).fetchone()
        return row is not None

    def get_model(self, model_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM model_records WHERE model_id=?", (model_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown model: {model_id}")
        payload = dict(row)
        payload["inspection"] = json.loads(payload.pop("inspection_json"))
        return payload

    def list_models(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT model_id FROM model_records ORDER BY updated_at DESC"
            ).fetchall()
        return [self.get_model(str(row["model_id"])) for row in rows]

    def put_deployment(self, record: dict[str, Any]) -> dict[str, Any]:
        now = self._now()
        with self._connect() as connection:
            connection.execute(
                """INSERT OR REPLACE INTO product_deployments VALUES
                   (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    record["deployment_id"],
                    record["server_id"],
                    record["job_id"],
                    record["model_id"],
                    record["model_version"],
                    record["key_id"],
                    record["version_id"],
                    record.get("status", DeploymentStatus.DRAFT.value),
                    int(record["remote_port"]),
                    record.get("previous_version_id"),
                    json.dumps(record.get("metadata", {})),
                    record.get("created_at", now),
                    now,
                ),
            )
        return self.get_deployment(str(record["deployment_id"]))

    def get_deployment(self, deployment_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM product_deployments WHERE deployment_id=?", (deployment_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown deployment: {deployment_id}")
        payload = dict(row)
        payload["metadata"] = json.loads(payload.pop("metadata_json"))
        return payload

    def list_deployments(self, *, healthy_only: bool = False) -> list[dict[str, Any]]:
        query = "SELECT deployment_id FROM product_deployments"
        parameters: tuple[object, ...] = ()
        if healthy_only:
            query += " WHERE status=?"
            parameters = (DeploymentStatus.HEALTHY.value,)
        query += " ORDER BY created_at DESC"
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [self.get_deployment(str(row["deployment_id"])) for row in rows]

    def set_deployment_status(
        self, deployment_id: str, status: DeploymentStatus
    ) -> dict[str, Any]:
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE product_deployments SET status=?, updated_at=? WHERE deployment_id=?",
                (status.value, self._now(), deployment_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"unknown deployment: {deployment_id}")
        return self.get_deployment(deployment_id)

    def remove_deployment(self, deployment_id: str) -> None:
        self.get_deployment(deployment_id)
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM health_checks WHERE deployment_id=?", (deployment_id,)
            )
            connection.execute(
                "DELETE FROM product_deployments WHERE deployment_id=?", (deployment_id,)
            )

    def record_health(
        self, deployment_id: str, passed: bool, payload: dict[str, Any]
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO health_checks
                   (deployment_id, checked_at, passed, payload_json) VALUES (?, ?, ?, ?)""",
                (deployment_id, self._now(), int(passed), json.dumps(payload)),
            )
