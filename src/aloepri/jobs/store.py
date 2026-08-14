from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any


class JobState(StrEnum):
    CREATED = "CREATED"
    PREFLIGHT = "PREFLIGHT"
    DOWNLOADING = "DOWNLOADING"
    CONVERTING = "CONVERTING"
    UPLOADING = "UPLOADING"
    VERIFYING = "VERIFYING"
    READY_TO_DEPLOY = "READY_TO_DEPLOY"
    DEPLOYING = "DEPLOYING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    ROLLED_BACK = "ROLLED_BACK"


_TRANSITIONS: dict[JobState, frozenset[JobState]] = {
    JobState.CREATED: frozenset({JobState.PREFLIGHT, JobState.CANCELLED}),
    JobState.PREFLIGHT: frozenset({JobState.DOWNLOADING, JobState.CONVERTING, JobState.FAILED}),
    JobState.DOWNLOADING: frozenset(
        {JobState.CONVERTING, JobState.PAUSED, JobState.FAILED, JobState.CANCELLED}
    ),
    JobState.CONVERTING: frozenset(
        {
            JobState.UPLOADING,
            JobState.VERIFYING,
            JobState.PAUSED,
            JobState.FAILED,
            JobState.CANCELLED,
        }
    ),
    JobState.UPLOADING: frozenset(
        {JobState.VERIFYING, JobState.PAUSED, JobState.FAILED, JobState.CANCELLED}
    ),
    JobState.VERIFYING: frozenset({JobState.READY_TO_DEPLOY, JobState.FAILED}),
    JobState.READY_TO_DEPLOY: frozenset({JobState.DEPLOYING, JobState.CANCELLED}),
    JobState.DEPLOYING: frozenset({JobState.RUNNING, JobState.FAILED, JobState.ROLLED_BACK}),
    JobState.RUNNING: frozenset({JobState.ROLLED_BACK, JobState.FAILED}),
    JobState.PAUSED: frozenset(
        {JobState.DOWNLOADING, JobState.CONVERTING, JobState.UPLOADING, JobState.CANCELLED}
    ),
    JobState.FAILED: frozenset({JobState.PREFLIGHT, JobState.CANCELLED}),
    JobState.CANCELLED: frozenset(),
    JobState.ROLLED_BACK: frozenset(),
}


def default_state_path() -> Path:
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return base / "AloePri" / "state.db"
    return Path.home() / ".local" / "share" / "aloepri" / "state.db"


class JobStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or default_state_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    state TEXT NOT NULL,
                    plan_json TEXT NOT NULL,
                    progress_json TEXT NOT NULL,
                    resume_state TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    state TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS tiles (
                    job_id TEXT NOT NULL,
                    tensor_name TEXT NOT NULL,
                    tile_index INTEGER NOT NULL,
                    sha256 TEXT NOT NULL,
                    completed_at TEXT NOT NULL,
                    PRIMARY KEY (job_id, tensor_name, tile_index)
                );
                CREATE TABLE IF NOT EXISTS deployments (
                    deployment_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    key_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    environment TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat()

    def create(self, job_id: str, plan: dict[str, Any]) -> None:
        now = self._now()
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO jobs VALUES (?, ?, ?, ?, NULL, NULL, ?, ?)",
                (job_id, JobState.CREATED.value, json.dumps(plan), "{}", now, now),
            )
            self._event(connection, job_id, JobState.CREATED, {})

    def _event(
        self,
        connection: sqlite3.Connection,
        job_id: str,
        state: JobState,
        payload: dict[str, Any],
    ) -> None:
        connection.execute(
            "INSERT INTO events(job_id, created_at, state, payload_json) VALUES (?, ?, ?, ?)",
            (job_id, self._now(), state.value, json.dumps(payload)),
        )

    def get(self, job_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown job: {job_id}")
        payload = dict(row)
        payload["plan"] = json.loads(payload.pop("plan_json"))
        payload["progress"] = json.loads(payload.pop("progress_json"))
        return payload

    def list_jobs(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute("SELECT job_id FROM jobs ORDER BY created_at DESC").fetchall()
        return [self.get(str(row["job_id"])) for row in rows]

    def events(self, job_id: str, *, after_id: int = 0) -> list[dict[str, Any]]:
        self.get(job_id)
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM events WHERE job_id = ? AND id > ? ORDER BY id",
                (job_id, after_id),
            ).fetchall()
        events: list[dict[str, Any]] = []
        for row in rows:
            payload = dict(row)
            payload["payload"] = json.loads(payload.pop("payload_json"))
            events.append(payload)
        return events

    def transition(
        self,
        job_id: str,
        target: JobState,
        *,
        progress: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        current_payload = self.get(job_id)
        current = JobState(current_payload["state"])
        if target not in _TRANSITIONS[current]:
            raise ValueError(f"illegal job transition: {current.value} -> {target.value}")
        resume_state = (
            current.value if target == JobState.PAUSED else current_payload["resume_state"]
        )
        progress_payload = progress if progress is not None else current_payload["progress"]
        with self._connect() as connection:
            connection.execute(
                """UPDATE jobs SET state = ?, progress_json = ?, resume_state = ?, error = ?,
                   updated_at = ? WHERE job_id = ?""",
                (
                    target.value,
                    json.dumps(progress_payload),
                    resume_state,
                    error,
                    self._now(),
                    job_id,
                ),
            )
            self._event(connection, job_id, target, progress_payload)

    def update_progress(self, job_id: str, progress: dict[str, Any]) -> None:
        """Persist progress without pretending that the job changed phase."""
        payload = self.get(job_id)
        state = JobState(payload["state"])
        with self._connect() as connection:
            connection.execute(
                "UPDATE jobs SET progress_json = ?, updated_at = ? WHERE job_id = ?",
                (json.dumps(progress), self._now(), job_id),
            )
            self._event(connection, job_id, state, progress)

    def resume(self, job_id: str) -> JobState:
        payload = self.get(job_id)
        if payload["state"] != JobState.PAUSED.value or not payload["resume_state"]:
            raise ValueError("only a paused job can resume")
        target = JobState(payload["resume_state"])
        self.transition(job_id, target)
        return target

    def record_tile(self, job_id: str, tensor_name: str, tile_index: int, sha256: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO tiles VALUES (?, ?, ?, ?, ?)",
                (job_id, tensor_name, tile_index, sha256, self._now()),
            )

    def completed_tiles(self, job_id: str, tensor_name: str) -> dict[int, str]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT tile_index, sha256 FROM tiles WHERE job_id = ? AND tensor_name = ?",
                (job_id, tensor_name),
            ).fetchall()
        return {int(row["tile_index"]): str(row["sha256"]) for row in rows}

    def put_deployment(
        self,
        deployment_id: str,
        *,
        job_id: str,
        model_id: str,
        key_id: str,
        status: str,
        environment: str,
        metadata: dict[str, Any],
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """INSERT OR REPLACE INTO deployments
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    deployment_id,
                    job_id,
                    model_id,
                    key_id,
                    status,
                    environment,
                    json.dumps(metadata),
                    self._now(),
                ),
            )

    def get_deployment(self, deployment_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM deployments WHERE deployment_id = ?", (deployment_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown deployment: {deployment_id}")
        payload = dict(row)
        payload["metadata"] = json.loads(payload.pop("metadata_json"))
        return payload

    def latest_running_deployment(self, *, exclude: str | None = None) -> dict[str, Any] | None:
        query = "SELECT deployment_id FROM deployments WHERE status = 'RUNNING'"
        parameters: tuple[object, ...] = ()
        if exclude is not None:
            query += " AND deployment_id != ?"
            parameters = (exclude,)
        query += " ORDER BY updated_at DESC LIMIT 1"
        with self._connect() as connection:
            row = connection.execute(query, parameters).fetchone()
        return None if row is None else self.get_deployment(str(row["deployment_id"]))

    def update_deployment_status(self, deployment_id: str, status: str) -> None:
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE deployments SET status = ?, updated_at = ? WHERE deployment_id = ?",
                (status, self._now(), deployment_id),
            )
        if cursor.rowcount != 1:
            raise KeyError(f"unknown deployment: {deployment_id}")
