from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from aloepri.keys.vault import protect_current_user, unprotect_current_user
from aloepri.product.paths import product_paths


class ChatHistoryStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or product_paths().chat_db
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """CREATE TABLE IF NOT EXISTS history_snapshots (
                    deployment_id TEXT PRIMARY KEY,
                    protected_payload BLOB NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )"""
            )

    def get(self, deployment_id: str) -> dict[str, Any]:
        with sqlite3.connect(self.path) as connection:
            row = connection.execute(
                "SELECT protected_payload FROM history_snapshots WHERE deployment_id=?",
                (deployment_id,),
            ).fetchone()
        if row is None:
            return {"saved": [], "active": None}
        payload = json.loads(unprotect_current_user(bytes(row[0])))
        if not isinstance(payload, dict):
            raise ValueError("chat history snapshot is invalid")
        return payload

    def put(self, deployment_id: str, payload: dict[str, Any]) -> None:
        protected = protect_current_user(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
        )
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                """INSERT INTO history_snapshots(deployment_id, protected_payload)
                   VALUES (?, ?) ON CONFLICT(deployment_id) DO UPDATE SET
                   protected_payload=excluded.protected_payload,
                   updated_at=CURRENT_TIMESTAMP""",
                (deployment_id, protected),
            )

    def delete(self, deployment_id: str) -> None:
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                "DELETE FROM history_snapshots WHERE deployment_id=?", (deployment_id,)
            )
            connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
