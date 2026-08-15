from __future__ import annotations

import os
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path


def _configured_path(new_name: str, legacy_name: str) -> Path | None:
    value = os.environ.get(new_name)
    if value:
        return Path(value)
    legacy = os.environ.get(legacy_name)
    if legacy:
        warnings.warn(
            f"{legacy_name} is deprecated; use {new_name}",
            DeprecationWarning,
            stacklevel=2,
        )
        return Path(legacy)
    return None


@dataclass(frozen=True)
class ProductPaths:
    root: Path
    state: Path
    chat: Path
    credentials: Path
    logs: Path
    cache: Path

    @property
    def state_db(self) -> Path:
        configured = _configured_path("YINBIAN_STATE_DB", "ALOEPRI_STATE_DB")
        return configured or self.state / "state.db"

    @property
    def chat_db(self) -> Path:
        configured = _configured_path("YINBIAN_CHAT_DB", "ALOEPRI_CHAT_DB")
        return configured or self.chat / "chat.db"

    def create(self) -> None:
        for path in (
            self.root,
            self.state,
            self.chat,
            self.credentials,
            self.logs,
            self.cache,
        ):
            path.mkdir(parents=True, exist_ok=True)


def product_paths() -> ProductPaths:
    configured = _configured_path("YINBIAN_HOME", "ALOEPRI_HOME")
    if configured is not None:
        root = configured
    elif os.name == "nt":
        local = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        root = local / "YinbianZhimo"
    else:
        root = Path.home() / ".local" / "share" / "yinbian"
    cache = _configured_path("YINBIAN_CACHE_DIR", "ALOEPRI_CACHE_DIR") or root / "cache"
    return ProductPaths(
        root=root,
        state=root / "state",
        chat=root / "chat",
        credentials=root / "credentials",
        logs=root / "logs",
        cache=cache,
    )


def legacy_state_path() -> Path:
    if os.name == "nt":
        local = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return local / "AloePri" / "state.db"
    return Path.home() / ".local" / "share" / "aloepri" / "state.db"


def prepare_product_state(paths: ProductPaths | None = None) -> Path:
    selected = paths or product_paths()
    selected.create()
    target = selected.state_db
    if target.exists():
        return target
    legacy = legacy_state_path()
    if legacy.is_file() and legacy.resolve() != target.resolve():
        backup = selected.state / "legacy-state.pre-migration.db"
        if not backup.exists():
            shutil.copy2(legacy, backup)
        partial = target.with_suffix(".db.partial")
        shutil.copy2(legacy, partial)
        os.replace(partial, target)
    return target
