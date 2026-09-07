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
    """Mutable application paths, kept outside the source checkout."""

    root: Path
    state: Path
    chat: Path
    credentials: Path
    logs: Path
    cache: Path
    data: Path
    source_models: Path
    private_models: Path
    evidence: Path

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
            self.data,
            self.source_models,
            self.private_models,
            self.evidence,
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
    configured_data = _configured_path("YINBIAN_DATA_DIR", "ALOEPRI_DATA_DIR")
    configured_cache = _configured_path("YINBIAN_CACHE_DIR", "ALOEPRI_CACHE_DIR")
    data = configured_data or root / "data"
    cache = configured_cache or data / "cache"
    # Preserve the 1.0 behavior for users who configured only the cache path.
    # A dedicated data root takes precedence for all new installations.
    private_models = (
        data / "private-models"
        if configured_data is not None or configured_cache is None
        else configured_cache / "private"
    )
    return ProductPaths(
        root=root,
        state=root / "state",
        chat=root / "chat",
        credentials=root / "credentials",
        logs=root / "logs",
        cache=cache,
        data=data,
        source_models=data / "source-models",
        private_models=private_models,
        evidence=data / "evidence",
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
