"""Built-in model catalog and architecture fingerprinting."""

from aloepri.catalog.models import (
    AdapterMatch,
    ArchitectureFingerprint,
    MatchStatus,
    ModelCatalogEntry,
)
from aloepri.catalog.registry import ModelCatalog, builtin_catalog

__all__ = [
    "AdapterMatch",
    "ArchitectureFingerprint",
    "MatchStatus",
    "ModelCatalog",
    "ModelCatalogEntry",
    "builtin_catalog",
]
