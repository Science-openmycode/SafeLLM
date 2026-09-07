"""Compatibility imports for TEE package integrity checks.

New code should import :mod:`aloepri.tee.package_integrity` directly.
"""

from aloepri.tee.package_integrity import (
    PackageInspection,
    inspect_server_package,
    verify_manifest_files,
    verify_signed_manifest,
)

__all__ = [
    "PackageInspection",
    "inspect_server_package",
    "verify_manifest_files",
    "verify_signed_manifest",
]
