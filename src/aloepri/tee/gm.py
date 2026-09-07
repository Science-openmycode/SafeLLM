"""Compatibility imports for the renamed GM cryptography module.

New code should import :mod:`aloepri.tee.gm_cryptography` directly.
"""

from aloepri.tee.gm_cryptography import GmCryptoHelper, TongsuoCapabilities, inspect_tongsuo

__all__ = ["GmCryptoHelper", "TongsuoCapabilities", "inspect_tongsuo"]
