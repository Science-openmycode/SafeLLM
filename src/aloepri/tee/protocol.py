"""Compatibility imports for TEE attestation request and response models.

New code should import :mod:`aloepri.tee.attestation_models` directly.
"""

from aloepri.tee.attestation_models import (
    AttestationRequest,
    AttestationResponse,
    ProvisionRequest,
    ProvisionResponse,
)

__all__ = [
    "AttestationRequest",
    "AttestationResponse",
    "ProvisionRequest",
    "ProvisionResponse",
]
