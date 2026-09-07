"""Compatibility imports for the renamed TEE attestation service.

New code should import :mod:`aloepri.tee.attestation_service` directly.
"""

from aloepri.tee.attestation_service import (
    TeeAttestationService,
    TeeDeploymentIdentity,
    TeeServiceState,
)

__all__ = ["TeeAttestationService", "TeeDeploymentIdentity", "TeeServiceState"]
