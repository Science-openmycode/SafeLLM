from __future__ import annotations


def test_legacy_tee_module_names_export_current_implementations() -> None:
    from aloepri.secure_head.masked_outsource_engine import (
        MaskedOutsourceHeadEngine as CurrentMaskedHead,
    )
    from aloepri.secure_head.outsourced import (
        MaskedOutsourceHeadEngine as LegacyMaskedHead,
    )
    from aloepri.tee.attestation_models import AttestationRequest as CurrentRequest
    from aloepri.tee.attestation_service import TeeAttestationService as CurrentService
    from aloepri.tee.boundary import SoftwareTrustedBoundary as LegacyBoundary
    from aloepri.tee.gm import GmCryptoHelper as LegacyGmHelper
    from aloepri.tee.gm_cryptography import GmCryptoHelper as CurrentGmHelper
    from aloepri.tee.package import verify_manifest_files as legacy_verify_manifest
    from aloepri.tee.package_integrity import verify_manifest_files as current_verify_manifest
    from aloepri.tee.protocol import AttestationRequest as LegacyRequest
    from aloepri.tee.service import TeeAttestationService as LegacyService
    from aloepri.tee.trusted_boundary import SoftwareTrustedBoundary as CurrentBoundary

    assert LegacyGmHelper is CurrentGmHelper
    assert legacy_verify_manifest is current_verify_manifest
    assert LegacyRequest is CurrentRequest
    assert LegacyService is CurrentService
    assert LegacyBoundary is CurrentBoundary
    assert LegacyMaskedHead is CurrentMaskedHead
