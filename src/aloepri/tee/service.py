from __future__ import annotations

import base64
from dataclasses import dataclass
from enum import StrEnum

from aloepri.tee.attestation import Attestor, build_report_data
from aloepri.tee.protocol import AttestationRequest, AttestationResponse


class TeeServiceState(StrEnum):
    STARTING = "STARTING"
    ATTESTATION_READY = "ATTESTATION_READY"
    WAITING_FOR_PROVISIONING = "WAITING_FOR_PROVISIONING"
    LOADING_BOUNDARY_PACKAGE = "LOADING_BOUNDARY_PACKAGE"
    READY = "READY"
    FAILED = "FAILED"


@dataclass(frozen=True)
class TeeDeploymentIdentity:
    model_id: str
    model_version: str
    key_id: str
    runtime_hash_sm3: str
    server_manifest_sm3: str
    sm2_public_key_der: bytes
    sm2_certificate_pem: str


class TeeAttestationService:
    def __init__(
        self,
        *,
        identity: TeeDeploymentIdentity,
        attestor: Attestor,
        initially_provisioned: bool = False,
    ) -> None:
        self.identity = identity
        self.attestor = attestor
        self.state = (
            TeeServiceState.READY
            if initially_provisioned
            else TeeServiceState.WAITING_FOR_PROVISIONING
        )

    def attest(self, request: AttestationRequest) -> AttestationResponse:
        if request.protocol_version != 1:
            raise ValueError("unsupported TEE attestation protocol version")
        if request.expected_model_id != self.identity.model_id:
            raise ValueError("attestation requested a different model")
        if request.expected_model_version != self.identity.model_version:
            raise ValueError("attestation requested a different model version")
        try:
            nonce = base64.b64decode(request.nonce, validate=True)
        except ValueError as error:
            raise ValueError("attestation nonce is not valid base64") from error
        report_data = build_report_data(
            nonce=nonce,
            sm2_public_key_der=self.identity.sm2_public_key_der,
            runtime_hash_sm3=self.identity.runtime_hash_sm3,
            server_manifest_sm3=self.identity.server_manifest_sm3,
            model_id=self.identity.model_id,
            model_version=self.identity.model_version,
            key_id=self.identity.key_id,
        )
        evidence = self.attestor.quote(report_data)
        return AttestationResponse(
            tee_type=evidence.tee_type,
            quote=base64.b64encode(evidence.quote).decode("ascii"),
            sm2_certificate=self.identity.sm2_certificate_pem,
            runtime_hash_sm3=self.identity.runtime_hash_sm3,
            server_manifest_sm3=self.identity.server_manifest_sm3,
            model_id=self.identity.model_id,
            model_version=self.identity.model_version,
            key_id=self.identity.key_id,
            hardware_attested=evidence.hardware_attested,
            debug=evidence.debug,
            tcb_status=evidence.tcb_status,
            mrtd=evidence.mrtd,
            rtmrs=list(evidence.rtmrs),
        )

    def mark_loading(self) -> None:
        if self.state != TeeServiceState.WAITING_FOR_PROVISIONING:
            raise ValueError(f"cannot provision TEE while state is {self.state}")
        self.state = TeeServiceState.LOADING_BOUNDARY_PACKAGE

    def mark_ready(self) -> None:
        if self.state != TeeServiceState.LOADING_BOUNDARY_PACKAGE:
            raise ValueError(f"cannot mark TEE ready while state is {self.state}")
        self.state = TeeServiceState.READY

    def mark_failed(self) -> None:
        self.state = TeeServiceState.FAILED

    def require_ready(self) -> None:
        if self.state != TeeServiceState.READY:
            raise RuntimeError(f"TEE boundary is not ready: {self.state}")
