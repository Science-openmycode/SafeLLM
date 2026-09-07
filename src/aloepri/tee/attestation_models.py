"""Validated request and response models for TEE attestation and provisioning."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class AttestationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    protocol_version: int = 1
    nonce: str
    expected_model_id: str
    expected_model_version: str


class AttestationResponse(BaseModel):
    tee_type: str
    quote: str
    sm2_certificate: str
    runtime_hash_sm3: str
    server_manifest_sm3: str
    model_id: str
    model_version: str
    key_id: str
    hardware_attested: bool
    debug: bool
    tcb_status: str
    mrtd: str
    rtmrs: list[str]


class ProvisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    encrypted_key_envelope: str = Field(min_length=1)
    expected_boundary_manifest_sm3: str = Field(min_length=64, max_length=64)


class ProvisionResponse(BaseModel):
    status: str
    model_id: str
    model_version: str
    key_id: str
