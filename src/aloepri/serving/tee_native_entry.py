from __future__ import annotations

import argparse
import json
from pathlib import Path

import uvicorn

from aloepri.serving.tee_app import create_tee_app
from aloepri.serving.tee_runtime import TeeSplitHFRuntime
from aloepri.tee.attestation import SoftwareAttestor, sm3_file, sm3_files
from aloepri.tee.service import TeeAttestationService, TeeDeploymentIdentity


def main() -> None:
    parser = argparse.ArgumentParser(description="Yinbian software-simulated TEE runtime")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tee-boundary", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--head-mode", choices=("local", "masked_outsource"), default="local")
    parser.add_argument("--initial-mask-count", type=int, default=0)
    args = parser.parse_args()

    runtime = TeeSplitHFRuntime(
        args.model,
        args.tee_boundary,
        device=args.device,
        head_mode=args.head_mode,
        initial_mask_count=args.initial_mask_count,
    )
    manifest = json.loads(
        (args.tee_boundary / "tee-manifest.json").read_text(encoding="utf-8")
    )
    service = TeeAttestationService(
        identity=TeeDeploymentIdentity(
            model_id=runtime.model_id,
            model_version=str(manifest.get("model_version", "unknown")),
            key_id=runtime.key_id,
            runtime_hash_sm3=sm3_files(
                [Path(__file__), Path(__file__).with_name("tee_runtime.py")]
            ),
            server_manifest_sm3=sm3_file(args.model / "server-manifest.json"),
            sm2_public_key_der=b"SOFTWARE-SIMULATION",
            sm2_certificate_pem="SOFTWARE SIMULATION - NO CERTIFICATE",
        ),
        attestor=SoftwareAttestor(),
        initially_provisioned=True,
    )
    uvicorn.run(
        create_tee_app(runtime, service, enable_software_encrypted_transport=True),
        host=args.host,
        port=args.port,
        access_log=False,
    )


if __name__ == "__main__":
    main()
