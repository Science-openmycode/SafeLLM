from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from aloepri.conversion.metadata import strip_secret_metadata
from aloepri.conversion.paper_qwen2 import (
    convert_qwen2_modules,
    frobenius_norm_ratio_proxy,
)
from aloepri.conversion.vocab_checkpoint import sha256_file
from aloepri.keys.generate import generate_vocab_key
from aloepri.models.configuration_aloepri_qwen2 import AloePriQwen2Config
from aloepri.models.modeling_aloepri_qwen2 import AloePriQwen2ForCausalLM
from aloepri.tee.attestation import sm3
from aloepri.tee.config import BoundaryMode, SecurityMode, SecurityProfile, TeeBackend
from aloepri.tee.gm_cryptography import GmCryptoHelper
from aloepri.transforms.paper_key_matrix import (
    make_compatible_inverse_family,
    make_paper_key_pair,
)
from aloepri.transforms.qwen_structural import (
    transform_glm_layers,
    transform_qwen3_layers,
    transform_qwen_layers,
)


def get_rope_theta(config: object) -> float:
    """Read RoPE theta from legacy and Transformers 5-style configs."""
    rope_theta = getattr(config, "rope_theta", None)
    if rope_theta is not None:
        return float(rope_theta)
    rope_parameters = getattr(config, "rope_parameters", None)
    if isinstance(rope_parameters, dict):
        return float(rope_parameters.get("rope_theta", 10000.0))
    return 10000.0


def resolve_noise_seeds(
    seed: int, embedding_noise_seed: int | None, head_noise_seed: int | None
) -> tuple[int, int]:
    """Resolve independent paper noise draws while preserving legacy defaults."""
    return (
        embedding_noise_seed if embedding_noise_seed is not None else seed + 2,
        head_noise_seed if head_noise_seed is not None else seed + 3,
    )


def public_metadata(metadata: dict[str, object]) -> dict[str, object]:
    """Strip every deterministic secret from metadata shipped with the server model."""
    return strip_secret_metadata(metadata)


def paper_alignment_profile(args: argparse.Namespace) -> dict[str, object]:
    """Describe formula-level departures without turning them into secret metadata.

    This profile is shipped with the server checkpoint so a reviewer can tell
    which paper construction is literal, corrected, disabled, or numerically
    stabilized without access to the offline key.
    """
    rms_representation = getattr(args, "rms_representation", "gram")
    return {
        "algorithm1": "shape-corrected-nullspaces",
        "rmsnorm": (
            (
                "corrected-exact-derived-metric"
                if rms_representation == "gram"
                else "corrected-exact-stable-factor"
            )
            if args.rms_mode == "exact-metric"
            else "paper-scalar-kappa-approximation"
        ),
        "blockperm": (
            "disabled-beta-one-due-noncommuting-rope"
            if args.block_beta == 1
            else "corrected-runtime-synchronized-rope-conjugation"
        ),
        "rope_frequencies": (
            "architecture-correct-qwen"
            if args.rope_frequency_mode == "qwen-actual"
            else "paper-literal-exponent"
        ),
        "uvo_sampling": (
            "paper-gaussian"
            if args.uvo_condition_max is None
            else "conditioned-paper-gaussian-for-numerical-stability"
        ),
        "attention_precision": args.attention_compute_dtype,
        "embedding_noise_active": args.alpha_e != 0.0,
        "head_noise_active": args.alpha_h != 0.0,
    }


def _map_token_id(value: int | list[int] | None, tau: torch.Tensor) -> int | list[int] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return [int(tau[item]) for item in value]
    return int(tau[value])


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _sm3_file(path: Path) -> str:
    try:
        digest = hashlib.new("sm3")
    except ValueError as error:  # pragma: no cover - depends on platform OpenSSL
        raise RuntimeError("the platform crypto provider has no SM3 implementation") from error
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path, *, root: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "sm3": _sm3_file(path),
    }


def _make_head_basis(hidden_size: int, *, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    gaussian = torch.randn(hidden_size, hidden_size, generator=generator, dtype=torch.float64)
    basis = torch.linalg.qr(gaussian).Q.float().contiguous()
    del gaussian
    return basis


def _write_signed_manifest(
    path: Path,
    payload: dict[str, object],
    *,
    signature_path: Path,
    helper: GmCryptoHelper | None,
    signing_key: str | None,
) -> None:
    encoded = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    path.write_bytes(encoded)
    if helper is None:
        return
    if not signing_key:
        raise ValueError("an SM2 signing key reference is required with the GM helper")
    signature_path.write_bytes(helper.sign_sm2_sm3(encoded, key_reference=signing_key))


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Qwen2 to the corrected-paper d+2h model")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--key-dir", type=Path, required=True)
    parser.add_argument(
        "--security-mode",
        choices=[item.value.replace("_", "-") for item in SecurityMode],
        default="permutation",
    )
    parser.add_argument(
        "--boundary-mode",
        choices=[item.value.replace("_", "-") for item in BoundaryMode],
        default="in-model",
    )
    parser.add_argument(
        "--tee-backend",
        choices=[item.value.replace("_", "-") for item in TeeBackend],
    )
    parser.add_argument(
        "--tee-output",
        type=Path,
        help="Trusted boundary package; required for tee-gm mode.",
    )
    parser.add_argument(
        "--gm-crypto-helper",
        type=Path,
        help="Reviewed Tongsuo-backed helper used for SM2 manifest signatures.",
    )
    parser.add_argument(
        "--gm-signing-key",
        help="Opaque SM2 signing-key reference understood by the GM helper.",
    )
    parser.add_argument(
        "--gm-boundary-key",
        help="Opaque SM4 boundary-key reference understood by the GM helper.",
    )
    parser.add_argument("--h", type=int, default=128)
    parser.add_argument("--lambda", dest="coefficient_lambda", type=float, default=0.3)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--embedding-noise-seed",
        type=int,
        help="Independent paper Gaussian seed for W_e; defaults to --seed + 2.",
    )
    parser.add_argument(
        "--head-noise-seed",
        type=int,
        help="Independent paper Gaussian seed for W_h; defaults to --seed + 3.",
    )
    parser.add_argument("--alpha-e", type=float, default=1.0)
    parser.add_argument("--alpha-h", type=float, default=0.2)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument(
        "--attention-compute-dtype",
        choices=["float32", "float64"],
        default="float32",
        help="Storage/runtime dtype for Q/K/V/O; float64 prioritizes function equivalence.",
    )
    parser.add_argument("--max-shard-size", default="2GB")
    parser.add_argument("--model-id")
    parser.add_argument("--key-id")
    parser.add_argument("--source-revision", default="unknown")
    parser.add_argument("--rms-calibration", type=Path)
    parser.add_argument("--kappa-override", type=float)
    parser.add_argument(
        "--kappa-mode",
        choices=["covariant-rms", "paper-norm-ratio-proxy", "paper-expectation"],
        default="paper-norm-ratio-proxy",
    )
    parser.add_argument(
        "--rms-mode",
        choices=["paper-kappa", "exact-metric"],
        default="paper-kappa",
        help="Paper approximation or exact corrected plaintext-metric RMSNorm.",
    )
    parser.add_argument(
        "--rms-representation",
        choices=["gram", "stable-factor"],
        default="gram",
        help=(
            "Store the exact Gram form or its PSD factor. The stable factor "
            "evaluates the same quadratic form without nullspace cancellation."
        ),
    )
    parser.add_argument("--algorithm2", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ffn-scale-min", type=float, default=0.5)
    parser.add_argument("--ffn-scale-max", type=float, default=2.0)
    parser.add_argument("--block-beta", type=int, default=8)
    parser.add_argument("--sampling-gamma", type=float, default=1000.0)
    parser.add_argument(
        "--blockperm-mode",
        choices=["paper-distribution-boundary-corrected", "gamma-corrected"],
        default="paper-distribution-boundary-corrected",
    )
    parser.add_argument(
        "--rope-frequency-mode",
        choices=["qwen-actual", "paper-literal"],
        default="qwen-actual",
    )
    parser.add_argument("--qk-scale-min", type=float, default=0.5)
    parser.add_argument("--qk-scale-max", type=float, default=2.0)
    parser.add_argument(
        "--uvo-condition-max",
        type=float,
        help="Reject Algorithm 2 Gaussian U_vo samples above this condition number.",
    )
    args = parser.parse_args()
    profile = SecurityProfile.from_mapping(
        {
            "security_mode": args.security_mode.replace("-", "_"),
            "boundary_mode": args.boundary_mode.replace("-", "_"),
            "tee_backend": (
                args.tee_backend.replace("-", "_") if args.tee_backend else None
            ),
        }
    )
    if profile.is_tee and args.tee_output is None:
        parser.error("--tee-output is required for tee-gm mode")
    if not profile.is_tee and args.tee_output is not None:
        parser.error("--tee-output is only valid for tee-gm mode")
    if profile.is_tee and (args.alpha_e != 0.0 or args.alpha_h != 0.0):
        parser.error("tee-gm exact boundary conversion currently requires alpha-e=alpha-h=0")
    if profile.tee_backend == TeeBackend.INTEL_TDX and args.gm_crypto_helper is None:
        parser.error("intel-tdx conversion requires --gm-crypto-helper for SM2 signatures")
    if profile.tee_backend == TeeBackend.INTEL_TDX and not args.gm_signing_key:
        parser.error("intel-tdx conversion requires --gm-signing-key")
    if profile.tee_backend == TeeBackend.INTEL_TDX and not args.gm_boundary_key:
        parser.error("intel-tdx conversion requires --gm-boundary-key for SM4-GCM encryption")
    gm_helper = GmCryptoHelper(args.gm_crypto_helper) if args.gm_crypto_helper else None
    embedding_noise_seed, head_noise_seed = resolve_noise_seeds(
        args.seed, args.embedding_noise_seed, args.head_noise_seed
    )

    if args.output.exists() or args.key_dir.exists() or (
        args.tee_output is not None and args.tee_output.exists()
    ):
        raise FileExistsError("output or key directory already exists")
    output_partial = args.output.with_name(f"{args.output.name}.partial")
    key_partial = args.key_dir.with_name(f"{args.key_dir.name}.partial")
    tee_partial = (
        args.tee_output.with_name(f"{args.tee_output.name}.partial")
        if args.tee_output is not None
        else None
    )
    if output_partial.exists() or key_partial.exists() or (
        tee_partial is not None and tee_partial.exists()
    ):
        raise FileExistsError("stale partial directory exists; inspect and remove it explicitly")
    output_partial.mkdir(parents=True)
    key_partial.mkdir(parents=True)
    if tee_partial is not None:
        tee_partial.mkdir(parents=True)

    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    source = AutoModelForCausalLM.from_pretrained(
        args.source,
        local_files_only=True,
        dtype=dtype,
        attn_implementation="eager",
    ).eval()
    config = AloePriQwen2Config.from_qwen2_config(
        source.config,
        expansion_h=args.h,
        rms_mode=args.rms_mode.replace("-", "_"),
        rms_representation=args.rms_representation.replace("-", "_"),
        attention_compute_dtype=args.attention_compute_dtype,
    )
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        target = AloePriQwen2ForCausalLM(config).eval()
    finally:
        torch.set_default_dtype(old_dtype)

    if profile.is_tee:
        # tee_gm sends ordinary token IDs inside the attested encrypted channel.
        # Identity is used only to reuse the unchanged body conversion routine;
        # neither tensor is emitted into a TEE artifact.
        tau = torch.arange(config.vocab_size, dtype=torch.int64)
        inverse_tau = tau
    else:
        tau, inverse_tau = generate_vocab_key(config.vocab_size, seed=args.seed + 1)
    key_pair = make_paper_key_pair(
        source.config.hidden_size,
        args.h,
        coefficient_lambda=args.coefficient_lambda,
        seed=args.seed,
    )
    inverse_family = make_compatible_inverse_family(key_pair, seed=args.seed + 5000)
    rms_kappas = None
    if args.rms_calibration:
        calibration = json.loads(args.rms_calibration.read_text(encoding="utf-8"))
        rms_kappas = {name: float(value) for name, value in calibration["kappas"].items()}
    elif args.kappa_mode == "paper-expectation":
        parser.error("--kappa-mode paper-expectation requires --rms-calibration")
    if args.kappa_override is not None:
        if args.kappa_override <= 0:
            raise ValueError("kappa override must be positive")
        rms_kappas = {
            **{
                f"layers.{index}.{norm}": args.kappa_override
                for index in range(source.config.num_hidden_layers)
                for norm in ("input_layernorm", "post_attention_layernorm")
            },
            "model.norm": args.kappa_override,
        }
    elif args.kappa_mode == "paper-norm-ratio-proxy":
        literal_kappa = frobenius_norm_ratio_proxy(key_pair.p)
        rms_kappas = {
            **{
                f"layers.{index}.{norm}": literal_kappa
                for index in range(source.config.num_hidden_layers)
                for norm in ("input_layernorm", "post_attention_layernorm")
            },
            "model.norm": literal_kappa,
        }
    stats = convert_qwen2_modules(
        source,
        target,
        key_pair=key_pair,
        inverse_family=inverse_family,
        tau=tau,
        alpha_e=args.alpha_e,
        alpha_h=args.alpha_h,
        embedding_noise_seed=embedding_noise_seed,
        head_noise_seed=head_noise_seed,
        rms_kappas=rms_kappas,
    )
    structural_key: dict[str, torch.Tensor] = {}
    if args.algorithm2:
        source_family = getattr(source.config, "aloepri_source_family", None)
        if source_family is None and getattr(source.config, "model_type", None) == "qwen3":
            source_family = "qwen3_dense"
        if source_family == "glm_dense":
            structural_key = transform_glm_layers(
                target,
                seed=args.seed + 10000,
                partial_rotary_factor=float(
                    getattr(source.config, "partial_rotary_factor", 0.5)
                ),
                ffn_scale_min=args.ffn_scale_min,
                ffn_scale_max=args.ffn_scale_max,
                qk_scale_min=args.qk_scale_min,
                qk_scale_max=args.qk_scale_max,
                value_condition_max=args.uvo_condition_max,
            )
        elif source_family == "qwen3_dense":
            structural_key = transform_qwen3_layers(
                target,
                seed=args.seed + 10000,
                ffn_scale_min=args.ffn_scale_min,
                ffn_scale_max=args.ffn_scale_max,
                qk_scale_min=args.qk_scale_min,
                qk_scale_max=args.qk_scale_max,
                value_condition_max=args.uvo_condition_max,
            )
        else:
            structural_key = transform_qwen_layers(
                target,
                seed=args.seed + 10000,
                coordinate_mode="dense_orthogonal",
                ffn_scale_min=args.ffn_scale_min,
                ffn_scale_max=args.ffn_scale_max,
                block_beta=args.block_beta,
                sampling_gamma=args.sampling_gamma,
                blockperm_mode=args.blockperm_mode,
                rope_frequency_mode=args.rope_frequency_mode,
                rope_theta=get_rope_theta(source.config),
                qk_scale_min=args.qk_scale_min,
                qk_scale_max=args.qk_scale_max,
                value_condition_max=args.uvo_condition_max,
            )
    embedding_private: torch.Tensor | None = None
    exact_head: torch.Tensor | None = None
    head_basis: torch.Tensor | None = None
    outsourced_head: torch.Tensor | None = None
    if profile.is_tee:
        embedding_private = (
            source.get_input_embeddings().weight.detach().float() @ key_pair.p.float()
        ).contiguous()
        exact_head = (
            source.get_output_embeddings().weight.detach().float()
            * source.model.norm.weight.detach().float().unsqueeze(0)
        ).contiguous()
        head_basis = _make_head_basis(source.config.hidden_size, seed=args.seed + 70000)
        # For row-vector h and x=hB, F.linear(x, W_B) equals the exact Head
        # when W_B = W_exact B for orthogonal B.
        outsourced_head = (exact_head @ head_basis).contiguous()
        with torch.no_grad():
            # The server package must not contain either boundary.  These
            # allocated tensors remain only to preserve the HF model schema.
            target.get_input_embeddings().weight.zero_()
            target.get_output_embeddings().weight.zero_()
    model_id = args.model_id or args.output.name
    key_id = args.key_id or args.key_dir.name
    metadata = {
        "schema_version": 2,
        "model_id": model_id,
        "key_id": key_id,
        "security_mode": profile.security_mode.value,
        "boundary_mode": profile.boundary_mode.value,
        "tee_backend": profile.tee_backend.value if profile.tee_backend else None,
        "hardware_attested": False,
        "transform_mode": "paper_d_plus_2h",
        "source": str(args.source.resolve()),
        "source_revision": args.source_revision,
        "plain_hidden_size": source.config.hidden_size,
        "private_hidden_size": config.hidden_size,
        "expansion_h": args.h,
        "lambda": args.coefficient_lambda,
        "alpha_e": args.alpha_e,
        "alpha_h": args.alpha_h,
        "embedding_noise_seed": embedding_noise_seed,
        "head_noise_seed": head_noise_seed,
        "seed": args.seed,
        "kappa": stats.kappa,
        "rms_mode": stats.rms_mode,
        "rms_representation": args.rms_representation,
        "kappa_mode": args.kappa_mode,
        "kappa_override": args.kappa_override,
        "rms_calibration": str(args.rms_calibration) if args.rms_calibration else None,
        "algorithm2": args.algorithm2,
        "attention_compute_dtype": args.attention_compute_dtype,
        "algorithm2_seed": args.seed + 10000 if args.algorithm2 else None,
        "attention_block_beta": args.block_beta if args.algorithm2 else 1,
        "attention_sampling_gamma": args.sampling_gamma,
        "attention_blockperm_mode": args.blockperm_mode,
        "attention_rope_frequency_mode": args.rope_frequency_mode,
        "qk_scale_min": args.qk_scale_min if args.algorithm2 else 1.0,
        "qk_scale_max": args.qk_scale_max if args.algorithm2 else 1.0,
        "uvo_condition_max": args.uvo_condition_max if args.algorithm2 else None,
        "ffn_scale_min": args.ffn_scale_min if args.algorithm2 else 1.0,
        "ffn_scale_max": args.ffn_scale_max if args.algorithm2 else 1.0,
        "b_condition_number": key_pair.condition_b,
        "pq_relative_error_fp64": key_pair.pq_relative_error,
        "spectral_norm_p": key_pair.spectral_norm_p,
        "spectral_norm_q": key_pair.spectral_norm_q,
        "inverse_key_mode": "algorithm1_shared_init_independent_d",
        "inverse_key_count": 6,
        "inverse_key_maximum_pq_relative_error_fp64": (inverse_family.maximum_relative_error),
        "paper_alignment": paper_alignment_profile(args),
        "git_commit": _git_commit(),
        "source_family": (
            getattr(source.config, "aloepri_source_family", None)
            or (
                "qwen3_dense"
                if getattr(source.config, "model_type", None) == "qwen3"
                else "qwen2"
            )
        ),
    }
    if metadata["source_family"] == "glm_dense":
        metadata["attention_block_beta"] = 1
        metadata["attention_blockperm_mode"] = "glm-partial-rope-commuting-map"
        metadata["paper_alignment"]["blockperm"] = (
            "glm-partial-rope-commuting-map-without-frequency-block-permutation"
        )
    elif metadata["source_family"] == "qwen3_dense":
        metadata["attention_block_beta"] = 1
        metadata["attention_blockperm_mode"] = "qwen3-qk-norm-commuting-map"
        metadata["paper_alignment"]["blockperm"] = (
            "qwen3-shared-qk-norm-signed-rope-map"
        )
    server_metadata = public_metadata(metadata)
    target.config.aloepri = server_metadata
    target.generation_config = source.generation_config
    for name in ("bos_token_id", "eos_token_id", "pad_token_id"):
        value = getattr(target.generation_config, name, None)
        setattr(target.generation_config, name, _map_token_id(value, tau))
        config_value = getattr(target.config, name, None)
        setattr(target.config, name, _map_token_id(config_value, tau))
    target.save_pretrained(
        output_partial, safe_serialization=True, max_shard_size=args.max_shard_size
    )
    if not profile.is_tee:
        tokenizer = AutoTokenizer.from_pretrained(args.source, local_files_only=True)
        tokenizer.save_pretrained(output_partial)

    if profile.is_tee:
        assert tee_partial is not None
        assert embedding_private is not None
        assert exact_head is not None
        assert head_basis is not None
        assert outsourced_head is not None
        save_file(
            {"embedding_private": embedding_private},
            tee_partial / "embedding-private.safetensors",
        )
        save_file(
            {"exact_head": exact_head},
            tee_partial / "exact-head-archive.safetensors",
        )
        save_file(
            {"q_final": inverse_family.head.float().contiguous()},
            tee_partial / "final-coordinate-key.safetensors",
        )
        save_file(
            {"head_basis": head_basis},
            tee_partial / "head-coordinate-key.safetensors",
        )
        save_file(
            {"outsourced_head": outsourced_head},
            output_partial / "masked-head-worker.safetensors",
        )
        special_tokens = {
            name: getattr(source.generation_config, name, None)
            for name in ("bos_token_id", "eos_token_id", "pad_token_id")
        }
        (tee_partial / "special-tokens.json").write_text(
            json.dumps(special_tokens, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        plaintext_records = [
            _file_record(path, root=tee_partial)
            for path in sorted(tee_partial.iterdir())
            if path.is_file()
        ]
        plaintext_manifest = json.dumps(
            plaintext_records,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        plaintext_manifest_sm3 = sm3(plaintext_manifest).hex()
        encryption_records: list[dict[str, object]] = []
        if profile.tee_backend == TeeBackend.INTEL_TDX:
            assert gm_helper is not None
            assert args.gm_boundary_key is not None
            for path in sorted(tee_partial.iterdir()):
                encrypted = path.with_name(f"{path.name}.gm-partial")
                result = gm_helper.encrypt_file_sm4_gcm(
                    path,
                    encrypted,
                    key_reference=args.gm_boundary_key,
                    aad={
                        "path": path.name,
                        "model_id": model_id,
                        "key_id": key_id,
                        "model_version": args.source_revision,
                        "plaintext_manifest_sm3": plaintext_manifest_sm3,
                    },
                )
                plaintext_bytes = path.stat().st_size
                path.unlink()
                os.replace(encrypted, path)
                encryption_records.append(
                    {
                        "path": path.name,
                        "algorithm": result["algorithm"],
                        "chunk_bytes": int(result["chunk_bytes"]),
                        "plaintext_bytes": plaintext_bytes,
                        "nonce_strategy": str(result["nonce_strategy"]),
                    }
                )

        tee_files = [
            _file_record(path, root=tee_partial)
            for path in sorted(tee_partial.iterdir())
            if path.is_file()
        ]
        tee_manifest = {
            "schema_version": 1,
            "security_mode": "tee_gm",
            "boundary_mode": "tee_split",
            "tee_backend": profile.tee_backend.value,
            "model_id": model_id,
            "key_id": key_id,
            "model_version": args.source_revision,
            "hardware_attested": False,
            "encrypted_at_rest": profile.tee_backend == TeeBackend.INTEL_TDX,
            "development_only": profile.tee_backend == TeeBackend.SOFTWARE_SIM,
            "signature_algorithm": "SM2-with-SM3" if gm_helper else None,
            "content_encryption": encryption_records,
            "plaintext_manifest_sm3": plaintext_manifest_sm3,
            "files": tee_files,
        }
        _write_signed_manifest(
            tee_partial / "tee-manifest.json",
            tee_manifest,
            signature_path=tee_partial / "tee-manifest.sm2sig",
            helper=gm_helper,
            signing_key=args.gm_signing_key,
        )

    key_tensors = {
        "p": key_pair.p,
        "q": key_pair.q,
        **(key_pair.algorithm1_base.tensors() if key_pair.algorithm1_base is not None else {}),
        **inverse_family.tensors(),
        **structural_key,
    }
    if not profile.is_tee:
        key_tensors["tau"] = tau
        key_tensors["inverse_tau"] = inverse_tau
    save_file(key_tensors, key_partial / "paper_key.safetensors")
    # key.json belongs to the trusted client package.  Seeds remain here only for
    # reproducibility; the server receives neither this directory nor its contents.
    key_metadata = {**metadata, "vocab_file": "paper_key.safetensors"}
    (key_partial / "key.json").write_text(
        json.dumps(key_metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    key_files = [
        {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in (key_partial / "key.json", key_partial / "paper_key.safetensors")
    ]
    (key_partial / "key_manifest.json").write_text(
        json.dumps({"key_id": key_id, "files": key_files}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    files = [
        {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(output_partial.iterdir())
        if path.is_file()
    ]
    (output_partial / "aloepri_manifest.json").write_text(
        json.dumps({"metadata": server_metadata, "files": files}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if profile.is_tee:
        server_files = [
            _file_record(path, root=output_partial)
            for path in sorted(output_partial.iterdir())
            if path.is_file()
            and path.name not in {"server-manifest.json", "server-manifest.sm2sig"}
        ]
        server_manifest = {
            "schema_version": 1,
            "security_mode": "tee_gm",
            "boundary_mode": "tee_split",
            "model_id": model_id,
            "key_id": key_id,
            "model_version": args.source_revision,
            "hardware_attested": False,
            "signature_algorithm": "SM2-with-SM3" if gm_helper else None,
            "files": server_files,
        }
        _write_signed_manifest(
            output_partial / "server-manifest.json",
            server_manifest,
            signature_path=output_partial / "server-manifest.sm2sig",
            helper=gm_helper,
            signing_key=args.gm_signing_key,
        )
    os.replace(output_partial, args.output)
    os.replace(key_partial, args.key_dir)
    if tee_partial is not None and args.tee_output is not None:
        os.replace(tee_partial, args.tee_output)
    del target, source
    if output_partial.exists():
        shutil.rmtree(output_partial)
    print(json.dumps(server_metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
