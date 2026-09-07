from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from aloepri.adapters.registry import default_adapter_registry
from aloepri.catalog.download import download_pinned_snapshot, find_catalog_entry
from aloepri.catalog.inspect import inspect_local_checkpoint
from aloepri.cloud import MockObjectStore
from aloepri.conversion.deepseek_streaming import convert_deepseek_checkpoint
from aloepri.conversion.family_normalization import (
    normalize_glm_dense_checkpoint,
    normalize_kimi_k2_text_checkpoint,
)
from aloepri.conversion.glm4_moe_streaming import convert_glm4_moe_checkpoint
from aloepri.conversion.resource_estimate import estimate_deepseek_host_memory
from aloepri.jobs.store import JobState, JobStore
from aloepri.keys.encryption import encrypt_offline_key_directory
from aloepri.packaging import split_key_package
from aloepri.planning import ConversionPlan
from aloepri.resources import ResourceBudgetMonitor


class ConversionPaused(RuntimeError):
    pass


class ConversionCancelled(RuntimeError):
    pass


def _subprocess_python() -> str:
    """Use a console interpreter so native crashes leave actionable diagnostics."""

    executable = Path(sys.executable).resolve()
    if executable.stem.casefold() == "pythonw":
        console = executable.with_name("python.exe")
        if console.is_file():
            return str(console)
    return str(executable)


def _local_output(plan: ConversionPlan) -> Path:
    uri = str(plan.output["uri"])
    if uri.startswith("s3://"):
        staging = plan.output.get("staging_path")
        if not staging:
            raise ValueError("S3 output requires output.staging_path for local conversion")
        return Path(str(staging))
    return Path(uri)


def _resolve_source(plan: ConversionPlan, store: JobStore) -> Path:
    source_type = str(plan.source.get("type", "local"))
    if source_type == "local":
        return Path(str(plan.source["path"]))
    if source_type == "s3":
        destination = Path(str(plan.source["cache_path"]))
        if JobState(store.get(plan.job_id)["state"]) == JobState.PREFLIGHT:
            store.transition(plan.job_id, JobState.DOWNLOADING)
        objects = MockObjectStore(store.path.parent / "mock-cloud" / "objects")
        files = objects.download_prefix(str(plan.source["uri"]), destination)
        store.update_progress(
            plan.job_id,
            {"download": {"source": "s3", "files_completed": len(files)}},
        )
        config, inventory = inspect_local_checkpoint(destination)
        match = default_adapter_registry().detect(config, inventory)
        if match.adapter_id != plan.adapter or match.status.value != "SUPPORTED":
            raise ValueError(f"downloaded S3 checkpoint adapter mismatch: {match.to_dict()}")
        if JobState(store.get(plan.job_id)["state"]) == JobState.DOWNLOADING:
            store.transition(plan.job_id, JobState.CONVERTING)
        return destination
    if source_type != "huggingface":
        raise ValueError(f"unsupported source type: {source_type}")
    entry = find_catalog_entry(str(plan.source["repo_id"]))
    if str(plan.source.get("revision")) != entry.revision:
        raise ValueError("plan revision no longer matches the pinned catalog revision")
    destination = Path(str(plan.source["cache_path"]))
    state = JobState(store.get(plan.job_id)["state"])
    if state == JobState.PREFLIGHT:
        store.transition(plan.job_id, JobState.DOWNLOADING)
    download_pinned_snapshot(
        entry,
        destination,
        progress=lambda phase: store.update_progress(
            plan.job_id, {"download": {"phase": phase, "revision": entry.revision}}
        ),
    )
    config, inventory = inspect_local_checkpoint(destination)
    match = default_adapter_registry().detect(config, inventory)
    if match.adapter_id != plan.adapter or match.status.value != "SUPPORTED":
        raise ValueError(f"downloaded checkpoint adapter mismatch: {match.to_dict()}")
    coverage = default_adapter_registry().get(plan.adapter).validate_inventory(config, inventory)
    if not coverage.pass_:
        raise ValueError(
            "downloaded checkpoint inventory rejected: "
            f"missing={list(coverage.missing)}, unknown={list(coverage.unknown)}"
        )
    if JobState(store.get(plan.job_id)["state"]) == JobState.DOWNLOADING:
        store.transition(plan.job_id, JobState.CONVERTING)
    return destination


def _key_paths(plan: ConversionPlan, output: Path) -> tuple[Path, Path, Path]:
    root = output.parent / f"{output.name}-keys"
    return (
        Path(str(plan.keys.get("full", root / "full"))),
        Path(str(plan.keys.get("online", root / "online"))),
        Path(str(plan.keys.get("offline", root / "offline"))),
    )


def finalize_qwen_key_package(
    plan: ConversionPlan,
    output: Path,
    *,
    offline_key_password: str | None = None,
) -> tuple[Path, Path, Path]:
    """Idempotently finish key splitting/encryption after a converter restart."""

    full_key, online_key, offline_key = _key_paths(plan, output)
    if not online_key.exists() and not offline_key.exists():
        if not full_key.is_dir():
            raise FileNotFoundError(f"full conversion key is missing: {full_key}")
        split_key_package(full_key, online_key, offline_key)
    elif not online_key.is_dir() or not offline_key.is_dir():
        raise FileNotFoundError("online/offline key package is only partially finalized")
    if bool(plan.security.get("offline_key_encrypted", True)):
        plaintext_master = offline_key / "offline_master_key.safetensors"
        encrypted_manifest = offline_key / "manifest.json"
        if plaintext_master.is_file():
            _encrypt_offline_if_required(
                plan, offline_key, password=offline_key_password
            )
        elif not encrypted_manifest.is_file():
            raise FileNotFoundError("encrypted offline-key manifest is missing")
    return full_key, online_key, offline_key


def finalize_tee_key_package(
    plan: ConversionPlan,
    output: Path,
    *,
    offline_key_password: str | None = None,
) -> Path:
    """Move the structural conversion key into one encrypted offline archive.

    TEE mode has no online TokenKey.  Its online boundary material is emitted
    separately under ``plan.keys.tee`` and is provisioned only after attestation.
    """

    full_key, _, offline_key = _key_paths(plan, output)
    if not offline_key.exists():
        if not full_key.is_dir():
            raise FileNotFoundError(f"full TEE conversion key is missing: {full_key}")
        offline_key.parent.mkdir(parents=True, exist_ok=True)
        os.replace(full_key, offline_key)
        source_master = offline_key / "paper_key.safetensors"
        target_master = offline_key / "offline_master_key.safetensors"
        if not source_master.is_file():
            raise FileNotFoundError("TEE structural key tensor archive is missing")
        os.replace(source_master, target_master)
        metadata_path = offline_key / "key.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError("TEE offline key metadata is not an object")
        metadata["vocab_file"] = target_master.name
        metadata["package_type"] = "tee_structural_offline_key"
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (offline_key / "key_manifest.json").unlink(missing_ok=True)
    if bool(plan.security.get("offline_key_encrypted", True)):
        plaintext_master = offline_key / "offline_master_key.safetensors"
        encrypted_manifest = offline_key / "manifest.json"
        if plaintext_master.is_file():
            _encrypt_offline_if_required(
                plan, offline_key, password=offline_key_password
            )
        elif not encrypted_manifest.is_file():
            raise FileNotFoundError("encrypted TEE offline-key manifest is missing")
    return offline_key


def _run_qwen(
    plan: ConversionPlan,
    output: Path,
    source: Path | None = None,
    *,
    offline_key_password: str | None = None,
) -> dict[str, Any]:
    source = source or Path(str(plan.source["path"]))
    full_key, online_key, offline_key = _key_paths(plan, output)
    security_profile = plan.security_profile()
    tee_boundary = Path(
        str(plan.keys.get("tee", output.parent / f"{output.name}-tee-boundary"))
    )
    conversion = plan.conversion
    root = Path(__file__).resolve().parents[3]
    output_partial = output.with_name(f"{output.name}.partial")
    key_partial = full_key.with_name(f"{full_key.name}.partial")
    tee_partial = tee_boundary.with_name(f"{tee_boundary.name}.partial")
    partials = (output_partial, key_partial, tee_partial) if security_profile.is_tee else (
        output_partial,
        key_partial,
    )
    for partial in partials:
        if partial.is_dir():
            try:
                partial.rmdir()
            except OSError as error:
                raise FileExistsError(
                    f"non-empty partial conversion directory requires inspection: {partial}"
                ) from error
    command = [
        _subprocess_python(),
        str(root / "scripts" / "convert_paper_qwen2_checkpoint.py"),
        "--source",
        str(source),
        "--output",
        str(output),
        "--key-dir",
        str(full_key),
        "--h",
        str(conversion["expansion_h"]),
        "--lambda",
        str(conversion["lambda"]),
        "--seed",
        str(conversion["seed"]),
        "--alpha-e",
        str(conversion["alpha_e"]),
        "--alpha-h",
        str(conversion["alpha_h"]),
        "--dtype",
        str(conversion["dtype"]),
        "--attention-compute-dtype",
        "float32",
        "--rms-mode",
        "exact-metric",
        "--rms-representation",
        "stable-factor",
        "--block-beta",
        str(conversion["block_beta"]),
        "--sampling-gamma",
        str(conversion["sampling_gamma"]),
        "--blockperm-mode",
        "paper-distribution-boundary-corrected",
        "--rope-frequency-mode",
        "qwen-actual",
        "--qk-scale-min",
        str(conversion["qk_scale_min"]),
        "--qk-scale-max",
        str(conversion["qk_scale_max"]),
        "--ffn-scale-min",
        str(conversion["ffn_scale_min"]),
        "--ffn-scale-max",
        str(conversion["ffn_scale_max"]),
        "--uvo-condition-max",
        str(conversion["value_condition_max"]),
        "--model-id",
        str(plan.output.get("model_id", output.name)),
        "--key-id",
        str(plan.output.get("key_id", f"key-{plan.job_id[:8]}")),
        "--algorithm2",
    ]
    if security_profile.is_tee:
        if security_profile.tee_backend is None:
            raise ValueError("tee_gm conversion has no tee_backend")
        command.extend(
            [
                "--security-mode",
                "tee-gm",
                "--boundary-mode",
                "tee-split",
                "--tee-backend",
                str(security_profile.tee_backend.value).replace("_", "-"),
                "--tee-output",
                str(tee_boundary),
            ]
        )
        gm_helper = plan.security.get("gm_crypto_helper")
        gm_signing_key = plan.security.get("gm_signing_key")
        gm_boundary_key = plan.security.get("gm_boundary_key")
        if gm_helper:
            command.extend(["--gm-crypto-helper", str(gm_helper)])
        if gm_signing_key:
            command.extend(["--gm-signing-key", str(gm_signing_key)])
        if gm_boundary_key:
            command.extend(["--gm-boundary-key", str(gm_boundary_key)])
    child_environment = os.environ.copy()
    child_environment.setdefault("PYTHONFAULTHANDLER", "1")
    child_environment.setdefault("OMP_NUM_THREADS", "4")
    child_environment.setdefault("MKL_NUM_THREADS", "4")
    log_path = output.parent / f"{output.name}-conversion.log"
    try:
        with log_path.open("a", encoding="utf-8", errors="backslashreplace") as log:
            subprocess.run(
                command,
                cwd=root,
                check=True,
                env=child_environment,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
    except subprocess.CalledProcessError as error:
        tail = ""
        if log_path.is_file():
            tail = "\n".join(
                log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-20:]
            )
        unsigned_code = int(error.returncode) & 0xFFFFFFFF
        raise RuntimeError(
            "Qwen checkpoint conversion failed: "
            f"exit={error.returncode} (0x{unsigned_code:08X}), log={log_path}"
            + (f", last output:\n{tail}" if tail else "")
        ) from error
    tee_offline_key: Path | None = None
    if not security_profile.is_tee:
        finalize_qwen_key_package(
            plan, output, offline_key_password=offline_key_password
        )
    else:
        tee_offline_key = finalize_tee_key_package(
            plan, output, offline_key_password=offline_key_password
        )
    return {
        "adapter": plan.adapter,
        "output": str(output.resolve()),
        "full_key": None if security_profile.is_tee else str(full_key.resolve()),
        "online_key": None if security_profile.is_tee else str(online_key.resolve()),
        "offline_key": (
            str(tee_offline_key.resolve())
            if tee_offline_key is not None
            else str(offline_key.resolve())
        ),
        "tee_boundary": str(tee_boundary.resolve()) if security_profile.is_tee else None,
        "security_mode": security_profile.security_mode.value,
        "hardware_attested": False,
    }


def convert_qwen_checkpoint(
    plan: ConversionPlan,
    output: Path,
    source: Path,
    *,
    offline_key_password: str | None = None,
) -> dict[str, Any]:
    """Public compatibility entry used by the product pipeline.

    Qwen2/Qwen2.5 checkpoints share the adapter.  The converter accepts both
    single-file and indexed Safetensors checkpoints; callers must still enforce
    the host-memory recommendation recorded by the catalog for larger models.
    """

    return _run_qwen(
        plan,
        output,
        source,
        offline_key_password=offline_key_password,
    )


def convert_model_checkpoint(
    plan: ConversionPlan,
    output: Path,
    source: Path,
    *,
    offline_key_password: str | None = None,
    store: JobStore | None = None,
    progress_callback: Callable[[str, int, str], None] | None = None,
) -> dict[str, Any]:
    """Dispatch a validated plan to its architecture-family converter.

    Product callers must use this entry point.  Keeping the dispatch here
    prevents a catalog entry from being accepted by one adapter and then being
    accidentally sent through another family's converter.
    """

    if plan.adapter in {"qwen2", "qwen3_dense"}:
        if offline_key_password is None:
            return _run_qwen(plan, output, source)
        return _run_qwen(plan, output, source, offline_key_password=offline_key_password)
    if plan.adapter == "glm_dense":
        normalized = output.parent / f".{output.name}-glm-canonical"
        normalization = normalize_glm_dense_checkpoint(source, normalized, resume=True)
        result = _run_qwen(
            plan,
            output,
            normalized,
            offline_key_password=offline_key_password,
        )
        result["normalization"] = normalization
        return result
    if plan.adapter == "glm4_moe":
        full_key, online_key, _ = _key_paths(plan, output)
        conversion = plan.conversion
        result = convert_glm4_moe_checkpoint(
            source_root=source,
            output_root=output,
            key_root=full_key,
            online_key_root=online_key,
            seed=int(conversion["seed"]),
            expansion_h=int(conversion["expansion_h"]),
            coefficient_lambda=float(conversion["lambda"]),
            resume=True,
            model_id=str(plan.output.get("model_id", output.name)),
            key_id=str(plan.output.get("key_id", f"key-{plan.job_id[:8]}")),
        )
        _encrypt_offline_if_required(
            plan,
            full_key,
            password=offline_key_password,
        )
        return {
            "adapter": plan.adapter,
            "output": str(output.resolve()),
            "result": result,
        }
    if plan.adapter == "kimi_k2":
        source_config = json.loads((source / "config.json").read_text(encoding="utf-8"))
        if source_config.get("model_type") == "kimi_k25":
            return _run_deepseek(
                plan,
                output,
                store,
                source,
                offline_key_password=offline_key_password,
            )
        normalized = output.parent / f".{output.name}-kimi-text-canonical"
        normalization = normalize_kimi_k2_text_checkpoint(source, normalized, resume=True)
        result = _run_deepseek(
            plan,
            output,
            store,
            normalized,
            offline_key_password=offline_key_password,
        )
        result["normalization"] = normalization
        return result
    if plan.adapter in {"deepseek_v2", "deepseek_v3"}:
        return _run_deepseek(
            plan,
            output,
            store,
            source,
            offline_key_password=offline_key_password,
            progress_callback=progress_callback,
        )
    raise ValueError(
        f"adapter {plan.adapter!r} has no executable checkpoint converter; "
        "inspection support is not conversion support"
    )


def _run_deepseek(
    plan: ConversionPlan,
    output: Path,
    store: JobStore | None = None,
    source: Path | None = None,
    *,
    offline_key_password: str | None = None,
    progress_callback: Callable[[str, int, str], None] | None = None,
) -> dict[str, Any]:
    source = source or Path(str(plan.source["path"]))
    full_key, online_key, _ = _key_paths(plan, output)
    conversion = plan.conversion
    raw_config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    config = (
        dict(raw_config["text_config"])
        if raw_config.get("model_type") == "kimi_k25"
        and isinstance(raw_config.get("text_config"), dict)
        else raw_config
    )
    estimate = estimate_deepseek_host_memory(
        config,
        expansion_h=int(conversion["expansion_h"]),
        host_budget_gib=plan.resources.host_memory_budget_gib,
    )
    if not estimate.pass_:
        raise MemoryError(
            f"conversion plan exceeds host budget: {estimate.estimated_peak_gib:.3f} GiB"
        )

    def progress(name: str, tile_index: int, phase: str) -> None:
        if progress_callback is not None:
            progress_callback(name, tile_index, phase)
        if store is None:
            return
        job = store.get(plan.job_id)
        state = JobState(job["state"])
        if state == JobState.PAUSED:
            raise ConversionPaused(f"conversion paused before {name}")
        if state == JobState.CANCELLED:
            raise ConversionCancelled(f"conversion cancelled before {name}")
        payload = {
            **job.get("progress", {}),
            "conversion": {"tensor": name, "tile": tile_index, "phase": phase},
        }
        store.update_progress(plan.job_id, payload)
        if phase in {"completed", "resumed"}:
            digest = hashlib.sha256(f"{name}:{tile_index}:{phase}".encode()).hexdigest()
            store.record_tile(plan.job_id, name, tile_index, digest)

    result = convert_deepseek_checkpoint(
        source_root=source,
        output_root=output,
        key_root=full_key,
        online_key_root=online_key,
        seed=int(conversion["seed"]),
        ffn_scale_min=float(conversion["ffn_scale_min"]),
        ffn_scale_max=float(conversion["ffn_scale_max"]),
        vocab_permutation=True,
        resume=True,
        paper_complete=plan.adapter in {"deepseek_v3", "kimi_k2"},
        expansion_h=int(conversion["expansion_h"]),
        coefficient_lambda=float(conversion["lambda"]),
        alpha_e=float(conversion["alpha_e"]),
        alpha_h=float(conversion["alpha_h"]),
        block_beta=int(conversion["block_beta"]),
        sampling_gamma=float(conversion["sampling_gamma"]),
        qk_scale_min=float(conversion["qk_scale_min"]),
        qk_scale_max=float(conversion["qk_scale_max"]),
        value_condition_max=float(conversion["value_condition_max"]),
        progress_callback=progress,
        model_id=str(plan.output.get("model_id", output.name)),
        key_id=str(plan.output.get("key_id", f"key-{plan.job_id[:8]}")),
    )
    _encrypt_offline_if_required(
        plan,
        full_key,
        password=offline_key_password,
    )
    return {
        "adapter": plan.adapter,
        "output": str(output.resolve()),
        "resource_estimate": estimate.to_dict(),
        "result": result,
    }


def _encrypt_offline_if_required(
    plan: ConversionPlan,
    offline_directory: Path,
    *,
    password: str | None = None,
) -> None:
    if not bool(plan.security.get("offline_key_encrypted", True)):
        return
    password = password or os.environ.get("YINBIAN_OFFLINE_KEY_PASSWORD") or os.environ.get(
        "ALOEPRI_OFFLINE_KEY_PASSWORD"
    )
    if not password:
        raise ValueError(
            "YINBIAN_OFFLINE_KEY_PASSWORD is required when offline_key_encrypted=true"
        )
    encrypt_offline_key_directory(offline_directory, password)


def execute_conversion_plan(plan: ConversionPlan, store: JobStore) -> dict[str, Any]:
    try:
        try:
            job = store.get(plan.job_id)
        except KeyError:
            store.create(plan.job_id, plan.to_dict())
            job = store.get(plan.job_id)
        state = JobState(job["state"])
        if state == JobState.CREATED:
            store.transition(plan.job_id, JobState.PREFLIGHT)
        elif state == JobState.FAILED:
            store.transition(plan.job_id, JobState.PREFLIGHT)
        elif state == JobState.PAUSED:
            store.resume(plan.job_id)
        elif state not in {JobState.PREFLIGHT, JobState.DOWNLOADING, JobState.CONVERTING}:
            raise ValueError(f"job cannot convert from state {state.value}")
        source = _resolve_source(plan, store)
        state = JobState(store.get(plan.job_id)["state"])
        if state == JobState.PREFLIGHT:
            store.transition(plan.job_id, JobState.CONVERTING)
        elif state == JobState.DOWNLOADING:
            store.transition(plan.job_id, JobState.CONVERTING)
        output = _local_output(plan)
        with ResourceBudgetMonitor(
            gpu_budget_gib=plan.resources.gpu_memory_budget_gib,
            minimum_free_gpu_gib=plan.resources.minimum_free_gpu_gib,
            host_budget_gib=plan.resources.host_memory_budget_gib,
        ) as monitor:
            result = convert_model_checkpoint(
                plan,
                output,
                source,
                store=store,
            )
        assert monitor.observation is not None
        result["resources"] = {
            "peak_rss_gib": monitor.observation.peak_rss_gib,
            "peak_gpu_allocated_gib": monitor.observation.peak_gpu_allocated_gib,
            "gpu_free_before_gib": monitor.observation.gpu_free_before_gib,
            "pass": monitor.observation.pass_,
        }
        # Local conversion is complete.  The explicit upload command owns the
        # UPLOADING -> VERIFYING -> READY_TO_DEPLOY transitions so a real object
        # store can record multipart evidence before deployment becomes legal.
        store.transition(plan.job_id, JobState.UPLOADING, progress=result)
        return result
    except (ConversionPaused, ConversionCancelled):
        raise
    except BaseException as error:
        try:
            current = JobState(store.get(plan.job_id)["state"])
            if current in {
                JobState.PREFLIGHT,
                JobState.DOWNLOADING,
                JobState.CONVERTING,
                JobState.UPLOADING,
                JobState.VERIFYING,
                JobState.DEPLOYING,
                JobState.RUNNING,
            }:
                store.transition(plan.job_id, JobState.FAILED, error=str(error))
        except (KeyError, ValueError):
            pass
        raise
