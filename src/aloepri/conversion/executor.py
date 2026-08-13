from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from aloepri.conversion.deepseek_streaming import convert_deepseek_checkpoint
from aloepri.jobs.store import JobState, JobStore
from aloepri.keys.encryption import encrypt_offline_key_directory
from aloepri.packaging import split_key_package
from aloepri.planning import ConversionPlan
from aloepri.resources import ResourceBudgetMonitor


def _local_output(plan: ConversionPlan) -> Path:
    uri = str(plan.output["uri"])
    if uri.startswith("s3://"):
        staging = plan.output.get("staging_path")
        if not staging:
            raise ValueError("S3 output requires output.staging_path for local conversion")
        return Path(str(staging))
    return Path(uri)


def _key_paths(plan: ConversionPlan, output: Path) -> tuple[Path, Path, Path]:
    root = output.parent / f"{output.name}-keys"
    return (
        Path(str(plan.keys.get("full", root / "full"))),
        Path(str(plan.keys.get("online", root / "online"))),
        Path(str(plan.keys.get("offline", root / "offline"))),
    )


def _run_qwen(plan: ConversionPlan, output: Path) -> dict[str, Any]:
    source = Path(str(plan.source["path"]))
    full_key, online_key, offline_key = _key_paths(plan, output)
    conversion = plan.conversion
    root = Path(__file__).resolve().parents[3]
    command = [
        sys.executable,
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
        "exact_metric",
        "--rms-representation",
        "stable_factor",
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
    subprocess.run(command, cwd=root, check=True)
    split_key_package(full_key, online_key, offline_key)
    _encrypt_offline_if_required(plan, offline_key)
    return {
        "adapter": plan.adapter,
        "output": str(output.resolve()),
        "full_key": str(full_key.resolve()),
        "online_key": str(online_key.resolve()),
        "offline_key": str(offline_key.resolve()),
    }


def _run_deepseek(plan: ConversionPlan, output: Path) -> dict[str, Any]:
    source = Path(str(plan.source["path"]))
    full_key, online_key, _ = _key_paths(plan, output)
    conversion = plan.conversion
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
        paper_complete=plan.adapter == "deepseek_v3",
        expansion_h=int(conversion["expansion_h"]),
        coefficient_lambda=float(conversion["lambda"]),
        alpha_e=float(conversion["alpha_e"]),
        alpha_h=float(conversion["alpha_h"]),
        block_beta=int(conversion["block_beta"]),
        sampling_gamma=float(conversion["sampling_gamma"]),
        qk_scale_min=float(conversion["qk_scale_min"]),
        qk_scale_max=float(conversion["qk_scale_max"]),
        value_condition_max=float(conversion["value_condition_max"]),
    )
    _encrypt_offline_if_required(plan, full_key)
    return {"adapter": plan.adapter, "output": str(output.resolve()), "result": result}


def _encrypt_offline_if_required(plan: ConversionPlan, offline_directory: Path) -> None:
    if not bool(plan.security.get("offline_key_encrypted", True)):
        return
    password = os.environ.get("ALOEPRI_OFFLINE_KEY_PASSWORD")
    if not password:
        raise ValueError(
            "ALOEPRI_OFFLINE_KEY_PASSWORD is required when offline_key_encrypted=true"
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
            store.transition(plan.job_id, JobState.CONVERTING)
        elif state == JobState.PAUSED:
            store.resume(plan.job_id)
        elif state != JobState.CONVERTING:
            raise ValueError(f"job cannot convert from state {state.value}")
        output = _local_output(plan)
        with ResourceBudgetMonitor(
            gpu_budget_gib=plan.resources.gpu_memory_budget_gib,
            minimum_free_gpu_gib=plan.resources.minimum_free_gpu_gib,
            host_budget_gib=plan.resources.host_memory_budget_gib,
        ) as monitor:
            result = (
                _run_qwen(plan, output)
                if plan.adapter == "qwen2"
                else _run_deepseek(plan, output)
            )
        assert monitor.observation is not None
        result["resources"] = {
            "peak_rss_gib": monitor.observation.peak_rss_gib,
            "peak_gpu_allocated_gib": monitor.observation.peak_gpu_allocated_gib,
            "gpu_free_before_gib": monitor.observation.gpu_free_before_gib,
            "pass": monitor.observation.pass_,
        }
        store.transition(plan.job_id, JobState.VERIFYING, progress=result)
        store.transition(plan.job_id, JobState.READY_TO_DEPLOY, progress=result)
        return result
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
