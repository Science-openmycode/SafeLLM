from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer
import uvicorn
import yaml

if TYPE_CHECKING:
    from aloepri.client.sdk import PrivateInferenceClient

from aloepri.product_cli import register_plan_job, register_product_commands

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="隐变智模：本地模型改造、私有部署和Token-ID问答工具。",
)
register_product_commands(app)


def _version_callback(value: bool) -> None:
    if value:
        from aloepri import __version__

        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def main_options(
    version: Annotated[
        bool | None,
        typer.Option("--version", callback=_version_callback, is_eager=True),
    ] = None,
) -> None:
    """隐变智模命令行。"""


def legacy_main() -> None:
    typer.echo(
        "提示：aloepri 命令将在下一个兼容周期移除，请改用 yinbian。",
        err=True,
    )
    app(prog_name="aloepri")


def load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise typer.BadParameter("configuration root must be a mapping")
    return payload


def run_checked(arguments: list[str]) -> None:
    typer.echo(f"$ {' '.join(arguments)}")
    subprocess.run(arguments, check=True)


def append_optional_noise_seed_arguments(
    command: list[str], conversion: dict[str, Any]
) -> None:
    for key, option in (
        ("embedding_noise_seed", "--embedding-noise-seed"),
        ("head_noise_seed", "--head-noise-seed"),
    ):
        value = conversion.get(key)
        if value is not None:
            command.extend((option, str(int(value))))


def summarize_product_verification(
    conversion: dict[str, Any],
    formula: dict[str, Any],
    layerwise: dict[str, Any],
    generation: dict[str, Any],
) -> dict[str, Any]:
    """Apply different correctness gates to exact and intentionally noisy models."""

    noise_active = float(conversion.get("alpha_e", 0.0)) != 0.0 or float(
        conversion.get("alpha_h", 0.0)
    ) != 0.0
    formula_pass = bool(formula.get("overall_pass", False))
    layerwise_pass = bool(layerwise.get("all_pass", False))
    runtime_checks = {
        "next_token_equal_after_inverse": bool(
            generation.get("next_token_equal_after_inverse", False)
        ),
        "decode_input_is_tau_plain_next": bool(
            generation.get("decode_input_is_tau_plain_next", False)
        ),
        "greedy_ids_equal": bool(generation.get("greedy_ids_equal", False)),
        "cache_lengths_advance": (
            int(generation.get("private_prefill_cache_length", -1))
            == int(generation.get("plain_prefill_cache_length", -2))
            and int(generation.get("decode_cache_length", -1))
            == int(generation.get("private_prefill_cache_length", -2)) + 1
        ),
    }
    runtime_required_checks = (
        {
            "decode_input_is_tau_plain_next": runtime_checks[
                "decode_input_is_tau_plain_next"
            ],
            "cache_lengths_advance": runtime_checks["cache_lengths_advance"],
        }
        if noise_active
        else runtime_checks
    )
    runtime_pass = all(runtime_required_checks.values())
    # Non-zero paper noise intentionally changes activations.  Its utility is
    # judged by task-level evaluation, not the zero-noise NRMSE identity gate.
    layerwise_required = not noise_active
    functional_pass = formula_pass and runtime_pass and (
        layerwise_pass if layerwise_required else True
    )
    return {
        "schema_version": 1,
        "noise_active": noise_active,
        "formula_pass": formula_pass,
        "layerwise": {
            "required": layerwise_required,
            "pass": layerwise_pass,
            "role": "gate" if layerwise_required else "diagnostic_only",
            "first_failure": layerwise.get("first_failure"),
        },
        "runtime_checks": runtime_checks,
        "runtime_required_checks": runtime_required_checks,
        "runtime_pass": runtime_pass,
        "functional_pass": functional_pass,
        "accuracy_gate": "external_task_evaluation_required" if noise_active else "not_applicable",
    }


@app.command()
def convert(
    config: Annotated[
        Path | None, typer.Option("--config", exists=True, dir_okay=False)
    ] = None,
    plan: Annotated[Path | None, typer.Option("--plan", exists=True, dir_okay=False)] = None,
    execute: Annotated[
        bool, typer.Option("--execute/--schedule-only")
    ] = True,
    password: Annotated[str | None, typer.Option("--password", hide_input=True)] = None,
    private_key_passphrase: Annotated[
        str | None, typer.Option("--private-key-passphrase", hide_input=True)
    ] = None,
    accept_license: Annotated[bool, typer.Option("--accept-license")] = False,
) -> None:
    """Convert through a product plan, or use the compatible legacy Qwen config."""

    from aloepri.conversion.executor import execute_conversion_plan
    from aloepri.jobs.store import JobState, JobStore
    from aloepri.packaging import split_key_package
    from aloepri.planning import ConversionPlan

    if plan is not None:
        if config is not None:
            raise typer.BadParameter("use either --plan or --config, not both")
        current_plan = ConversionPlan.load(plan)
        if current_plan.schema_version >= 2:
            if current_plan.source.get("type") != "huggingface":
                legacy_store = JobStore(
                    Path(os.environ["YINBIAN_STATE_DB"])
                    if os.environ.get("YINBIAN_STATE_DB")
                    else (
                        Path(os.environ["ALOEPRI_STATE_DB"])
                        if os.environ.get("ALOEPRI_STATE_DB")
                        else None
                    )
                )
                try:
                    legacy_store.get(current_plan.job_id)
                except KeyError:
                    legacy_store.create(current_plan.job_id, current_plan.to_dict())
                if JobState(legacy_store.get(current_plan.job_id)["state"]) == JobState.CREATED:
                    legacy_store.transition(current_plan.job_id, JobState.PREFLIGHT)
                if execute:
                    execute_conversion_plan(current_plan, legacy_store)
                typer.echo(
                    json.dumps(
                        legacy_store.get(current_plan.job_id), ensure_ascii=False, indent=2
                    )
                )
                return
            from aloepri.cloud.ssh import SSHProfile
            from aloepri.product.pipeline import (
                ProgressiveConversionPipeline,
                SSHDirectorySink,
            )
            from aloepri.product.state import ProductJobStatus, ProductStore

            state_path = os.environ.get("YINBIAN_STATE_DB") or os.environ.get(
                "ALOEPRI_STATE_DB"
            )
            product = ProductStore(None if state_path is None else Path(state_path))
            pipeline = ProgressiveConversionPipeline(product)
            mode = str(current_plan.output.get("deployment_mode", "local-only"))
            if not execute:
                try:
                    product.create_job(
                        current_plan.job_id,
                        {
                            "schema_version": 2,
                            "mode": mode,
                            "conversion": current_plan.to_dict(),
                        },
                    )
                    product.transition_job(
                        current_plan.job_id, ProductJobStatus.AWAITING_CONFIRMATION
                    )
                except Exception as error:
                    if "UNIQUE constraint" not in str(error):
                        raise
                typer.echo(
                    json.dumps(product.get_job(current_plan.job_id), ensure_ascii=False, indent=2)
                )
                return
            sink = None
            if mode == "direct-deploy":
                server_id = current_plan.output.get("server_id")
                if not server_id:
                    raise typer.BadParameter("direct-deploy plan has no server_id")
                server = product.get_server(str(server_id))
                private_key = server.get("private_key_path")
                profile = SSHProfile(
                    host=str(server["host"]),
                    port=int(server["port"]),
                    username=str(server["username"]),
                    password=password,
                    private_key=None if not private_key else Path(str(private_key)),
                    private_key_passphrase=private_key_passphrase,
                    host_key_fingerprint=server.get("host_key_fingerprint"),
                    sudo_mode=str(server["sudo_mode"]),
                    model_root=str(server["model_root"]),
                )
                sink = SSHDirectorySink(
                    profile,
                    f"{profile.model_root}/uploads/{current_plan.job_id}",
                )
            if str(current_plan.source.get("type")) == "local":
                result = pipeline.run_local_model(
                    current_plan,
                    mode=mode,
                    sink=sink,
                    offline_key_password=os.environ.get("YINBIAN_OFFLINE_KEY_PASSWORD"),
                )
            else:
                result = pipeline.run_catalog_model(
                    current_plan,
                    mode=mode,
                    token=os.environ.get("YINBIAN_HF_TOKEN"),
                    sink=sink,
                    clean_source_after_commit=bool(
                        current_plan.security.get("clean_source_after_commit", False)
                    ),
                    accept_license=accept_license,
                )
            typer.echo(json.dumps(result, ensure_ascii=False, indent=2))
            return
        job = register_plan_job(plan)
        store = JobStore(
            Path(os.environ["ALOEPRI_STATE_DB"])
            if os.environ.get("ALOEPRI_STATE_DB")
            else None
        )
        state = JobState(job["state"])
        if state == JobState.CREATED:
            store.transition(job["job_id"], JobState.PREFLIGHT)
        if execute:
            execute_conversion_plan(current_plan, store)
        typer.echo(json.dumps(store.get(job["job_id"]), ensure_ascii=False, indent=2))
        return
    if config is None:
        raise typer.BadParameter("one of --plan or --config is required")

    cfg = load_config(config)
    conversion = cfg["conversion"]
    command = [
        sys.executable,
        "scripts/convert_paper_qwen2_checkpoint.py",
        "--source",
        str(cfg["source_model"]),
        "--output",
        str(cfg["output_model"]),
        "--key-dir",
        str(cfg["full_key_dir"]),
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
        str(conversion.get("attention_compute_dtype", "float32")),
        "--rms-mode",
        str(conversion["rms_mode"]),
        "--rms-representation",
        str(conversion.get("rms_representation", "gram")),
        "--block-beta",
        str(conversion["block_beta"]),
        "--sampling-gamma",
        str(conversion["sampling_gamma"]),
        "--blockperm-mode",
        str(conversion["blockperm_mode"]),
        "--rope-frequency-mode",
        str(conversion["rope_frequency_mode"]),
        "--qk-scale-min",
        str(conversion["qk_scale_min"]),
        "--qk-scale-max",
        str(conversion["qk_scale_max"]),
        "--ffn-scale-min",
        str(conversion["ffn_scale_min"]),
        "--ffn-scale-max",
        str(conversion["ffn_scale_max"]),
        "--uvo-condition-max",
        str(conversion["uvo_condition_max"]),
        "--model-id",
        str(cfg["model_id"]),
        "--key-id",
        str(cfg["key_id"]),
        "--algorithm2",
    ]
    append_optional_noise_seed_arguments(command, conversion)
    run_checked(command)
    split_key_package(
        Path(cfg["full_key_dir"]),
        Path(cfg["online_key_dir"]),
        Path(cfg["offline_key_dir"]),
    )
    typer.echo("conversion and key split completed")


@app.command()
def verify(
    config: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)],
) -> None:
    """Run formula, layer-wise, cache, and generation verification."""

    cfg = load_config(config)
    evidence = Path(cfg.get("evidence_dir", "artifacts/verification/product"))
    evidence.mkdir(parents=True, exist_ok=True)
    run_checked(
        [
            sys.executable,
            "scripts/verify_paper_formula_checkpoint.py",
            "--source",
            str(cfg["source_model"]),
            "--private",
            str(cfg["output_model"]),
            "--key-dir",
            str(cfg["full_key_dir"]),
            "--out",
            str(evidence / "formula.json"),
        ]
    )
    run_checked(
        [
            sys.executable,
            "scripts/verify_layerwise_equivalence.py",
            "--source",
            str(cfg["source_model"]),
            "--private",
            str(cfg["output_model"]),
            "--key-dir",
            str(cfg["full_key_dir"]),
            "--out",
            str(evidence / "layerwise.json"),
            "--private-attention-compute-dtype",
            str(cfg["conversion"].get("attention_compute_dtype", "float32")),
        ]
    )
    run_checked(
        [
            sys.executable,
            "scripts/verify_paper_qwen2_checkpoint.py",
            "--source",
            str(cfg["source_model"]),
            "--private",
            str(cfg["output_model"]),
            "--key-dir",
            str(cfg["full_key_dir"]),
            "--out",
            str(evidence / "generation.json"),
            "--dtype",
            str(cfg["conversion"]["dtype"]),
        ]
    )
    summary = summarize_product_verification(
        cfg["conversion"],
        json.loads((evidence / "formula.json").read_text(encoding="utf-8")),
        json.loads((evidence / "layerwise.json").read_text(encoding="utf-8")),
        json.loads((evidence / "generation.json").read_text(encoding="utf-8")),
    )
    summary_path = evidence / "verification_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    typer.echo(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["functional_pass"]:
        raise typer.Exit(1)


@app.command()
def serve(
    config: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)],
) -> None:
    """Start the token-ID-only HF inference service."""

    from aloepri.serving.app import create_app
    from aloepri.serving.hf_runtime import PrivateHFRuntime

    cfg = load_config(config)
    server = cfg.get("server", {})
    host = str(server.get("host", "127.0.0.1"))
    bearer_env = str(server.get("bearer_token_env", "ALOEPRI_BEARER_TOKEN"))
    bearer_token = os.environ.get(bearer_env)
    local_host = host in {"127.0.0.1", "localhost", "::1"}
    tls_certfile = server.get("tls_certfile")
    tls_keyfile = server.get("tls_keyfile")
    proxy_tls = bool(server.get("tls_terminated_by_proxy", False))
    if not local_host:
        if not bearer_token:
            raise typer.BadParameter(f"remote binding requires bearer token in {bearer_env}")
        if not proxy_tls and not (tls_certfile and tls_keyfile):
            raise typer.BadParameter(
                "remote binding requires a TLS certificate/key or tls_terminated_by_proxy=true"
            )
    runtime = PrivateHFRuntime(
        Path(server.get("model_path", cfg["output_model"])),
        device=str(server.get("device", "auto")),
        dtype=str(server.get("dtype", cfg["conversion"]["dtype"])),
        max_input_tokens=int(server.get("max_input_tokens", 2048)),
        max_output_tokens=int(server.get("max_output_tokens", 512)),
        gpu_memory_fraction=float(server.get("gpu_memory_fraction", 0.70)),
    )
    api = create_app(
        runtime,
        bearer_token=bearer_token,
        max_request_bytes=int(server.get("max_request_bytes", 1_000_000)),
        enable_text_compat=bool(server.get("enable_text_compat", False)),
    )
    uvicorn.run(
        api,
        host=host,
        port=int(server.get("port", 8000)),
        access_log=False,
        ssl_certfile=str(tls_certfile) if tls_certfile else None,
        ssl_keyfile=str(tls_keyfile) if tls_keyfile else None,
    )


@app.command("serve-tee-sim")
def serve_tee_sim(
    model: Annotated[Path, typer.Option("--model", exists=True, file_okay=False)],
    tee_boundary: Annotated[
        Path, typer.Option("--tee-boundary", exists=True, file_okay=False)
    ],
    port: Annotated[int, typer.Option("--port", min=1, max=65535)] = 8001,
    device: Annotated[str, typer.Option("--device")] = "auto",
    head_mode: Annotated[str, typer.Option("--head-mode")] = "local",
) -> None:
    """Run the explicit non-production TEE functional simulator."""

    from aloepri.serving.tee_app import create_tee_app
    from aloepri.serving.tee_runtime import TeeSplitHFRuntime
    from aloepri.tee.attestation import SoftwareAttestor, sm3_files
    from aloepri.tee.service import TeeAttestationService, TeeDeploymentIdentity

    runtime = TeeSplitHFRuntime(
        model,
        tee_boundary,
        device=device,
        head_mode=head_mode,
    )
    manifest = json.loads((tee_boundary / "tee-manifest.json").read_text(encoding="utf-8"))
    service = TeeAttestationService(
        identity=TeeDeploymentIdentity(
            model_id=runtime.model_id,
            model_version=str(manifest.get("model_version", "unknown")),
            key_id=runtime.key_id,
            runtime_hash_sm3=sm3_files(
                [Path(__file__), Path(__file__).parent / "serving" / "tee_runtime.py"]
            ),
            server_manifest_sm3=sm3_files([model / "server-manifest.json"]),
            sm2_public_key_der=b"SOFTWARE-SIMULATION",
            sm2_certificate_pem="SOFTWARE SIMULATION - NO CERTIFICATE",
        ),
        attestor=SoftwareAttestor(),
        initially_provisioned=True,
    )
    uvicorn.run(
        create_tee_app(runtime, service),
        host="127.0.0.1",
        port=port,
        access_log=False,
    )


@app.command("chat-tee")
def chat_tee(
    tokenizer: Annotated[Path, typer.Option("--tokenizer", exists=True, file_okay=False)],
    model_id: Annotated[str, typer.Option("--model-id")],
    model_version: Annotated[str, typer.Option("--model-version")],
    key_id: Annotated[str, typer.Option("--key-id")],
    vocab_size: Annotated[int, typer.Option("--vocab-size", min=1)],
    transport_mode: Annotated[str, typer.Option("--transport-mode")] = "tee_gm",
    tee_backend: Annotated[str, typer.Option("--tee-backend")] = "intel_tdx",
    server: Annotated[str | None, typer.Option("--server")] = None,
    gm_client: Annotated[Path | None, typer.Option("--gm-client")] = None,
    gm_profile: Annotated[Path | None, typer.Option("--gm-profile")] = None,
    prompt: Annotated[str | None, typer.Option("--prompt")] = None,
    max_new_tokens: Annotated[int, typer.Option("--max-new-tokens", min=1, max=2048)] = 128,
) -> None:
    """Chat through ordinary Token IDs inside a TEE/GM transport (never a TokenKey)."""

    from transformers import AutoTokenizer

    from aloepri.client.tee_sdk import (
        NativeGmSession,
        SoftwareSimSession,
        TeeDeployment,
        TeeGmSession,
        TeeInferenceClient,
    )

    if transport_mode != "tee_gm":
        raise typer.BadParameter("--transport-mode must be tee_gm")
    session: TeeGmSession
    if tee_backend == "software_sim":
        if server is None:
            raise typer.BadParameter("software_sim requires --server http://127.0.0.1:PORT")
        session = SoftwareSimSession(server)
    elif tee_backend == "intel_tdx":
        if gm_client is None or gm_profile is None:
            raise typer.BadParameter("intel_tdx requires --gm-client and --gm-profile")
        session = NativeGmSession(gm_client, gm_profile)
    else:
        raise typer.BadParameter("--tee-backend must be software_sim or intel_tdx")
    client = TeeInferenceClient(
        tokenizer=AutoTokenizer.from_pretrained(tokenizer, local_files_only=True),
        deployment=TeeDeployment(
            model_id=model_id,
            model_version=model_version,
            key_id=key_id,
            vocab_size=vocab_size,
        ),
        session=session,
    )
    history: list[dict[str, str]] = []

    def run_turn(user_prompt: str) -> None:
        history.append({"role": "user", "content": user_prompt})
        pieces: list[str] = []
        for chunk in client.stream_chat(
            history,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
        ):
            typer.echo(chunk.text, nl=False)
            pieces.append(chunk.text)
        typer.echo()
        history.append({"role": "assistant", "content": "".join(pieces)})

    try:
        if prompt is not None:
            run_turn(prompt)
            return
        while True:
            user_prompt = typer.prompt("你")
            if user_prompt.strip() in {"/exit", "/quit"}:
                break
            if user_prompt.strip() == "/clear":
                history.clear()
                typer.echo("本地上下文已清除。")
                continue
            run_turn(user_prompt)
    finally:
        client.close()


@app.command("chat-direct", hidden=True)
def chat_direct(
    server: Annotated[str, typer.Option("--server")],
    key_dir: Annotated[Path, typer.Option("--key-dir", exists=True, file_okay=False)],
    tokenizer: Annotated[Path, typer.Option("--tokenizer", exists=True, file_okay=False)] = Path(
        "data/models/qwen2.5-0.5b"
    ),
    privacy_mode: Annotated[str, typer.Option("--privacy-mode")] = "permutation",
    epsilon1: Annotated[float | None, typer.Option("--epsilon1")] = None,
    bearer_token_env: Annotated[str, typer.Option("--bearer-token-env")] = "ALOEPRI_BEARER_TOKEN",
    max_new_tokens: Annotated[int, typer.Option("--max-new-tokens", min=1, max=2048)] = 128,
    assume_yes: Annotated[
        bool, typer.Option("--yes", help="Do not ask before each local M1 perturbation")
    ] = False,
) -> None:
    """Run an interactive client; plaintext and conversation history remain local."""

    from aloepri.client.sdk import PrivateInferenceClient
    from aloepri.privacy.rmdp import expected_m1_change_rate

    if privacy_mode == "rmdp" and epsilon1 is None:
        raise typer.BadParameter("--epsilon1 is required for rmdp mode")
    client = PrivateInferenceClient.from_directories(
        base_url=server,
        tokenizer_dir=tokenizer,
        key_dir=key_dir,
        bearer_token=os.environ.get(bearer_token_env),
    )
    history: list[dict[str, str]] = []
    last_stats: dict[str, object] | None = None
    typer.echo("隐变智模对话：/clear /stats /privacy /quit")
    try:
        while True:
            prompt = typer.prompt("you")
            if prompt == "/quit":
                break
            if prompt == "/clear":
                history.clear()
                typer.echo("local history cleared")
                continue
            if prompt == "/stats":
                typer.echo(json.dumps(last_stats or {}, ensure_ascii=False))
                continue
            if prompt == "/privacy":
                detail = {"privacy_mode": privacy_mode, "epsilon1": epsilon1}
                if privacy_mode == "rmdp" and epsilon1 is not None:
                    detail["expected_token_change_rate"] = expected_m1_change_rate(
                        client.key.tau.numel(), epsilon1
                    )
                typer.echo(json.dumps(detail, ensure_ascii=False))
                continue
            if privacy_mode == "rmdp" and epsilon1 is not None:
                expected_rate = expected_m1_change_rate(client.key.tau.numel(), epsilon1)
                typer.echo(
                    "local M1 expected token change rate: "
                    f"{expected_rate:.6f} ({expected_rate * 100:.2f}%)"
                )
                if not assume_yes and not typer.confirm("send this perturbed request"):
                    typer.echo("request cancelled locally")
                    continue
            history.append({"role": "user", "content": prompt})
            typer.echo("assistant: ", nl=False)
            chunks = client.stream_chat(
                history,
                max_new_tokens=max_new_tokens,
                privacy_mode=privacy_mode,
                epsilon1=epsilon1,
            )
            answer = ""
            elapsed: list[float] = []
            privacy_ledger: dict[str, float | int | str] | None = None
            for chunk in chunks:
                typer.echo(chunk.text, nl=False)
                answer = chunk.accumulated_text
                elapsed.append(chunk.elapsed_ms)
                privacy_ledger = chunk.privacy
            typer.echo()
            history.append({"role": "assistant", "content": answer})
            last_stats = {
                "output_tokens": len(elapsed),
                "ttft_ms": elapsed[0] if elapsed else 0.0,
                "tpot_ms": sum(elapsed[1:]) / max(1, len(elapsed) - 1),
            }
            if privacy_ledger is not None:
                last_stats["privacy"] = privacy_ledger
    finally:
        client.close()


chat_app = typer.Typer(no_args_is_help=False, invoke_without_command=True)


@chat_app.callback()
def chat_callback(
    context: typer.Context,
    server: Annotated[str | None, typer.Option("--server")] = None,
    key_dir: Annotated[Path | None, typer.Option("--key-dir")] = None,
    tokenizer: Annotated[Path, typer.Option("--tokenizer")] = Path(
        "data/models/qwen2.5-0.5b"
    ),
    privacy_mode: Annotated[str, typer.Option("--privacy-mode")] = "permutation",
    epsilon1: Annotated[float | None, typer.Option("--epsilon1")] = None,
    bearer_token_env: Annotated[str, typer.Option("--bearer-token-env")] = (
        "YINBIAN_BEARER_TOKEN"
    ),
    max_new_tokens: Annotated[int, typer.Option("--max-new-tokens")] = 128,
    assume_yes: Annotated[bool, typer.Option("--yes")] = False,
) -> None:
    if context.invoked_subcommand is not None:
        return
    if server is None or key_dir is None:
        raise typer.BadParameter(
            "use a chat subcommand, or provide --server and --key-dir for direct mode"
        )
    chat_direct(
        server,
        key_dir,
        tokenizer,
        privacy_mode,
        epsilon1,
        bearer_token_env,
        max_new_tokens,
        assume_yes,
    )


def _deployment_client(
    deployment_id: str,
    *,
    password: str | None,
    private_key_passphrase: str | None,
) -> tuple[PrivateInferenceClient, object | None]:
    from aloepri.client.sdk import PrivateInferenceClient
    from aloepri.cloud.ssh import SSHProfile
    from aloepri.cloud.tunnel import ManagedTunnel
    from aloepri.keys.directory_vault import materialize_online_key_directory
    from aloepri.keys.vault import CredentialVault
    from aloepri.product.paths import product_paths
    from aloepri.product.state import DeploymentStatus, ProductStore

    state_path = os.environ.get("YINBIAN_STATE_DB") or os.environ.get("ALOEPRI_STATE_DB")
    store = ProductStore(None if state_path is None else Path(state_path))
    deployment = store.get_deployment(deployment_id)
    if deployment["status"] != DeploymentStatus.HEALTHY.value:
        raise typer.BadParameter("deployment is not healthy")
    metadata = deployment["metadata"]
    endpoint = metadata.get("local_server_url")
    tunnel = None
    if endpoint is None:
        server = store.get_server(str(deployment["server_id"]))
        private_key = server.get("private_key_path")
        profile = SSHProfile(
            host=str(server["host"]),
            port=int(server["port"]),
            username=str(server["username"]),
            password=password,
            private_key=None if not private_key else Path(str(private_key)),
            private_key_passphrase=private_key_passphrase,
            host_key_fingerprint=server.get("host_key_fingerprint"),
            sudo_mode=str(server["sudo_mode"]),
            model_root=str(server["model_root"]),
        )
        tunnel = ManagedTunnel(
            deployment_id, profile, remote_port=int(deployment["remote_port"])
        )
        status = tunnel.open()
        if status.local_port is None:
            raise typer.BadParameter("SSH tunnel did not expose a local port")
        endpoint = f"http://127.0.0.1:{status.local_port}"
    tokenizer_dir = metadata.get("tokenizer_dir")
    online_key_dir = metadata.get("online_key_dir")
    online_key_credential = metadata.get("online_key_credential_id")
    if not tokenizer_dir or (not online_key_dir and not online_key_credential):
        if tunnel is not None:
            tunnel.close()
        raise typer.BadParameter("deployment has no local tokenizer or online key path")
    bearer = None
    credential_id = metadata.get("bearer_credential_id")
    if credential_id:
        bearer = CredentialVault(product_paths().credentials).get(str(credential_id))["secret"]
    if online_key_credential:
        with tempfile.TemporaryDirectory(prefix="yinbian-cli-online-key-") as temporary:
            key_dir = materialize_online_key_directory(
                CredentialVault(product_paths().credentials),
                str(online_key_credential),
                Path(temporary) / "key",
            )
            client = PrivateInferenceClient.from_directories(
                base_url=str(endpoint),
                tokenizer_dir=Path(str(tokenizer_dir)),
                key_dir=key_dir,
                bearer_token=bearer,
            )
    else:
        client = PrivateInferenceClient.from_directories(
            base_url=str(endpoint),
            tokenizer_dir=Path(str(tokenizer_dir)),
            key_dir=Path(str(online_key_dir)),
            bearer_token=bearer,
        )
    return client, tunnel


@chat_app.command("deployments")
def chat_deployments() -> None:
    from aloepri.product.state import ProductStore

    state_path = os.environ.get("YINBIAN_STATE_DB") or os.environ.get("ALOEPRI_STATE_DB")
    store = ProductStore(None if state_path is None else Path(state_path))
    typer.echo(
        json.dumps(store.list_deployments(healthy_only=True), ensure_ascii=False, indent=2)
    )


def _chat_once(
    deployment_id: str,
    prompt: str,
    *,
    stream: bool,
    password: str | None,
    private_key_passphrase: str | None,
    history: list[dict[str, str]] | None = None,
    max_new_tokens: int = 128,
) -> str:
    client, tunnel = _deployment_client(
        deployment_id,
        password=password,
        private_key_passphrase=private_key_passphrase,
    )
    messages = [*(history or []), {"role": "user", "content": prompt}]
    try:
        if not stream:
            response = client.chat(messages, max_new_tokens=max_new_tokens)
            typer.echo(response.text)
            return response.text
        answer = ""
        rendered = ""
        for chunk in client.stream_chat(messages, max_new_tokens=max_new_tokens):
            answer = chunk.accumulated_text
            delta = answer[len(rendered) :] if answer.startswith(rendered) else answer
            typer.echo(delta, nl=False)
            rendered = answer
        typer.echo()
        return answer
    finally:
        client.close()
        if tunnel is not None:
            tunnel.close()  # type: ignore[attr-defined]


@chat_app.command("send")
def chat_send(
    deployment_id: Annotated[str, typer.Option("--deployment")],
    prompt: Annotated[str, typer.Option("--prompt")],
    password: Annotated[str | None, typer.Option("--password", hide_input=True)] = None,
    private_key_passphrase: Annotated[
        str | None, typer.Option("--private-key-passphrase", hide_input=True)
    ] = None,
    max_new_tokens: Annotated[int, typer.Option("--max-new-tokens")] = 128,
) -> None:
    _chat_once(
        deployment_id,
        prompt,
        stream=False,
        password=password,
        private_key_passphrase=private_key_passphrase,
        max_new_tokens=max_new_tokens,
    )


@chat_app.command("stream")
def chat_stream(
    deployment_id: Annotated[str, typer.Option("--deployment")],
    prompt: Annotated[str, typer.Option("--prompt")],
    password: Annotated[str | None, typer.Option("--password", hide_input=True)] = None,
    private_key_passphrase: Annotated[
        str | None, typer.Option("--private-key-passphrase", hide_input=True)
    ] = None,
    max_new_tokens: Annotated[int, typer.Option("--max-new-tokens")] = 128,
) -> None:
    _chat_once(
        deployment_id,
        prompt,
        stream=True,
        password=password,
        private_key_passphrase=private_key_passphrase,
        max_new_tokens=max_new_tokens,
    )


@chat_app.command("interactive")
def chat_interactive(
    deployment_id: Annotated[str, typer.Option("--deployment")],
    password: Annotated[str | None, typer.Option("--password", hide_input=True)] = None,
    private_key_passphrase: Annotated[
        str | None, typer.Option("--private-key-passphrase", hide_input=True)
    ] = None,
    max_new_tokens: Annotated[int, typer.Option("--max-new-tokens")] = 128,
) -> None:
    history: list[dict[str, str]] = []
    typer.echo("隐变智模多轮对话：/clear /quit")
    while True:
        prompt = typer.prompt("you")
        if prompt == "/quit":
            return
        if prompt == "/clear":
            history.clear()
            continue
        answer = _chat_once(
            deployment_id,
            prompt,
            stream=True,
            password=password,
            private_key_passphrase=private_key_passphrase,
            history=history,
            max_new_tokens=max_new_tokens,
        )
        history.extend(
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer},
            ]
        )


app.add_typer(chat_app, name="chat")


@app.command()
def demo(
    server: Annotated[str, typer.Option("--server")] = "http://127.0.0.1:8000",
    key_dir: Annotated[Path, typer.Option("--key-dir", exists=True, file_okay=False)] = Path(
        "data/keys/qwen05b-product-v31-blockperm8-online"
    ),
    tokenizer: Annotated[Path, typer.Option("--tokenizer", exists=True, file_okay=False)] = Path(
        "data/models/qwen2.5-0.5b"
    ),
    port: Annotated[int, typer.Option("--port", min=1024, max=65535)] = 7860,
    bearer_token_env: Annotated[str, typer.Option("--bearer-token-env")] = "ALOEPRI_BEARER_TOKEN",
    access_code_env: Annotated[str, typer.Option("--access-code-env")] = "ALOEPRI_DEMO_ACCESS_CODE",
    session_secret_env: Annotated[
        str, typer.Option("--session-secret-env")
    ] = "ALOEPRI_DEMO_SESSION_SECRET",
) -> None:
    """Start the trusted localhost visualization client."""

    from aloepri.demo.app import DemoGateway, create_demo_app

    gateway = DemoGateway(
        model_server=server,
        tokenizer_dir=tokenizer,
        key_dir=key_dir,
        bearer_token=os.environ.get(bearer_token_env),
    )
    uvicorn.run(
        create_demo_app(
            gateway,
            access_code=os.environ.get(access_code_env),
            session_secret=os.environ.get(session_secret_env),
        ),
        host="127.0.0.1",
        port=port,
        access_log=False,
    )


@app.command("inspect-package")
def inspect_package(
    server_package: Annotated[Path, typer.Option("--server-package", exists=True)],
) -> None:
    from aloepri.packaging import inspect_server_package

    result = inspect_server_package(server_package)
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["pass"]:
        raise typer.Exit(1)


@app.command("build-server-package")
def build_server(
    source_checkpoint: Annotated[
        Path, typer.Option("--source-checkpoint", exists=True, file_okay=False)
    ],
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    from aloepri.packaging import build_server_package

    build_server_package(source_checkpoint, output)
    typer.echo(f"sanitized server package: {output}")


@app.command("split-key")
def split_key(
    source_key_dir: Annotated[Path, typer.Option("--source-key-dir", exists=True, file_okay=False)],
    online_dir: Annotated[Path, typer.Option("--online-dir")],
    offline_dir: Annotated[Path, typer.Option("--offline-dir")],
) -> None:
    from aloepri.packaging import split_key_package

    split_key_package(source_key_dir, online_dir, offline_dir)
    typer.echo(f"online key: {online_dir}")
    typer.echo(f"offline master key: {offline_dir}")


@app.command("build-release")
def build_release(
    output: Annotated[Path, typer.Option("--output")],
    server_package: Annotated[Path, typer.Option("--server-package", exists=True)],
    config: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)],
    wheel: Annotated[Path | None, typer.Option("--wheel", exists=True, dir_okay=False)] = None,
    report: Annotated[list[Path] | None, typer.Option("--report", exists=True)] = None,
    evidence: Annotated[list[Path] | None, typer.Option("--evidence", exists=True)] = None,
) -> None:
    from aloepri.release import build_product_release

    result = build_product_release(
        output=output,
        server_package=server_package,
        config_path=config,
        wheel=wheel,
        reports=report,
        evidence=evidence,
    )
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))


@app.command("inspect-release")
def inspect_release(
    release: Annotated[Path, typer.Option("--release", exists=True, file_okay=False)],
) -> None:
    from aloepri.release import inspect_product_release

    result = inspect_product_release(release)
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["pass"]:
        raise typer.Exit(1)


@app.command("rmdp-budget")
def rmdp_budget(
    epsilon1: Annotated[float, typer.Option("--epsilon1")],
    vocab_size: Annotated[
        int, typer.Option("--vocab-size", "--sequence-length", min=2)
    ],
    embedding_sigma: Annotated[float, typer.Option("--embedding-sigma")],
    head_sigma: Annotated[float, typer.Option("--head-sigma")],
    embedding_top1: Annotated[float, typer.Option("--embedding-top1")],
    embedding_top2: Annotated[float, typer.Option("--embedding-top2")],
    head_top1: Annotated[float, typer.Option("--head-top1")],
    head_top2: Annotated[float, typer.Option("--head-top2")],
) -> None:
    from aloepri.privacy.rmdp import calculate_rmdp_budget

    result = calculate_rmdp_budget(
        epsilon1=epsilon1,
        vocab_size=vocab_size,
        embedding_sigma=embedding_sigma,
        head_sigma=head_sigma,
        embedding_singular_values=(embedding_top1, embedding_top2),
        head_singular_values=(head_top1, head_top2),
    )
    typer.echo(json.dumps(result.__dict__, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    app()
