from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Annotated, Any

import typer
import uvicorn
import yaml

from aloepri.client.sdk import PrivateInferenceClient
from aloepri.packaging import build_server_package, inspect_server_package, split_key_package
from aloepri.privacy.rmdp import calculate_rmdp_budget, expected_m1_change_rate
from aloepri.serving.app import create_app
from aloepri.serving.hf_runtime import PrivateHFRuntime

app = typer.Typer(no_args_is_help=True, add_completion=False)


def load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise typer.BadParameter("configuration root must be a mapping")
    return payload


def run_checked(arguments: list[str]) -> None:
    typer.echo(f"$ {' '.join(arguments)}")
    subprocess.run(arguments, check=True)


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
    runtime_pass = all(runtime_checks.values())
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
        "runtime_pass": runtime_pass,
        "functional_pass": functional_pass,
        "accuracy_gate": "external_task_evaluation_required" if noise_active else "not_applicable",
    }


@app.command()
def convert(
    config: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)],
) -> None:
    """Convert Qwen2.5-0.5B and split online/offline keys."""

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

    cfg = load_config(config)
    server = cfg.get("server", {})
    host = str(server.get("host", "127.0.0.1"))
    bearer_env = str(server.get("bearer_token_env", "ALOEPRI_BEARER_TOKEN"))
    bearer_token = os.environ.get(bearer_env)
    if host not in {"127.0.0.1", "localhost", "::1"} and not bearer_token:
        raise typer.BadParameter(f"remote binding requires bearer token in {bearer_env}")
    runtime = PrivateHFRuntime(
        Path(cfg["output_model"]),
        device=str(server.get("device", "auto")),
        dtype=str(server.get("dtype", cfg["conversion"]["dtype"])),
        max_input_tokens=int(server.get("max_input_tokens", 2048)),
        max_output_tokens=int(server.get("max_output_tokens", 512)),
    )
    api = create_app(
        runtime,
        bearer_token=bearer_token,
        max_request_bytes=int(server.get("max_request_bytes", 1_000_000)),
    )
    uvicorn.run(api, host=host, port=int(server.get("port", 8000)), access_log=False)


@app.command()
def chat(
    server: Annotated[str, typer.Option("--server")],
    key_dir: Annotated[Path, typer.Option("--key-dir", exists=True, file_okay=False)],
    tokenizer: Annotated[Path, typer.Option("--tokenizer", exists=True, file_okay=False)] = Path(
        "data/models/qwen2.5-0.5b"
    ),
    privacy_mode: Annotated[str, typer.Option("--privacy-mode")] = "permutation",
    epsilon1: Annotated[float | None, typer.Option("--epsilon1")] = None,
    bearer_token_env: Annotated[str, typer.Option("--bearer-token-env")] = "ALOEPRI_BEARER_TOKEN",
    max_new_tokens: Annotated[int, typer.Option("--max-new-tokens", min=1, max=2048)] = 128,
) -> None:
    """Run an interactive client; plaintext and conversation history remain local."""

    if privacy_mode == "rmdp" and epsilon1 is None:
        raise typer.BadParameter("--epsilon1 is required for rmdp mode")
    client = PrivateInferenceClient.from_directories(
        base_url=server,
        tokenizer_dir=tokenizer,
        key_dir=key_dir,
        bearer_token=os.environ.get(bearer_token_env),
    )
    history: list[dict[str, str]] = []
    last_stats: dict[str, float | int] | None = None
    typer.echo("AloePri chat: /clear /stats /privacy /quit")
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
            for chunk in chunks:
                typer.echo(chunk.text, nl=False)
                answer = chunk.accumulated_text
                elapsed.append(chunk.elapsed_ms)
            typer.echo()
            history.append({"role": "assistant", "content": answer})
            last_stats = {
                "output_tokens": len(elapsed),
                "ttft_ms": elapsed[0] if elapsed else 0.0,
                "tpot_ms": sum(elapsed[1:]) / max(1, len(elapsed) - 1),
            }
    finally:
        client.close()


@app.command("inspect-package")
def inspect_package(
    server_package: Annotated[Path, typer.Option("--server-package", exists=True)],
) -> None:
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
    build_server_package(source_checkpoint, output)
    typer.echo(f"sanitized server package: {output}")


@app.command("split-key")
def split_key(
    source_key_dir: Annotated[Path, typer.Option("--source-key-dir", exists=True, file_okay=False)],
    online_dir: Annotated[Path, typer.Option("--online-dir")],
    offline_dir: Annotated[Path, typer.Option("--offline-dir")],
) -> None:
    split_key_package(source_key_dir, online_dir, offline_dir)
    typer.echo(f"online key: {online_dir}")
    typer.echo(f"offline master key: {offline_dir}")


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
