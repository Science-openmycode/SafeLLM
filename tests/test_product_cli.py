from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from typer.testing import CliRunner

from aloepri.cli import app
from aloepri.conversion import executor
from aloepri.jobs.store import JobState, JobStore
from aloepri.planning import ConversionPlan


def _tiny_qwen(path: Path) -> None:
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen2",
                "num_hidden_layers": 0,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "torch_dtype": "float32",
            }
        ),
        encoding="utf-8",
    )
    save_file(
        {
            "model.embed_tokens.weight": torch.zeros(4, 2),
            "model.norm.weight": torch.ones(2),
            "lm_head.weight": torch.zeros(4, 2),
        },
        path / "model.safetensors",
    )


def _write_server_manifest(path: Path) -> None:
    files = []
    for item in sorted(path.iterdir()):
        if item.is_file() and item.name != "aloepri_manifest.json":
            files.append(
                {
                    "path": item.name,
                    "bytes": item.stat().st_size,
                    "sha256": hashlib.sha256(item.read_bytes()).hexdigest(),
                }
            )
    (path / "aloepri_manifest.json").write_text(
        json.dumps({"metadata": {"schema_version": 1}, "files": files}),
        encoding="utf-8",
    )


def test_product_cli_plan_job_upload_and_mock_deploy(tmp_path: Path) -> None:
    runner = CliRunner()
    model = tmp_path / "model"
    _tiny_qwen(model)
    plan_path = tmp_path / "plan.yaml"
    environment = {"ALOEPRI_STATE_DB": str(tmp_path / "state.db")}
    result = runner.invoke(app, ["models", "inspect", "--model", str(model)], env=environment)
    assert result.exit_code == 0, result.output
    assert '"adapter_id": "qwen2"' in result.output
    result = runner.invoke(
        app,
        [
            "plan",
            "--model",
            str(model),
            "--output",
            str(plan_path),
            "--destination",
            str(tmp_path / "private"),
        ],
        env=environment,
    )
    assert result.exit_code == 0, result.output
    plan = ConversionPlan.load(plan_path)
    result = runner.invoke(
        app, ["convert", "--plan", str(plan_path), "--schedule-only"], env=environment
    )
    assert result.exit_code == 0, result.output
    assert '"state": "PREFLIGHT"' in result.output
    rejected = runner.invoke(
        app, ["upload", plan.job_id, "--cloud-profile", "mock"], env=environment
    )
    assert rejected.exit_code != 0
    assert "cannot upload" in rejected.output
    private = tmp_path / "private"
    private.mkdir()
    package = private / "model.safetensors"
    save_file({"tau": torch.arange(4)}, package)
    _write_server_manifest(private)
    store = JobStore(tmp_path / "state.db")
    store.transition(plan.job_id, JobState.CONVERTING)
    store.transition(plan.job_id, JobState.UPLOADING)
    secret_rejected = runner.invoke(
        app, ["upload", plan.job_id, "--cloud-profile", "mock"], env=environment
    )
    assert secret_rejected.exit_code != 0
    assert "secret scan" in secret_rejected.output
    package.unlink()
    save_file({"private.weight": torch.zeros(1)}, package)
    _write_server_manifest(private)
    result = runner.invoke(
        app, ["upload", plan.job_id, "--cloud-profile", "mock"], env=environment
    )
    assert result.exit_code == 0, result.output
    assert '"environment": "mock-cloud"' in result.output
    assert (tmp_path / "mock-cloud" / "objects" / "mock" / "jobs").is_dir()
    result = runner.invoke(
        app, ["deploy", plan.job_id, "--cloud-profile", "mock"], env=environment
    )
    assert result.exit_code == 0, result.output
    assert '"real_cloud_validated": false' in result.output


def test_convert_requires_exactly_one_input() -> None:
    result = CliRunner().invoke(app, ["convert"])
    assert result.exit_code != 0
    assert "one of --plan or --config is required" in result.output


def test_yinbian_product_command_surface(tmp_path: Path) -> None:
    environment = {"YINBIAN_STATE_DB": str(tmp_path / "state.db")}
    runner = CliRunner()
    for arguments in (
        ["keys", "list"],
        ["servers", "list"],
        ["deploy", "list"],
        ["chat", "deployments"],
    ):
        result = runner.invoke(app, arguments, env=environment)
        assert result.exit_code == 0, f"{arguments}: {result.output}"
    result = runner.invoke(app, ["tunnel", "status", "missing"], env=environment)
    assert result.exit_code == 0, result.output
    assert '"connected": false' in result.output


def test_servers_add_accepts_pasted_ssh_command(tmp_path: Path) -> None:
    environment = {"YINBIAN_STATE_DB": str(tmp_path / "state.db")}
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "servers",
            "add",
            "--name",
            "Rental GPU",
            "--ssh-command",
            "ssh -p 51838 root@gpu.example",
            "--auth-type",
            "password",
        ],
        env=environment,
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["host"] == "gpu.example"
    assert payload["port"] == 51838
    assert payload["username"] == "root"
    assert "password" not in payload


def test_completed_local_conversion_waits_for_explicit_upload(
    tmp_path: Path, monkeypatch: object
) -> None:
    plan = ConversionPlan(
        schema_version=1,
        job_id="job-execute",
        source={"type": "local", "path": str(tmp_path / "source")},
        adapter="qwen2",
        fingerprint={"weight_format": "float32"},
        output={"type": "local", "uri": str(tmp_path / "private")},
    )

    def fake_run(current: ConversionPlan, output: Path, source: Path) -> dict[str, str]:
        assert current is plan
        assert source == tmp_path / "source"
        return {"output": str(output)}

    monkeypatch.setattr(executor, "_run_qwen", fake_run)  # type: ignore[attr-defined]
    store = JobStore(tmp_path / "state.db")
    executor.execute_conversion_plan(plan, store)
    assert store.get(plan.job_id)["state"] == "UPLOADING"


def test_generic_converter_dispatches_deepseek_to_deepseek_path(
    tmp_path: Path, monkeypatch: object
) -> None:
    plan = ConversionPlan(
        schema_version=1,
        job_id="job-deepseek-dispatch",
        source={"type": "local", "path": str(tmp_path / "source")},
        adapter="deepseek_v2",
        fingerprint={"weight_format": "bfloat16"},
        output={"type": "local", "uri": str(tmp_path / "private")},
    )
    called: dict[str, object] = {}

    def fake_deepseek(
        current: ConversionPlan,
        output: Path,
        store: JobStore | None,
        source: Path,
        *,
        offline_key_password: str | None = None,
        progress_callback: object = None,
    ) -> dict[str, str]:
        called.update(
            plan=current,
            output=output,
            store=store,
            source=source,
            password=offline_key_password,
            progress_callback=progress_callback,
        )
        return {"adapter": current.adapter}

    monkeypatch.setattr(executor, "_run_deepseek", fake_deepseek)  # type: ignore[attr-defined]
    result = executor.convert_model_checkpoint(
        plan,
        tmp_path / "private",
        tmp_path / "source",
        offline_key_password="correct horse battery staple",
    )
    assert result == {"adapter": "deepseek_v2"}
    assert called["source"] == tmp_path / "source"
