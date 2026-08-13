from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import load_file, save_file
from transformers import DeepseekV3Config, DeepseekV3ForCausalLM
from transformers.models.deepseek_v2.configuration_deepseek_v2 import DeepseekV2Config
from transformers.models.deepseek_v2.modeling_deepseek_v2 import DeepseekV2ForCausalLM

from aloepri.conversion import deepseek_streaming
from aloepri.conversion.deepseek_streaming import (
    audit_deepseek_checkpoint,
    convert_deepseek_checkpoint,
)
from aloepri.evidence import key_directory_identity, sha256_file
from aloepri.models.modeling_aloepri_deepseek_v3 import register_aloepri_deepseek_v3
from aloepri.packaging import inspect_server_package


def tiny_config(q_lora_rank: int | None = None) -> DeepseekV2Config:
    return DeepseekV2Config(
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=16,
        qk_nope_head_dim=8,
        qk_rope_head_dim=8,
        v_head_dim=16,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        first_k_dense_replace=1,
        vocab_size=100,
        n_group=1,
        topk_group=1,
    )


def test_streaming_converter_preserves_tiny_v2_checkpoint(tmp_path) -> None:
    torch.manual_seed(31)
    source_model = DeepseekV2ForCausalLM(tiny_config()).eval()
    source = tmp_path / "source"
    output = tmp_path / "private"
    key_dir = tmp_path / "keys"
    source_model.save_pretrained(source, safe_serialization=True)

    audit = audit_deepseek_checkpoint(source)
    assert audit["pass"] is True
    assert audit["expert_layout"] == "individual"

    result = convert_deepseek_checkpoint(
        source_root=source,
        output_root=output,
        key_root=key_dir,
        seed=701,
        ffn_scale_min=0.5,
        ffn_scale_max=2.0,
    )
    assert result["pass"] is True
    assert (output / "model.safetensors.index.json").is_file()
    assert (output / "aloepri_manifest.json").is_file()
    assert (key_dir / "offline_master_key.safetensors").is_file()
    offline = load_file(key_dir / "offline_master_key.safetensors")
    assert "layers.0.kv_latent_order" in offline
    assert "layers.0.q_latent_order" not in offline
    assert "layers.0.nonrouted_ffn_order" in offline
    assert "layers.1.nonrouted_ffn_order" in offline

    private_model = DeepseekV2ForCausalLM.from_pretrained(
        output, local_files_only=True
    ).eval()
    input_ids = torch.randint(0, source_model.config.vocab_size, (2, 8))
    with torch.inference_mode():
        expected = source_model(input_ids).logits
        actual = private_model(input_ids).logits
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


def test_streaming_converter_resumes_only_verified_partial_files(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch.manual_seed(37)
    source = tmp_path / "source"
    output = tmp_path / "private"
    key_dir = tmp_path / "keys"
    DeepseekV2ForCausalLM(tiny_config()).save_pretrained(source, safe_serialization=True)

    original_save = deepseek_streaming._save_tensor
    calls = 0

    def interrupted_save(path, name, tensor):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        if calls == 6:
            raise RuntimeError("simulated interruption")
        return original_save(path, name, tensor)

    monkeypatch.setattr(deepseek_streaming, "_save_tensor", interrupted_save)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        convert_deepseek_checkpoint(
            source_root=source,
            output_root=output,
            key_root=key_dir,
            seed=709,
        )
    assert output.with_name(output.name + ".partial").is_dir()

    monkeypatch.setattr(deepseek_streaming, "_save_tensor", original_save)
    result = convert_deepseek_checkpoint(
        source_root=source,
        output_root=output,
        key_root=key_dir,
        seed=709,
        resume=True,
    )
    assert result["pass"] is True
    private_model = DeepseekV2ForCausalLM.from_pretrained(output, local_files_only=True).eval()
    reference = DeepseekV2ForCausalLM.from_pretrained(source, local_files_only=True).eval()
    input_ids = torch.randint(0, reference.config.vocab_size, (1, 5))
    with torch.inference_mode():
        torch.testing.assert_close(
            private_model(input_ids).logits,
            reference(input_ids).logits,
            atol=2e-5,
            rtol=2e-5,
        )


def test_streaming_converter_preserves_q_lora_latent_path(tmp_path) -> None:
    torch.manual_seed(35)
    source_model = DeepseekV2ForCausalLM(tiny_config(q_lora_rank=16)).eval()
    source = tmp_path / "source"
    output = tmp_path / "private"
    key_dir = tmp_path / "keys"
    source_model.save_pretrained(source, safe_serialization=True)
    convert_deepseek_checkpoint(
        source_root=source,
        output_root=output,
        key_root=key_dir,
        seed=707,
    )
    offline = load_file(key_dir / "offline_master_key.safetensors")
    assert "layers.0.q_latent_order" in offline
    private_model = DeepseekV2ForCausalLM.from_pretrained(
        output, local_files_only=True
    ).eval()
    input_ids = torch.randint(0, source_model.config.vocab_size, (1, 6))
    with torch.inference_mode():
        torch.testing.assert_close(
            private_model(input_ids).logits,
            source_model(input_ids).logits,
            atol=2e-5,
            rtol=2e-5,
        )


def test_streaming_converter_rejects_unpinned_source_revision(tmp_path) -> None:
    source = tmp_path / "source"
    DeepseekV2ForCausalLM(tiny_config()).save_pretrained(source, safe_serialization=True)
    (source / "download_receipt.json").write_text(
        json.dumps(
            {
                "repo": "example/model",
                "revision": "revision-a",
                "complete": True,
                "files": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="source revision mismatch"):
        convert_deepseek_checkpoint(
            source_root=source,
            output_root=tmp_path / "private",
            key_root=tmp_path / "keys",
            seed=711,
            expected_source_revision="revision-b",
        )


def test_streaming_converter_accepts_complete_verified_source_receipt(tmp_path) -> None:
    source = tmp_path / "source"
    DeepseekV2ForCausalLM(tiny_config()).save_pretrained(source, safe_serialization=True)
    records = [
        {
            "path": path.relative_to(source).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(source.rglob("*"))
        if path.is_file()
    ]
    (source / "download_receipt.json").write_text(
        json.dumps(
            {
                "repo": "example/model",
                "revision": "locked-revision",
                "complete": True,
                "files": records,
            }
        ),
        encoding="utf-8",
    )
    result = convert_deepseek_checkpoint(
        source_root=source,
        output_root=tmp_path / "private",
        key_root=tmp_path / "keys",
        seed=713,
        expected_source_revision="locked-revision",
    )
    assert result["pass"] is True


def test_streaming_converter_vocab_round_trip(tmp_path) -> None:
    torch.manual_seed(41)
    source_model = DeepseekV2ForCausalLM(tiny_config()).eval()
    source = tmp_path / "source"
    output = tmp_path / "private"
    key_dir = tmp_path / "offline"
    online_dir = tmp_path / "online"
    source_model.save_pretrained(source, safe_serialization=True)
    convert_deepseek_checkpoint(
        source_root=source,
        output_root=output,
        key_root=key_dir,
        online_key_root=online_dir,
        seed=719,
        vocab_permutation=True,
    )
    key = load_file(online_dir / "online_key.safetensors")
    assert key_directory_identity(online_dir)["files"]
    assert key_directory_identity(key_dir)["files"]
    tau = key["tau"]
    private_model = DeepseekV2ForCausalLM.from_pretrained(output, local_files_only=True).eval()
    assert private_model.config.aloepri["server_token_space"] == "private"
    private_manifest = json.loads(
        (output / "aloepri_manifest.json").read_text(encoding="utf-8")
    )
    online_metadata = json.loads((online_dir / "key.json").read_text(encoding="utf-8"))
    assert private_manifest["metadata"]["key_id"] == online_metadata["key_id"]
    assert private_model.config.aloepri["key_id"] == online_metadata["key_id"]
    assert inspect_server_package(output)["pass"] is True
    plain_ids = torch.randint(0, source_model.config.vocab_size, (2, 6))
    with torch.inference_mode():
        expected = source_model(plain_ids).logits
        private_logits = private_model(tau[plain_ids]).logits
    restored_logits = private_logits.index_select(-1, tau)
    torch.testing.assert_close(restored_logits, expected, atol=2e-5, rtol=2e-5)


def test_paper_complete_v3_conversion_expands_every_residual_boundary(tmp_path) -> None:
    torch.manual_seed(43)
    config = DeepseekV3Config(
        hidden_size=32,
        intermediate_size=48,
        moe_intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        q_lora_rank=None,
        kv_lora_rank=8,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=4,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        first_k_dense_replace=1,
        vocab_size=64,
        n_group=1,
        topk_group=1,
        rope_interleave=True,
    )
    source_model = DeepseekV3ForCausalLM(config).eval()
    source = tmp_path / "source-v3"
    output = tmp_path / "private-v3"
    offline = tmp_path / "offline-v3"
    online = tmp_path / "online-v3"
    source.mkdir()
    config.save_pretrained(source)
    save_file(
        {
            name: tensor.detach().contiguous()
            for name, tensor in source_model.state_dict().items()
        },
        source / "model.safetensors",
    )
    result = convert_deepseek_checkpoint(
        source_root=source,
        output_root=output,
        key_root=offline,
        online_key_root=online,
        seed=727,
        ffn_scale_min=0.5,
        ffn_scale_max=2.0,
        vocab_permutation=True,
        paper_complete=True,
        expansion_h=4,
        coefficient_lambda=0.3,
        alpha_e=0.0,
        alpha_h=0.0,
        block_beta=2,
        router_normalize=False,
    )
    assert result["pass"] is True
    private_config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    assert private_config["model_type"] == "aloepri_deepseek_v3"
    assert private_config["plain_hidden_size"] == 32
    assert private_config["hidden_size"] == 40
    assert private_config["tie_word_embeddings"] is False
    register_aloepri_deepseek_v3()
    private_model = DeepseekV3ForCausalLM  # keep native class import exercised
    del private_model
    from transformers import AutoModelForCausalLM

    loaded = AutoModelForCausalLM.from_pretrained(output, local_files_only=True).eval()
    assert tuple(loaded.model.embed_tokens.weight.shape) == (64, 40)
    assert tuple(loaded.model.layers[0].self_attn.q_proj.weight.shape) == (16, 40)
    assert tuple(loaded.model.layers[1].mlp.gate.weight.shape) == (4, 40)
    assert tuple(loaded.model.layers[1].mlp.experts.gate_up_proj.shape) == (
        4,
        32,
        40,
    )
    assert tuple(loaded.model.layers[1].mlp.experts.down_proj.shape) == (
        4,
        40,
        16,
    )
    key = load_file(offline / "offline_master_key.safetensors")
    assert tuple(key["p.residual"].shape) == (32, 40)
    assert tuple(key["q.attention_q"].shape) == (40, 32)
    assert inspect_server_package(output)["pass"] is True
    online_key = load_file(online / "online_key.safetensors")
    tau = online_key["tau"]
    plain_ids = torch.randint(0, config.vocab_size, (2, 5))
    with torch.inference_mode():
        expected = source_model(plain_ids).logits
        private_logits = loaded(tau[plain_ids]).logits
    restored = private_logits.index_select(-1, tau)
    assert torch.equal(restored.argmax(dim=-1), expected.argmax(dim=-1))
