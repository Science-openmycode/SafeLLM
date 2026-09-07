from __future__ import annotations

import hashlib
import json

import torch
from safetensors.torch import save_file
from transformers import Qwen2Config, Qwen2ForCausalLM

from aloepri.conversion.paper_qwen2 import convert_qwen2_modules
from aloepri.models.configuration_aloepri_qwen2 import AloePriQwen2Config
from aloepri.models.modeling_aloepri_qwen2 import AloePriQwen2ForCausalLM
from aloepri.serving.protocol import GenerateRequest
from aloepri.serving.tee_runtime import TeeSplitHFRuntime
from aloepri.tee.attestation import sm3
from aloepri.transforms.paper_key_matrix import PaperKeyPair


def _build_tiny_split(tmp_path) -> tuple[Qwen2ForCausalLM, TeeSplitHFRuntime]:
    plain_dim = 16
    expansion_h = 4
    private_dim = plain_dim + 2 * expansion_h
    config = Qwen2Config(
        vocab_size=64,
        hidden_size=plain_dim,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        rms_norm_eps=0.0,
        bos_token_id=1,
        eos_token_id=63,
        pad_token_id=0,
    )
    torch.manual_seed(19)
    source = Qwen2ForCausalLM(config).eval()
    private_config = AloePriQwen2Config.from_qwen2_config(
        config,
        expansion_h=expansion_h,
    )
    target = AloePriQwen2ForCausalLM(private_config).eval()
    p = torch.zeros((plain_dim, private_dim), dtype=torch.float64)
    p[:, :plain_dim] = torch.eye(plain_dim, dtype=torch.float64)
    q = p.mT.contiguous()
    key = PaperKeyPair(
        p=p,
        q=q,
        b=torch.eye(plain_dim, dtype=torch.float64),
        condition_b=1.0,
        pq_relative_error=0.0,
        spectral_norm_p=1.0,
        spectral_norm_q=1.0,
    )
    identity = torch.arange(config.vocab_size)
    convert_qwen2_modules(
        source,
        target,
        key_pair=key,
        tau=identity,
        alpha_e=0.0,
        alpha_h=0.0,
        embedding_noise_seed=1,
        head_noise_seed=2,
    )
    target.config.aloepri = {
        "model_id": "tiny-tee",
        "key_id": "tiny-key",
        "security_mode": "tee_gm",
        "boundary_mode": "tee_split",
        "tee_backend": "software_sim",
        "hardware_attested": False,
    }
    target.generation_config.eos_token_id = 63
    target.generation_config.bos_token_id = 1

    boundary = tmp_path / "boundary"
    body = tmp_path / "body"
    boundary.mkdir()
    embedding = source.get_input_embeddings().weight.detach().float() @ p.float()
    exact_head = (
        source.get_output_embeddings().weight.detach().float()
        * source.model.norm.weight.detach().float().unsqueeze(0)
    )
    save_file({"embedding_private": embedding}, boundary / "embedding-private.safetensors")
    save_file({"exact_head": exact_head}, boundary / "exact-head-archive.safetensors")
    save_file({"q_final": q.float()}, boundary / "final-coordinate-key.safetensors")
    save_file(
        {"head_basis": torch.eye(plain_dim)},
        boundary / "head-coordinate-key.safetensors",
    )
    files = []
    for path in sorted(boundary.iterdir()):
        content = path.read_bytes()
        files.append(
            {
                "path": path.name,
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
                "sm3": sm3(content).hex(),
            }
        )
    (boundary / "tee-manifest.json").write_text(
        json.dumps(
            {
                "security_mode": "tee_gm",
                "boundary_mode": "tee_split",
                "model_id": "tiny-tee",
                "key_id": "tiny-key",
                "files": files,
            }
        ),
        encoding="utf-8",
    )
    with torch.no_grad():
        target.get_input_embeddings().weight.zero_()
        target.get_output_embeddings().weight.zero_()
    target.save_pretrained(body, safe_serialization=True)
    save_file({"outsourced_head": exact_head}, body / "masked-head-worker.safetensors")
    runtime = TeeSplitHFRuntime(body, boundary, device="cpu", dtype="float32")
    return source, runtime


def test_split_runtime_matches_plain_greedy_prefill_and_decode(tmp_path) -> None:
    source, runtime = _build_tiny_split(tmp_path)
    input_ids = torch.tensor([[1, 4, 6]])
    expected: list[int] = []
    current = input_ids
    attention_mask = torch.ones_like(input_ids)
    cache = None
    with torch.inference_mode():
        for _ in range(4):
            result = source(
                current,
                attention_mask=attention_mask,
                past_key_values=cache,
                use_cache=True,
            )
            token = int(result.logits[:, -1].argmax())
            expected.append(token)
            cache = result.past_key_values
            current = torch.tensor([[token]])
            attention_mask = torch.cat((attention_mask, torch.ones((1, 1), dtype=torch.long)), -1)

    response = runtime.generate(
        GenerateRequest(
            model_id="tiny-tee",
            key_id="tiny-key",
            input_ids=[1, 4, 6],
            max_new_tokens=4,
            temperature=0.0,
        )
    )
    assert response.output_ids == expected
    assert runtime.readiness()["hardware_attested"] is False


def test_masked_head_runtime_matches_local_and_safely_falls_back(tmp_path) -> None:
    _, local_runtime = _build_tiny_split(tmp_path)
    masked_runtime = TeeSplitHFRuntime(
        tmp_path / "body",
        tmp_path / "boundary",
        device="cpu",
        dtype="float32",
        head_mode="masked_outsource",
        initial_mask_count=1,
    )
    request = GenerateRequest(
        model_id="tiny-tee",
        key_id="tiny-key",
        input_ids=[1, 4, 6],
        max_new_tokens=3,
        temperature=0.0,
    )
    # First token may use the one-time outsourced mask; subsequent tokens must
    # fall back to the exact local TEE Head without changing the result.
    assert masked_runtime.generate(request).output_ids == local_runtime.generate(request).output_ids
    assert masked_runtime.mask_pool is not None
    assert masked_runtime.mask_pool.counts()["CONSUMED"] == 1


def test_traced_request_reports_actual_embedding_body_and_head_boundaries(tmp_path) -> None:
    _build_tiny_split(tmp_path)
    runtime = TeeSplitHFRuntime(
        tmp_path / "body",
        tmp_path / "boundary",
        device="cpu",
        dtype="float32",
        head_mode="masked_outsource",
        initial_mask_count=0,
    )
    events = list(
        runtime.iter_token_ids(
            GenerateRequest(
                model_id="tiny-tee",
                key_id="tiny-key",
                input_ids=[1, 4, 6],
                max_new_tokens=1,
                temperature=0.0,
                include_execution_trace=True,
            )
        )
    )
    assert len(events) == 1
    by_stage = {event["stage"]: event for event in runtime.last_execution_trace}
    assert "tee_embedding_lookup" in by_stage
    assert "tee_to_gpu_transfer" in by_stage
    assert "gpu_transformer_body" in by_stage
    assert "gpu_to_tee_transfer" in by_stage
    assert "tee_recover_hidden" in by_stage
    assert "masked_head_outbound" in by_stage
    assert "masked_head_worker" in by_stage
    assert "masked_head_inbound" in by_stage
    assert "masked_head_freivalds" in by_stage
    assert "token_step_complete" in by_stage
    embedding_evidence = by_stage["tee_embedding_lookup"]["evidence"]
    assert embedding_evidence["decrypted_token_ids_head"] == [1, 4, 6]
    assert embedding_evidence["decrypted_token_ids_tail"] == []
    recover_evidence = by_stage["tee_recover_hidden"]["evidence"]
    assert recover_evidence["q_final_shape"] == [24, 16]
    coordinate_evidence = by_stage["masked_head_coordinate"]["evidence"]
    assert coordinate_evidence["basis_shape"] == [16, 16]
    body_evidence = by_stage["gpu_transformer_body"]["evidence"]
    assert body_evidence["mode"] == "prefill"
    assert body_evidence["last_hidden_state"]["sha256_16"]
    assert body_evidence["input"]["sample_head"]
    assert body_evidence["input"]["sample_tail"]
    assert len(body_evidence["input"]["sample_head"]) <= 4
    assert len(body_evidence["input"]["sample_tail"]) <= 4
    assert runtime.mask_pool is not None
    counts = runtime.mask_pool.counts()
    assert counts["CONSUMED"] + counts["BURNED"] == 1
