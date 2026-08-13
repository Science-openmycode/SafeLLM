from __future__ import annotations

import torch

from aloepri.adapters.deepseek_plan import build_deepseek_v3_static_plan
from aloepri.transforms.mtp import TinyMTPRuntime, transform_mtp_eh_projection


def test_mtp_eh_projection_preserves_private_coordinates() -> None:
    generator = torch.Generator().manual_seed(47)
    hidden = 4
    p = torch.randn(hidden, hidden, generator=generator)
    while torch.linalg.det(p).abs() < 0.1:
        p = torch.randn(hidden, hidden, generator=generator)
    q = torch.linalg.inv(p)
    weight = torch.randn(hidden, 2 * hidden, generator=generator)
    enorm = torch.rand(hidden, generator=generator) + 0.5
    hnorm = torch.rand(hidden, generator=generator) + 0.5
    embedding = torch.randn(3, hidden, generator=generator)
    previous_hidden = torch.randn(3, hidden, generator=generator)
    private_weight = transform_mtp_eh_projection(
        weight,
        embedding_norm_weight=enorm,
        hidden_norm_weight=hnorm,
        p=p,
        q=q,
    )
    plain = torch.nn.functional.linear(
        torch.cat((embedding * enorm, previous_hidden * hnorm), dim=-1), weight
    )
    private = torch.nn.functional.linear(
        torch.cat((embedding @ p, previous_hidden @ p), dim=-1), private_weight
    )
    assert torch.allclose(private, plain @ p, atol=1e-5, rtol=1e-5)


def test_official_v3_static_plan_covers_61_layers_256_experts_and_mtp() -> None:
    config = {
        "model_type": "deepseek_v3",
        "num_hidden_layers": 61,
        "num_nextn_predict_layers": 1,
        "n_routed_experts": 256,
        "first_k_dense_replace": 3,
        "moe_layer_freq": 1,
        "quantization_config": {
            "quant_method": "fp8",
            "weight_block_size": [128, 128],
        },
    }
    plan = build_deepseek_v3_static_plan(config)
    assert plan.main_layers == 61
    assert plan.routed_experts == 256
    assert plan.mtp_layers == 1
    assert plan.fp8_block_size == (128, 128)
    assert any(module.layer == 61 and module.component == "eh_proj" for module in plan.modules)
    mtp_experts = {
        module.expert for module in plan.modules if module.layer == 61 and module.expert is not None
    }
    assert mtp_experts == set(range(256))
    assert plan.to_dict()["tensor_coverage"] == 1.0
    assert plan.to_dict()["copied_unknown"] == []


def test_tiny_mtp_runtime_generates_and_validates_candidate() -> None:
    runtime = TinyMTPRuntime(torch.eye(3, 6), torch.arange(15.0).reshape(5, 3))
    embedding = torch.tensor([[1.0, 0.0, 0.0]])
    hidden = torch.tensor([[0.0, 1.0, 0.0]])
    candidate = runtime.candidate(embedding, hidden)
    logits = runtime(embedding, hidden)
    returned, accepted = runtime.validate_candidate(candidate, logits)
    assert torch.equal(returned, candidate)
    assert accepted.item() is True
