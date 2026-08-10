from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file
from torch import Tensor, nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv

from aloepri.models.modeling_aloepri_qwen2 import register_aloepri_qwen2


def first_tensor(value: object) -> Tensor:
    if isinstance(value, Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            if isinstance(item, Tensor):
                return item
    raise TypeError(f"hook value does not contain a tensor: {type(value)!r}")


def capture_model(
    model: nn.Module, input_ids: Tensor, *, preserve_dtype: bool = False
) -> dict[str, Any]:
    captures: dict[str, Tensor] = {}
    handles: list[Any] = []

    def captured_tensor(value: object) -> Tensor:
        tensor = first_tensor(value).detach()
        return tensor.cpu() if preserve_dtype else tensor.float().cpu()

    def add(name: str, module: nn.Module, *, capture_input: bool = False) -> None:
        if capture_input:
            handles.append(
                module.register_forward_pre_hook(
                    lambda _module, args, key=f"{name}.input": captures.__setitem__(
                        key, captured_tensor(args)
                    )
                )
            )
        handles.append(
            module.register_forward_hook(
                lambda _module, _args, output, key=f"{name}.output": captures.__setitem__(
                    key, captured_tensor(output)
                )
            )
        )

    add("embedding", model.model.embed_tokens)
    for index, layer in enumerate(model.model.layers):
        prefix = f"layer.{index}"
        add(f"{prefix}.input_norm", layer.input_layernorm, capture_input=True)
        for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
            add(f"{prefix}.{projection}", getattr(layer.self_attn, projection))
        add(f"{prefix}.attention", layer.self_attn)
        add(f"{prefix}.post_norm", layer.post_attention_layernorm, capture_input=True)
        for projection in ("gate_proj", "up_proj", "down_proj"):
            add(f"{prefix}.{projection}", getattr(layer.mlp, projection))
        add(f"{prefix}.mlp", layer.mlp)
        add(prefix, layer)
    add("final_norm", model.model.norm, capture_input=True)
    add("lm_head", model.lm_head)

    with torch.inference_mode():
        output = model(input_ids=input_ids, use_cache=True)
    for handle in handles:
        handle.remove()
    cache = [
        {
            "key": (
                layer.keys.detach().cpu()
                if preserve_dtype
                else layer.keys.detach().float().cpu()
            ),
            "value": (
                layer.values.detach().cpu()
                if preserve_dtype
                else layer.values.detach().float().cpu()
            ),
        }
        for layer in output.past_key_values.layers
    ]
    captures["logits"] = (
        output.logits.detach().cpu()
        if preserve_dtype
        else output.logits.detach().float().cpu()
    )
    return {"captures": captures, "cache": cache}


def metrics(reference: Tensor, candidate: Tensor) -> dict[str, float]:
    reference = reference.double()
    candidate = candidate.double()
    delta = candidate - reference
    rmse = torch.sqrt(torch.mean(delta.square()))
    reference_rms = torch.sqrt(torch.mean(reference.square()))
    reference_l2 = torch.linalg.vector_norm(reference)
    return {
        "max_abs": float(delta.abs().max()),
        "mean_abs": float(delta.abs().mean()),
        "relative_l2": float(torch.linalg.vector_norm(delta) / reference_l2.clamp_min(1e-30)),
        "nrmse": float(rmse / reference_rms.clamp_min(1e-30)),
    }


def restore_heads(value: Tensor, order: Tensor, maps: Tensor, group_size: int) -> Tensor:
    input_dtype = value.dtype
    shaped = value.view(*value.shape[:-1], order.numel(), maps.shape[-1]).double()
    restored = torch.empty_like(shaped)
    for new_head, old_head in enumerate(order.tolist()):
        group = old_head // group_size
        restored[..., old_head, :] = shaped[..., new_head, :] @ torch.linalg.inv(
            maps[group].double()
        )
    return restored.to(input_dtype).reshape_as(value)


def restore_permuted_features(value: Tensor, order: Tensor, scales: Tensor | None = None) -> Tensor:
    restored = torch.empty_like(value)
    working = value if scales is None else value * scales.to(value.dtype)
    restored[..., order] = working
    return restored


def attention_internals(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    causal: bool,
    float64_scores: bool = False,
) -> dict[str, Tensor]:
    batch, length, _ = q.shape
    query = q.view(batch, length, num_heads, head_dim).transpose(1, 2)
    key = k.view(batch, length, num_kv_heads, head_dim).transpose(1, 2)
    value = v.view(batch, length, num_kv_heads, head_dim).transpose(1, 2)
    positions = torch.arange(length, dtype=torch.float32)
    inv_freq = 1_000_000.0 ** (-torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    angles = torch.outer(positions, inv_freq)
    rotary = torch.cat((angles, angles), dim=-1).unsqueeze(0)
    cos, sin = rotary.cos(), rotary.sin()
    query_rope, key_rope = apply_rotary_pos_emb(query, key, cos, sin)
    repeated_key = repeat_kv(key_rope, num_heads // num_kv_heads)
    repeated_value = repeat_kv(value, num_heads // num_kv_heads)
    score_query = query_rope.double() if float64_scores else query_rope
    score_key = repeated_key.double() if float64_scores else repeated_key
    scores = torch.matmul(score_query, score_key.transpose(2, 3)) * (head_dim**-0.5)
    if causal:
        causal_mask = torch.full((length, length), float("-inf"), dtype=scores.dtype)
        causal_mask = torch.triu(causal_mask, diagonal=1)
        probabilities = torch.softmax(scores + causal_mask, dim=-1, dtype=torch.float32).to(
            query.dtype
        )
    else:
        probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    aggregate = torch.matmul(probabilities, repeated_value).transpose(1, 2).contiguous()
    return {
        "q_rope": query_rope,
        "k_rope": key_rope,
        "scores": scores,
        "softmax": probabilities,
        "aggregate": aggregate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Layer-wise AloePri coordinate equivalence audit")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--private", type=Path, required=True)
    parser.add_argument("--key-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prompt", default="用一句话解释矩阵乘法。")
    parser.add_argument("--nrmse-limit", type=float, default=1e-5)
    parser.add_argument(
        "--private-attention-compute-dtype",
        choices=("float32", "float64", "float64_scores"),
        default="float32",
    )
    args = parser.parse_args()

    register_aloepri_qwen2()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.source, local_files_only=True)
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    plain_ids = encoded["input_ids"].to(device)
    key = load_file(args.key_dir / "paper_key.safetensors", device="cpu")
    tau = key["tau"].to(device)
    inverse = key["q"].float()

    common = {"local_files_only": True, "dtype": torch.float32, "attn_implementation": "eager"}
    original = AutoModelForCausalLM.from_pretrained(args.source, **common).to(device).eval()
    plain = capture_model(original, plain_ids)
    source_config = original.config
    norm_weights = {
        **{
            f"layer.{i}.input_norm": layer.input_layernorm.weight.detach().float().cpu()
            for i, layer in enumerate(original.model.layers)
        },
        **{
            f"layer.{i}.post_norm": layer.post_attention_layernorm.weight.detach().float().cpu()
            for i, layer in enumerate(original.model.layers)
        },
        "final_norm": original.model.norm.weight.detach().float().cpu(),
    }
    del original
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    private_config = AutoConfig.from_pretrained(args.private, local_files_only=True)
    private_config.aloepri_attention_compute_dtype = args.private_attention_compute_dtype
    private_load = dict(common)
    if args.private_attention_compute_dtype == "float64":
        private_load.pop("dtype")
    private_model = (
        AutoModelForCausalLM.from_pretrained(
            args.private, config=private_config, **private_load
        )
        .to(device)
        .eval()
    )
    private = capture_model(
        private_model,
        tau[plain_ids],
        preserve_dtype=args.private_attention_compute_dtype == "float64",
    )
    del private_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    records: list[dict[str, object]] = []

    def compare(name: str, reference: Tensor, candidate: Tensor) -> None:
        result = metrics(reference, candidate)
        records.append({"operator": name, **result, "pass": result["nrmse"] <= args.nrmse_limit})

    pc = plain["captures"]
    vc = private["captures"]
    def restore_residual(value: Tensor) -> Tensor:
        if value.dtype == torch.float64:
            return value @ inverse.double()
        return value @ inverse

    compare("embedding", pc["embedding.output"], restore_residual(vc["embedding.output"]))
    num_heads = int(source_config.num_attention_heads)
    num_kv_heads = int(source_config.num_key_value_heads)
    head_dim = int(
        getattr(
            source_config,
            "head_dim",
            source_config.hidden_size // source_config.num_attention_heads,
        )
    )
    group_size = num_heads // num_kv_heads
    structural = "layers.0.q_order" in key
    for layer in range(int(source_config.num_hidden_layers)):
        prefix = f"layer.{layer}"
        key_prefix = f"layers.{layer}"
        compare(
            f"{prefix}.input_norm.input",
            pc[f"{prefix}.input_norm.input"],
            restore_residual(vc[f"{prefix}.input_norm.input"]),
        )
        compare(
            f"{prefix}.input_norm.output",
            pc[f"{prefix}.input_norm.output"],
            restore_residual(vc[f"{prefix}.input_norm.output"])
            * norm_weights[f"{prefix}.input_norm"],
        )
        restored: dict[str, Tensor] = {}
        for projection, size in (
            ("q_proj", group_size),
            ("k_proj", 1),
            ("v_proj", 1),
        ):
            value = vc[f"{prefix}.{projection}.output"]
            if structural:
                order_name = "q_order" if projection == "q_proj" else "kv_order"
                map_name = {"q_proj": "q_maps", "k_proj": "k_maps", "v_proj": "value_maps"}[
                    projection
                ]
                value = restore_heads(
                    value, key[f"{key_prefix}.{order_name}"], key[f"{key_prefix}.{map_name}"], size
                )
            restored[projection] = value
            compare(f"{prefix}.{projection}", pc[f"{prefix}.{projection}.output"], value)

        plain_attention = attention_internals(
            pc[f"{prefix}.q_proj.output"],
            pc[f"{prefix}.k_proj.output"],
            pc[f"{prefix}.v_proj.output"],
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            causal=True,
        )
        private_attention = attention_internals(
            vc[f"{prefix}.q_proj.output"],
            vc[f"{prefix}.k_proj.output"],
            vc[f"{prefix}.v_proj.output"],
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            causal=True,
            float64_scores=args.private_attention_compute_dtype == "float64_scores",
        )
        if structural:
            q_order = key[f"{key_prefix}.q_order"]
            kv_order = key[f"{key_prefix}.kv_order"]
            q_maps = key[f"{key_prefix}.q_maps"]
            k_maps = key[f"{key_prefix}.k_maps"]
            value_maps = key[f"{key_prefix}.value_maps"]
            private_q_rope = (
                restore_heads(
                    private_attention["q_rope"]
                    .transpose(1, 2)
                    .reshape_as(vc[f"{prefix}.q_proj.output"]),
                    q_order,
                    q_maps,
                    group_size,
                )
                .view_as(pc[f"{prefix}.q_proj.output"])
                .view(-1, plain_ids.shape[-1], num_heads, head_dim)
                .transpose(1, 2)
            )
            private_k_rope = (
                restore_heads(
                    private_attention["k_rope"]
                    .transpose(1, 2)
                    .reshape_as(vc[f"{prefix}.k_proj.output"]),
                    kv_order,
                    k_maps,
                    1,
                )
                .view(-1, plain_ids.shape[-1], num_kv_heads, head_dim)
                .transpose(1, 2)
            )
            private_scores = torch.empty_like(private_attention["scores"])
            private_softmax = torch.empty_like(private_attention["softmax"])
            for new_head, old_head in enumerate(q_order.tolist()):
                private_scores[:, old_head] = private_attention["scores"][:, new_head]
                private_softmax[:, old_head] = private_attention["softmax"][:, new_head]
            aggregate_flat = private_attention["aggregate"].reshape(
                *private_attention["aggregate"].shape[:2], -1
            )
            private_aggregate = restore_heads(
                aggregate_flat, q_order, value_maps, group_size
            ).view_as(plain_attention["aggregate"])
        else:
            private_q_rope = private_attention["q_rope"]
            private_k_rope = private_attention["k_rope"]
            private_scores = private_attention["scores"]
            private_softmax = private_attention["softmax"]
            private_aggregate = private_attention["aggregate"]
        compare(f"{prefix}.q_rope", plain_attention["q_rope"], private_q_rope)
        compare(f"{prefix}.k_rope", plain_attention["k_rope"], private_k_rope)
        compare(f"{prefix}.attention_scores", plain_attention["scores"], private_scores)
        compare(f"{prefix}.attention_softmax", plain_attention["softmax"], private_softmax)
        compare(
            f"{prefix}.attention_value_aggregate", plain_attention["aggregate"], private_aggregate
        )
        compare(
            f"{prefix}.o_proj",
            pc[f"{prefix}.o_proj.output"],
            restore_residual(vc[f"{prefix}.o_proj.output"]),
        )
        compare(
            f"{prefix}.attention_residual",
            pc[f"{prefix}.post_norm.input"],
            restore_residual(vc[f"{prefix}.post_norm.input"]),
        )
        compare(
            f"{prefix}.post_norm.output",
            pc[f"{prefix}.post_norm.output"],
            restore_residual(vc[f"{prefix}.post_norm.output"])
            * norm_weights[f"{prefix}.post_norm"],
        )
        gate = vc[f"{prefix}.gate_proj.output"]
        up = vc[f"{prefix}.up_proj.output"]
        if structural:
            order = key[f"{key_prefix}.ffn_order"]
            scales = key[f"{key_prefix}.ffn_scales"]
            gate = restore_permuted_features(gate, order)
            up = restore_permuted_features(up, order, scales)
        compare(f"{prefix}.gate_proj", pc[f"{prefix}.gate_proj.output"], gate)
        compare(f"{prefix}.up_proj", pc[f"{prefix}.up_proj.output"], up)
        compare(
            f"{prefix}.down_proj",
            pc[f"{prefix}.down_proj.output"],
            restore_residual(vc[f"{prefix}.down_proj.output"]),
        )
        compare(
            f"{prefix}.ffn_residual",
            pc[f"{prefix}.output"],
            restore_residual(vc[f"{prefix}.output"]),
        )

        plain_cache = plain["cache"][layer]
        private_cache = private["cache"][layer]
        private_cache_key = (
            private_cache["key"].transpose(1, 2).reshape_as(vc[f"{prefix}.k_proj.output"])
        )
        private_cache_value = (
            private_cache["value"].transpose(1, 2).reshape_as(vc[f"{prefix}.v_proj.output"])
        )
        if structural:
            private_cache_key = restore_heads(private_cache_key, kv_order, k_maps, 1)
            private_cache_value = restore_heads(private_cache_value, kv_order, value_maps, 1)
        compare(
            f"{prefix}.kv_cache.key",
            plain_cache["key"],
            private_cache_key.view(-1, plain_ids.shape[-1], num_kv_heads, head_dim).transpose(1, 2),
        )
        compare(
            f"{prefix}.kv_cache.value",
            plain_cache["value"],
            private_cache_value.view(-1, plain_ids.shape[-1], num_kv_heads, head_dim).transpose(
                1, 2
            ),
        )

    compare("final_norm.input", pc["final_norm.input"], restore_residual(vc["final_norm.input"]))
    compare(
        "final_norm.output",
        pc["final_norm.output"],
        restore_residual(vc["final_norm.output"]) * norm_weights["final_norm"],
    )
    compare("logits", pc["logits"], vc["logits"][..., key["tau"]])
    first_failure = next((item for item in records if not item["pass"]), None)
    payload = {
        "schema_version": 1,
        "source": str(args.source.resolve()),
        "private": str(args.private.resolve()),
        "key": str((args.key_dir / "paper_key.safetensors").resolve()),
        "prompt_sha256": hashlib.sha256(args.prompt.encode("utf-8")).hexdigest(),
        "input_tokens": int(plain_ids.numel()),
        "nrmse_limit": args.nrmse_limit,
        "private_attention_compute_dtype": args.private_attention_compute_dtype,
        "operator_count": len(records),
        "pass_count": sum(bool(item["pass"]) for item in records),
        "all_pass": first_failure is None,
        "first_failure": first_failure,
        "records": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {k: v for k, v in payload.items() if k != "records"}, ensure_ascii=False, indent=2
        )
    )


if __name__ == "__main__":
    main()
