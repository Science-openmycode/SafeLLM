from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections.abc import Callable
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizerBase

from aloepri.attacks.mapping import rowsort_nearest_product
from aloepri.evidence import model_identity as verified_model_identity

CACHE_ALGORITHM_VERSION = 3


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    contiguous = value.detach().cpu().contiguous()
    return hashlib.sha256(contiguous.numpy().tobytes()).hexdigest()


def prepare_cache_manifest(
    cache_dir: Path | None,
    inputs: dict[str, object],
    *,
    bind_existing_unversioned_cache: bool,
) -> None:
    if cache_dir is None:
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "cache_manifest.json"
    input_digest = hashlib.sha256(
        json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    existing_predictions = sorted(cache_dir.glob("*.pt"))
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("input_sha256") != input_digest:
            raise ValueError(f"VMA cache input fingerprint mismatch: {manifest_path}")
        for name, expected in manifest.get("prediction_files", {}).items():
            path = cache_dir / name
            if not path.is_file() or sha256_file(path) != expected["sha256"]:
                raise ValueError(f"VMA cache file fingerprint mismatch: {path}")
        listed = set(manifest.get("prediction_files", {}))
        actual = {path.name for path in existing_predictions}
        extra = sorted(actual - listed)
        if extra:
            raise ValueError(f"unregistered VMA cache prediction files: {extra}")
        return
    if existing_predictions and not bind_existing_unversioned_cache:
        raise ValueError(
            "unversioned VMA prediction cache exists; rerun with an empty cache or "
            "explicitly pass --bind-existing-unversioned-cache after provenance review"
        )
    payload = {
        "schema_version": 1,
        "cache_algorithm_version": CACHE_ALGORITHM_VERSION,
        "input_sha256": input_digest,
        "inputs": inputs,
        "prediction_files": {
            path.name: {"size": path.stat().st_size, "sha256": sha256_file(path)}
            for path in existing_predictions
        },
        "bound_existing_unversioned_cache": bool(existing_predictions),
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def finalize_cache_manifest(cache_dir: Path | None) -> None:
    if cache_dir is None:
        return
    manifest_path = cache_dir / "cache_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["prediction_files"] = {
        path.name: {"size": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(cache_dir.glob("*.pt"))
    }
    partial = manifest_path.with_name(manifest_path.name + ".partial")
    partial.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    partial.replace(manifest_path)


def register_cache_prediction(cache_dir: Path, path: Path) -> None:
    manifest_path = cache_dir / "cache_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload.setdefault("prediction_files", {})[path.name] = {
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    partial = manifest_path.with_name(manifest_path.name + ".partial")
    partial.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    partial.replace(manifest_path)


def wilson_interval(successes: int, total: int) -> list[float]:
    """Two-sided 95% Wilson score interval for a reported recovery proportion."""
    if total <= 0:
        return [0.0, 0.0]
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return [center - half, center + half]


def cached_prediction(
    cache_dir: Path | None,
    name: str,
    expected_count: int,
    compute: Callable[[], torch.Tensor],
) -> torch.Tensor:
    if cache_dir is None:
        return compute()
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{name}.pt"
    if path.is_file():
        prediction = torch.load(path, map_location="cpu", weights_only=True)
        if prediction.shape != (expected_count,) or prediction.dtype != torch.int64:
            raise ValueError(f"invalid cached prediction: {path}")
        return prediction
    prediction = compute().to(device="cpu", dtype=torch.int64)
    partial = path.with_name(path.name + ".partial")
    torch.save(prediction, partial)
    partial.replace(path)
    register_cache_prediction(cache_dir, path)
    return prediction


def qk_cross_product(
    left_embedding: torch.Tensor,
    right_embedding: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    if num_heads % num_kv_heads:
        raise ValueError("num_heads must be divisible by num_kv_heads")
    query = (left_embedding @ q_weight.mT).reshape(-1, num_heads, head_dim)
    key = (right_embedding @ k_weight.mT).reshape(-1, num_kv_heads, head_dim)
    key = key.repeat_interleave(num_heads // num_kv_heads, dim=1)
    return query.flatten(1) @ key.flatten(1).mT


def qk_project(
    embedding: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    query = (embedding @ q_weight.mT).reshape(-1, num_heads, head_dim)
    key = (embedding @ k_weight.mT).reshape(-1, num_kv_heads, head_dim)
    key = key.repeat_interleave(num_heads // num_kv_heads, dim=1)
    return query.flatten(1), key.flatten(1)


def load_tensor(model_dir: Path, name: str) -> torch.Tensor:
    for path in sorted(model_dir.glob("*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as tensors:
            if name in tensors.keys():
                return tensors.get_tensor(name)
    raise KeyError(f"{name} not found in {model_dir}")


def load_head(model_dir: Path) -> torch.Tensor:
    try:
        return load_tensor(model_dir, "lm_head.weight")
    except KeyError:
        config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
        if not bool(getattr(config, "tie_word_embeddings", False)):
            raise
        return load_tensor(model_dir, "model.embed_tokens.weight")


def pupa_tokens(
    tokenizer_path: Path,
) -> tuple[PreTrainedTokenizerBase, list[list[int]], list[list[int]], torch.Tensor]:
    from datasets import load_dataset

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    dataset = load_dataset("Columbia-NLP/PUPA", "pupa_tnb", split="train")
    units: list[list[int]] = []
    texts: list[list[int]] = []
    for row in dataset:
        text_ids = tokenizer.encode(str(row["user_query"]), add_special_tokens=False)
        if text_ids:
            texts.append(text_ids)
        for text in str(row["pii_units"]).split("||"):
            token_ids = tokenizer.encode(text.strip(), add_special_tokens=False)
            if token_ids:
                units.append(token_ids)
    unique = torch.tensor(
        sorted({token for sequence in (*texts, *units) for token in sequence}),
        dtype=torch.int64,
    )
    return tokenizer, texts, units, unique


def score_mapping(
    recovered: dict[int, int],
    tau: torch.Tensor,
    texts: list[list[int]],
    units: list[list[int]],
    *,
    tokenizer: PreTrainedTokenizerBase,
    plaintext_embedding: torch.Tensor,
) -> dict[str, float | int | str]:
    import sacrebleu

    token_total = sum(len(text) for text in texts)
    token_correct = sum(
        recovered.get(token, -1) == int(tau[token]) for text in texts for token in text
    )
    pii_correct = sum(
        all(recovered.get(token, -1) == int(tau[token]) for token in unit) for unit in units
    )
    inverse: dict[int, int] = {}
    conflicts: set[int] = set()
    for plain_id, private_id in recovered.items():
        if private_id in inverse and inverse[private_id] != plain_id:
            conflicts.add(private_id)
        else:
            inverse[private_id] = plain_id
    for private_id in conflicts:
        inverse.pop(private_id, None)
    unk_token_id = getattr(tokenizer, "unk_token_id", None)
    if unk_token_id is None:
        unk_token_id = getattr(tokenizer, "eos_token_id", 0)
    recovered_text_tokens = [
        [inverse.get(int(tau[token]), int(unk_token_id)) for token in text] for text in texts
    ]
    original_strings = [tokenizer.decode(text, skip_special_tokens=True) for text in texts]
    recovered_strings = [
        tokenizer.decode(text, skip_special_tokens=True) for text in recovered_text_tokens
    ]
    bleu4 = float(sacrebleu.corpus_bleu(recovered_strings, [original_strings]).score)
    similarities = []
    embedding = plaintext_embedding.float()
    for original_ids, recovered_ids in zip(texts, recovered_text_tokens, strict=True):
        original_vector = embedding[torch.tensor(original_ids, device=embedding.device)].mean(0)
        recovered_vector = embedding[torch.tensor(recovered_ids, device=embedding.device)].mean(0)
        similarities.append(
            torch.nn.functional.cosine_similarity(
                original_vector.unsqueeze(0), recovered_vector.unsqueeze(0)
            )[0]
        )
    cossim = float(torch.stack(similarities).mean()) if similarities else 0.0
    return {
        "token_occurrences": token_total,
        "recovered_token_occurrences": token_correct,
        "ttrsr": token_correct / token_total,
        "ttrsr_wilson_95_percent_ci": wilson_interval(token_correct, token_total),
        "pii_units": len(units),
        "recovered_pii_units": pii_correct,
        "piirsr": pii_correct / len(units),
        "piirsr_wilson_95_percent_ci": wilson_interval(pii_correct, len(units)),
        "proportion_ci_note": (
            "Wilson interval over token occurrences or PII units; it does not measure "
            "between-key uncertainty because this artifact uses one transform key"
        ),
        "bleu4": bleu4,
        "cossim": cossim,
        "bleu4_definition": "SacreBLEU corpus BLEU-4 on decoded PUPA user_query text",
        "cossim_definition": (
            "mean cosine similarity of mean Qwen input-embedding vectors per PUPA user_query"
        ),
    }


def main() -> None:
    started_at = time.perf_counter()
    parser = argparse.ArgumentParser(description="PUPA-scoped paper VMA for dense Qwen")
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--private", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--key-dir", type=Path)
    source.add_argument("--candidate-observations", type=Path)
    parser.add_argument("--candidate-sizes", type=int, nargs="+", default=[1024, 4096])
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 4, 8, 12, 16, 20, 23])
    parser.add_argument(
        "--combinations",
        nargs="+",
        choices=["We_Wh", "We_Wq_We_WkT", "We_Wgate", "We_Wup", "Wdown_Wh"],
        default=["We_Wh", "We_Wq_We_WkT", "We_Wgate", "We_Wup", "Wdown_Wh"],
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prediction-cache-dir", type=Path)
    parser.add_argument("--query-batch-size", type=int, default=64)
    parser.add_argument("--candidate-batch-size", type=int, default=256)
    parser.add_argument(
        "--stream-known-from-cpu",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--bind-existing-unversioned-cache",
        action="store_true",
        help="Bind legacy .pt caches to the current full input fingerprint after review.",
    )
    args = parser.parse_args()

    tokenizer, texts, units, query_ids = pupa_tokens(args.original)
    candidate_artifact: dict[str, object] | None = None
    if args.candidate_observations is not None:
        candidate_artifact = json.loads(
            args.candidate_observations.read_text(encoding="utf-8")
        )
        if (
            candidate_artifact.get("schema") != "aloepri-vma-candidate-observations-v1"
            or candidate_artifact.get("mapping_disclosed") is not False
            or candidate_artifact.get("query_ids") != query_ids.tolist()
        ):
            parser.error("candidate observation artifact is incompatible with PUPA queries")
        tau = None
        vocabulary_size = int(candidate_artifact["vocabulary_size"])
    else:
        tau = load_file(args.key_dir / "paper_key.safetensors", device="cpu")["tau"]
        vocabulary_size = tau.numel()
    max_candidates = max(args.candidate_sizes)
    if max_candidates > vocabulary_size:
        parser.error("candidate size exceeds vocabulary")
    if min(args.candidate_sizes) < query_ids.numel():
        parser.error(
            f"candidate size must cover all {query_ids.numel()} unique PUPA text/PII tokens"
        )

    generator = torch.Generator().manual_seed(20260805)
    if candidate_artifact is not None:
        plain_candidates = torch.tensor(candidate_artifact["plain_candidates"])
        if plain_candidates.numel() < max_candidates:
            parser.error("candidate artifact does not cover requested maximum")
    else:
        query_mask = torch.zeros(vocabulary_size, dtype=torch.bool)
        query_mask[query_ids] = True
        decoys = torch.randperm(vocabulary_size, generator=generator)
        decoys = decoys[~query_mask[decoys]][: max_candidates - query_ids.numel()]
        plain_candidates = torch.cat((query_ids, decoys))

    cache_inputs: dict[str, object] = {
        "algorithm_version": CACHE_ALGORITHM_VERSION,
        "original": verified_model_identity(args.original),
        "private": verified_model_identity(args.private),
        "candidate_observations_sha256": (
            sha256_file(args.candidate_observations) if args.candidate_observations else None
        ),
        "key_manifest_sha256": (
            sha256_file(args.key_dir / "key_manifest.json") if args.key_dir else None
        ),
        "paper_key_sha256": (
            sha256_file(args.key_dir / "paper_key.safetensors") if args.key_dir else None
        ),
        "query_ids_sha256": tensor_sha256(query_ids),
        "plain_candidates_sha256": tensor_sha256(plain_candidates),
        "candidate_sizes": args.candidate_sizes,
        "layers": args.layers,
        "combinations": args.combinations,
        "text_count": len(texts),
        "pii_unit_count": len(units),
        "dataset_content_sha256": hashlib.sha256(
            json.dumps(
                {"texts": texts, "pii_units": units},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "script_sha256": sha256_file(Path(__file__)),
    }
    prepare_cache_manifest(
        args.prediction_cache_dir,
        cache_inputs,
        bind_existing_unversioned_cache=args.bind_existing_unversioned_cache,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    # Keep full-vocabulary matrices on CPU. Only evaluated query/candidate rows are
    # transferred to the GPU so a 6 GiB device retains a material safety margin.
    original_embedding = load_tensor(args.original, "model.embed_tokens.weight").float()
    private_embedding = load_tensor(args.private, "model.embed_tokens.weight").float()
    original_head = load_head(args.original).float()
    private_head = load_head(args.private).float()
    final_norm = load_tensor(args.original, "model.norm.weight").float()
    config = AutoConfig.from_pretrained(args.original, local_files_only=True)
    num_heads = int(config.num_attention_heads)
    num_kv_heads = int(config.num_key_value_heads)
    head_dim = int(getattr(config, "head_dim", config.hidden_size // num_heads))

    payload: dict[str, object] = {
        "attack": "VMA_PUPA_candidate_scaling",
        "dataset": "Columbia-NLP/PUPA:pupa_tnb:train",
        "text_tokenization": "user_query, tokenizer.encode(add_special_tokens=False)",
        "pii_tokenization": (
            "each pii_units entry, stripped, tokenizer.encode(add_special_tokens=False)"
        ),
        "unique_text_and_pii_token_ids": query_ids.numel(),
        "vocabulary_size": vocabulary_size,
        "target_key_loaded": tau is not None,
        "candidate_space_note": (
            "all PUPA user_query/PII tokens plus deterministic random vocabulary decoys"
        ),
        "layers": args.layers,
        "combinations": args.combinations,
        "results": {},
        "device": device,
        "prediction_cache_dir": (
            str(args.prediction_cache_dir) if args.prediction_cache_dir else None
        ),
    }

    for candidate_size in sorted(set(args.candidate_sizes)):
        candidates = plain_candidates[:candidate_size]
        if candidate_artifact is not None:
            private_candidates = torch.tensor(
                candidate_artifact["private_candidates_by_size"][str(candidate_size)],
                dtype=torch.int64,
            )
        else:
            private_candidates = tau[candidates]
            private_candidates = private_candidates[
                torch.randperm(candidate_size, generator=generator)
            ]
        query_embedding = original_embedding[query_ids].to(device)
        candidate_embedding = original_embedding[candidates].to(device)
        private_candidate_embedding = private_embedding[private_candidates].to(device)
        final_norm_device = final_norm.to(device)
        fused_head_query = original_head[query_ids].to(device) * final_norm_device
        fused_head_candidate = original_head[candidates].to(device) * final_norm_device
        private_candidate_head = private_head[private_candidates].to(device)
        combination_predictions: dict[str, list[torch.Tensor]] = {
            combination: [] for combination in args.combinations
        }

        if "We_Wh" in combination_predictions:
            known_wewh = (query_embedding @ fused_head_candidate.mT).cpu()
            combination_predictions["We_Wh"].append(
                cached_prediction(
                    args.prediction_cache_dir,
                    f"c{candidate_size}.We_Wh",
                    query_ids.numel(),
                    lambda known_wewh=known_wewh,
                    private_candidate_embedding=private_candidate_embedding,
                    private_candidate_head=private_candidate_head: rowsort_nearest_product(
                        known_wewh,
                        private_candidate_embedding,
                        private_candidate_head.mT,
                        query_batch_size=args.query_batch_size,
                        candidate_batch_size=args.candidate_batch_size,
                        device=device,
                        stream_known_from_cpu=args.stream_known_from_cpu,
                    ),
                )
            )
            del known_wewh

        for layer in args.layers:
            layer_prefix = f"model.layers.{layer}"
            norm_name = f"model.layers.{layer}.post_attention_layernorm.weight"
            original_norm = load_tensor(args.original, norm_name).to(device).float()
            for projection, combination in (
                ("gate_proj", "We_Wgate"),
                ("up_proj", "We_Wup"),
            ):
                if combination not in combination_predictions:
                    continue
                name = f"{layer_prefix}.mlp.{projection}.weight"
                original_weight = load_tensor(args.original, name).to(device).float()
                private_weight = load_tensor(args.private, name).to(device).float()
                fused_weight = original_weight * original_norm.unsqueeze(0)
                known = (query_embedding @ fused_weight.mT).cpu()
                combination_predictions[combination].append(
                    cached_prediction(
                        args.prediction_cache_dir,
                        f"c{candidate_size}.{combination}.layer{layer}",
                        query_ids.numel(),
                        lambda known=known,
                        private_candidate_embedding=private_candidate_embedding,
                        private_weight=private_weight: rowsort_nearest_product(
                            known,
                            private_candidate_embedding,
                            private_weight.mT,
                            query_batch_size=args.query_batch_size,
                            candidate_batch_size=args.candidate_batch_size,
                            device=device,
                            stream_known_from_cpu=args.stream_known_from_cpu,
                        ),
                    )
                )
                del original_weight, private_weight, fused_weight, known

            if "Wdown_Wh" in combination_predictions:
                down_name = f"{layer_prefix}.mlp.down_proj.weight"
                original_down = load_tensor(args.original, down_name).to(device).float()
                private_down = load_tensor(args.private, down_name).to(device).float()
                known_down_head = (fused_head_query @ original_down).cpu()
                combination_predictions["Wdown_Wh"].append(
                    cached_prediction(
                        args.prediction_cache_dir,
                        f"c{candidate_size}.Wdown_Wh.layer{layer}",
                        query_ids.numel(),
                        lambda known_down_head=known_down_head,
                        private_candidate_head=private_candidate_head,
                        private_down=private_down: rowsort_nearest_product(
                            known_down_head,
                            private_candidate_head,
                            private_down,
                            query_batch_size=args.query_batch_size,
                            candidate_batch_size=args.candidate_batch_size,
                            device=device,
                            stream_known_from_cpu=args.stream_known_from_cpu,
                        ),
                    )
                )
                del original_down, private_down, known_down_head

            if "We_Wq_We_WkT" in combination_predictions:
                input_norm = load_tensor(
                    args.original, f"{layer_prefix}.input_layernorm.weight"
                ).to(device).float()
                original_q = load_tensor(
                    args.original, f"{layer_prefix}.self_attn.q_proj.weight"
                ).to(device).float()
                original_k = load_tensor(
                    args.original, f"{layer_prefix}.self_attn.k_proj.weight"
                ).to(device).float()
                private_q = load_tensor(
                    args.private, f"{layer_prefix}.self_attn.q_proj.weight"
                ).to(device).float()
                private_k = load_tensor(
                    args.private, f"{layer_prefix}.self_attn.k_proj.weight"
                ).to(device).float()
                known_qk = qk_cross_product(
                    query_embedding * input_norm,
                    candidate_embedding * input_norm,
                    original_q,
                    original_k,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                ).cpu()
                observed_query, observed_key = qk_project(
                    private_candidate_embedding,
                    private_q,
                    private_k,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                )
                combination_predictions["We_Wq_We_WkT"].append(
                    cached_prediction(
                        args.prediction_cache_dir,
                        f"c{candidate_size}.We_Wq_We_WkT.layer{layer}",
                        query_ids.numel(),
                        lambda known_qk=known_qk,
                        observed_query=observed_query,
                        observed_key=observed_key: rowsort_nearest_product(
                            known_qk,
                            observed_query,
                            observed_key.mT,
                            query_batch_size=args.query_batch_size,
                            candidate_batch_size=args.candidate_batch_size,
                            device=device,
                            stream_known_from_cpu=args.stream_known_from_cpu,
                        ),
                    )
                )
                del (
                    input_norm,
                    original_q,
                    original_k,
                    private_q,
                    private_k,
                    known_qk,
                    observed_query,
                    observed_key,
                )
            del original_norm
            if device == "cuda":
                torch.cuda.empty_cache()

        del (
            query_embedding,
            candidate_embedding,
            private_candidate_embedding,
            final_norm_device,
            fused_head_query,
            fused_head_candidate,
            private_candidate_head,
        )

        size_results: dict[str, object] = {}
        for combination, predictions in combination_predictions.items():
            voted_positions = torch.mode(torch.stack(predictions), dim=0).values
            predicted_private_ids = private_candidates[voted_positions]
            recovered = {
                int(token): int(predicted_private_ids[index])
                for index, token in enumerate(query_ids)
            }
            if tau is None:
                result = {
                    "plain_token_ids": query_ids.tolist(),
                    "predicted_private_ids": predicted_private_ids.tolist(),
                    "target_key_loaded": False,
                    "vote_count": len(predictions),
                }
            else:
                result = score_mapping(
                    recovered,
                    tau,
                    texts,
                    units,
                    tokenizer=tokenizer,
                    plaintext_embedding=original_embedding,
                )
                result["vote_count"] = len(predictions)
                result["unique_mapping_recovery_rate"] = sum(
                    recovered[int(token)] == int(tau[token]) for token in query_ids
                ) / query_ids.numel()
            size_results[combination] = result
        payload["results"][str(candidate_size)] = size_results  # type: ignore[index]

    payload["runtime"] = {
        "elapsed_seconds": time.perf_counter() - started_at,
        "device": device,
        "query_batch_size": args.query_batch_size,
        "candidate_batch_size": args.candidate_batch_size,
        "stream_known_from_cpu": args.stream_known_from_cpu,
        "peak_gpu_allocated_bytes": (
            torch.cuda.max_memory_allocated() if device == "cuda" else None
        ),
        "peak_gpu_reserved_bytes": (
            torch.cuda.max_memory_reserved() if device == "cuda" else None
        ),
    }
    finalize_cache_manifest(args.prediction_cache_dir)
    if args.prediction_cache_dir is not None:
        cache_manifest_path = args.prediction_cache_dir / "cache_manifest.json"
        cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
        payload["provenance"] = {
            "schema_version": 1,
            "formal_run_binding": not bool(
                cache_manifest.get("bound_existing_unversioned_cache")
            ),
            "inputs": cache_inputs,
            "cache_manifest_sha256": sha256_file(cache_manifest_path),
            "cache_bound_existing_unversioned": bool(
                cache_manifest.get("bound_existing_unversioned_cache")
            ),
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
