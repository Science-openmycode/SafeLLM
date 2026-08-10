from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM

from aloepri.keys.generate import generate_vocab_key
from aloepri.transforms.vocab import permute_vocab_rows

SPECIAL_TOKEN_FIELDS = (
    "bos_token_id",
    "eos_token_id",
    "pad_token_id",
    "decoder_start_token_id",
)


def _map_token_value(value: object, tau: torch.Tensor) -> object:
    if value is None:
        return None
    if isinstance(value, int):
        return int(tau[value])
    if isinstance(value, list):
        return [int(tau[item]) for item in value]
    return value


def _map_special_tokens(config: object, tau: torch.Tensor) -> None:
    for field in SPECIAL_TOKEN_FIELDS:
        if hasattr(config, field):
            setattr(config, field, _map_token_value(getattr(config, field), tau))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def convert_vocab_checkpoint(
    source: Path,
    output: Path,
    key_dir: Path,
    *,
    model_id: str,
    source_revision: str,
    key_id: str,
    seed: int,
    max_shard_size: str = "2GB",
    resume: bool = False,
) -> Path:
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    partial = output.with_name(f"{output.name}.partial")
    if partial.exists():
        if resume:
            from aloepri.conversion.verify import verify_manifest

            verification = verify_manifest(partial)
            if not verification.ok:
                raise ValueError(
                    "partial checkpoint is incomplete or corrupt: "
                    + ", ".join(verification.failures)
                )
            os.replace(partial, output)
            return output
        raise FileExistsError(f"partial output already exists: {partial}")
    partial.mkdir(parents=True)
    key_dir.mkdir(parents=True, exist_ok=False)

    model = AutoModelForCausalLM.from_pretrained(
        source,
        local_files_only=True,
        dtype=torch.float32,
        attn_implementation="eager",
    )
    model.eval()
    embedding = model.get_input_embeddings()
    lm_head = model.get_output_embeddings()
    vocab_size = embedding.weight.shape[0]
    tau, inverse_tau = generate_vocab_key(vocab_size, seed=seed)
    tied = embedding.weight.data_ptr() == lm_head.weight.data_ptr()
    original_embedding = embedding.weight.detach().clone()
    with torch.no_grad():
        embedding.weight.copy_(permute_vocab_rows(original_embedding, tau))
        if not tied:
            original_head = lm_head.weight.detach().clone()
            lm_head.weight.copy_(permute_vocab_rows(original_head, tau))
    _map_special_tokens(model.config, tau)
    _map_special_tokens(model.generation_config, tau)
    model.config.aloepri = {
        "schema_version": 1,
        "model_id": model_id,
        "source_revision": source_revision,
        "key_id": key_id,
        "transform": "vocab_permutation",
        "server_token_space": "private",
    }
    model.save_pretrained(partial, safe_serialization=True, max_shard_size=max_shard_size)
    del model

    save_file({"tau": tau, "inverse_tau": inverse_tau}, key_dir / "vocab.safetensors")
    key_metadata = {
        "schema_version": 1,
        "key_id": key_id,
        "model_id": model_id,
        "source_revision": source_revision,
        "vocab_size": vocab_size,
        "tied_embeddings": tied,
        "vocab_file": "vocab.safetensors",
    }
    (key_dir / "key.json").write_text(json.dumps(key_metadata, indent=2), encoding="utf-8")
    files = []
    for path in sorted(partial.iterdir()):
        if path.is_file():
            files.append(
                {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            )
    manifest = {
        "schema_version": 1,
        "model_id": model_id,
        "source_revision": source_revision,
        "key_id": key_id,
        "transform_mode": "vocab_permutation",
        "files": files,
    }
    (partial / "aloepri_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(partial, output)
    return output


def remove_partial_conversion(path: Path) -> None:
    """Explicit recovery helper; callers must pass an exact `.partial` directory."""
    if path.suffix != ".partial" or not path.is_dir():
        raise ValueError("only an existing .partial directory can be removed")
    shutil.rmtree(path)
