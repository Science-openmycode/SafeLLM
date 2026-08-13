from __future__ import annotations

from collections.abc import Iterable

from aloepri.catalog.models import ModelCatalogEntry


class ModelCatalog:
    def __init__(self, entries: Iterable[ModelCatalogEntry]) -> None:
        self._entries = {entry.catalog_id: entry for entry in entries}

    def list(self) -> tuple[ModelCatalogEntry, ...]:
        return tuple(self._entries[name] for name in sorted(self._entries))

    def get(self, catalog_id: str) -> ModelCatalogEntry:
        try:
            return self._entries[catalog_id]
        except KeyError as error:
            raise KeyError(f"unknown catalog model: {catalog_id}") from error

    def recommend(self) -> tuple[ModelCatalogEntry, ...]:
        order = ("qwen2.5-0.5b-instruct", "deepseek-v2-lite-chat", "deepseek-v3")
        return tuple(self._entries[name] for name in order)


def builtin_catalog() -> ModelCatalog:
    return ModelCatalog(
        (
            ModelCatalogEntry(
                catalog_id="qwen2.5-0.5b-instruct",
                display_name="Qwen2.5-0.5B-Instruct",
                repo_id="Qwen/Qwen2.5-0.5B-Instruct",
                revision="7ae557604adf67be50417f59c2c2f167def9a775",
                adapter_id="qwen2",
                status="supported",
                parameter_summary="0.49B dense",
                expected_bytes=988_097_824,
                capabilities={"gqa": True, "mla": False, "moe": False, "mtp": False},
                source={"format": "safetensors", "dtype": "bfloat16"},
                conversion={"output_dtype": "bfloat16", "expansion_h": 128, "tile_mib": 256},
                runtime={"preferred": "hf", "fallback": "vllm"},
                license="Apache-2.0",
            ),
            ModelCatalogEntry(
                catalog_id="deepseek-v2-lite-chat",
                display_name="DeepSeek-V2-Lite-Chat",
                repo_id="deepseek-ai/DeepSeek-V2-Lite-Chat",
                revision="85864749cd611b4353ce1decdb286193298f64c7",
                adapter_id="deepseek_v2",
                status="supported",
                parameter_summary="16B total / 2.4B activated",
                expected_bytes=31_000_000_000,
                capabilities={"gqa": False, "mla": True, "moe": True, "mtp": False},
                source={"format": "safetensors", "dtype": "bfloat16"},
                conversion={"output_dtype": "bfloat16", "expansion_h": 0, "tile_mib": 256},
                runtime={"preferred": "hf", "fallback": "sglang"},
                license="DeepSeek Model License",
            ),
            ModelCatalogEntry(
                catalog_id="deepseek-v3",
                display_name="DeepSeek-V3",
                repo_id="deepseek-ai/DeepSeek-V3",
                revision="bb399fea3bbfbea55d71cb018e12cdfb6b215179",
                adapter_id="deepseek_v3",
                status="supported",
                parameter_summary="671B main + MTP weights",
                expected_bytes=689_000_000_000,
                capabilities={"gqa": False, "mla": True, "moe": True, "mtp": True, "fp8": True},
                source={
                    "format": "safetensors",
                    "dtype": "fp8_e4m3fn",
                    "weight_block_size": [128, 128],
                },
                conversion={"output_dtype": "fp8_e4m3fn", "expansion_h": 128, "tile_mib": 256},
                runtime={"preferred": "sglang", "fallback": None},
                license="DeepSeek Model License",
            ),
        )
    )
