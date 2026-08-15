from __future__ import annotations

from collections.abc import Iterable

from aloepri.catalog.models import ModelCatalogEntry


def _qwen25_instruct_entry(
    *,
    size: str,
    revision: str,
    expected_bytes: int | None,
    license_id: str,
    validated: bool = False,
) -> ModelCatalogEntry:
    """Build a pinned Qwen2.5 Instruct family entry.

    The architecture adapter is shared across the family.  Only the 0.5B
    checkpoint is a product acceptance baseline; larger entries remain
    executable but are labelled family-compatible until their checkpoint
    smoke gate has been recorded.
    """

    catalog_id = f"qwen2.5-{size.lower()}-instruct"
    repo_name = f"Qwen2.5-{size}-Instruct"
    minimum_ram_gib = {
        "0.5B": 8,
        "1.5B": 16,
        "3B": 32,
        "7B": 64,
        "14B": 128,
        "32B": 256,
        "72B": 512,
    }[size]
    return ModelCatalogEntry(
        catalog_id=catalog_id,
        display_name=repo_name,
        repo_id=f"Qwen/{repo_name}",
        revision=revision,
        adapter_id="qwen2",
        status="supported" if validated else "family-compatible",
        parameter_summary=f"{size} dense Qwen2.5",
        expected_bytes=expected_bytes,
        capabilities={"gqa": True, "mla": False, "moe": False, "mtp": False},
        source={"format": "safetensors", "dtype": "bfloat16"},
        conversion={
            "output_dtype": "bfloat16",
            "expansion_h": 128,
            "tile_mib": 256,
            "minimum_host_ram_gib": minimum_ram_gib,
            "estimated_output_ratio": 1.15,
        },
        runtime={"preferred": "hf", "fallback": "vllm"},
        license=license_id,
        family_id="qwen2",
        family_name="Qwen2 / Qwen2.5",
        conversion_ready=True,
        deployment_ready=validated,
        support_note=(
            "已完成转换、部署和问答验收"
            if validated
            else "同族转换器可执行；部署前必须完成该参数规模的冒烟验收"
        ),
        max_stage="chat" if validated else "convert",
        visibility="recommended" if validated else "family",
        source_url=f"https://huggingface.co/Qwen/{repo_name}",
        last_validated_revision=revision if validated else None,
    )


def _qwen3_dense_entry(
    *, size: str, revision: str, expected_bytes: int
) -> ModelCatalogEntry:
    repo_name = f"Qwen3-{size}"
    minimum_ram_gib = {
        "0.6B": 8,
        "1.7B": 16,
        "4B": 32,
        "14B": 128,
        "32B": 256,
    }[size]
    return ModelCatalogEntry(
        catalog_id=f"qwen3-{size.lower()}",
        display_name=repo_name,
        repo_id=f"Qwen/{repo_name}",
        revision=revision,
        adapter_id="qwen3_dense",
        status="family-compatible",
        parameter_summary=f"{size} dense GQA with Q/K normalization",
        expected_bytes=expected_bytes,
        capabilities={
            "gqa": True,
            "mla": False,
            "moe": False,
            "mtp": False,
            "qk_norm": True,
        },
        source={"format": "safetensors", "dtype": "bfloat16"},
        conversion={
            "output_dtype": "bfloat16",
            "expansion_h": 128,
            "tile_mib": 256,
            "minimum_host_ram_gib": minimum_ram_gib,
            "estimated_output_ratio": 1.15,
        },
        runtime={"preferred": "hf", "fallback": "vllm"},
        license="Apache-2.0",
        family_id="qwen3",
        family_name="Qwen3",
        conversion_ready=True,
        deployment_ready=False,
        support_note=(
            "Qwen3 Q/K Norm 已由专用坐标变换同步处理；部署前会按当前服务器资源"
            "执行完整 checkpoint 检查和加载冒烟。"
        ),
        max_stage="convert",
        visibility="family",
        source_url=f"https://huggingface.co/Qwen/{repo_name}",
    )


def _deepseek_entry(
    *,
    repo_name: str,
    revision: str,
    generation: str,
    parameter_summary: str,
    expected_bytes: int | None,
) -> ModelCatalogEntry:
    v3 = generation == "v3"
    adapter_id = "deepseek_v3" if v3 else "deepseek_v2"
    dtype = "fp8_e4m3fn" if v3 else "bfloat16"
    source: dict[str, object] = {"format": "safetensors", "dtype": dtype}
    if v3:
        source["weight_block_size"] = [128, 128]
    return ModelCatalogEntry(
        catalog_id=repo_name.lower().replace(".", "-").replace("_", "-"),
        display_name=repo_name,
        repo_id=f"deepseek-ai/{repo_name}",
        revision=revision,
        adapter_id=adapter_id,
        status="family-compatible",
        parameter_summary=parameter_summary,
        expected_bytes=expected_bytes,
        capabilities={
            "gqa": False,
            "mla": True,
            "moe": True,
            "mtp": v3,
            "fp8": v3,
        },
        source=source,
        conversion={
            "output_dtype": dtype,
            "expansion_h": 128 if v3 else 0,
            "tile_mib": 256,
            "minimum_host_ram_gib": 32,
        },
        runtime={"preferred": "sglang", "fallback": "hf"},
        license="DeepSeek Model License",
        family_id=adapter_id,
        family_name="DeepSeek MLA / MoE",
        conversion_ready=True,
        deployment_ready=False,
        support_note=(
            "官方同架构版本已接入统一MLA/MoE转换器；下载后仍会逐张量检查并在目标服务器强制加载冒烟"
        ),
        max_stage="convert",
        visibility="family",
        source_url=f"https://huggingface.co/deepseek-ai/{repo_name}",
    )


def _glm_dense_entry(
    *, repo_name: str, revision: str, parameter_summary: str, expected_bytes: int
) -> ModelCatalogEntry:
    branch_norm = "0414" in repo_name.upper()
    return ModelCatalogEntry(
        catalog_id=repo_name.lower(),
        display_name=repo_name,
        repo_id=f"zai-org/{repo_name}",
        revision=revision,
        adapter_id="glm_dense",
        status="operator-gap" if branch_norm else "family-compatible",
        parameter_summary=parameter_summary,
        expected_bytes=expected_bytes,
        capabilities={"gqa": True, "mla": False, "moe": False, "mtp": False},
        source={"format": "safetensors", "dtype": "bfloat16"},
        conversion={
            "output_dtype": "bfloat16",
            "expansion_h": 128,
            "tile_mib": 256,
            "minimum_host_ram_gib": 48 if expected_bytes < 40_000_000_000 else 128,
        },
        runtime={"preferred": "hf", "fallback": "vllm"},
        license="GLM-4 License",
        family_id="glm_dense",
        family_name="GLM",
        conversion_ready=not branch_norm,
        deployment_ready=False,
        support_note=(
            "该检查点含Attention/MLP分支后置RMSNorm；当前轻量变换无法在不引入逐层D×D矩阵的情况下忠实消除，已禁止转换"
            if branch_norm
            else "GLM Dense同族转换路径已接入；下载后严格校验融合FFN、GQA和Partial RoPE布局"
        ),
        max_stage="inspect" if branch_norm else "convert",
        visibility="family",
        source_url=f"https://huggingface.co/zai-org/{repo_name}",
    )


def _glm_moe_entry(
    *,
    repo_name: str,
    revision: str,
    parameter_summary: str,
    expected_bytes: int,
    fp8: bool,
) -> ModelCatalogEntry:
    return ModelCatalogEntry(
        catalog_id=repo_name.lower(),
        display_name=repo_name,
        repo_id=f"zai-org/{repo_name}",
        revision=revision,
        adapter_id="glm4_moe",
        status="family-compatible",
        parameter_summary=parameter_summary,
        expected_bytes=expected_bytes,
        capabilities={"gqa": True, "mla": False, "moe": True, "mtp": True, "fp8": fp8},
        source={"format": "safetensors", "dtype": "fp8" if fp8 else "bfloat16"},
        conversion={
            "output_dtype": "bfloat16",
            "expansion_h": 128,
            "tile_mib": 256,
            "minimum_host_ram_gib": 64,
            "estimated_output_ratio": 2.2 if fp8 else 1.15,
        },
        runtime={"preferred": "sglang", "fallback": "hf"},
        license="MIT",
        family_id="glm4_moe",
        family_name="GLM",
        conversion_ready=True,
        deployment_ready=False,
        support_note="GLM4-MoE同族权重已接入FP8/BF16、专家路由、Partial RoPE和MTP转换路径",
        max_stage="convert",
        visibility="family",
        source_url=f"https://huggingface.co/zai-org/{repo_name}",
    )


def _kimi_text_entry(
    *,
    repo_name: str,
    revision: str,
    parameter_summary: str,
    expected_bytes: int | None,
    multimodal_int4: bool = False,
    packed_int4: bool = False,
) -> ModelCatalogEntry:
    uses_int4 = multimodal_int4 or packed_int4
    return ModelCatalogEntry(
        catalog_id=repo_name.lower(),
        display_name=repo_name,
        repo_id=f"moonshotai/{repo_name}",
        revision=revision,
        adapter_id="kimi_k2",
        status="text-private-supported" if uses_int4 else "family-compatible",
        parameter_summary=parameter_summary,
        expected_bytes=expected_bytes,
        capabilities={
            "gqa": False,
            "mla": True,
            "moe": True,
            "mtp": False,
            "fp8": not uses_int4,
            "int4": uses_int4,
            "multimodal": multimodal_int4,
        },
        source={
            "format": "safetensors",
            "dtype": "mixed_int4_bfloat16" if uses_int4 else "fp8_e4m3fn",
        },
        conversion={
            "output_dtype": "bfloat16" if uses_int4 else "fp8_e4m3fn",
            "expansion_h": 128,
            "tile_mib": 64 if uses_int4 else 256,
            "minimum_host_ram_gib": 32,
            "mode": "text-backbone" if multimodal_int4 else "text",
            "estimated_output_ratio": 4.2 if uses_int4 else 1.15,
        },
        runtime={"preferred": "sglang" if not uses_int4 else "hf", "fallback": None},
        license="Modified MIT",
        family_id="kimi_k2",
        family_name="Kimi",
        conversion_ready=True,
        deployment_ready=False,
        support_note=(
            "多模态外壳中的完整文本骨干已接入INT4解码；当前私有协议只开放文本问答"
            if multimodal_int4
            else (
                "Kimi-K2 Thinking的packed INT4专家权重会先按scale解码，再进入统一转换路径"
                if packed_int4
                else "Kimi-K2同族MLA、384专家与FP8权重已接入统一转换路径"
            )
        ),
        max_stage="convert",
        visibility="family",
        source_url=f"https://huggingface.co/moonshotai/{repo_name}",
    )


def _moonlight_entry(
    *, repo_name: str, revision: str, parameter_summary: str
) -> ModelCatalogEntry:
    return ModelCatalogEntry(
        catalog_id=repo_name.lower(),
        display_name=repo_name,
        repo_id=f"moonshotai/{repo_name}",
        revision=revision,
        adapter_id="deepseek_v3",
        status="family-compatible",
        parameter_summary=parameter_summary,
        expected_bytes=32_000_000_000,
        capabilities={"gqa": False, "mla": True, "moe": True, "mtp": False, "fp8": False},
        source={"format": "safetensors", "dtype": "bfloat16"},
        conversion={"output_dtype": "bfloat16", "expansion_h": 0, "tile_mib": 256},
        runtime={"preferred": "sglang", "fallback": "hf"},
        license="MIT",
        family_id="kimi_moonlight",
        family_name="Kimi",
        conversion_ready=True,
        deployment_ready=False,
        support_note="官方配置为DeepSeek-V3式MLA/MoE文本模型，复用统一DeepSeek转换器并执行严格张量门禁",
        max_stage="convert",
        visibility="family",
        source_url=f"https://huggingface.co/moonshotai/{repo_name}",
    )


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
        order = ("qwen2.5-0.5b-instruct",)
        return tuple(self._entries[name] for name in order)


def builtin_catalog() -> ModelCatalog:
    return ModelCatalog(
        (
            _qwen25_instruct_entry(
                size="0.5B",
                revision="7ae557604adf67be50417f59c2c2f167def9a775",
                expected_bytes=988_097_824,
                license_id="Apache-2.0",
                validated=True,
            ),
            _qwen25_instruct_entry(
                size="1.5B",
                revision="989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
                expected_bytes=3_087_467_144,
                license_id="Apache-2.0",
            ),
            _qwen25_instruct_entry(
                size="3B",
                revision="aa8e72537993ba99e69dfaafa59ed015b17504d1",
                expected_bytes=6_171_926_992,
                license_id="Qwen Research License",
            ),
            _qwen25_instruct_entry(
                size="7B",
                revision="a09a35458c702b33eeacc393d103063234e8bc28",
                expected_bytes=15_231_271_888,
                license_id="Apache-2.0",
            ),
            _qwen25_instruct_entry(
                size="14B",
                revision="cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8",
                expected_bytes=29_540_134_000,
                license_id="Apache-2.0",
            ),
            _qwen25_instruct_entry(
                size="32B",
                revision="5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd",
                expected_bytes=65_527_841_856,
                license_id="Apache-2.0",
            ),
            _qwen25_instruct_entry(
                size="72B",
                revision="495f39366efef23836d0cfae4fbe635880d2be31",
                expected_bytes=145_412_519_312,
                license_id="Qwen Research License",
            ),
            _deepseek_entry(
                repo_name="DeepSeek-V2-Lite",
                revision="604d5664dddd88a0433dbae533b7fe9472482de0",
                generation="v2",
                parameter_summary="16B total / 2.4B activated · Base",
                expected_bytes=31_413_526_576,
            ),
            _deepseek_entry(
                repo_name="DeepSeek-V2",
                revision="4461458f186c35188585855f28f77af5661ad489",
                generation="v2",
                parameter_summary="236B total / 21B activated · Base",
                expected_bytes=472_000_000_000,
            ),
            _deepseek_entry(
                repo_name="DeepSeek-V2-Chat",
                revision="8e3f5f6c2226787e41ba3e9283a06389d178c926",
                generation="v2",
                parameter_summary="236B total / 21B activated · Chat",
                expected_bytes=472_000_000_000,
            ),
            _deepseek_entry(
                repo_name="DeepSeek-V2-Chat-0628",
                revision="5d09e272c2b223830f4e84359cd9dd047a5d7c78",
                generation="v2",
                parameter_summary="236B total / 21B activated · Chat 0628",
                expected_bytes=472_000_000_000,
            ),
            _deepseek_entry(
                repo_name="DeepSeek-V2.5",
                revision="c85b5ede86f2a598af339624cac5723861e557ed",
                generation="v2",
                parameter_summary="236B total / 21B activated · V2.5",
                expected_bytes=472_000_000_000,
            ),
            _deepseek_entry(
                repo_name="DeepSeek-V2.5-1210",
                revision="6f134cbe88cb9284a8ce696e8ac8eefd0bc24ede",
                generation="v2",
                parameter_summary="236B total / 21B activated · V2.5 1210",
                expected_bytes=472_000_000_000,
            ),
            _deepseek_entry(
                repo_name="DeepSeek-V3-Base",
                revision="afb92e1fa402c2be2a9eb085312bb02e0384d6c7",
                generation="v3",
                parameter_summary="671B total / 37B activated · Base",
                expected_bytes=689_000_000_000,
            ),
            _deepseek_entry(
                repo_name="DeepSeek-V3-0324",
                revision="e9b33add76883f293d6bf61f6bd89b497e80e335",
                generation="v3",
                parameter_summary="671B total / 37B activated · 0324",
                expected_bytes=689_000_000_000,
            ),
            _deepseek_entry(
                repo_name="DeepSeek-V3.1",
                revision="c0781d039fb7a1ba2abc4add0bdc293e92d2b8db",
                generation="v3",
                parameter_summary="671B total / 37B activated · V3.1",
                expected_bytes=689_000_000_000,
            ),
            _deepseek_entry(
                repo_name="DeepSeek-V3.1-Base",
                revision="d3d4eafdc470de44bbf6f0a74f852eb522357be8",
                generation="v3",
                parameter_summary="671B total / 37B activated · V3.1 Base",
                expected_bytes=689_000_000_000,
            ),
            _deepseek_entry(
                repo_name="DeepSeek-V3.1-Terminus",
                revision="19510d6dc61f79dbd925bd51ee8a9081c509a4b6",
                generation="v3",
                parameter_summary="671B total / 37B activated · Terminus",
                expected_bytes=689_000_000_000,
            ),
            _deepseek_entry(
                repo_name="DeepSeek-R1",
                revision="56d4cbbb4d29f4355bab4b9a39ccb717a14ad5ad",
                generation="v3",
                parameter_summary="671B total / 37B activated · Reasoning",
                expected_bytes=689_000_000_000,
            ),
            _deepseek_entry(
                repo_name="DeepSeek-R1-Zero",
                revision="72234287cbc67dbf474d911359ae32b61a2fdc7e",
                generation="v3",
                parameter_summary="671B total / 37B activated · R1 Zero",
                expected_bytes=689_000_000_000,
            ),
            _deepseek_entry(
                repo_name="DeepSeek-R1-0528",
                revision="4236a6af538feda4548eca9ab308586007567f52",
                generation="v3",
                parameter_summary="671B total / 37B activated · R1 0528",
                expected_bytes=689_000_000_000,
            ),
            ModelCatalogEntry(
                catalog_id="openseek-small-v1-sft",
                display_name="OpenSeek-Small-v1-SFT",
                repo_id="BAAI/OpenSeek-Small-v1-SFT",
                revision="1515c184e6fe4a91e6061be513a79d607e8787cb",
                adapter_id="deepseek_v3",
                status="supported",
                parameter_summary="3.18 GB trained MLA + MoE checkpoint",
                expected_bytes=3_178_063_824,
                capabilities={
                    "gqa": False,
                    "mla": True,
                    "moe": True,
                    "mtp": False,
                    "fp8": False,
                },
                source={"format": "safetensors", "dtype": "bfloat16"},
                conversion={
                    "output_dtype": "bfloat16",
                    "expansion_h": 128,
                    "estimated_output_ratio": 1.21,
                    "tile_mib": 256,
                },
                runtime={"preferred": "hf", "fallback": None},
                license="Apache-2.0",
                family_id="deepseek_v3",
                family_name="DeepSeek MLA / MoE",
                conversion_ready=True,
                deployment_ready=True,
                support_note="真实检查点已完成MLA/MoE转换；目标服务器加载仍执行强制冒烟",
                max_stage="chat",
                visibility="advanced",
                source_url="https://huggingface.co/BAAI/OpenSeek-Small-v1-SFT",
                last_validated_revision="1515c184e6fe4a91e6061be513a79d607e8787cb",
            ),
            ModelCatalogEntry(
                catalog_id="deepseek-v2-lite-chat",
                display_name="DeepSeek-V2-Lite-Chat",
                repo_id="deepseek-ai/DeepSeek-V2-Lite-Chat",
                revision="85864749cd611b4353ce1decdb286193298f64c7",
                adapter_id="deepseek_v2",
                status="experimental",
                parameter_summary="16B total / 2.4B activated",
                expected_bytes=31_000_000_000,
                capabilities={"gqa": False, "mla": True, "moe": True, "mtp": False},
                source={"format": "safetensors", "dtype": "bfloat16"},
                conversion={"output_dtype": "bfloat16", "expansion_h": 0, "tile_mib": 256},
                runtime={"preferred": "hf", "fallback": "sglang"},
                license="DeepSeek Model License",
                family_id="deepseek_v2",
                family_name="DeepSeek MLA / MoE",
                conversion_ready=True,
                deployment_ready=False,
                support_note="DeepSeek-V2统一转换与HF部署路径已接入；目标服务器加载仍执行强制冒烟",
                max_stage="convert",
                visibility="advanced",
                source_url="https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat",
            ),
            ModelCatalogEntry(
                catalog_id="deepseek-v3",
                display_name="DeepSeek-V3",
                repo_id="deepseek-ai/DeepSeek-V3",
                revision="bb399fea3bbfbea55d71cb018e12cdfb6b215179",
                adapter_id="deepseek_v3",
                status="family-compatible",
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
                family_id="deepseek_v3",
                family_name="DeepSeek MLA / MoE",
                conversion_ready=True,
                deployment_ready=False,
                support_note="MLA/MoE/FP8/MTP转换器和部署入口已接入；671B物理转换尚未执行",
                max_stage="convert",
                visibility="developer",
                source_url="https://huggingface.co/deepseek-ai/DeepSeek-V3",
            ),
            _glm_dense_entry(
                repo_name="glm-4-9b-chat-1m-hf",
                revision="c6e9cd8555fe037a26d9113f14258f2600023690",
                parameter_summary="9B Dense · Chat · 1M context",
                expected_bytes=18_800_000_000,
            ),
            _glm_dense_entry(
                repo_name="GLM-4-9B-0414",
                revision="645b8482494e31b6b752272bf7f7f273ef0f3caf",
                parameter_summary="9B Dense · 0414",
                expected_bytes=18_800_000_000,
            ),
            _glm_dense_entry(
                repo_name="GLM-4-32B-Base-0414",
                revision="7675abea82951aaaedeb19014bab4e8f88c2d7a5",
                parameter_summary="32B Dense · Base 0414",
                expected_bytes=65_000_000_000,
            ),
            _glm_dense_entry(
                repo_name="GLM-4-32B-0414",
                revision="077b5c2f5c43bd3239fd605a0600229e8facbd4a",
                parameter_summary="32B Dense · Chat 0414",
                expected_bytes=65_000_000_000,
            ),
            _glm_dense_entry(
                repo_name="GLM-Z1-32B-0414",
                revision="8eb2858992c1f749e2a6d4075455decc2484722d",
                parameter_summary="32B Dense · Reasoning 0414",
                expected_bytes=65_000_000_000,
            ),
            _glm_moe_entry(
                repo_name="GLM-4.5-Air-Base",
                revision="888c873d4eca81f28d0ef420aa2d96457c28b959",
                parameter_summary="106B total / 12B activated · Base BF16",
                expected_bytes=212_000_000_000,
                fp8=False,
            ),
            _glm_moe_entry(
                repo_name="GLM-4.5-Air",
                revision="a24ceef6ce4f3536971efe9b778bdaa1bab18daa",
                parameter_summary="106B total / 12B activated · BF16",
                expected_bytes=212_000_000_000,
                fp8=False,
            ),
            _glm_moe_entry(
                repo_name="GLM-4.5-Air-FP8",
                revision="f9a9c5acf5e543cd24d659a056c5dbcda78ffcfc",
                parameter_summary="106B total / 12B activated · FP8",
                expected_bytes=113_000_000_000,
                fp8=True,
            ),
            _glm_moe_entry(
                repo_name="GLM-4.5-Base",
                revision="922a0cee7f137cf3b64c186f0bee77882e4a4e80",
                parameter_summary="355B total / 32B activated · Base BF16",
                expected_bytes=716_000_000_000,
                fp8=False,
            ),
            _glm_moe_entry(
                repo_name="GLM-4.5",
                revision="cbb2c7cfb52fa128a9660cb1a7a78e017899e115",
                parameter_summary="355B total / 32B activated · BF16",
                expected_bytes=716_000_000_000,
                fp8=False,
            ),
            _glm_moe_entry(
                repo_name="GLM-4.5-FP8",
                revision="8cc290ee4c7cbfa38d3a2db9bd0b7371773ece81",
                parameter_summary="355B total / 32B activated · FP8",
                expected_bytes=358_000_000_000,
                fp8=True,
            ),
            _glm_moe_entry(
                repo_name="GLM-4.6",
                revision="be72194883d968d7923a07e2f61681ea9a2826d1",
                parameter_summary="357B total / 32B activated · BF16",
                expected_bytes=714_000_000_000,
                fp8=False,
            ),
            _glm_moe_entry(
                repo_name="GLM-4.6-FP8",
                revision="c064d336a8d0b0f59071f77eafdcdfca40f4b54c",
                parameter_summary="357B total / 32B activated · FP8",
                expected_bytes=357_000_000_000,
                fp8=True,
            ),
            _glm_moe_entry(
                repo_name="GLM-4.7",
                revision="602d01efcdd332c5238ca4bcede555defbe83eb7",
                parameter_summary="358B total / 32B activated · BF16",
                expected_bytes=716_000_000_000,
                fp8=False,
            ),
            ModelCatalogEntry(
                catalog_id="glm-4-9b-chat-hf",
                display_name="GLM-4-9B-Chat-HF",
                repo_id="zai-org/glm-4-9b-chat-hf",
                revision="8599336fc6c125203efb2360bfaf4c80eef1d1bf",
                adapter_id="glm_dense",
                status="family-compatible",
                parameter_summary="9B dense GQA / fused SwiGLU",
                expected_bytes=18_799_902_720,
                capabilities={"gqa": True, "mla": False, "moe": False, "mtp": False},
                source={"format": "safetensors", "dtype": "bfloat16"},
                conversion={
                    "output_dtype": "bfloat16",
                    "expansion_h": 128,
                    "tile_mib": 256,
                    "minimum_host_ram_gib": 48,
                },
                runtime={"preferred": "hf", "fallback": "vllm"},
                license="GLM-4 License",
                family_id="glm_dense",
                family_name="GLM",
                conversion_ready=True,
                deployment_ready=False,
                support_note=(
                    "融合Gate/Up会先规范化为等价Dense GQA图，再进入私有Qwen运行时；"
                    "正式9B检查点仍需目标机器冒烟"
                ),
                max_stage="convert",
                visibility="family",
                source_url="https://huggingface.co/zai-org/glm-4-9b-chat-hf",
            ),
            ModelCatalogEntry(
                catalog_id="glm-4.7-fp8",
                display_name="GLM-4.7-FP8",
                repo_id="zai-org/GLM-4.7-FP8",
                revision="7b3b5f81eee81be12a6f8da2710eac4bafb0166a",
                adapter_id="glm4_moe",
                status="family-compatible",
                parameter_summary="MoE + QK-Norm + partial RoPE + MTP + FP8",
                expected_bytes=None,
                capabilities={"gqa": True, "mla": False, "moe": True, "mtp": True, "fp8": True},
                source={"format": "safetensors", "dtype": "fp8"},
                conversion={
                    "output_dtype": "bfloat16",
                    "expansion_h": 128,
                    "tile_mib": 256,
                    "minimum_host_ram_gib": 16,
                    "estimated_output_ratio": 2.2,
                },
                runtime={"preferred": "sglang", "fallback": None},
                license="MIT",
                family_id="glm4_moe",
                family_name="GLM",
                conversion_ready=True,
                deployment_ready=False,
                support_note=(
                    "已实现GLM MoE专用FP8解码、Q/K Norm、部分RoPE、"
                    "路由器、逐专家与MTP权重转换；HF主模型问答不启用MTP加速"
                ),
                max_stage="convert",
                visibility="developer",
                source_url="https://huggingface.co/zai-org/GLM-4.7-FP8",
            ),
            ModelCatalogEntry(
                catalog_id="qwen3-8b",
                display_name="Qwen3-8B",
                repo_id="Qwen/Qwen3-8B",
                revision="b968826d9c46dd6066d109eabc6255188de91218",
                adapter_id="qwen3_dense",
                status="family-compatible",
                parameter_summary="8B dense GQA with Q/K normalization",
                expected_bytes=16_381_470_720,
                capabilities={
                    "gqa": True,
                    "mla": False,
                    "moe": False,
                    "mtp": False,
                    "qk_norm": True,
                },
                source={"format": "safetensors", "dtype": "bfloat16"},
                conversion={
                    "output_dtype": "bfloat16",
                    "expansion_h": 128,
                    "tile_mib": 256,
                    "minimum_host_ram_gib": 64,
                    "estimated_output_ratio": 1.15,
                },
                runtime={"preferred": "hf", "fallback": "vllm"},
                license="Apache-2.0",
                family_id="qwen3",
                family_name="Qwen3",
                conversion_ready=True,
                deployment_ready=False,
                support_note=(
                    "Qwen3 Q/K Norm 已由专用坐标变换同步处理；"
                    "正式 8B 部署前仍需在目标 GPU 完成检查点冒烟测试"
                ),
                max_stage="convert",
                visibility="family",
                source_url="https://huggingface.co/Qwen/Qwen3-8B",
            ),
            _qwen3_dense_entry(
                size="0.6B",
                revision="c1899de289a04d12100db370d81485cdf75e47ca",
                expected_bytes=1_503_264_768,
            ),
            _qwen3_dense_entry(
                size="1.7B",
                revision="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
                expected_bytes=4_063_479_808,
            ),
            _qwen3_dense_entry(
                size="4B",
                revision="1cfa9a7208912126459214e8b04321603b3df60c",
                expected_bytes=8_044_936_192,
            ),
            _qwen3_dense_entry(
                size="14B",
                revision="40c069824f4251a91eefaf281ebe4c544efd3e18",
                expected_bytes=29_536_614_400,
            ),
            _qwen3_dense_entry(
                size="32B",
                revision="9216db5781bf21249d130ec9da846c4624c16137",
                expected_bytes=65_524_246_528,
            ),
            _moonlight_entry(
                repo_name="Moonlight-16B-A3B",
                revision="476b36a473d4467f94469414bef6cee75c9c8172",
                parameter_summary="16B total / 3B activated · Base",
            ),
            _moonlight_entry(
                repo_name="Moonlight-16B-A3B-Instruct",
                revision="4e735b07a89f73647dfab71ab91b840f362ede5b",
                parameter_summary="16B total / 3B activated · Instruct",
            ),
            _kimi_text_entry(
                repo_name="Kimi-K2-Base",
                revision="ce72df012259dcc55d945e890f815fe7ef69159c",
                parameter_summary="1T total / 32B activated · Base FP8",
                expected_bytes=1_026_408_235_864,
            ),
            _kimi_text_entry(
                repo_name="Kimi-K2-Instruct-0905",
                revision="ac6c49f04883bd0a0598b790693a72061c676629",
                parameter_summary="1T total / 32B activated · Instruct 0905 FP8",
                expected_bytes=1_026_408_235_864,
            ),
            _kimi_text_entry(
                repo_name="Kimi-K2-Thinking",
                revision="a51ccc050d73dab088bf7b0e2dd9b30ae85a4e55",
                parameter_summary="1T total / 32B activated · Thinking packed INT4",
                expected_bytes=None,
                packed_int4=True,
            ),
            _kimi_text_entry(
                repo_name="Kimi-K2.5",
                revision="4d01dfe0332d63057c186e0b262165819efb6611",
                parameter_summary="1.1T multimodal · INT4 text backbone",
                expected_bytes=595_000_000_000,
                multimodal_int4=True,
            ),
            ModelCatalogEntry(
                catalog_id="kimi-k2-instruct",
                display_name="Kimi-K2-Instruct",
                repo_id="moonshotai/Kimi-K2-Instruct",
                revision="fd1984e2b7a3350dbf7305fe73a4ede25c14de50",
                adapter_id="kimi_k2",
                status="family-compatible",
                parameter_summary="1T total / 32B activated, MLA + 384 experts",
                expected_bytes=1_026_408_235_864,
                capabilities={
                    "gqa": False,
                    "mla": True,
                    "moe": True,
                    "mtp": False,
                    "fp8": True,
                },
                source={
                    "format": "safetensors",
                    "dtype": "fp8_e4m3fn",
                    "weight_block_size": [128, 128],
                },
                conversion={
                    "output_dtype": "fp8_e4m3fn",
                    "expansion_h": 128,
                    "tile_mib": 256,
                    "minimum_host_ram_gib": 16,
                },
                runtime={"preferred": "sglang", "fallback": None},
                license="Modified MIT",
                family_id="kimi_k2",
                family_name="Kimi",
                conversion_ready=True,
                deployment_ready=False,
                support_note=(
                    "纯文本Kimi-K2通过零拷贝结构规范化复用已验证的"
                    "DeepSeek-V3 MLA/MoE/FP8私有转换器"
                ),
                max_stage="convert",
                visibility="family",
                source_url="https://huggingface.co/moonshotai/Kimi-K2-Instruct",
            ),
            ModelCatalogEntry(
                catalog_id="kimi-k2.6",
                display_name="Kimi-K2.6",
                repo_id="moonshotai/Kimi-K2.6",
                revision="7eb5002f6aadc958aed6a9177b7ed26bb94011bb",
                adapter_id="kimi_k2",
                status="text-private-supported",
                parameter_summary=(
                    "multimodal wrapper + DeepSeek-like MLA/MoE text backbone "
                    "+ INT4 experts"
                ),
                expected_bytes=595_148_192_736,
                capabilities={
                    "gqa": False,
                    "mla": True,
                    "moe": True,
                    "mtp": False,
                    "multimodal": True,
                    "int4": True,
                },
                source={"format": "safetensors", "dtype": "mixed_int4_bfloat16"},
                conversion={
                    "output_dtype": "bfloat16",
                    "expansion_h": 128,
                    "tile_mib": 64,
                    "minimum_host_ram_gib": 16,
                    "mode": "text-backbone",
                    "estimated_output_ratio": 4.2,
                },
                runtime={"preferred": "hf", "fallback": None},
                license="Modified MIT",
                family_id="kimi_k2",
                family_name="Kimi",
                conversion_ready=True,
                deployment_ready=False,
                support_note=(
                    "完整转换61层MLA/MoE文本骨干，并按官方compressed-tensors规则"
                    "解码384专家INT4权重；当前私有协议只开放文本问答。"
                ),
                max_stage="convert",
                visibility="family",
                source_url="https://huggingface.co/moonshotai/Kimi-K2.6",
            ),
        )
    )
