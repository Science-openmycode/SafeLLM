from __future__ import annotations

from collections.abc import Iterable

from aloepri.catalog.models import ModelCatalogEntry


def _qwen25_instruct_entry(
    *,
    size: str,
    revision: str,
    expected_bytes: int,
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
        deployment_ready=True,
        support_note=(
            "已完成转换、部署和问答验收"
            if validated
            else "同族转换器可执行；部署前必须完成该参数规模的冒烟验收"
        ),
        max_stage="chat" if validated else "chat-after-smoke",
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
        deployment_ready=True,
        support_note=(
            "Qwen3 Q/K Norm 已由专用坐标变换同步处理；部署前会按当前服务器资源"
            "执行完整 checkpoint 检查和加载冒烟。"
        ),
        max_stage="chat-after-smoke",
        visibility="family",
        source_url=f"https://huggingface.co/Qwen/{repo_name}",
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
            ModelCatalogEntry(
                catalog_id="openseek-small-v1-sft",
                display_name="OpenSeek-Small-v1-SFT",
                repo_id="BAAI/OpenSeek-Small-v1-SFT",
                revision="1515c184e6fe4a91e6061be513a79d607e8787cb",
                adapter_id="deepseek_v3",
                status="validated-conversion",
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
                conversion={"output_dtype": "bfloat16", "expansion_h": 0, "tile_mib": 256},
                runtime={"preferred": "hf", "fallback": None},
                license="Apache-2.0",
                family_id="deepseek_v3",
                family_name="DeepSeek MLA / MoE",
                conversion_ready=True,
                deployment_ready=True,
                support_note="真实检查点已完成MLA/MoE转换；目标服务器加载仍执行强制冒烟",
                max_stage="chat-after-smoke",
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
                deployment_ready=True,
                support_note="DeepSeek-V2统一转换与HF部署路径已接入；目标服务器加载仍执行强制冒烟",
                max_stage="chat-after-smoke",
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
                runtime={"preferred": "hf", "fallback": "sglang"},
                license="DeepSeek Model License",
                family_id="deepseek_v3",
                family_name="DeepSeek MLA / MoE",
                conversion_ready=True,
                deployment_ready=True,
                support_note="MLA/MoE/FP8/MTP转换器和部署入口已接入；671B物理转换尚未执行",
                max_stage="chat-after-smoke",
                visibility="developer",
                source_url="https://huggingface.co/deepseek-ai/DeepSeek-V3",
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
                deployment_ready=True,
                support_note=(
                    "融合Gate/Up会先规范化为等价Dense GQA图，再进入私有Qwen运行时；"
                    "正式9B检查点仍需目标机器冒烟"
                ),
                max_stage="chat-after-smoke",
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
                runtime={"preferred": "hf", "fallback": "sglang"},
                license="MIT",
                family_id="glm4_moe",
                family_name="GLM",
                conversion_ready=True,
                deployment_ready=True,
                support_note=(
                    "已实现GLM MoE专用FP8解码、Q/K Norm、部分RoPE、"
                    "路由器、逐专家与MTP权重转换；HF主模型问答不启用MTP加速"
                ),
                max_stage="chat-after-smoke",
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
                conversion={"output_dtype": "bfloat16", "expansion_h": 128, "tile_mib": 256},
                runtime={"preferred": "hf", "fallback": "vllm"},
                license="Apache-2.0",
                family_id="qwen3",
                family_name="Qwen3",
                conversion_ready=True,
                deployment_ready=True,
                support_note=(
                    "Qwen3 Q/K Norm 已由专用坐标变换同步处理；"
                    "正式 8B 部署前仍需在目标 GPU 完成检查点冒烟测试"
                ),
                max_stage="chat-after-smoke",
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
                runtime={"preferred": "hf", "fallback": "sglang"},
                license="Modified MIT",
                family_id="kimi_k2",
                family_name="Kimi",
                conversion_ready=True,
                deployment_ready=True,
                support_note=(
                    "纯文本Kimi-K2通过零拷贝结构规范化复用已验证的"
                    "DeepSeek-V3 MLA/MoE/FP8私有转换器"
                ),
                max_stage="chat-after-smoke",
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
                deployment_ready=True,
                support_note=(
                    "完整转换61层MLA/MoE文本骨干，并按官方compressed-tensors规则"
                    "解码384专家INT4权重；当前私有协议只开放文本问答。"
                ),
                max_stage="chat-after-smoke",
                visibility="family",
                source_url="https://huggingface.co/moonshotai/Kimi-K2.6",
            ),
        )
    )
