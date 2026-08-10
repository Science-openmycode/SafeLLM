from __future__ import annotations

import hashlib
import json
from pathlib import Path

CHINESE_CONCEPTS = [
    "矩阵乘法",
    "二分查找",
    "哈希表",
    "注意力机制",
    "KV Cache",
    "RMSNorm",
    "浮点舍入",
    "事务隔离",
    "幂等接口",
    "缓存淘汰",
    "图的广度优先搜索",
    "动态规划",
    "张量并行",
    "流水线并行",
    "数字签名",
    "SHA-256",
    "容器化部署",
    "GPU显存",
    "单元测试",
    "流式文本生成",
]

ENGLISH_CONCEPTS = [
    "matrix multiplication",
    "binary search",
    "hash tables",
    "attention mechanisms",
    "KV cache",
    "RMS normalization",
    "floating-point rounding",
    "transaction isolation",
    "idempotent APIs",
    "cache eviction",
    "breadth-first graph search",
    "dynamic programming",
    "tensor parallelism",
    "pipeline parallelism",
    "digital signatures",
    "SHA-256",
    "containerized deployment",
    "GPU memory",
    "unit testing",
    "streaming text generation",
]


def main() -> None:
    output = Path("configs/eval/gate1_prompts_200.json")
    manifest_path = Path("configs/eval/gate1_prompts_200.manifest.json")
    chinese_templates = [
        "用两句话解释{concept}。",
        "给出一个关于{concept}的具体例子。",
        "列出{concept}的三个关键检查点。",
        "说明{concept}最常见的一个错误及修复方法。",
        "写出验证{concept}是否正确的最小测试步骤。",
    ]
    english_templates = [
        "Explain {concept} in two sentences.",
        "Give one concrete example of {concept}.",
        "List three key checks for {concept}.",
        "Describe one common failure in {concept} and how to fix it.",
        "Write the minimum test steps needed to verify {concept}.",
    ]
    prompts = [
        template.format(concept=concept)
        for concept in CHINESE_CONCEPTS
        for template in chinese_templates
    ] + [
        template.format(concept=concept)
        for concept in ENGLISH_CONCEPTS
        for template in english_templates
    ]
    if len(prompts) != 200 or len(set(prompts)) != 200:
        raise AssertionError("Gate 1 prompt corpus must contain 200 unique prompts")
    rendered = json.dumps(prompts, ensure_ascii=False, indent=2) + "\n"
    output.write_text(rendered, encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "purpose": "Gate 1 fixed bilingual greedy-equivalence corpus",
        "prompt_count": len(prompts),
        "languages": {"zh": 100, "en": 100},
        "data_file": str(output),
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "generation_script": str(Path(__file__)),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
