# C05：跨后端比较必须锁定完整生成参数

## 现象

v30 首次比较 HF 与 vLLM 时，前三个私有 token 相同，第 4 个 token 不同。直接读取 HF
原始 logits 后发现，vLLM 选择的是该位置的最高 logit，因此差异不是 Attention、RoPE
或 KV Cache 错误。

checkpoint 的 `generation_config.json` 含：

```json
{"repetition_penalty": 1.1}
```

HF `generate()` 自动读取该值；vLLM `SamplingParams` 默认使用 `1.0`。两条路径实际比较了
不同的生成策略。

## 修改

数学和后端一致性测试统一显式设置：

```text
temperature = 0.0
repetition_penalty = 1.0
max_new_tokens = 32
```

对应文件：

- `scripts/vllm_smoke.py`
- `scripts/compare_hf_vllm_smoke.py`
- `scripts/sglang_smoke.py`

其中 HF 比较器不再依赖 checkpoint 的隐式 generation defaults。SGLang 同样显式传入
`repetition_penalty=1.0`。这项修改只统一解码器策略，不改变模型权重、论文变换或 logits。

## 结果

固定参数后：

- v30：HF 与 vLLM 32/32 私有 token 相同；
- v31 beta=8：HF、vLLM、SGLang 32/32 私有 token 相同；
- HF 比较路径启用 KV Cache。

证据：

- `artifacts/hf-vllm-greedy-v30-32tokens.json`
- `artifacts/hf-vllm-greedy-v31-blockperm8-32tokens.json`
- `artifacts/sglang-smoke-v31-blockperm8-32tokens.json`

以后若开放 `top_k`、`top_p`、随机采样或重复惩罚，协议、HF、vLLM 和 SGLang 必须显式
传递同一组值和 seed；不得用各框架默认值做一致性结论。
