# Qwen2.5-0.5B 非平凡 BlockPerm 功能完成记录

更新时间：2026-08-09

## 1. 本轮实现对象

唯一模型：`Qwen2.5-0.5B-Instruct`。

本轮 checkpoint：

```text
data/packages/qwen05b-product-v31-blockperm8
```

配置：

```text
configs/product/qwen05b_v31_blockperm8.yaml
```

相对 v30，噪声和 P/Q 参数不变，唯一算法性变化是把 `block_beta` 从 1 改为论文
默认值 8，并同步修改运行时 RoPE：

$$
R_B(t)=B^{\mathsf T}R(t)B.
$$

该修正的原因、证明和不修改的论文步骤见
`docs/engineering_corrections/C01_SYNCHRONIZED_ROPE_BLOCKPERM.md`。

## 2. 从客户端到模型回答的实际路径

1. 客户端使用原始 Qwen tokenizer 和 Chat Template，把问题编码成明文 token ID $x$。
2. 客户端计算混淆输入 $\widetilde x=\tau(x)$。
3. HTTP 请求只发送 $\widetilde x$、`model_id`、`key_id` 和生成参数。
4. 服务端加载扩维后的混淆 checkpoint，隐藏维从 896 变为 1152。
5. Embedding 使用 $\Pi(W_e+\alpha_eE_e)P_e$。
6. 每层 Attention 使用变换后的 Q/K/V/O、GQA head 置换和非平凡 BlockPerm。
7. RoPE 在私有频率块坐标中执行 $B^{\mathsf T}R(t)B$；RoPE 后的 K 直接进入私有 KV Cache。
8. FFN 同步执行 gate/up/down 中间维置换和缩放。
9. RMSNorm 使用 $G=QQ^{\mathsf T}$ 恢复原坐标均方根；Residual 两支在同一私有坐标相加。
10. LM Head 生成混淆 token $\widetilde y$；服务端不做逆置换。
11. 客户端逐 token 计算 $y=\tau^{-1}(\widetilde y)$，再用原始 tokenizer 输出文本。

## 3. 代码修改

| 文件 | 实现内容 |
|---|---|
| `src/aloepri/models/modeling_aloepri_qwen2.py` | HF 同步 BlockPerm RoPE；prefill/decode 共用；持久化逐层顺序表 |
| `src/aloepri/transforms/qwen_structural.py` | 离线 Q/K BlockPerm；按 KV-head 新顺序生成运行时顺序表 |
| `src/aloepri/serving/vllm_qwen2.py` | vLLM 同步 RoPE；严格接收 `aloepri_block_orders` |
| `src/aloepri/serving/sglang_models/aloepri_qwen2.py` | SGLang 同步 RoPE；严格接收审计张量 |
| `scripts/verify_paper_formula_checkpoint.py` | checkpoint、config、离线 key 三方核对；每行置换合法性检查 |
| `src/aloepri/packaging.py` | 服务器包密钥扫描；tokenizer 词面与秘密元数据键分开判断 |
| `scripts/wsl/run_vllm_smoke.sh` | vLLM 0.26、75% 显存、batch 1 的固定运行命令 |
| `scripts/wsl/run_sglang_smoke.sh` | SGLang 0.5.17、CUDA 13 JIT 环境与固定运行命令 |

## 4. 功能验收结果

| 检查对象 | 检查方法 | 实测结果 | 判定 |
|---|---|---:|---|
| 词表置换 | `inverse_tau[tau]` 全词表核对 | 全词表互逆 | 通过 |
| Algorithm 1 | 从 $B,E,F,Z$ 重建 $P,Q_j$ | 最大约束误差低于 `1e-10` | 通过 |
| Algorithm 2 | 逐层重建 head、Q/K、V/O、FFN 变换 | 24/24 层通过 | 通过 |
| 全 checkpoint 公式覆盖 | 独立读取源权重、私有权重和 key | 316/316 张量；失败 0；漏检 0 | 通过 |
| BlockPerm | `beta=8`；核对 checkpoint/config/key | 24/24 层一致 | 通过 |
| HF prefill cache | 真实 GPU forward | 长度 36 | 通过 |
| HF decode cache | 使用 prefill cache 单 token decode | 长度 37 | 通过 |
| HF greedy | 私有输出逆置换后比较明文生成 | 16-token 序列相同 | 通过 |
| FastAPI | `/v1/private/generate` | 返回正常混淆 token | 通过 |
| SSE | `/v1/private/generate/stream` | token 顺序与非流式相同 | 通过 |
| 错误 key | 同一 API 使用错误 `key_id` | HTTP 400 | 通过 |
| vLLM | 与 HF 私有 greedy 序列比较 | 32/32 token 相同 | 通过 |
| SGLang | 与 vLLM 私有 greedy 序列比较 | 32/32 token 相同 | 通过 |
| 服务包扫描 | 扫描 JSON、safetensors 和 manifest | findings 为空 | 通过 |

有噪声模型不要求中间张量与明文模型逐元素相等。本 checkpoint 的首个逐层偏差在
Embedding，NRMSE 为 `0.0098750917`，与噪声加入位置一致。最终功能门禁检查的是私有
权重是否严格由既定公式生成、私有坐标是否连贯、cache 是否正确推进，以及输出 token
是否能由客户端恢复。

## 5. 本轮实际资源数据

| 项目 | 实测值 |
|---|---:|
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU，6GB |
| HF 明文阶段峰值 | 2,115,368,448 bytes |
| HF 私有阶段峰值 | 3,692,555,776 bytes |
| vLLM 权重显存 | 3.08GiB |
| vLLM peak activation | 0.63GiB |
| vLLM KV cache | 0.75GiB |
| SGLang 权重显存 | 3.11GB |
| SGLang 加载后可用显存 | 1.86GB |
| vLLM/SGLang 显存配置 | 75% |
| 后端上下文 / batch | 128 / 1 |

## 6. 与论文和甲方要求同表比较

本表只列本轮 v31-blockperm8 已实际运行的项目。完整精度、攻击和延迟统计没有沿用
v30，也没有用未测试值填表。

| 项目 | 论文方法或原始目标 | 甲方目标 | v31-blockperm8 实测 | 差距 |
|---|---|---|---|---|
| 扩维 | $d\rightarrow d+2h$ | 实现论文结构 | $896\rightarrow1152$，$h=128$ | 无 |
| P/Q | $PQ=I_d$ | 函数坐标可恢复 | FP64 相对误差 `2.2734e-14` | 无 |
| BlockPerm | 默认 $\beta=8$ | 实现 Attention 混淆 | 24 层 beta=8，三方核对通过 | 无 |
| RoPE | BlockPerm 后仍须正确推理 | 回答和 cache 正常 | 使用 $B^{\mathsf T}R(t)B$；HF/vLLM/SGLang 对齐 | 无 |
| 客户端/服务端 | 客户端置换，服务器混淆推理 | 服务端不接收明文 prompt | API/SSE 实测通过 | 无 |
| 精度 | 论文报告下游任务 | 单项下降不超过 3.5pp | 本 checkpoint 未跑完整任务集 | 待测 |
| 隐私攻击 | 论文 VMA/IA/TFMA/SDA 等 | 指标低于各门禁 | 本 checkpoint 未重跑完整攻击矩阵 | 待测 |
| 性能 | 明文/私有同分布比较 | TTFT、TPOT 劣化不超过 15% | 仅记录加载与显存，未做 ABBA 延迟实验 | 待测 |

## 7. 复现命令

```powershell
cd E:\AloePri
uv sync --frozen
uv run aloepri convert --config configs/product/qwen05b_v31_blockperm8.yaml
uv run aloepri verify --config configs/product/qwen05b_v31_blockperm8.yaml
uv run aloepri inspect-package --server-package data/packages/qwen05b-product-v31-blockperm8
```

真实 API 测试：

```powershell
$env:ALOEPRI_RUN_MODEL_TESTS='1'
$env:ALOEPRI_SOURCE_MODEL='data/models/qwen2.5-0.5b'
$env:ALOEPRI_PRIVATE_MODEL='data/packages/qwen05b-product-v31-blockperm8'
$env:ALOEPRI_KEY_DIR='data/keys/qwen05b-product-v31-blockperm8-online'
uv run pytest -q tests/integration/test_real_private_api.py
```

加速后端：

```powershell
wsl -e bash /mnt/e/AloePri/scripts/wsl/run_vllm_smoke.sh
wsl -e bash /mnt/e/AloePri/scripts/wsl/run_sglang_smoke.sh
```

全仓检查：

```powershell
uv run ruff check src scripts tests
uv run mypy src
uv run pytest -q
```

## 8. 证据文件

```text
artifacts/verification/qwen05b-product-v31-blockperm8/formula.json
artifacts/verification/qwen05b-product-v31-blockperm8/layerwise.json
artifacts/verification/qwen05b-product-v31-blockperm8/generation.json
artifacts/verification/qwen05b-product-v31-blockperm8/verification_summary.json
artifacts/verification/qwen05b-product-v31-blockperm8/private_api.json
artifacts/vllm-smoke-v31-blockperm8-32tokens.json
artifacts/hf-vllm-greedy-v31-blockperm8-32tokens.json
artifacts/sglang-smoke-v31-blockperm8-32tokens.json
artifacts/verification/qwen05b-product-v31-blockperm8/functional_evidence_index.json
```

证据索引重新计算命令：

```powershell
uv run python scripts/build_v31_functional_evidence_index.py
```
