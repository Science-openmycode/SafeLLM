# Qwen2.5-0.5B 噪声调参与产品状态

更新时间：2026-08-09

## 1. 论文噪声与当前实现

论文分别对 Embedding 权重和 LM Head 权重采样独立高斯噪声：

$$
E_{\mathrm{embed}} \sim \mathcal{N}(0, \sigma_e^2 I),
\qquad
E_{\mathrm{head}} \sim \mathcal{N}(0, \sigma_h^2 I),
$$

其中 $\sigma_e=\operatorname{Std}(W_e)$、$\sigma_h=\operatorname{Std}(W_h)$。加噪权重为：

$$
W'_{\mathrm{embed}}=W_e+\alpha_eE_{\mathrm{embed}},
\qquad
W'_{\mathrm{head}}=W_h+\alpha_hE_{\mathrm{head}}.
$$

随后执行词表置换和扩维矩阵变换：

$$
\widetilde W_{\mathrm{embed}}=\Pi W'_{\mathrm{embed}}\widehat P_{\mathrm{embed}},
\qquad
\widetilde W_{\mathrm{head}}=\widehat Q_{\mathrm{head}}W'_{\mathrm{head}}\Pi^T.
$$

代码位置：

- `src/aloepri/transforms/paper_noise.py`：按源权重总体标准差采样高斯矩阵并乘以 $\alpha$；
- `src/aloepri/conversion/paper_qwen2.py`：Embedding 和 Head 使用独立 seed，加噪后再执行 $\Pi/P/Q$ 变换；
- `scripts/verify_paper_formula_checkpoint.py`：从原始权重和密钥重新计算私有 checkpoint，逐张量比较。

本轮没有改变噪声分布、加噪位置或计算顺序，只调整 $\alpha_e$ 和 $\alpha_h$。

## 2. 两个可复核档位

| 档位 | Checkpoint | $\alpha_e$ | $\alpha_h$ | 用途 | 当前结论 |
|---|---|---:|---:|---|---|
| Utility 产品档 | `qwen2.5-0.5b-product-v30-noise001-0002` | 0.01 | 0.002 | 实际问答、API、客户端置换 | 功能通过；校准精度保持；VMA 不通过 |
| Paper-default 研究档 | `qwen2.5-0.5b-product-v34-paper-noise10-02` | 1.0 | 0.2 | 论文推荐噪声强度、攻击实验 | VMA 通过；MMLU 归一化精度不通过 |

Utility 产品档的服务端不持有 `tau`、`inverse_tau`、P/Q 或随机种子。它满足“服务端请求中不出现明文 prompt 和明文 token ID”的产品边界。它不满足“恶意服务器通过权重攻击也无法恢复置换”的强攻击门禁。

## 3. 参数搜索结果

### 3.1 VMA/PUPA，16,384 候选，单 key

| 候选 | $\alpha_e$ | $\alpha_h$ | We–Wh TTRSR | We–Wh PIIRSR | We–Wgate TTRSR | We–Wgate PIIRSR | 15%/3%门禁 |
|---|---:|---:|---:|---:|---:|---:|---|
| v30 | 0.01 | 0.002 | 100.00% | 100.00% | 100.00% | 100.00% | 不通过 |
| 扫描点 | 0.60 | 0.20 | 26.58% | 2.26% | 10.06% | 0.60% | 不通过 |
| 扫描点 | 0.80 | 0.20 | 16.72% | 0.15% | 2.59% | 0.30% | 不通过 |
| v32 | 0.80 | 0.40 | 14.09% | 0.15% | 2.59% | 0.30% | 通过 |
| v33 | 0.90 | 0.30 | 13.75% | 0.15% | 1.24% | 0.30% | 通过 |
| v34 | 1.00 | 0.20 | 13.43% | 0.15% | 1.05% | 0.15% | 通过 |
| v31 | 1.00 | 0.50 | 7.85% | 0.15% | 1.05% | 0.15% | 通过 |

证据文件：

- `artifacts/privacy/qwen05b-product-grid-screen-seed20260803.json`
- `artifacts/privacy/qwen05b-v31-noise-breakpoint-screen.json`
- `artifacts/privacy/qwen05b-v32-noise-fine-screen.json`
- `artifacts/privacy/qwen05b-v34-paper-default-noise-screen.json`

以上置信区间是 token occurrence/PII unit 的 Wilson 区间；当前表只有一个变换 key，不代表 key 间不确定性。

### 3.2 固定校准题精度

表中分数为答对数/样本数。括号内为相对原模型的绝对百分点变化。

| 候选 | MMLU acc | MMLU acc_norm | C-Eval acc | PIQA acc | PIQA acc_norm |
|---|---:|---:|---:|---:|---:|
| 原模型 | 20/60 | 23/60 | 20/58 | 72/100 | 70/100 |
| v30 0.01/0.002 | 20/60 (0.00) | 23/60 (0.00) | 20/58 (0.00) | 72/100 (0.00) | 70/100 (0.00) |
| v31 1.0/0.5 | 18/60 (-3.33) | 18/60 (-8.33) | 15/58 (-8.62) | 63/100 (-9.00) | 64/100 (-6.00) |
| v32 0.8/0.4 | 19/60 (-1.67) | 19/60 (-6.67) | 17/58 (-5.17) | 60/100 (-12.00) | 62/100 (-8.00) |
| v33 0.9/0.3 | 19/60 (-1.67) | 16/60 (-11.67) | 未测试 | 未测试 | 未测试 |
| v34 1.0/0.2 | 18/60 (-3.33) | 18/60 (-8.33) | 未测试 | 未测试 | 未测试 |

v34 在 MMLU 归一化指标上已超过 3.5 个百分点停止线，因此没有继续运行 C-Eval 和 PIQA。表中没有使用未测试值补齐结论。

### 3.3 生成回归

| 候选 | 20条逐序列完全相同 | 生成 token 一致率 |
|---|---:|---:|
| v30 0.01/0.002 | 19/20 | 99.82% |
| v31 1.0/0.5 | 0/20 | 75.47% |
| v32 0.8/0.4 | 0/20 | 75.78% |
| v33 0.9/0.3 | 0/20 | 76.09% |
| v34 1.0/0.2 | 0/20 | 76.41% |

生成一致率只描述与原模型的 token 重合，不等同于任务准确率。

## 4. v30 产品闭环结果

### 4.1 服务包与密钥

| 工件 | 路径 | 内容 |
|---|---|---|
| 服务端模型 | `data/packages/qwen05b-product-v30-utility` | 私有权重、配置、manifest；包扫描通过 |
| 客户端在线密钥 | `data/keys/qwen05b-product-v30-online` | `tau`、`inverse_tau`、特殊 token 映射 |
| 离线主密钥 | `data/keys/qwen05b-product-v30-offline` | P/Q、Attention/FFN 变换与重建材料 |
| 产品配置 | `configs/product/qwen05b_v30_utility.yaml` | 转换参数、服务限制和路径 |

服务包扫描结果：`pass=true`，未发现在线密钥、P/Q、随机种子或原始权重。

### 4.2 功能验证

| 检查 | 结果 |
|---|---|
| 完整公式重建 | 292/292 张量通过 |
| Algorithm 1 | 通过 |
| Algorithm 2 | 通过 |
| 私有 next token 逆置换 | 通过 |
| decode 输入等于 `tau(plain_next)` | 通过 |
| KV Cache：36 → 37 | 通过 |
| 固定问答 greedy IDs | 通过 |
| 有噪声逐层差异 | 记录为诊断；首个差异为 Embedding，NRMSE 0.009875 |

有噪声模型不要求逐层输出与明文模型相等。必须相等的是公式生成的私有权重、私有坐标流、token 逆置换和 cache 状态；回答质量由任务精度评测决定。

### 4.3 真实问答

输入：`请用一句话解释什么是光合作用。`

输出：`光合作用是植物、藻类和某些细菌利用阳光、二氧化碳和水进行的化学反应，将二氧化碳和水转化为葡萄糖和氧气的过程。`

| 指标 | 实测 |
|---|---:|
| 输入 token | 38 |
| 输出 token | 34 |
| TTFT | 660.31 ms |
| TPOT | 66.81 ms |
| 明文/私有输入 ID 是否相同 | 否 |

机器证据：`artifacts/product/qwen05b-v30-private-chat-smoke.json`。

## 5. 复现命令

```powershell
uv sync --frozen
uv run aloepri inspect-package --server-package data/packages/qwen05b-product-v30-utility
uv run aloepri verify --config configs/product/qwen05b_v30_utility.yaml

$env:ALOEPRI_BEARER_TOKEN='replace-with-local-token'
uv run aloepri serve --config configs/product/qwen05b_v30_utility.yaml
```

另开终端：

```powershell
$env:PYTHONUTF8='1'
uv run python scripts/run_product_chat_smoke.py `
  --server http://127.0.0.1:8000 `
  --tokenizer data/models/qwen2.5-0.5b `
  --online-key data/keys/qwen05b-product-v30-online `
  --prompt '请用一句话解释什么是光合作用。' `
  --bearer-token replace-with-local-token `
  --max-new-tokens 48 `
  --out artifacts/product/qwen05b-v30-private-chat-smoke.json
```

交互式客户端：

```powershell
uv run aloepri chat `
  --server http://127.0.0.1:8000 `
  --key-dir data/keys/qwen05b-product-v30-online `
  --tokenizer data/models/qwen2.5-0.5b
```

## 6. 当前结论

Qwen2.5-0.5B 已实现论文稠密架构范围内的转换、独立 Embedding/Head 高斯噪声、词表置换、扩维 P/Q、Attention、FFN、RMSNorm 修正式、Residual、KV Cache、checkpoint、服务端 API、客户端恢复和真实问答。

当前可运行产品采用 v30：功能闭环和校准精度通过，服务端直接请求中没有明文 prompt。当前不能宣称 v30 抵抗权重级 VMA。达到 VMA 15% 门禁的 v32/v34 在0.5B客观题上超过精度停止线；该矛盾来自同一全权重高斯噪声在0.5B上的实测效用—隐私折中。
