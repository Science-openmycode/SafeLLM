# Qwen2.5-0.5B v29 正确性优先功能工件

## 1. 工件定义

v29 使用论文的词表置换、$d\to d+2h$ 的 P/Q 坐标变换、Attention 与 FFN 结构变换、
精确度量 RMSNorm 修正式和客户端逆置换。参数如下：

| 参数 | 值 |
|---|---:|
| 原模型 | `Qwen2.5-0.5B-Instruct` |
| $d$ / $h$ / 私有维度 | 896 / 128 / 1152 |
| $\lambda$ | 0.3 |
| key seed | 20260803 |
| Algorithm 2 | 开启 |
| $\beta$ | 1 |
| Q/K scale | 0.5–2.0 |
| FFN scale | 0.5–2.0 |
| $\operatorname{cond}(U_{vo})$ | 不超过 42 |
| Embedding/Head 噪声 | 0 / 0 |
| Attention Q/K/V/O | 权重 FP64、计算 FP64 |
| 其余权重 | FP32 |

Attention 使用 FP64 是同一组论文矩阵公式的数值实现，不增加新的可训练层，不加载明文
模型，也不向服务器提供 P/Q、$\tau$ 或随机 seed。服务端只加载转换后的混合精度权重。

## 2. 代码改动

| 文件 | 功能 |
|---|---|
| `src/aloepri/models/configuration_aloepri_qwen2.py` | checkpoint 记录 Attention 计算精度 |
| `src/aloepri/models/modeling_aloepri_qwen2.py` | FP64 Q/K/V/O 与 FP64 score 两种可验证运行路径 |
| `src/aloepri/conversion/paper_qwen2.py` | 按目标权重 dtype 执行 Attention P/Q 离线变换 |
| `src/aloepri/transforms/qwen_structural.py` | Q/K/V 结构矩阵以 FP64 写入离线 key |
| `src/aloepri/serving/hf_runtime.py` | `dtype=auto` 保留 checkpoint 混合精度 |
| `src/aloepri/cli.py` | convert/verify 传递 Attention 计算精度 |
| `scripts/verify_layerwise_equivalence.py` | 记录 FP32、FP64 score、完整 FP64 的逐层证据 |
| `scripts/search_uvo_key_seeds.py` | 在预定 seed 集合内筛选高斯 $U_{vo}$ 条件数 |
| `scripts/upgrade_structural_key_precision.py` | 确定性重建 FP64 结构 key 并更新 manifest |

## 3. 功能结果

| 检查 | v25 FP32 | v29 | 判定 |
|---|---:|---:|---|
| 460 个内部边界通过数 | 457 | 457 | 未达到 460/460 |
| 最终 logits NRMSE | $6.155\times10^{-6}$ | $4.756\times10^{-6}$ | v29 更低，均通过 $10^{-5}$ |
| Prefill mean absolute error | $1.521\times10^{-5}$ | $1.168\times10^{-5}$ | v29 更低 |
| Decode mean absolute error | $9.074\times10^{-6}$ | $5.439\times10^{-6}$ | v29 更低 |
| Prefill Top-1 | 100% | 100% | 通过 |
| 单样本 greedy | 一致 | 一致 | 通过 |
| 固定 200 prompt | 200/200 | 200/200 | 通过 |
| 新生成 token | 3200/3200 | 3200/3200 | 通过 |
| KV Cache 长度 | 正确 | 36→37 | 通过 |
| 私有模型 CUDA peak | 3,451,209,216 B | 3,692,543,488 B | 约占 6 GiB 的 57% |
| Server package 扫描 | 通过 | 通过 | `findings=[]` |
| HTTP 隐私边界 | 通过 | 全部检查 true | 通过 |
| HTTP/SSE token 顺序 | 通过 | 完全一致 | 通过 |
| 实际恢复回答 | 正常中文 | “矩阵乘法是一种将一个矩阵与另一个矩阵相乘的运算，其” | 通过 |

v29 仍超过 $10^{-5}$ 的 3 个内部边界为：

| 算子 | NRMSE |
|---|---:|
| layer 0 Attention softmax | $1.0881\times10^{-5}$ |
| layer 22 Attention value aggregate | $1.5809\times10^{-5}$ |
| layer 22 O projection | $1.2784\times10^{-5}$ |

Attention residual、FFN residual、final RMSNorm、KV Cache、最终 logits 和生成 token 均在门禁内。

## 4. 工件路径

| 工件 | 路径 |
|---|---|
| checkpoint | `data/checkpoints/qwen2.5-0.5b-product-v29-attnfp64-uvo42-no-noise` |
| full key | `data/keys/dev-qwen05b-product-v29-attnfp64-uvo42-no-noise` |
| online key | `data/keys/qwen05b-product-v29-online` |
| offline key | `data/keys/qwen05b-product-v29-offline` |
| server package | `data/packages/qwen05b-functional-v29` |
| 产品配置 | `configs/product/qwen05b_v29_functional.yaml` |
| 逐层证据 | `artifacts/verification/qwen05b-product-v29-layerwise-fp64-key.json` |
| 200 prompt 证据 | `artifacts/verification/qwen05b-product-v29-greedy200.json` |
| prefill/decode 证据 | `artifacts/verification/qwen05b-product-v29-checkpoint.json` |
| 边界证据 | `artifacts/verification/qwen05b-functional-v29/privacy-boundary.json` |
| HTTP/SSE 证据 | `artifacts/verification/qwen05b-functional-v29/http-sse-smoke.json` |

## 5. 启动与人工检查

终端 A：

```powershell
cd E:\AloePri
$OutputEncoding = [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$env:PYTHONUTF8 = '1'
uv run aloepri inspect-package --server-package data/packages/qwen05b-functional-v29
uv run aloepri serve --config configs/product/qwen05b_v29_functional.yaml
```

终端 B：

```powershell
cd E:\AloePri
$OutputEncoding = [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$env:PYTHONUTF8 = '1'
uv run aloepri chat `
  --server http://127.0.0.1:8000 `
  --key-dir data/keys/qwen05b-product-v29-online `
  --tokenizer data/models/qwen2.5-0.5b `
  --max-new-tokens 128
```

人工检查顺序：输入中文问题；确认客户端显示正常回答；执行 `/stats` 查看 TTFT/TPOT；
检查服务端终端只出现请求状态，不出现明文问题；执行 `/clear` 后确认本地历史被清空。

## 6. 发布边界

v29 的 `alpha_e=0`、`alpha_h=0`，用于证明论文变换在 Qwen2.5-0.5B 上的功能闭环。
它通过 token 置换、拆钥、server package 扫描和网络边界检查；没有通过“非零噪声同时满足
精度与攻击阈值”的安全发布条件。非零噪声 checkpoint 必须单独转换、测试和出具攻击证据，
不得复用 v29 的 200/200 结果。
