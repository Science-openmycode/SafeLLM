# Qwen2.5-0.5B exact-metric RMS 长序列数值修复

## 1. 故障

v47 候选 checkpoint 在 IFEval 长序列解码中出现全量 logits 为 NaN。两个独立失败样本分别在第 641 和第 609 个生成步首次出现：

```text
step=641 nan=151936 finite_bounds=None
step=609 nan=151936 finite_bounds=None
```

NaN 之后，CUDA `argmax` 产生非词表整数，Transformers 的 repetition-penalty `gather/scatter` 才报告索引越界。因此索引越界是后果，不是根因。

## 2. 原公式与故障实现

设明文残差为行向量 `x`，私有残差为：

```text
z = x P
```

且矩形左右逆满足：

```text
P Q = I
```

明文 RMS 方差应为：

```text
variance = ||z Q||^2 / d
         = z (Q Q^T) z^T / d
```

旧实现把：

```text
G = Q Q^T
```

以 FP32 写入 checkpoint，并直接计算 `z G z^T`。因为 `Q` 的形状为 `1152 × 896`，所以 `G` 必然秩亏，至少有 256 维零空间。FP32 落盘和二次型求和会在零空间附近发生严重消减，使理论上非负的方差成为负值，随后 `rsqrt(variance + eps)` 产生 NaN。

当前 v47 实测：

| 项目 | 结果 |
|---|---:|
| FP32 `G` 最小特征值 | `-2.1550705699364847e-08` |
| FP32 `G` 负特征值数量 | `123` |
| 由 FP64 `Q` 计算的最小特征值 | `-1.4255221391468246e-15` |
| FP64 结果中小于 `-1e-14` 的特征值数量 | `0` |

`1e-15` 量级属于双精度特征值算法的机器误差；`2.16e-8` 的负值足以在隐藏状态幅值增大后破坏 RMS 方差。

## 3. 修复

对精确 Gram 矩阵做特征分解：

```text
G = U diag(lambda) U^T
```

取 896 个非负特征值和对应特征向量，构造：

```text
F = U_positive diag(sqrt(lambda_positive))
```

于是：

```text
F F^T = G = Q Q^T
```

运行时改为：

```text
variance = ||z F||^2 / d
```

这是同一二次型的平方和计算，不会通过大正负交叉项相减得到方差。服务器只保存 `F`；`F` 与旧 `G` 数学等价，因为双方都能互相恢复同一个 Gram 矩阵，不额外披露离线 Q 的具体右逆基。

## 4. 工件与验证

新 checkpoint：

```text
data/packages/qwen05b-candidate-v47-stable-factor
```

新增张量：

```text
aloepri_rms_factor: float64[1152, 896]
```

构建命令：

```powershell
uv run python scripts/upgrade_exact_rms_stable_factor.py `
  --source data/packages/qwen05b-candidate-v47-best-single `
  --key data/keys/dev-qwen05b-candidate-v47-best-single/paper_key.safetensors `
  --output data/packages/qwen05b-candidate-v47-stable-factor
```

构建结果：

| 检查 | 结果 |
|---|---:|
| `F` 形状 | `1152 × 896` |
| 旧 FP32 Gram 与 FP64 `Q Q^T` 最大误差 | `4.5807683646259534e-08` |
| 构建时 `F F^T` 与 `Q Q^T` 最大误差 | `2.6645352591003757e-15` |
| 公式验证张量数 | `317` |
| Algorithm 1 | 通过 |
| Algorithm 2 | 通过 |
| 缺失公式覆盖 | `0` |
| 原第 609 步失败长样本 | 完整生成并保存 |

完整验证命令：

```powershell
uv run aloepri verify `
  --config configs/product/qwen05b_v47_best_single_candidate.yaml

uv run python scripts/run_hf_ifeval.py `
  --model data/packages/qwen05b-candidate-v47-stable-factor `
  --tokenizer data/models/qwen2.5-0.5b `
  --key data/keys/dev-qwen05b-candidate-v47-best-single/paper_key.safetensors `
  --dataset-json data/eval/ifeval_inputs_v47.json `
  --out artifacts/debug/v47-stable-factor-local-shard0.json `
  --num-shards 5 --shard-index 0 --max-samples-per-run 1 `
  --max-new-tokens 1280 --batch-size 1 --dtype float32 `
  --attn-implementation sdpa --deterministic `
  --gpu-memory-fraction 0.65 --minimum-free-gpu-gib 0.2
```

## 5. 验收影响

该修复改变 checkpoint、config 和运行时 RMS 计算，因此以下候选模型结果必须重新计算并绑定新 manifest：MMLU、C-Eval、PIQA、IFEval、HumanEval、产品问答、攻击和性能。明文基线不加载 AloePri exact-metric RMS，可保留，但必须由验收器重新核对其数据集、脚本、运行时和模型哈希。

该问题属于有限精度工程实现缺陷，不是论文公式错误。论文的目标二次型保持不变。
