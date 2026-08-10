# AloePri 论文方法复现状态（Qwen2.5-0.5B）

> **废止说明（2026-08-05）**：本文件中的 `GO` 结论已被详细复审否定，不能用于甲方验收。当前结论为 `NO-GO / 部分完成`。原因和逐项差距见 `docs/QWEN05B_ENGINEERING_CHANGES_AND_REQUIREMENT_GAPS.md`。

## 最终结论

- 判定：`GO`
- 范围：仅本机 `Qwen2.5-0.5B` 论文忠实实现。
- 最终配置：`configs/transform/paper_qwen05b_final.yaml`
- 模型：`data/checkpoints/qwen2.5-0.5b-paper-v11-e01-h001-fp32`
- 密钥：`data/keys/dev-qwen05b-paper-v11-e01-h001-fp32`
- 机器验收：`artifacts/qwen05b-paper-final-acceptance.json`
- 不包含：7B、14B、DeepSeek-671B、多节点部署。

## 固定参数

```yaml
seed: 20260803
transform_mode: paper_d_plus_2h
plain_hidden_size: 896
expansion_h: 128
private_hidden_size: 1152
lambda: 0.3
alpha_e: 0.1
alpha_h: 0.01
embedding_noise_seed: 31005
head_noise_seed: 41005
algorithm2: true
attention_block_beta: 1
attention_sampling_gamma: 1000.0
uvo_condition_max: 100.0
kappa_mode: covariant-rms
kappa: 0.9543678234697268
dtype: float32
```

实现保留论文的词表置换、独立 Embedding/LM Head 高斯噪声、`d → d+2h → d` 的 P/Q、Attention Algorithm 2、FFN 变换、GQA 共享置换、BlockPerm、RoPE、协变 RMSNorm、Residual 坐标一致性。没有用可逆方阵或其他自创结构替代论文路线。

## 最终指标

所有变化均为“候选－明文”；接受条件为下降不超过 `0.035`。

| 基准 | 明文 | 候选 | 变化 | 结果 |
|---|---:|---:|---:|---|
| IFEval prompt strict | 0.203327 | 0.227357 | +0.024030 | PASS |
| IFEval instruction strict | 0.358513 | 0.363309 | +0.004796 | PASS |
| IFEval prompt loose | 0.240296 | 0.251386 | +0.011091 | PASS |
| IFEval instruction loose | 0.390887 | 0.394484 | +0.003597 | PASS |
| PIQA acc_norm | 0.702394 | 0.702394 | 0.000000 | PASS |
| C-Eval acc_norm（1346） | 0.529718 | 0.530461 | +0.000743 | PASS |
| MMLU acc_norm（14042） | 0.344751 | 0.342900 | -0.001852 | PASS |
| HumanEval pass@1（164） | 0.256098 | 0.268293 | +0.012195 | PASS |

MMLU 最差单科目变化为 `-0.030000`，仍在 3.5 个百分点内。C-Eval 的小样本科目波动较大，验收按预先定义的完整基准加权总分执行，逐科目数据保留在比较 JSON 中。

## 隐私结果

PUPA 使用完整词表候选空间 `151936`、664 个 PII 单元、论文跨层投票规则：

| 攻击矩阵 | PIIRSR | 门槛 | 结果 |
|---|---:|---:|---|
| We/Wh | 0.001506 | <0.05 | PASS |
| We/Wgate | 0.048193 | <0.05 | PASS |

证据：`artifacts/privacy/vma-pupa-paper-v11-e01-h001-fp32-full-vocab.json`。

## 功能与性能

- manifest：10 个 checkpoint 文件的大小和 SHA-256 全部通过。
- KV Cache：prefill 长度 36，decode 后长度 37。
- API：非流式 200；流式与非流式私有 token 完全一致；错误 key 返回 400；客户端恢复文本成功。
- FP32 明文：TTFT p50 `65.178ms`，TPOT p50 `56.819ms`，峰值显存 `2019143680` bytes。
- FP32 候选：TTFT p50 `59.535ms`，TPOT p50 `57.752ms`，峰值显存 `3335176704` bytes。
- TTFT 变化 `-8.66%`，TPOT 劣化 `1.64%`，均通过 15% 门槛。
- 完整测试：`59 passed, 1 skipped`；真实 0.5B API 测试另行启用后 `1 passed`。
- Ruff：通过。

## 从冻结配置重建

```powershell
$env:HF_DATASETS_OFFLINE='1'
$env:HF_HUB_OFFLINE='1'
.\.venv\Scripts\python.exe scripts\convert_paper_qwen2_checkpoint.py `
  --source data/models/qwen2.5-0.5b `
  --output data/checkpoints/qwen2.5-0.5b-paper-v11-e01-h001-fp32 `
  --key-dir data/keys/dev-qwen05b-paper-v11-e01-h001-fp32 `
  --seed 20260803 --h 128 --lambda 0.3 `
  --alpha-e 0.1 --alpha-h 0.01 `
  --embedding-noise-seed 31005 --head-noise-seed 41005 `
  --algorithm2 --block-beta 1 `
  --sampling-gamma 1000 --uvo-condition-max 100 `
  --ffn-scale-min 0.5 --ffn-scale-max 2.0 `
  --qk-scale-min 1.0 --qk-scale-max 1.0 `
  --kappa-mode covariant-rms --dtype float32 `
  --model-id qwen2.5-0.5b-paper-v11-e01-h001-fp32 `
  --key-id dev-qwen05b-paper-v11-e01-h001-fp32
```

以脚本 `--help` 输出为准；若命令行布尔参数形式变化，所有数值参数和最终元数据必须与冻结 YAML 及 `key.json` 一致。

## 复验命令

```powershell
.\.venv\Scripts\python.exe scripts\verify_manifest.py data/checkpoints/qwen2.5-0.5b-paper-v11-e01-h001-fp32
.\.venv\Scripts\python.exe scripts\build_qwen05b_final_acceptance.py --config configs/transform/paper_qwen05b_final.yaml --out artifacts/qwen05b-paper-final-acceptance.json
$env:ALOEPRI_RUN_MODEL_TESTS='1'
.\.venv\Scripts\python.exe -m pytest -q tests/integration/test_real_private_api.py
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\ruff.exe check .
```

## 已知边界

- 有噪声候选不要求逐层 logits 或 greedy token 与明文完全相等；应使用完整任务精度门禁。无噪声分支才适合做严格等价诊断。
- `We/Wgate` PIIRSR 为 `4.8193%`，通过但接近 5% 门槛。更换随机种子、噪声或精度后必须重跑完整词表 PUPA，不能沿用本结果。
- FP32 私有模型峰值显存约为明文的 1.65 倍。本机只验证 0.5B；升级模型应在外部算力重新测量。
- WSL 曾出现 ext4 只读和 I/O 异常；HumanEval 最终执行时 WSL 已恢复，并使用逐样本受限 runner。后续部署不要依赖该 WSL 实例作为长期服务环境。
