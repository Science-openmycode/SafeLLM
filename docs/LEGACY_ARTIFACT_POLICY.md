# 历史配置与工件规则

## 正式目标

只有以下配置进入当前验收：

```text
configs/product/qwen05b_v47_best_single_candidate.yaml
configs/acceptance/qwen05b_v47.yaml
```

验收器逐项读取 `configs/acceptance/qwen05b_v47.yaml` 中的路径，不搜索目录，也不自动选择
“最新”JSON。因此同目录中存在 v15-v38 文件不会改变 v47 结论。

## 保留的历史内容

- `configs/transform/paper_qwen05b_complete.yaml`：论文默认参数历史复现。
- `configs/transform/paper_qwen05b_ablation_no_noise.yaml`：无噪声消融。
- `configs/product/qwen05b_v25_*.yaml` 至 `qwen05b_v38_*.yaml`：参数搜索轨迹。
- README 后半部分的旧结果表：用于追踪公式修正和参数选择过程。
- `run_gate_ia.py`、`run_tfma_curve.py`、`train_sda_smoke.py`：旧的单进程攻击实验。

这些文件不删除，因为论文勘误、参数调优和历史报告引用了它们；也不得把它们的结果复制或
重命名成 v47 工件。

## 当前正式攻击入口

| 历史入口 | 当前入口 | 变化 |
|---|---|---|
| `run_direct_attack.py` | `run_direct_weight_match_isolated.py` | 攻击与目标密钥评分分离 |
| `run_vma_pupa.py --key-dir` | `run_vma_pupa.py --candidate-observations` | 候选准备、攻击、评分分离 |
| `run_gate_ia.py` | `run_gate_ia_isolated.py` | 全词表候选，攻击不读取 key |
| `run_attn_ia.py` | `run_attn_ia_corrected.py` | 修正维度和 Qwen RoPE 坐标 |
| `train_ima_smoke.py` | `build_ima_training_pairs.py` + `train_ima_inverter.py` | 禁止目标 key 参与训练 |
| `run_isa_hidden_state.py` | `run_isa_attention_score.py` / `run_isa_hidden_state_isolated.py` | 目标观察和真值评分分离 |
| `run_tfma_curve.py` | `run_tfma_isolated.py` | 候选搜索不读取 `inverse_tau` |
| `train_sda_smoke.py` | `train_sda_transformer.py` + `run_sda_target_attack.py` | 训练、攻击、评分分离 |

## 提交前检查

```powershell
rg -n "qwen05b.*v(15|16|17|25|29|30|31|35|36|37|38)" configs/acceptance
uv run python scripts/build_qwen05b_v47_acceptance.py `
  --config configs/acceptance/qwen05b_v47.yaml `
  --out artifacts/acceptance/qwen05b-v47-current.json
```

第一条命令必须无输出。第二条命令在证据未齐时应返回退出码 2 和 `NO-GO`，不得读取旧工件
补齐缺项。
