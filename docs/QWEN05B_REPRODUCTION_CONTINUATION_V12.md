# Qwen2.5-0.5B AloePri 继续复现记录（v12）

## 固定范围

- 唯一模型：`Qwen2.5-0.5B-Instruct`。
- 本机不再执行 7B、14B、DeepSeek 或多节点实验。
- 只实现论文已有机制；论文缺项必须单独标注，不伪装成作者实现。

## 本轮代码修正

| 项目 | 修改 | 验证 |
|---|---|---|
| Algorithm 1 | 保持形状一致解释：`rows(C) in null(F^T)`、`columns(D) in null(E)` | 100 个 seed 的 `P@Q=I` 单测 |
| Algorithm 2 `BlockPerm` | 补上论文伪代码遗漏的游标推进 | block order 为双射且终止 |
| Algorithm 2 `gamma` | 使用 `softmax(gamma * frequency_delta)`；否则论文给出的 `gamma=1e3` 完全无效 | gamma 差异单测 |
| 服务端密钥隔离 | 从模型 config 和 manifest 删除主 seed、Embedding noise seed、Head noise seed | manifest 递归秘密字段检查 |
| 测试入口 | 将 `scripts` 设为可导入包，并固定 pytest 根路径 | `62 passed, 1 skipped`（gamma 修正前全量基线） |

`gamma` 的乘法位置属于对论文缺失项的最小解释，不宣称是作者未公开代码的原始写法。

## v12 候选配置

配置：`configs/transform/paper_qwen05b_v12_secure.yaml`

| 参数 | 值 |
|---|---:|
| `h` | 128 |
| `lambda` | 0.3 |
| `alpha_e` | 0.1 |
| `alpha_h` | 0.01 |
| `beta` | 8 |
| `gamma` | 1000 |
| Q/K scale | log-uniform `[0.5, 2.0]` |
| FFN scale | log-uniform `[0.5, 2.0]` |
| `U_vo` | Gaussian `N(0,1/d_head)`，condition ≤ 100 |
| checkpoint dtype | FP32 |

`alpha_e/alpha_h` 是在论文允许调节的噪声机制内针对 0.5B 的定标，不等于论文 14B/671B 默认值 `1.0/0.2`。

## v12 实测结果

| 检查 | 结果 | 判断 |
|---|---:|---|
| manifest SHA-256 | 10/10 文件通过 | PASS |
| 服务端 seed/tau 元数据 | 0 个 | PASS |
| prefill cache length | plain=36, private=36 | PASS |
| decode cache length | 37 | PASS |
| prefill top-1 agreement | 83.33% | 未完全等价 |
| 20 prompts generation exact | 55.00% | 未完全等价 |
| 20 prompts token agreement | 95.23% | 功能可用，未达严格等价 |
| PUPA VMA `We/Wh`, 4096 candidates, TTRSR | 56.56% | FAIL |
| PUPA VMA `We/Wgate`, 4096 candidates, TTRSR | 52.83% | FAIL |
| PUPA VMA `We/Wh`, PIIRSR | 13.55% | FAIL |
| PUPA VMA `We/Wgate`, PIIRSR | 13.10% | FAIL |

证据：

- `artifacts/verification/paper-qwen05b-v12-secure-beta8-fp32.json`
- `artifacts/verification/prompt-regression-paper-v12-secure-beta8-fp32.json`
- `artifacts/privacy/vma-pupa-paper-v12-secure-beta8-fp32-c4096.json`

## 当前结论

v12 修复了服务端直接泄露随机种子的工程错误，并恢复了论文 `beta=8/gamma=1000` 的动态 block permutation。0.5B 的生成质量明显优于论文默认噪声配置，但 VMA 隐私门禁失败。

不能通过选择单个有利 seed 宣称复现成功。下一轮必须至少对 5 个独立 key seed 报告均值、标准差和最坏值，并采用同一候选集合和攻击参数。只有最坏 seed 同时满足 TTRSR ≤ 15%、PIIRSR ≤ 3%，才能通过隐私门禁。

## 下一轮执行顺序

1. 固定 `h=128, lambda=0.3, beta=8, gamma=1000`。
2. 对噪声网格 `(alpha_e,alpha_h)` 运行小样本精度与 PUPA VMA，不更换机制。
3. 对 Pareto 候选运行 5 个独立 key seed；禁止按单 seed 挑结果。
4. 仅对跨 seed 通过的候选运行 PIQA、C-Eval、MMLU、IFEval 和 HumanEval 全量测试。
5. 若不存在同时满足精度与隐私门禁的候选，正式记录“论文参数在 Qwen2.5-0.5B 上不可同时复现论文指标”，不引入论文外算法。

## 后续更正：VMA 必须包含已融合的 RMSNorm

旧 `run_vma_pupa.py` 使用 `We @ Wh` 和 `We @ Wgate` 作为已知明文矩阵，但转换器已经把相应 RMSNorm 权重融合进 Head/Gate。攻击者知道原始模型，可以同样计算融合后的明文权重。因此旧攻击比较了不同函数，系统性低估恢复率。

已更正为：

```text
We @ (Wh * final_norm)
We @ (Wgate_l * post_attention_norm_l)
```

本节之前记录的旧 VMA 数字只保留为历史，不再作为验收证据。

### 三 seed 正确 VMA 筛选

候选规模为 2048，覆盖全部 1397 个 PUPA PII token；每组使用相同 decoy 集合，分别生成独立 `tau`、P/Q 和噪声。

| alpha_e/alpha_h | 最坏 We/Wh TTRSR | 最坏 We/Wh PIIRSR | 最坏 We/Wgate TTRSR | 最坏 We/Wgate PIIRSR |
|---|---:|---:|---:|---:|
| 0.1/0.01 | 100.00% | 100.00% | 100.00% | 100.00% |
| 0.4/0.1 | 76.02% | 36.14% | 52.62% | 13.55% |
| 0.5/0.1 | 61.63% | 19.58% | 29.66% | 5.72% |
| 0.8/0.5 | 25.96% | 2.41% | 9.35% | 0.30% |
| 1.0/0.2 | 23.31% | 3.01% | 6.24% | 0.45% |

证据：`artifacts/privacy/qwen05b-multiseed-vma-grid.json`。

结论：Gate 路径可由较强 Embedding 噪声压到门禁以内，剩余瓶颈是 We/Wh。论文默认噪声仍未达到 TTRSR ≤ 15%。

### FP32 排除实验

生成论文默认噪声的 FP32 v13 checkpoint，以排除历史 BF16 数值误差：

| 指标 | v13 FP32 |
|---|---:|
| prefill top-1 agreement | 19.71% |
| 20 prompts generation exact | 0% |
| generation token agreement | 76.25% |

FP32 没有恢复默认噪声下的生成一致性，说明主要损失来自 0.5B 对论文噪声幅度不够鲁棒，而不是 BF16 舍入误差。
