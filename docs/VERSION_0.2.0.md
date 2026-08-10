# AloePri 0.2.0 版本说明

## 版本目标

本版本只验收 `Qwen2.5-0.5B-Instruct` 的 v47 checkpoint：

```text
data/packages/qwen05b-candidate-v47-best-single
data/keys/dev-qwen05b-candidate-v47-best-single
data/keys/qwen05b-candidate-v47-best-single-online
data/keys/qwen05b-candidate-v47-best-single-offline
```

转换参数来自 `configs/product/qwen05b_v47_best_single_candidate.yaml`。任何旧 checkpoint 的
精度、攻击或性能 JSON 都不能填入 `configs/acceptance/qwen05b_v47.yaml`。

## 攻击协议变更

| 攻击 | 0.2.0 攻击入口 | 目标密钥读取位置 | 输出 |
|---|---|---|---|
| Gate-IA | `scripts/run_gate_ia_isolated.py` | 独立 mapping scorer | plain ID → predicted private ID |
| Attention-IA | `scripts/run_attn_ia_corrected.py` | 独立 mapping scorer | plain ID → predicted private ID |
| IMA | `scripts/run_ima_target_attack.py` | 独立 inversion scorer | private ID → predicted plain ID |
| ISA-Attention | `scripts/run_isa_attention_score.py` | 独立 inversion scorer | private ID → predicted plain ID |
| ISA-Hidden | `scripts/run_isa_hidden_state_isolated.py` | 独立 inversion scorer | private ID → predicted plain ID |
| TFMA | `scripts/run_tfma_isolated.py` | `scripts/score_tfma_predictions.py` | private ID → Top-K plain IDs |
| SDA | `scripts/run_sda_target_attack.py` | `scripts/score_sda_predictions.py` | private ID → predicted plain ID |

`scripts/build_private_token_observations.py` 是可信测试夹具：它读取在线词表密钥，把锁定测试
语料转换成服务器能观察到的私有 token 工件。攻击程序只读取该工件，不读取密钥。评分程序
最后读取目标密钥计算 TTRSR、Top-K 恢复率和 BLEU-4。

## 公式修正

Attention-IA 原打印式在 Qwen 的 Q 投影形状上不能直接相乘。0.2.0 按每个 attention head
计算正则化 leverage：

$$
s_{t,J}=q_{t,J}(Q_J^TQ_J+\varepsilon I)^{-1}q_{t,J}^T.
$$

Qwen RoPE 的一对坐标使用 $(j,j+d_h/2)$。每层、每 head 的特征排序后再进行最近邻匹配，
避免把 head 和 block 的排列顺序当成已知信息。该修正的代码位于
`src/aloepri/attacks/attention_ia.py`。

Gate-IA 的均值投影采用完全等价的结合律：

$$
\operatorname{mean}(EW^T,\mathrm{dim}=1)=E\,\operatorname{mean}(W,\mathrm{dim}=0).
$$

它只降低峰值显存，不改变论文攻击特征。

## 当前可核验证据

`artifacts/acceptance/qwen05b-v47-current.json` 由验收器生成。当前已有公式与运行时两类 v47
证据；其余项目必须生成本版本定义的新工件后才能通过。验收器的退出码为：

- `0`：全部项目通过，结论 `GO`；
- `2`：存在失败或未测试项目，结论 `NO-GO`；
- 其他：配置、文件或程序执行错误。
