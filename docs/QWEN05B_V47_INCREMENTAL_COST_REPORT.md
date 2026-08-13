# Qwen2.5-0.5B v47 增量实验与费用报告

> 最终代码检查、4090 时长推导和费用口径见 `docs/QWEN05B_V47_FINAL_CODE_REVIEW_AND_4090_COST.md`。28 小时是 23.25 小时基础计划加 20% 故障准备金后按整小时取整，不是必须持续运行 28 小时。

_审计时间：2026-08-12；报价基线：用户于 2026-08-11 提供；币种：人民币_

## 1. 计费结论

本报告只计算当前 v47 尚未完成，或虽然生成过原始结果但不满足正式验收协议、必须重新生成的项目。
已经完成并与当前 checkpoint/key 绑定的实验不重跑；实验已经完成但指标失败，也不为了改变结论而重跑。

| 方案 | 基础实例小时 | 20%故障余量 | 预留计费小时 | GPU费用 |
| --- | ---: | ---: | ---: | ---: |
| RTX 4090单机完成全部增量项 | 23.25 h | 4.65 h | 28 h | ¥50.40 |
| RTX 3090完成HF增量项 | 21.75 h | 4.35 h | 27 h | ¥23.76 |
| RTX 4090只补vLLM/SGLang | 5.00 h | 1.00 h | 6 h | ¥10.80 |
| RTX 3090＋RTX 4090组合 | 26.75 h | 5.35 h | 33 h | ¥34.56 |

推荐：

- 操作最简单：驱动580的RTX 4090保留28小时，GPU预算`¥50.40`。
- GPU费用最低：RTX 3090跑27小时，RTX 4090跑6小时，合计`¥34.56`。

费用公式：

```text
基础实例小时 = 各增量阶段计划小时之和
含余量小时 = 基础实例小时 × 1.20
预留计费小时 = ceil(含余量小时 / 1小时) × 1小时
GPU费用 = 预留计费小时 × GPU数量 × 每小时单价
```

## 2. 当前工件审计与是否计费

当前验收审计为`6 PASS / 9 FAIL / 6 NOT_TESTED`。FAIL不等于没运行；下表按“是否还缺实验”重新分类。

| 模块 | 当前证据 | 当前状态 | 是否再跑 | 计费处理 |
| --- | --- | --- | --- | --- |
| 公式重建与运行时验证 | `artifacts/verification/qwen05b-candidate-v47-best-single/` | 已完成 | 否 | ¥0 |
| Direct Match | `artifacts/privacy/v47/direct-score.json` | 已完成，PASS | 否 | ¥0 |
| VMA | `artifacts/privacy/v47/vma-score.json` | 已完成，TTRSR失败 | 否 | ¥0 |
| Gate-IA | `artifacts/privacy/v47/gate-ia-score.json` | 已完成，指标失败 | 否 | ¥0 |
| Attention-IA | `artifacts/privacy/v47/attention-ia-score.json` | 已完成，PASS | 否 | ¥0 |
| IMA | `artifacts/privacy/v47/ima-score.json` | 已完成，PASS | 否 | ¥0 |
| ISA | `artifacts/privacy/v47/isa-attention-score.json` | 已完成，PASS | 否 | ¥0 |
| Known-plaintext | `artifacts/privacy/v47/known-plaintext-score.json` | 已完成，指标失败 | 否 | ¥0 |
| TFMA | `formal_corpus_complete=false` | 正式语料协议未完成 | 是，仅正式语料版本 | 计入 |
| SDA | `formal_corpus_complete=false` | 正式语料协议未完成 | 是，仅正式语料版本 | 计入 |
| MMLU | 14,042题两侧均已生成 | BF16/FP32不一致，当前脚本身份不匹配 | 是，两侧正式重建 | 计入 |
| C-Eval | 1,346题两侧均已生成 | BF16/FP32不一致，当前脚本身份不匹配 | 是，两侧正式重建 | 计入 |
| PIQA | 1,838题两侧均已生成 | BF16/FP32不一致，当前脚本身份不匹配 | 是，两侧正式重建 | 计入 |
| IFEval baseline | 旧541条为BF16/eager | 不符合当前FP32/SDPA协议 | 是，541条 | 计入 |
| IFEval candidate | 当前8/541 | 还缺533条；跨Linux主机不能混用运行时证据 | 是，服务器统一生成541条 | 计入 |
| HumanEval | 当前v47正式文件不存在 | 未测试 | 是，164＋164 | 计入 |
| 产品100问 | 当前v47正式文件不存在 | 未测试 | 是，100问 | 计入 |
| 传输隐私边界 | 当前v47正式文件不存在 | 未测试 | 是 | 计入 |
| HF平衡性能 | 当前v47正式文件不存在 | 未测试 | 是，80请求 | 计入 |
| HF/vLLM | 旧冒烟已运行但token不一致 | 新引擎栈尚未形成当前证据 | 是 | 计入 |
| vLLM/SGLang | 当前正式文件不存在 | 未测试 | 是 | 计入 |

MMLU、C-Eval、PIQA不是为了提高分数而重复运行。现有比较的baseline是BF16、candidate实际为FP32，
`dtype_match=false`；同时`run_lm_eval.py`已经修复有效dtype记录，旧工件保存的脚本SHA-256不再等于当前脚本。
若不重新生成两侧，最终验收器会继续把这三项判定为证据绑定失败。

本机8条IFEval没有被算成533条续跑，是因为当前工件将Windows绝对模型路径、运行时、GPU和脚本哈希写入统一
`run_provenance`。将它直接接到Linux生成会形成两种运行环境混合的单一证据。服务器重新生成这8条约占
IFEval总量的1.48%，费用已包含在服务器541条candidate预算中。

## 3. RTX 4090单机增量明细

机器：RTX 4090 24GB，驱动`580.159.04`，单价`¥1.80/h`。

| 阶段 | 数量或退出条件 | 计划小时 | 未取整费用 |
| --- | --- | ---: | ---: |
| 创建实例并挂载数据盘 | `/data`可用空间不少于90GB | 0.25 | ¥0.45 |
| 上传、解压、SHA-256 | 源码、模型、密钥、已有证据四包 | 0.50 | ¥0.90 |
| CUDA12.1核心环境 | torch2.5.1＋评测依赖 | 1.00 | ¥1.80 |
| preflight＋静态测试 | 硬件、输入、ruff、mypy、pytest | 0.25 | ¥0.45 |
| 净化server package | 构建并扫描密钥泄漏 | 0.25 | ¥0.45 |
| CCI3/MedDialog下载与标准化 | `formal_corpus_complete=true` | 0.50 | ¥0.90 |
| 重建private observations | 绑定正式语料manifest | 0.10 | ¥0.18 |
| TFMA正式运行与评分 | 只重跑TFMA | 0.15 | ¥0.27 |
| SDA正式训练 | 3,000 steps | 0.40 | ¥0.72 |
| SDA攻击与评分 | target attack＋score | 0.10 | ¥0.18 |
| MMLU FP32两侧 | 14,042×2 | 4.30 | ¥7.74 |
| C-Eval FP32两侧 | 1,346×2 | 0.50 | ¥0.90 |
| PIQA FP32两侧 | 1,838×2 | 0.40 | ¥0.72 |
| IFEval FP32 baseline | 541条，SDPA | 3.50 | ¥6.30 |
| IFEval FP32 candidate | 541条，SDPA | 4.50 | ¥8.10 |
| HumanEval baseline生成 | 164题 | 0.25 | ¥0.45 |
| HumanEval candidate生成 | 164题 | 0.35 | ¥0.63 |
| HumanEval隔离执行与比较 | 328份代码 | 0.10 | ¥0.18 |
| 传输隐私边界 | 请求、响应、日志、目录扫描 | 0.10 | ¥0.18 |
| 产品问答门禁 | 固定100问 | 0.50 | ¥0.90 |
| HF平衡性能 | ABBA＋BAAB，80请求 | 0.50 | ¥0.90 |
| CUDA13引擎环境 | vLLM0.26＋SGLang0.5.17 | 2.50 | ¥4.50 |
| vLLM兼容性 | 32-token＋IFEval smoke | 0.75 | ¥1.35 |
| SGLang兼容性 | 32-token与vLLM比较 | 1.00 | ¥1.80 |
| 最终验收、打包、下载 | acceptance JSON及日志回传 | 0.50 | ¥0.90 |
| **基础合计** |  | **23.25** | **¥41.85** |
| **20%故障余量** |  | **4.65** | **¥8.37** |
| **按整小时预留** |  | **28.00** | **¥50.40** |

## 4. RTX 3090＋RTX 4090组合明细

RTX 3090执行HF、TFMA/SDA、产品和性能，基础`21.75h`，加20%后为`26.10h`，按整小时预留`27h`：

```text
27 h × ¥0.88/h = ¥23.76
```

RTX 4090只安装CUDA13引擎并运行兼容性，基础`5.00h`，加20%后刚好`6.00h`：

| RTX 4090引擎阶段 | 计划小时 |
| --- | ---: |
| 建机、上传、交接校验 | 0.50 |
| vLLM/SGLang两个隔离环境 | 2.50 |
| vLLM兼容性 | 0.75 |
| SGLang兼容性 | 1.00 |
| 工件打包下载 | 0.25 |
| **基础合计** | **5.00** |
| **含20%余量** | **6.00** |

```text
6 h × ¥1.80/h = ¥10.80
组合总费用 = ¥23.76 + ¥10.80 = ¥34.56
```

组合方案比4090单机预算少`¥15.84`，代价是增加一次工件交接和第二台机器环境安装。

## 5. 小时估算依据

选择题预算直接使用当前RTX 3060 Laptop工件的`date`到文件落盘时间，不按理论算力虚构：

| 任务 | baseline实测 | candidate实测 | 实测合计 | 4090计划 |
| --- | ---: | ---: | ---: | ---: |
| MMLU | 1.180 h | 3.090 h | 4.270 h | 4.30 h |
| C-Eval | 0.168 h | 0.260 h | 0.428 h | 0.50 h |
| PIQA | 0.153 h | 0.205 h | 0.357 h | 0.40 h |

其余依据：

- 当前TFMA从private observations落盘到评分完成约7.5分钟，计划0.15小时。
- 当前SDA训练、攻击和评分约23.5分钟，正式版本合计计划0.50小时。
- 当前IFEval在6GB卡上最近两条用时约5.25分钟；考虑541条长度差异后，4090两侧共预留8小时。
- 旧HumanEval 164题两侧生成在本机合计约0.394小时；当前FP32生成、隔离执行和比较共预留0.70小时。
- 旧HF 80请求平衡性能运行约12分钟；新环境包含模型重载和异常余量，预留0.50小时。
- CUDA13两个隔离引擎环境可能下载或编译扩展，固定预留2.50小时，是非推理阶段最大不确定项。

IFEval和引擎环境的小时数仍是预算，不是尚未发生的实测。云端入口会把每个阶段的真实开始时间、结束时间、
秒数和退出码写入`artifacts/cloud/qwen05b-v47/stage-timings.tsv`。

## 6. 服务器只跑增量项

已有正式证据随第四个增量证据包上传。服务器不得执行攻击脚本的`all`阶段。

RTX 4090单机：

```bash
cd /data/AloePri
bash scripts/cloud/bootstrap_qwen05b_cuda121.sh
bash scripts/cloud/run_qwen05b_v47_incremental.sh core
bash scripts/cloud/bootstrap_qwen05b_engines_cuda13.sh
bash scripts/cloud/run_qwen05b_v47_incremental.sh engines
```

两机方案：RTX 3090只执行`core`；把新工件同步到RTX 4090后只执行`engines`。

```bash
# RTX 3090
bash scripts/cloud/run_qwen05b_v47_incremental.sh core

# RTX 4090
bash scripts/cloud/run_qwen05b_v47_incremental.sh engines
```

增量入口明确不调用以下已完成阶段：`direct-vma`、`ia`、`ima`、`isa`、`known-plaintext`。

## 7. 实际结算

实验完成后按真实计时文件计算：

```bash
.venv-qwen05b-cu121/bin/python scripts/summarize_cloud_runtime_cost.py \
  --timings artifacts/cloud/qwen05b-v47/stage-timings.tsv \
  --hourly-rate 1.80 --gpu-count 1 --billing-increment-hours 1.0 \
  --out artifacts/cloud/qwen05b-v47/actual-cost.json
```

费用配置和计划计算结果：

```text
configs/cloud/qwen05b_v47_incremental_cost_plan.yaml
artifacts/cloud/qwen05b-v47/incremental-cost-plan.json
```

GPU预算不包含数据盘扩容和公网流量。用户给出的机器信息没有这两项单价，因此不能将其写成0元；下单页出现单价后，
追加计算为：

```text
最终费用 = GPU实例费 + 数据盘GB×小时费 + 出网GB费
```

达到预留小时上限时停止新增任务，先下载已有工件，再决定是否续租。已完成但指标FAIL的实验不自动重跑。
