# AloePri Qwen2.5-0.5B 独立客观审核报告

日期：2026-08-07  
审核方式：只读代码、配置、论文源码和现有实验 JSON 交叉核对；未修改工程实现、配置、实验结果或既有文档。  
审核范围：仅 `Qwen2.5-0.5B-Instruct`，不评价 7B、14B、MoE 或 671B 可行性。

## 1. 结论

当前工程的基础代码质量检查通过，现有 v15/v16/v17 精度数字与 JSON 一致，v15 的最终验收结论仍是 `NO-GO`，未把 HumanEval、IFEval、TFMA、SDA、IMA、ISA 和正式性能误写为已经完成。

但是，现状还不能称为“按论文默认方法可复现”：v15 实际是“论文数值超参数 + corrected-paper 修正式实现”，README 和 HTML 却把它称为“论文默认参数”；README/HTML 给出的 checkpoint 转换命令不能被当前 CLI 解析；最终验收脚本对多类攻击证据只检查文件存在，空文件或失败结果也能被判为通过。这三项会直接影响复现结论、他人执行和最终验收，必须先修复。

## 2. 审核基线与验证结果

### 2.1 基线文件

- 论文方法：`E:\AloePri\data\papers\arxiv-2603.01499v2-source\chap\4.method.tex`
- 论文实验：`E:\AloePri\data\papers\arxiv-2603.01499v2-source\chap\6.exp.tex`
- 当前方法配置：`E:\AloePri\configs\transform\paper_qwen05b_complete.yaml`
- 当前报告：`E:\AloePri\README.md`
- 当前 HTML：`E:\AloePri\docs\AloePri_0.5B_Research_Progress_Report.html`
- 当前验收结果：`E:\AloePri\artifacts\acceptance\paper-qwen05b-v15-complete-bf16.json`

### 2.2 只读验证命令与结果

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
.\.venv\Scripts\ruff.exe check --no-cache src scripts tests
.\.venv\Scripts\mypy.exe --no-incremental src
```

结果：

| 检查 | 结果 | 说明 |
|---|---:|---|
| Pytest | 76 passed，1 skipped | 跳过的是 `tests/integration/test_real_private_api.py:22`，需要 `ALOEPRI_RUN_MODEL_TESTS=1` |
| Ruff | 通过 | `All checks passed!` |
| Mypy | 通过 | 39 个源码文件无类型错误 |

这些结果只能证明已纳入测试的代码通过，不能替代真实 0.5B checkpoint、完整攻击、性能和精度验收。

## 3. P0：会造成复现或验收结论失真的问题

### P0-1 最终验收脚本把“攻击证据文件存在”直接当作通过

**位置**

- `E:\AloePri\scripts\build_qwen05b_final_acceptance.py:187-199`
- 核心语句：`checks.append({"id": label, "artifact": path, "status": "PRESENT", "pass": True})`（第 199 行）

**证据**

Gate IA、Attention IA proxy、IMA、ISA、TFMA 和 SDA 的分支只执行路径存在性判断；没有读取 JSON，没有检查 `success`、样本量、模型 ID、key ID、dtype、阈值、正式/冒烟范围或失败字段。

因此以下任一情况都会被误判为通过：

1. JSON 是空对象；
2. JSON 明确记录训练失败；
3. JSON 来自其他模型或其他 key；
4. JSON 只是 smoke test；
5. 指标超过甲方阈值。

当前 v15 因这些文件不存在仍诚实得到 `NO-GO`，但一旦放入任何同名占位文件，验收结果就可能错误变为 `GO`。

**建议**

为每种证据定义 schema 和独立解析器，至少校验：`model_id`、checkpoint SHA-256、key SHA-256、dtype、dataset hash、sample_count、method、success、metric、threshold、CI、scope=full`。无法解析、字段缺失、模型/key 不匹配、仅 smoke 或指标失败均必须是 `FAIL`，而不是 `PRESENT/pass=true`。补充用空 JSON、失败 JSON、错模型 JSON、smoke JSON 验证必失败的单元测试。

### P0-2 v15 被称为“论文默认参数”，但配置和实现明确是修正版

**位置**

- `E:\AloePri\README.md:214`、`:465`
- `E:\AloePri\docs\AloePri_0.5B_Research_Progress_Report.html:20-21`、`:57`
- `E:\AloePri\configs\transform\paper_qwen05b_complete.yaml:2-4`、`:26-31`、`:38-39`
- `E:\AloePri\src\aloepri\transforms\qwen_structural.py:109-111`、`:156-218`
- `E:\AloePri\data\papers\arxiv-2603.01499v2-source\chap\4.method.tex:132-148`、`:179`

**证据**

配置自己声明 `method_profile: corrected-paper` 和 `claim_scope: qwen2.5_0.5b_corrected_paper_method_local_validation`。实际实现还采用：

- `blockperm_mode: gamma-corrected`；
- `rope_frequency_mode: qwen-actual`；
- Q/K 使用同一 block map，而不是论文伪代码给 K 的 `Z_block^T`；
- 基于 Qwen 实际 RoPE 频率的指数；
- 经验性逐层 RMS 校准；
- 论文未规定的 Q/K、FFN 缩放采样区间；
- 独立采样的相容右逆实现策略。

这些修正可能是必要且正确的工程处理，但它们不是论文源码逐字给出的默认算法。把 v15 简写成“论文默认参数”会让精度失败被错误归因于论文默认方法，也使“论文错误”和“工程修正”的边界消失。

**建议**

统一更名为“v15：论文数值超参数的 corrected-paper 实现”。验收结果拆成两个字段：

- `paper_numeric_hyperparameters_match`；
- `paper_literal_algorithm_match`。

当前前者可按逐项证据判断，后者必须为 `false`。论文原式、修正公式和实现公式要并列展示，不能用同一个“论文默认”标签覆盖。

## 4. P1：会阻止复现、污染证据或改变指标解释的问题

### P1-1 README/HTML 的 checkpoint 转换命令不可执行

**位置**

- `E:\AloePri\README.md:219-220`、`:235`、`:244`
- `E:\AloePri\docs\AloePri_0.5B_Research_Progress_Report.html:57-58`
- `E:\AloePri\scripts\convert_paper_qwen2_streaming.py:185-225`

**证据**

文档执行：

```powershell
.\.venv\Scripts\python.exe scripts\convert_paper_qwen2_streaming.py `
  --config configs\transform\paper_qwen05b_complete.yaml
```

而脚本没有 `--config` 参数，且强制要求 `--source`、`--output`、`--key-dir` 和 `--seed`。运行 `--help` 已独立确认。v15/v16/v17 的文档命令无法启动转换，也无法复现现有 checkpoint。

此外，当前 v15 checkpoint 是两个约 1 GB/0.63 GB 的 shard；流式脚本的 `save_tensor()`（`E:\AloePri\scripts\convert_paper_qwen2_streaming.py:93-128`）按 tensor 逐文件保存。现有成品形态更符合 `save_pretrained(max_shard_size=...)` 路线（`E:\AloePri\scripts\convert_paper_qwen2_checkpoint.py:269-270`），与文档声称的命令缺少可验证的 provenance。

**建议**

二选一：给 streaming CLI 实现 `--config` 并把最终解析参数写入 manifest；或在 README/HTML 展开全部实际参数。对现有 v15 明确记录真正执行的脚本、完整命令、Git commit、输入 checkpoint hash、输出 hash、开始/结束时间。若转换后要生成标准 shard，给 streaming 工具增加确定性 reshard 阶段并测试断点续跑。

### P1-2 README 的 PIQA 命令缺少必需 tokenizer 和私有 key

**位置**

- `E:\AloePri\README.md:280-286`
- `E:\AloePri\scripts\run_lm_eval.py:16-19`

**证据**

脚本强制要求 `--tokenizer`，但 README 命令没有该参数，会在 argparse 阶段失败。私有模型还需要 `--key` 才会把明文 token 映射为 `tau(token)`；README 同样没有提供。即使只补 tokenizer 而不补 key，得到的也不是 AloePri 私有 token 流评测。

**建议**

提供一条实际在干净 PowerShell 中验证过的完整明文命令和私有命令，私有命令同时包含原 tokenizer 路径、key 目录和输出路径；把命令执行日志和 CLI 参数固化进结果 JSON。

### P1-3 RMS 校准被标成论文期望，但计算的量与论文公式不同

**位置**

- 论文：`E:\AloePri\data\papers\arxiv-2603.01499v2-source\chap\4.method.tex:179`、`:238-245`
- 实现：`E:\AloePri\src\aloepri\transforms\rms_calibration.py:20-28`
- 校准脚本：`E:\AloePri\scripts\calibrate_paper_rms.py:65-73`
- 证据：`E:\AloePri\artifacts\calibration\qwen05b-paper-seed20260803-rms-expectation-bf16-20.json:5`
- 配置：`E:\AloePri\configs\transform\paper_qwen05b_complete.yaml:38-39`

**证据与推导**

论文第 179 行写的是：

```text
kappa_paper = E[ ||xP||_2 / ||x||_2 ]
```

实现计算的是：

```text
kappa_impl = mean( RMS(xP) / RMS(x) )
```

若原维度为 `d`、扩维后为 `D=d+2h`，则：

```text
RMS(xP) / RMS(x)
= (||xP||_2 / sqrt(D)) / (||x||_2 / sqrt(d))
= sqrt(d/D) * ||xP||_2 / ||x||_2
```

两者相差 `sqrt(d/D)`。本项目 `d=896`、`D=1152`，因子约为 `0.8819`，不是记号差异。实现的 RMS 比值更符合后续 RMSNorm 等价推导，但 JSON 写“matching the paper expectation”、配置写 `paper-expectation` 均不准确。

**建议**

把当前 estimator 改名为 `covariant-rms-empirical`。同时实现 `paper-literal-l2-ratio` 仅用于对照，禁止二者共享 `paper_parameter_match=true`。这一项属于论文公式内部不一致，应单独形成公式推导 PDF。

### P1-4 论文 Attention 原式与实际修正式没有在报告中明确分开

**位置**

- 论文：`E:\AloePri\data\papers\arxiv-2603.01499v2-source\chap\4.method.tex:132-136`
- README：`E:\AloePri\README.md:61-65`
- HTML：`E:\AloePri\docs\AloePri_0.5B_Research_Progress_Report.html:27`
- 实现：`E:\AloePri\src\aloepri\transforms\qwen_structural.py:109-111`

**证据与推导**

论文/报告写 Q 右乘 `Z`、K 右乘 `Z^T`。忽略 RoPE 其余项，注意力内积的 block 部分变为：

```text
(qZ)(kZ^T)^T = q Z Z k^T = q Z^2 k^T
```

一般置换矩阵只满足 `ZZ^T=I`，不满足 `Z^2=I`，所以该式不能一般性保持注意力分数。当前实现让 Q/K 使用同一 `block_map`，从而：

```text
(qZ)(kZ)^T = q Z Z^T k^T = qk^T
```

当前代码修正是合理的，但 README/HTML 只展示论文原式，紧接着便说“实现同时处理”，读者会误以为代码实现了原式。

**建议**

在公式区并列列出“论文打印式”和“工程修正式”，明确说明 v15 使用修正式；单独生成该论文错误的推导 PDF，并用非对合置换构造最小反例测试。

### P1-5 BlockPerm 的 `gamma` 和 Qwen RoPE 频率均属于论文歧义/错误修正，不能隐去

**位置**

- 论文：`E:\AloePri\data\papers\arxiv-2603.01499v2-source\chap\4.method.tex:103`、`:142-148`
- 实现：`E:\AloePri\src\aloepri\transforms\qwen_structural.py:156-218`
- 配置：`E:\AloePri\configs\transform\paper_qwen05b_complete.yaml:27-29`

**证据**

论文把 `gamma` 声明为 BlockPerm 输入，却在第 148 行的 softmax 公式中完全未使用。论文的频率指数写 `-2(i-1)/m_blocks`；Qwen 实际 rotary pair 的频率索引与这个写法不一致。实现明确提供 `gamma-corrected` 和 `qwen-actual` 两个修正模式，v15 使用修正模式。

**建议**

分别输出两份独立论文问题推导文档：一份证明 `gamma` 在打印算法中是死参数；一份从 Qwen `inv_freq` 定义推导 paper-literal 与 qwen-actual 的指数差异。报告中给出 literal/corrected 的最小消融结果，不再统称“论文默认”。

### P1-6 VMA 预测缓存缺少证据绑定，可能静默复用旧结果

**位置**

- `E:\AloePri\scripts\run_vma_pupa.py:38-57`、`:300-424`
- 报告：`E:\AloePri\README.md:420`

**证据**

`cached_prediction()` 只用调用者给出的名字生成 `.pt` 文件，并验证 tensor 的 shape/dtype。缓存键没有绑定：checkpoint SHA-256、key SHA-256、候选 token 列表/hash、查询数据 hash、代码版本、攻击算法版本和变换参数。README 报告的 v15 17.27 秒是 97 个缓存全部命中的报告重算时间，不是完整攻击计算耗时。

**建议**

为缓存写 sidecar manifest，完整输入指纹不一致就拒绝加载；结果 JSON记录每个缓存的 SHA-256 和 hit/miss。验收用的正式结果至少执行一次空缓存全量运行，并把完整运行时间、峰值显存和日志作为证据。当前“17.27 秒”只能标为 cache-hit aggregation time。

### P1-7 CUDA HF runtime 强制 BF16，不能保持 v16 FP32 实验口径

**位置**

- `E:\AloePri\src\aloepri\serving\hf_runtime.py:15-26`

**证据**

代码在 CUDA 上无条件设置 `torch.bfloat16`，CPU 上无条件设置 `torch.float32`，不读取 checkpoint 实际 tensor dtype，也没有显式 CLI/runtime dtype 配置。v16 是 FP32 checkpoint；若通过该 runtime/API 在 CUDA 加载，会被转换为 BF16。任何未来标记为“v16 FP32 API/性能”的证据都会口径错误。

**建议**

增加显式 `dtype` 参数，默认从 checkpoint tensor 或经验证的 manifest 读取；启动日志和 API 证据同时记录 `checkpoint_dtype` 与 `runtime_dtype`。验收脚本要求二者与实验声明一致。

### P1-8 VMA 的 Wilson 区间没有处理同一 query 内相关性，也不覆盖随机 key 波动

**位置**

- `E:\AloePri\README.md:420`
- `E:\AloePri\docs\AloePri_0.5B_Research_Progress_Report.html:89-90`

**证据**

当前区间把 62,634 个 token occurrence 视作二项样本。同一 query 内 token、重复 token 和同一变换 key 下的预测不独立，因此普通 Wilson 区间通常会低估不确定性。文档已经正确说明区间不表示不同 key 波动，但表格仍以“95% CI”呈现，容易被当作整体攻击不确定性。

**建议**

保留 Wilson 区间但明确改名为“occurrence-level descriptive binomial interval”；正式科研区间按 query 做 cluster bootstrap，并对多个独立 key 重复实验后报告 key-level 分布/区间。

### P1-9 `paper_parameter_match` 实际只验证项目自定义修正版字段

**位置**

- `E:\AloePri\scripts\build_qwen05b_final_acceptance.py:54-75`、`:276-278`
- `E:\AloePri\artifacts\acceptance\paper-qwen05b-v15-complete-bf16.json:5`

**证据**

验收脚本把 `gamma-corrected`、`qwen-actual`、`paper-expectation` 等项目约定硬编码为 expected，然后输出 `paper_parameter_match: true`。该字段没有证明论文 literal 算法匹配，只证明 YAML 符合本项目预设 corrected profile。

**建议**

将字段改为 `corrected_profile_match`，另建逐条 paper claim matrix；每一论文公式标记 `literal`、`corrected`、`not_applicable` 或 `not_implemented`，并链接测试和推导文档。

## 5. P2：可维护性、可追溯性和表述精度问题

### P2-1 校准语料缺少完整 provenance

**位置**

- `E:\AloePri\scripts\calibrate_paper_rms.py:13-22`、`:65-73`
- `E:\AloePri\artifacts\calibration\qwen05b-paper-seed20260803-rms-expectation-bf16-20.json:1-5`

**证据**

脚本内置默认 prompt 数量与 artifact 的 `prompt_count=20` 不一致，说明运行时用了外部/修改后的语料，但 artifact 没有保存 prompt 文件路径、内容 hash、顺序 hash 或每条 prompt ID。当前 kappa 不能由 artifact 单独复现。

**建议**

保存 calibration dataset 的规范化 SHA-256、逐条 ID/hash、tokenizer hash、抽样种子和完整命令；校准 JSON 不应只记录 prompt_count。

### P2-2 README 的仓库树与实际位置/能力表述不完全一致

**位置**

- `E:\AloePri\README.md:138-148`
- `E:\AloePri\src\aloepri\models\modeling_aloepri_qwen2.py`
- `E:\AloePri\src\aloepri\attacks\`

**证据**

README 树将 YAML 表现为直接位于 `configs/`，实际在 `configs/transform/`。README 称 models 为“自定义 Qwen2 forward”，但模型类主要继承 HF Qwen2 forward；README 又把 VMA/IA/IMA/ISA/TFMA/SDA 都归入 attacks 目录，实际多项入口位于 `scripts/`。

**建议**

用自动生成的树替换手写树；把描述改成“基于 HF Qwen2 继承、通过变换后配置/权重运行”，逐项列出真实入口路径。

### P2-3 manifest 校验不拒绝未列出的额外文件

**位置**

- `E:\AloePri\src\aloepri\integrity.py:36-58`（manifest 校验函数）

**证据**

校验器验证 manifest 中列出的文件，却不比较目录实际文件集合。额外 tensor/shard 或旧文件不会导致失败。当前 v15 目录未发现由此造成的实测错误，但这是通用完整性缺口。

**建议**

定义允许的非 manifest 文件白名单，实际文件集合与 manifest 集合不一致即失败；增加“注入额外 shard”测试。

### P2-4 HTML 的“工程闭环完成”范围过宽

**位置**

- `E:\AloePri\docs\AloePri_0.5B_Research_Progress_Report.html:21`
- 未测清单：同文件 `:99`

**证据**

HTML 同时写“工程闭环完成”，又列出 HumanEval、IFEval、TFMA、SDA、IMA、ISA、vLLM 和 HF 正式性能未运行。后文清单是诚实的，但标题易被理解为整个甲方验收闭环完成。

**建议**

改成“核心 token-ID 推理链路、checkpoint/KV Cache/API 与 PUPA VMA 闭环完成；完整验收未完成”。

## 6. 已核对且未发现抄写错误的结果

### 6.1 精度数字

README/HTML 中以下数值与现有 comparison JSON 一致：

| 配置 | 任务 | 明文 | 候选 | 变化 | 95% 区间 |
|---|---|---:|---:|---:|---:|
| v15 BF16 | MMLU | 34.475% | 27.261% | -7.214 pp | [-8.049, -6.386] |
| v15 BF16 | C-Eval | 52.972% | 26.003% | -26.969 pp | [-30.314, -23.551] |
| v15 BF16 | PIQA | 70.239% | 58.868% | -11.371 pp | [-13.765, -9.032] |
| v16 FP32 | PIQA | 70.131% | 70.239% | +0.109 pp | [-0.490, +0.707] |
| v17 BF16 | PIQA | 70.239% | 69.042% | -1.197 pp | [-2.448, +0.054] |

v15 三项均未达到甲方“绝对下降不超过 3.5 个百分点”目标；文档当前结论与数据一致。

### 6.2 VMA 数字

现有 PUPA v15 artifact 的五组结果与 README/HTML 表格一致，均在当前甲方阈值内。报告也已明确：

- 只有一个 transform key；
- 候选集为 16,384，不是完整词表；
- 区间不覆盖 key 间波动；
- 17.27 秒是缓存全命中的重算时间。

这些限制没有被完全隐瞒，但仍需按 P1-6/P1-8 加强证据完整性和统计口径。

### 6.3 未测项和最终结论

`E:\AloePri\artifacts\acceptance\paper-qwen05b-v15-complete-bf16.json` 当前为 `NO-GO`，并列出 HumanEval、IFEval、Gate IA、Attention IA proxy、IMA、ISA、TFMA、SDA 和 performance 缺失。README/HTML 的未测清单与此基本一致，没有发现把这些项目编造成实测结果的情况。

## 7. 建议修复顺序与复审门槛

1. 修复 P0-1：验收脚本按 schema、模型/key/hash、指标阈值验证真实证据，并补负向测试。
2. 修复 P0-2、P1-3、P1-4、P1-5、P1-9：把 paper-literal、corrected-paper 和 numeric-hyperparameters 三种口径完全拆开。
3. 修复 P1-1、P1-2：所有 README/HTML 命令在干净 PowerShell 实际执行到参数解析和启动阶段；记录真实 v15 生成命令。
4. 修复 P1-6、P1-7：绑定缓存/实验 provenance，显式控制 runtime dtype。
5. 补真实模型集成测试；当前被跳过的 `test_real_private_api.py` 必须在最终复审中实际运行。
6. 重新生成 v15 acceptance；在缺失攻击和性能实验未完成前，结论必须继续是 `NO-GO`。

复审通过至少需要：无占位证据可通过验收、文档命令可执行、v15 标签准确、RMS/Attention/BlockPerm/RoPE 的论文原式与修正式有独立推导、实验 JSON 能追溯到 checkpoint/key/dataset/code/dtype 的 hash。

## 8. 需要单独形成 PDF 的论文问题

根据本次源码与公式核对，至少应分别建立以下独立文档，不应合并成一份笼统的“论文有误”说明：

1. `Paper_Error_01_Attention_BlockPerm_Transpose.pdf`：证明 Q 乘 `Z`、K 乘 `Z^T` 会产生 `Z^2`，一般不保持注意力内积；给出非对合置换反例。
2. `Paper_Error_02_RMSNorm_Kappa_Dimension_Factor.pdf`：推导 L2 norm ratio 与 RMS ratio 相差 `sqrt(d/(d+2h))`，解释论文第 179 行与第 241-245 行的内部不一致。
3. `Paper_Error_03_BlockPerm_Gamma_Unused.pdf`：逐行证明 Algorithm 2 声明了 `gamma` 输入但概率分布不含 `gamma`，因此打印算法对 gamma 的偏导恒为零、参数不起作用。
4. `Paper_Error_04_RoPE_Frequency_Index_Qwen.pdf`：从 Qwen rotary embedding 的 `inv_freq` 定义推导 paper-literal 指数与 Qwen 实际 pair index 的差异。

这些 PDF 应只证明可由论文源码、公开 Qwen 定义和代数推导直接支持的问题；论文未公开采样分布、右逆采样细节等属于“实现信息缺失”，不应写成已经证明的论文错误。
