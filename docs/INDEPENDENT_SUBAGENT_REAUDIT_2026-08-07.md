# AloePri Qwen2.5-0.5B 第二轮独立复核

日期：2026-08-07。范围仅为 Qwen2.5-0.5B；只读核对论文、甲方 PDF/PPT、代码、CLI、artifact、README、HTML 和输出 PDF。除本文外未修改文件。

## 结论

首轮发现的公式口径、不可执行命令、manifest 额外文件、runtime dtype 和“仅文件存在即通过”等问题已有实质修复。当前 v15 数字在 artifact、README、HTML、PDF 间一致，最终结论仍为 `NO-GO`，未测试的 HumanEval/IFEval 和非论文精确代理大多已显式标注。

仍有 1 项 P0、3 项 P1、2 项 P2。P0 不改变当前 `NO-GO`，但会使未来验收可能被错误或移用的 JSON 骗过；修复前不能把 acceptance 脚本作为甲方最终签收器。

## P0

### P0-1 acceptance 仍未把证据绑定到 checkpoint/key/dataset，且不要求完整 VMA 协议

- 位置：`E:\AloePri\scripts\build_qwen05b_final_acceptance.py:91-118,156-185,187-298,349-368`。
- 精度、Gate/Attention IA、IMA、ISA、TFMA、SDA、性能均直接信任路径下 JSON 的自报字段；攻击 artifact 本身也没有 checkpoint SHA-256、key SHA-256、dataset hash 和代码 hash。
- VMA 只遍历 artifact 里已有组合，不要求五个组合齐全，不要求 24 层，不校验 cache manifest；删除四个组合、保留一个低恢复率组合仍可通过。
- 性能只读取 comparison JSON；一个来自其他模型/请求集的同结构文件也能通过。
- 当前 artifact 真实存在且当前总判定为 `NO-GO`，没有发现本次结果已被伪造；问题是验收器不能证明证据属于 v15。
- 修复：所有 artifact 增加统一 provenance；acceptance 对模型/manifest/key/dataset/prompt/code/dtype/device 逐项核 hash，并固定 VMA 五组合、24 层、候选集和正式缓存要求；增加错模型、错 key、缺组合、空/伪造 comparison 的负向测试。

## P1

### P1-1 ISA 的 token-Gram 适配不是论文攻击等价实现，却被 acceptance 判为 VALIDATED

- 位置：`E:\AloePri\scripts\run_isa_hidden_state.py:15-17,119-130,151-167`；`E:\AloePri\scripts\build_qwen05b_final_acceptance.py:235-252`。
- `token_gram(h)` 只在 `h -> hR` 且 `R` 正交时保持不变；本项目隐藏态近似 `hP`，其中 `P` 是 `896x1152` 的一般矩阵，不满足 `PP^T=cI`。因此匹配 `Gram(h_candidate)` 与 `Gram(h_private)` 没有论文给出的恢复等价性。
- artifact 没有 `paper_exact=false`，scope 反而写 `paper_attack_with_dimension_invariant_adaptation`；acceptance 只检查 loss 下降、步数和 TTRSR，当前将其记为 `VALIDATED/pass=true`。
- README/PDF 写“当前实现口径达到”，比“论文精确通过”克制，但验收字段仍过强。
- 修复：artifact 固定 `paper_exact=false`、`proxy_reason`；acceptance 对 ISA 返回诊断状态且不得计入论文精确通过，除非给出适用于一般矩形 P 的可证明不变量或按论文同维协议重做。

### P1-2 HF 性能证据可被错误配对，固定明文→v15 顺序不足以支持稳定通过

- 位置：`E:\AloePri\scripts\benchmark_hf.py:50-128`；`E:\AloePri\scripts\compare_performance.py:38-59`；acceptance `:349-368`。
- benchmark artifact 不保存 prompt hash、模型/manifest hash、key hash、GPU 型号/驱动、运行序号；comparison 只检查记录数量相同，不检查 prompt 身份、input/output token、dtype、device 或顺序。
- 当前两份 artifact 的 20 个 input length、100 output token、CUDA/BF16 均一致；数字与报告一致。但明文固定先跑，出现三个约 108-114 ms TTFT 离群点，v15 p95 仅 43.489 ms。报告已承认顺序效应，acceptance 仍把 p50/p95/p99 全部判通过。
- 修复：artifact 保存请求 ID/prompt hash及完整环境；comparison 严格校验；至少做 ABBA/BAAB 多轮独立进程顺序平衡，分层按 request ID 配对，再决定 15% 门禁。

### P1-3 当前 v15 VMA 正式结果仍来自“绑定旧缓存”，新 cache 机制存在未登记文件漏洞

- 位置：`E:\AloePri\scripts\run_vma_pupa.py:53-103,124-143,335-353,588-601`；`E:\AloePri\artifacts\privacy\cache\v15-c16384\cache_manifest.json`。
- v15/v16 cache manifest 均为 `bound_existing_unversioned_cache=true`；它们把历史 `.pt` 绑定到当前输入，不能证明这些预测最初由当前 checkpoint/key/code 计算。
- 有 manifest 时只校验 `prediction_files` 中列出的文件，不拒绝目录中新出现但未列出的 `.pt`；`cached_prediction()` 会直接读取这种额外文件，之后 finalize 才把它加入 manifest。
- private fingerprint 只 hash `aloepri_manifest.json`，运行 VMA 前不验证 manifest 所列权重，权重被改但 manifest 未改时仍可复用旧预测。
- 报告已诚实把完整未命中显存/时间归于 v16，没有把 v15 cache-hit 时间当全量运行；但 v15 隐私正式证据仍需一次空缓存重算。
- 修复：拒绝任何未登记 `.pt`；读取缓存前校验模型 manifest 及实际 shard；正式验收要求 `bound_existing_unversioned_cache=false`，对 v15 空缓存全量重跑并保存运行日志。

## P2

### P2-1 部分名称仍把 corrected 实现写成 paper-faithful

- `E:\AloePri\scripts\convert_paper_qwen2_checkpoint.py:72` 的 CLI 描述仍为 `paper-faithful d+2h model`，实际包含 gamma、Q/K 同侧 Z、Qwen RoPE、RMS 等修正。
- 建议统一为 `corrected-paper d+2h model`，避免命令输出与报告口径冲突。

### P2-2 README 仓库树仍高估 attacks 包内容

- `E:\AloePri\README.md` 与输出 PDF 把 `src/aloepri/attacks/` 标成 `VMA/IA/IMA/ISA/TFMA/SDA`；实际包主要是 mapping/recurrence，正式入口在 `scripts/`。
- 不影响结果，但应列真实入口，避免接手者在错误目录寻找实现。

## 首轮问题修复复核

| 首轮问题 | 第二轮结论 |
|---|---|
| 攻击文件存在即通过 | 已修复为字段/规模/阈值解析，但 provenance 与完整协议仍有 P0-1 |
| v15 写成论文默认算法 | 已修复为“论文数值超参数 + corrected-paper”，literal=false |
| checkpoint/PIQA 命令不可执行 | 已修复；当前 README 参数与 CLI `--help` 一致 |
| RMS、Q/K、gamma、RoPE 混写 | 已拆分论文原式和修正式；实现方向与报告一致 |
| CUDA runtime 强制 BF16 | 已增加 auto/float32/bfloat16 |
| manifest 接受额外模型文件 | 已修复并有负向测试 |
| VMA cache 无输入指纹 | 已增加 manifest/hash，但历史绑定和额外 `.pt` 漏洞见 P1-3 |
| “工程闭环完成”过宽 | 已限定已运行模块，最终明确 NO-GO |

## 公式、形状与当前数字核对

- 已核对 `paper_key_matrix.py`、`paper_qwen2.py`、`qwen_structural.py`：P 为 `d x (d+2h)`、Q 为 `(d+2h) x d`，输入投影用 `Q^T`、输出投影用 `P^T`；FFN 置换/缩放成对抵消；Q/K 使用同侧 block map；GQA 组重排一致。
- tiny 测试覆盖无噪声 logits、prefill/decode、KV cache；真实 v15 API 集成测试本轮实际通过。v15 生成本身不等价：prefill top-1 19.44%、greedy 不一致，报告未掩盖。
- v15 MMLU/C-Eval/PIQA、v16/v17 PIQA、五组 VMA、Gate/Attention IA、IMA/ISA/TFMA/SDA、HF TTFT/TPOT 数字与 artifact、README、HTML、`E:\AloePri\output\pdf\AloePri_0.5B_Research_Progress_Report.pdf` 一致。
- 甲方 PPT 的精度下降≤3.5 pp、VMA 四指标和效率劣化≤15%已正确列入；未把论文 Qwen3-14B 数据当作 0.5B 数据。

## 已运行命令与残余限制

- `pytest -q -p no:cacheprovider`：79 passed，1 skipped；真实模型单测另以 `ALOEPRI_RUN_MODEL_TESTS=1` 运行：1 passed。
- Ruff：通过；manifest：8 文件、0 failure；关键 converter/eval/VMA/ISA/performance CLI `--help` 均成功。
- 本轮并发 `mypy --no-incremental src` 超时，不能把本轮结果记为独立通过；主流程此前报告通过，但本审计不据此新增结论。
- 未重跑耗时的 MMLU/C-Eval/PIQA、空缓存 v15 VMA、HumanEval、IFEval 或顺序平衡性能；本轮只验证现有 artifact 与代码/报告的一致性。

## 最终补丁复核

P1-1 已关闭：ISA artifact 固定 `paper_exact=false`，acceptance 输出 `DIAGNOSTIC_PROXY/pass=false`，README、HTML、PDF 均不再把 token-Gram 代理计为论文精确通过。P1-3 已关闭：`formal-v3` 为 24 层、五组合、97 个从空缓存生成的预测文件，`formal_run_binding=true`、`cache_bound_existing_unversioned=false`；运行前拒绝未登记 `.pt`，并验证实际模型 shard。

### P0-1 仍部分未关闭：普通攻击 provenance 未与配置指定模型/key 做路径绑定

`E:\AloePri\scripts\build_qwen05b_final_acceptance.py:405,489` 只调用 `verify_run_provenance()`；`E:\AloePri\src\aloepri\evidence.py:123-142` 证明 artifact 指向的某个模型/key/数据/脚本当前存在且 hash 一致，但没有证明其路径就是 config 的 `source_model/output_model/key_dir`。当前 Gate/Attention/IMA/ISA/TFMA/SDA artifacts 的 provenance 实际均属于 v15，因此当前 `NO-GO` 未失真；但把另一 checkpoint/key 生成的、同 schema 且指标更低的正式 artifact 放到配置路径，仍可能通过。应像 `vma_provenance_ok()` 一样把三条期望路径传入并逐项 `same_path()`。

### P1-2 仍部分未关闭：8-run 协议未把 dtype 和 tokenizer 内容纳入强校验

`E:\AloePri\scripts\benchmark_hf.py:126-162` 把 dtype 放在顶层，provenance 只保存 `tokenizer_path` 字符串；`E:\AloePri\scripts\compare_performance.py:61-76` 只校验模型、prompt、key、max token、warmup 和 runtime，没有比较八次运行的顶层 `dtype`，也没有 tokenizer 文件 hash；acceptance 的 `performance_provenance_ok()` 同样未补此检查。当前八个 artifact 实际全部是 `torch.bfloat16`、同一原 tokenizer，ABBA+BAAB、80 对请求和报告/PDF 数字一致，因此当前结果未发现错配；但 validator 仍可接受混合 dtype 或内容不同但路径字段可用的 tokenizer。应把 dtype 和 `model_identity/file_identity` 形式的 tokenizer 指纹写入 provenance，并在 comparison 与 acceptance 两处强制一致。

## 最终窄范围复核（2026-08-07 最新补丁）

本节只复核上一节仍保留的 P0-1 路径绑定和 P1-2 性能运行一致性，并以最新代码覆盖上一节对应的代码状态判断。

### P0-1：代码约束已关闭；现有最终 acceptance JSON 尚未由新代码重建

- `scripts/build_qwen05b_final_acceptance.py:49-70` 的 `scoped_run_provenance_ok()` 先调用 `verify_run_provenance()` 校验模型、key、数据和脚本指纹，再将 `original_model`、`private_model`、`key` 的实际路径逐项与当前 config 的期望路径比较；不应出现的可选输入只要存在也会拒绝。
- Gate-IA、Attention-IA proxy、IMA、ISA、TFMA、SDA 在 `:446-459` 按各自所需输入调用该函数；API 在 `:573-578`、checkpoint 在 `:602-607` 同样使用该函数。因此，把其他 checkpoint/key 的同结构 JSON 放入当前 evidence 路径，不能再通过 provenance gate。
- `tests/unit/test_acceptance_provenance.py` 包含错误 private model、错误 key 和意外可选输入的负向测试。
- 对当前 v15 八份 attack/API/checkpoint artifact 做只读路径核对，所需路径全部等于 `configs/transform/paper_qwen05b_complete.yaml` 的 `source_model`、`output_model`、`key_dir`，不需要的槽位均为 `null`。
- `artifacts/qwen05b-paper-final-acceptance.json` 的时间为 2026-08-06 14:36:06，早于 acceptance 脚本的 2026-08-07 16:57:18；旧 JSON 也没有新版本输出的 `provenance_bound` 字段。因此，路径绑定的代码缺陷已关闭，但该旧 JSON 不能作为“新 gate 已实际执行”的证据，仍需用当前脚本重建最终 acceptance 文件。

### P1-2：新运行链路已约束 dtype/device/tokenizer；现有八个旧运行的 tokenizer 运行时指纹仍未闭合

- `scripts/benchmark_hf.py:148-168` 现在把 `tokenizer_identity`、dtype 和 device 写入每次运行的 provenance；以后新生成的运行 artifact 能保存运行时 tokenizer 文件指纹。
- `scripts/compare_performance.py:77-102` 比较八次运行的 dtype、device，并验证及比较 tokenizer identity；`:125-135` 将三者写入 comparison protocol。最新 comparison JSON 的协议为 BF16、CUDA、ABBA+BAAB，并含 4 个 tokenizer 文件的 SHA-256 指纹。
- acceptance 的 `performance_provenance_ok()` 在 `scripts/build_qwen05b_final_acceptance.py:122-170` 要求 protocol 为 `torch.bfloat16`/`cuda`，验证 tokenizer 指纹，重新读取并绑定 8 个 source artifact，校验模型角色、模型路径、key 路径和运行序号。
- 现有 8 个 source run JSON 的顶层值实际全部是 `torch.bfloat16` 和 `cuda`，模型/key 路径也与 v15 config 一致；但它们均没有 provenance 内的 `tokenizer` 指纹，只保存相同的 `tokenizer_path`。comparison 与 acceptance 在缺少该字段时会从当前路径重新计算指纹（分别见 `compare_performance.py:81-84` 和 acceptance `:144-150`）。这只能证明复核时该路径下 tokenizer 内容一致，不能证明八次运行执行当时使用的 tokenizer 文件内容一致。
- 因而，混合 dtype/device 已被拒绝，未来新运行的 tokenizer 指纹也已闭合；但就现有八个历史运行而言，P1-2 的“每次运行时 tokenizer 文件指纹一致”仍缺直接证据。严格关闭需要用新 `benchmark_hf.py` 重新产生八个运行 artifact，或取消缺失指纹时的路径回填并拒绝旧 artifact。当前单元测试也尚未覆盖混合 dtype、混合 device、缺失/不一致 tokenizer 指纹的负向情形。

## 正式重跑后的最终窄范围复核

本节复核 `artifacts/performance/hf-balanced-v15-v2/`、覆盖后的 performance comparison、当前 acceptance 代码和磁盘上的最终 acceptance JSON。未修改代码，未重跑模型实验。

### P0-1：校验代码已严格闭合；发布用 acceptance JSON 仍未重建

- 当前 `scoped_run_provenance_ok()` 及 attack/API/checkpoint 三处调用仍保持上一节确认的 config 路径绑定，未发现回退。
- 当前 v15 的 Gate-IA、Attention-IA proxy、IMA、ISA、TFMA、SDA、API、checkpoint artifact 所需模型/key 路径与 config 完全一致，不需要的输入槽位为 `null`。
- 但是磁盘上的 `artifacts/qwen05b-paper-final-acceptance.json` 修改时间仍为 2026-08-06 14:36:06，早于 2026-08-07 的 acceptance 补丁及本次性能重跑。文件仍只有旧的两个 p50 performance check，所有 check 均没有 `provenance_bound` 字段；因此它不是当前 `build_qwen05b_final_acceptance.py` 的重建产物。
- 客观结论：P0-1 的代码缺陷已关闭；“最终 acceptance 已用新代码重建”与当前磁盘证据不符，发布证据层尚未关闭。

### P1-2：八次新运行及 comparison 证据已闭合；严格校验器仍有 dtype/device provenance 回退

- `hf-balanced-v15-v2` 的 8 个 `run-*.json` 均各自包含顶层和 provenance 内的 `torch.bfloat16`、`cuda`，包含相同的 4 个 tokenizer 文件 SHA-256 指纹及相同的 `benchmark_hf.py` 文件指纹；上述 tokenizer/script 指纹按当前文件重新计算均有效。
- 覆盖后的 `artifacts/performance/hf-paper-v15-complete-bf16-comparison.json` 修改时间为 2026-08-07 17:19:16。其 8 个 `source_artifacts` 与新目录的 8 个 run 文件集合完全相等，artifact SHA-256 均有效；运行序号为 1 至 8，角色严格为 ABBA+BAAB，protocol tokenizer 与每个 run 的 tokenizer identity 完全相等，comparison script 指纹有效。
- tokenizer 的旧路径回填已从 comparison 和 acceptance 删除：缺少或无效 `provenance.tokenizer` 会直接失败。两处也都验证 benchmark/comparison script 文件指纹。
- `compare_performance.py:78-81` 对 provenance 内缺失 dtype/device 仍使用顶层值回退：`provenance.get("dtype", item.get("dtype"))` 和对应 device。也就是说，当前 8 个新 artifact 本身证据完整，但 comparison validator 仍不会因为单独删除 `provenance.dtype` 或 `provenance.device` 而失败；当前测试中也没有这一负向用例。若“严格闭合”要求这两个运行环境字段必须存在于 provenance，应改为直接读取并要求字段存在。
- 当前旧 acceptance JSON 没有引用本次新 comparison，也没有新 `provenance_bound` 结果。因此，新运行和 comparison 层面的 P1-2 已闭合，但最终 acceptance 发布链路尚未闭合；按“comparison 与 acceptance 均已实际重建并严格拒绝缺字段”的标准，P1-2 仍不能整体标为严格关闭。

## 最终状态纠正与关闭结论

上一节核对的是仓库根部遗留文件 `artifacts/qwen05b-paper-final-acceptance.json`，遗漏了随后生成在正式目录中的 `artifacts/acceptance/paper-qwen05b-v15-complete-bf16.json`。以下结论以磁盘上的最新正式文件为准，并纠正上一节关于“acceptance 未重建”的判断。

### 最新磁盘证据

- 正式 acceptance 为 `artifacts/acceptance/paper-qwen05b-v15-complete-bf16.json`，修改时间 2026-08-07 17:27:05，晚于 acceptance/compare 脚本补丁及 17:25:59 的 comparison；决策仍为 `NO-GO`。
- 该 acceptance 的 `performance.ttft_ms.p50.degradation` 为 `0.002038155823756327`，`provenance_bound=true`。TTFT/TPOT 的 p50、p95、p99 六项均带 `provenance_bound=true`。
- Gate-IA、Attention-IA proxy、IMA、ISA、TFMA、SDA、API、checkpoint 的相关正式 check 均为 `provenance_bound=true`。这表示当前 config 的 source model、private model、key 路径及相应文件指纹已经由新 acceptance gate 实际核验，而不是只在代码中存在。
- `configs/transform/paper_qwen05b_complete.yaml:71` 指向 `artifacts/performance/hf-paper-v15-complete-bf16-comparison.json`；该 comparison 的第一个 source artifact 是 `hf-balanced-v15-v2/run-01-baseline-abba-1.json`，完整 source set 正是此次新跑的 8 个文件。
- `compare_performance.py:78-81` 现为顶层字段与 `provenance.get("dtype")`、`provenance.get("device")` 的直接比较，不再以顶层值作为缺失字段的默认值。缺少 provenance dtype/device 会失败。
- acceptance 的 `performance_provenance_ok()` 同时要求 run 顶层 dtype/device 与 protocol 相等，并要求 `run_prov.dtype/device` 与 protocol 相等；tokenizer 也必须是有效且与 protocol 完全相同的文件指纹，不存在路径回填。

### 最终客观结论

- **P0-1 严格关闭。** 路径/指纹绑定既已在代码中强制，也已在最新正式 acceptance 上实际执行；attack、API、checkpoint 的 provenance 均通过当前 config 绑定。
- **P1-2 严格关闭。** 8 个新运行各自记录且一致的 dtype、device、tokenizer 文件指纹和 benchmark script 指纹；comparison 强制逐项一致并绑定 8 个 source artifact；最新正式 acceptance 再次读取、验 hash、核模型/key/角色/顺序及顶层与 provenance 字段，六个性能 check 均显示 provenance 已绑定。
- 仓库根部的 `artifacts/qwen05b-paper-final-acceptance.json` 仍是旧遗留文件，但不是本次正式重建输出；它不改变上述两项在 `artifacts/acceptance/paper-qwen05b-v15-complete-bf16.json` 上的关闭结论。
