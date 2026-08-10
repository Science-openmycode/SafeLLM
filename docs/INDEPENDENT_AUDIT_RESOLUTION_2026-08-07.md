# 独立审计整改记录

基线审计：`docs/INDEPENDENT_SUBAGENT_AUDIT_2026-08-07.md`

第二轮审计：`docs/INDEPENDENT_SUBAGENT_REAUDIT_2026-08-07.md`

## 第二轮复核补丁

| 第二轮问题 | 修复 | 证据 |
|---|---|---|
| P0：验收证据未绑定 checkpoint/key/dataset/hash，VMA 可缺组合或层 | 验收器验证模型 manifest 与实际 shard、key、数据文件、入口脚本、源 artifact、cache manifest 和 SHA-256；VMA 固定要求 16,384 候选、五组合、24 层和 97 个登记文件；加入缺组合、截层和伪 comparison 负向测试 | `src/aloepri/evidence.py`；`scripts/build_qwen05b_final_acceptance.py`；`tests/unit/test_acceptance_provenance.py` |
| P1：ISA token-Gram 不适用于一般矩形 P，却被记为通过 | artifact 固定 `paper_exact=false` 并写明不变量失效原因；验收状态改为 `DIAGNOSTIC_PROXY` | `scripts/run_isa_hidden_state.py`；`artifacts/privacy/isa-paper-v15-complete-bf16.json` |
| P1：性能固定明文→v15，comparison 不验证请求和环境身份 | 改为 8 个独立进程 ABBA+BAAB；80 对请求按 `run_id`/prompt hash 配对；重新运行 v2 时逐次记录并验证模型、key、prompt、tokenizer、dtype、device、benchmark 脚本和执行序号 | `scripts/run_balanced_hf_performance.ps1`；`artifacts/performance/hf-balanced-v15-v2/` |
| P1：v15 VMA 使用历史绑定缓存，允许未登记 `.pt` | 新预测写完即原子登记；manifest 拒绝所有未登记 `.pt`；私有模型先验证 manifest 和实际 shard；v15 在全新 v3 cache 上重算 97/97 文件 | `artifacts/privacy/cache/v15-c16384-formal-v3/cache_manifest.json`；`artifacts/privacy/vma-pupa-paper-v15-complete-bf16-c16384.json` |
| P2：CLI 仍写 paper-faithful | 改为 `corrected-paper d+2h model` | `scripts/convert_paper_qwen2_checkpoint.py` |
| P2：仓库树高估 `attacks/` 包内容 | 明确基础算子在 `src/aloepri/attacks/`，正式攻击入口在 `scripts/` | `README.md`；HTML/PDF 报告 |

| 编号 | 审计问题 | 处理结果 | 证据 |
|---|---|---|---|
| P0-1 | 验收脚本只检查攻击文件存在 | 已修复。Gate-IA、Attention-IA、IMA、ISA、TFMA、SDA 分别校验 schema、规模、攻击者是否使用目标 key、论文精确标记和指标阈值；占位文件不能通过 | `scripts/build_qwen05b_final_acceptance.py`；`artifacts/acceptance/paper-qwen05b-v15-complete-bf16.json` |
| P0-2 | v15 被写成论文默认算法 | 已修复。统一写为「论文数值超参数的 corrected-paper 实现」；验收拆成 `paper_numeric_hyperparameters_match=true` 和 `paper_literal_algorithm_match=false` | `README.md`；HTML；acceptance JSON |
| P1-1 | checkpoint 转换命令不接受 `--config` | 已修复。README 和 HTML 改为 `convert_paper_qwen2_checkpoint.py` 的完整实参；`--help` 已验证 | `README.md` 4.1 至 4.3；HTML 第 4 节 |
| P1-2 | PIQA 命令缺 tokenizer 和私有 key | 已修复。明文与私有命令分开，私有命令带 `--tokenizer` 和 `--key` | `README.md` 5.3 |
| P1-3 | RMS estimator 与论文 L2 公式不同 | 已修复表述与 provenance。报告并列 L2 和 RMS 公式；历史 `paper-expectation` 标签注明旧字段名；校准记录 prompt、模型、P、脚本和锁文件 SHA-256 | `README.md` 1.6、4.1；HTML 第 1 节；校准 JSON；E07 PDF |
| P1-4 | Attention 原式和修正式未分开 | 已修复。报告列出论文的 K 侧转置式及 `qZ^2k^T` 反例，v15 明确使用 Q/K 同侧 Z | `README.md` 1.4；HTML 第 1 节；E06 PDF |
| P1-5 | BlockPerm gamma 和 Qwen RoPE 修正边界不清 | 已修复。v15 标明 `gamma-corrected` 与 `qwen-actual`，不记为论文 literal | config、acceptance JSON、E03/E10 PDF |
| P1-6 | VMA cache 未绑定输入证据 | 已修复。cache manifest 绑定模型、key、数据、候选 ID、层、组合和算法版本，并校验每个缓存文件 SHA-256 | `scripts/run_vma_pupa.py`；`tests/unit/test_vma_pupa.py`；两个 cache manifest |
| P1-7 | CUDA HF runtime 强制 BF16 | 已修复。运行时支持 `auto/float32/bfloat16`；CUDA 的 `auto` 保留 checkpoint dtype，CPU 的 `auto` 使用 FP32 | `src/aloepri/serving/hf_runtime.py`；`scripts/serve_private.py` |
| P1-8 | Wilson 区间忽略文本内相关和 key 间波动 | 已修正报告口径。区间标为 occurrence/PII 单元层面的描述性区间，不代表文本聚类或 key 间波动。多 key 和 cluster bootstrap 尚未运行 | `README.md` 7.5；HTML 第 6 节 |
| P1-9 | `paper_parameter_match` 名称误导 | 已修复。字段改为 corrected profile、论文数值超参数和论文 literal 三项 | acceptance JSON |
| P2-1 | RMS 校准 provenance 不完整 | 已修复。确定性 P 由 seed/h/lambda 直接生成，消除校准与最终 key 的循环依赖；49 个 κ 与历史 v15 最大绝对差为 0 | `scripts/calibrate_paper_rms.py`；校准 JSON |
| P2-2 | README 目录树和能力描述有偏差 | 已修复配置实际路径；转换、攻击和评测入口逐项列出 | `README.md` 第 2 节 |
| P2-3 | manifest 不拒绝额外文件 | 已修复。目录中未列入 manifest 的文件返回 `unlisted:*` 并使验证失败 | `src/aloepri/conversion/verify.py`；`tests/unit/test_manifest_verify.py` |
| P2-4 | HTML 完成范围过宽 | 已修复。结论限定 0.5B corrected-paper 已运行模块；HumanEval、IFEval、论文医疗语料和 vLLM 单列 | HTML 第 8、9 节 |

## 新增实测

| 项目 | 结果 |
|---|---|
| Gate-IA | 512 token、24 层；Top-1 17.383%，未通过 15% 门槛 |
| Attention-IA 代理 | 256 token、4 层；Top-1 7.031%；`paper_exact=false` |
| IMA | 8192 token、2000 步；恢复率 0%；协议等价未确认 |
| ISA | 20 prompt、125 token、100 步；TTRSR 0%；两个模型分阶段加载 |
| SDA | MMLU 10,000/2,000、2000 步；BLEU-4 13.120，未通过 2.5 门槛 |
| HF 性能 | 8 个独立进程、80 对请求、每个 100 token；TTFT p50 与 TPOT p50/p95/p99 通过联合门禁，TTFT p95/p99 未通过 |
| 显存 | HF v15 推理峰值 1,649,150,976 bytes；ISA 两阶段峰值分别为 1,633,619,456 和 1,551,377,408 bytes |
| IFEval v15 | 541 prompt / 834 instruction 全量 BF16；四项下降 14.603-18.465 pp；四项均未通过；正式 provenance 已绑定 |
| HumanEval v15 | 164 题 BF16；42/164→0/164；-25.610 pp，95% CI [-32.317, -18.902]；正式 provenance 已绑定 |

## 仍未完成或无法按论文精确口径确认

| 项目 | 状态 |
|---|---|
| 论文医疗语料 TFMA/SDA | 语料和切分未公开；当前只有 MMLU 代理 |
| vLLM 正式性能 | 未运行 |
| 多 key 置信区间 | 未运行；现有 VMA 使用一个 key |

2026-08-08 重建的 acceptance 为 `NO-GO`：57 项检查，41 项通过、16 项失败、缺失证据 0 项。失败项包括 MMLU、C-Eval、PIQA、IFEval 四项、HumanEval、Gate-IA、SDA、prefill top-1 和部分 TTFT 分位数。该结论直接由逐项检查生成。

## 最终闭环复核

| 项目 | 最终状态 | 证据 |
|---|---|---|
| 攻击/API/checkpoint 与当前 config 的模型、key 绑定 | 严格关闭 | `scoped_run_provenance_ok()` 同时验证文件指纹和期望路径；最新 acceptance 已重建 |
| 性能 dtype/device/tokenizer/script 绑定 | 严格关闭 | `hf-balanced-v15-v2/` 的 8 个运行均在运行时记录指纹；comparison 与 acceptance 不允许字段回填 |
| 独立复核 | 通过上述两项窄范围复核 | `docs/INDEPENDENT_SUBAGENT_REAUDIT_2026-08-07.md` 最终复核段 |
