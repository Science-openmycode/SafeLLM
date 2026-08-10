# AloePri 论文实现逐项追溯审计

> 历史审计（2026-08-06）。其中 checkpoint、证据和 Algorithm 1 inverse-family
> 描述已被 v18 替代。当前结论、逐行分类和人工命令以
> `docs/PAPER_CODE_LINE_BY_LINE_AUDIT_0.5B.md`、同名 HTML 及
> `artifacts/audit/paper-formula-checkpoint-v18.json` 为准。

审计对象：`01_client_technical_route.pdf`（arXiv:2603.01499v2）与当前 `Qwen2.5-0.5B-Instruct` 仓库。

审计日期：2026-08-06。

## 1. 状态定义

| 状态 | 判定规则 |
|---|---|
| A：已实现且已在 0.5B 运行 | 有真实 0.5B checkpoint、运行工件或模型级测试 |
| B：已实现且仅完成局部验证 | 有代码和单元/Toy/小规模攻击验证，尚无对应论文协议的 0.5B 完整工件 |
| C：部分实现或协议不同 | 核心功能可运行，但公式解释、接口或实验协议与论文不完全相同 |
| D：未实现 | 仓库中没有对应实现或没有可执行证据 |
| N/A：0.5B 不适用 | 论文覆盖的其他架构或模型规模，Qwen2.5-0.5B 本身不包含该结构 |

状态不折算成百分比。一个模块只有同时具备代码、测试和对应运行证据时才标为 A。

## 2. 总结

| 范围 | 审计结论 |
|---|---|
| 0.5B Dense Qwen 核心离线变换 | 词表置换、Embedding/Head 噪声、Algorithm 1、扩维 checkpoint、Dense Attention、Dense FFN 和 RMSNorm 融合均已实现 |
| 0.5B 在线推理 | HF forward、generate、KV Cache、输入置换、输出逆置换和私有服务均已运行 |
| 与论文完全一致的部分 | 词表映射、噪声尺度、P/Q 形状与相消关系、Embedding/Head 变换、Dense FFN 置换/缩放、自回归混淆 Token 回灌 |
| 当前主要协议差异 | 在线接口直接发送 Token ID；RMSNorm 默认 κ 使用扩维 RMS 修正；BlockPerm 的 γ 采用显式乘法解释 |
| 当前主要未完成实验 | SST2、论文三组 TFMA/SDA 医疗语料、BLEU-4、CosSim、完整 Table 9 VMA 多 seed、完整 IMA 论文数据协议、论文效率协议重复 |
| 0.5B 不包含的结构 | MoE Router、专家置换和 MLA；仓库只有 Toy/DeepSeek 适配验证，不属于 0.5B 结果 |
| 现有覆盖清单状态 | `artifacts/current-implementation-coverage.json` 由旧脚本硬编码生成，仍把多个已实现模块标为 pending，不能继续作为完成状态来源 |

## 3. 论文 Section 5：模型与在线推理逐项审计

| 论文位置 | 论文要求 | 当前实现位置 | 测试或运行证据 | 状态 | 审计结论 |
|---|---|---|---|---|---|
| 5.1 | 生成秘密词表置换 τ 与逆置换 | `src/aloepri/keys/generate.py`、`transforms/vocab.py` | `tests/unit/test_vocab.py`、`test_special_tokens.py` | A | 全词表双射、逆置换和特殊 Token 同步已验证 |
| 5.1 | 输入使用 τ，输出使用 τ⁻¹ | `src/aloepri/client/sdk.py` | `test_client_sdk.py`、`test_real_private_api.py` | A | Token ID 输入输出闭环已实现；真实模型集成测试默认跳过，已有单独运行工件 |
| 5.2 | 对模型全部相关权重执行离线混淆 | `conversion/paper_qwen2.py`、`convert_paper_qwen2_checkpoint.py` | `artifacts/verification/paper-qwen05b-v12-secure-beta8-fp32.json` | A | 真实 0.5B checkpoint 已转换、保存和重载 |
| Algorithm 1 INIT-2 | 从 O(d) 均匀采样 U | `paper_key_matrix._orthogonal` | `test_algorithm_1_shapes_and_identity_for_100_seeds` | A | 通过 Gaussian QR 生成正交矩阵 |
| Algorithm 1 INIT-3 | V∼N(0,1/d)，B=U+λV，计算 B⁻¹ | `make_paper_key_pair` | 100 seed 与 float32 门禁 | A | 另增加 B 条件数门禁 |
| Algorithm 1 INIT-4 | 低秩 E=E₁E₂ | `make_paper_key_pair` | P/Q 形状与恒等测试 | A | 维度为 d×h |
| Algorithm 1 INIT-5 | 低秩 F=F₁F₂ | `make_paper_key_pair` | P/Q 形状与恒等测试 | A | 维度为 h×d |
| Algorithm 1 INIT-6 | 采样 Z∈O(d+2h) | `make_paper_key_pair` | P/Q 形状测试 | A | 扩维为 d+2h |
| Algorithm 1 KEYMATGEN | 构造 C 并生成 P=[B C E]Z | `make_paper_key_pair` | `P@Q≈I`，100 seed | C | 论文对 C 的“列属于 null(Fᵀ)”描述与矩阵形状冲突；代码采用满足 CF=0 的形状一致解释 |
| Algorithm 1 INVKEYMATGEN | 构造 D 并生成 Q=Zᵀ[B⁻¹ F D]ᵀ | `make_paper_key_pair` | `P@Q≈I`，100 seed | C | 代码采用满足 ED=0 的形状一致解释 |
| 5.2.2 Noise | Embedding 噪声标准差为其权重标准差的 αe 倍 | `transforms/paper_noise.py` | `test_paper_noise.py`、真实转换 manifest | A | 独立随机源已实现 |
| 5.2.2 Noise | Head 噪声标准差为其权重标准差的 αh 倍 | `transforms/paper_noise.py` | `test_paper_noise.py`、真实转换 manifest | A | 与 Embedding 使用不同 seed |
| 5.2.2 | W̃embed=ΠW★embedP̂embed | `transform_embedding` | `test_embedding_and_head_token_covariance` | A | Token 行置换和残差扩维同时实现 |
| 5.2.2 | W̃head=Q̂headW★headΠᵀ | `transform_head` | `test_embedding_and_head_token_covariance` | A | PyTorch `[vocab,hidden]` 存储方向已换算 |
| Algorithm 2-1 | 每个 head 采样二维旋转 Rqk | `make_rope_commuting_map` | `test_qwen_structural.py` | A | 针对 Qwen 分半存储的 RoPE 坐标生成 |
| Algorithm 2-2 | 成对采样 Q/K 缩放 Hqk 与 Hqk⁻¹ | `make_attention_key` | `test_qwen_structural.py`、v13 配置 | A | 使用对数均匀正缩放，论文没有给出缩放分布 |
| Algorithm 2-3/9-19 | Dynamic BlockPerm(β,γ,ζ,mblocks) | `make_dynamic_rope_block_order` | `test_qwen_structural.py` | C | 原伪代码不终止且有边界问题；`paper-distribution-boundary-corrected` 修复控制流但不使用 γ，`gamma-corrected` 再使用 `softmax(γ·offset)` |
| Algorithm 2-4 | Uvo∼N(0,1/dhead) | `make_attention_key` | 结构单测、v13 配置 | A | 增加条件数筛选保证可逆和数值稳定 |
| Algorithm 2-5 | 为 Q/K/V/O 采样输入输出 P/Q | `convert_qwen2_modules` | `test_projection_identities`、真实 checkpoint | C | 论文未写明是否共享同一次 Algorithm 1 Init；当前 0.5B 明确使用全局残差 P/Q，属于兼容且保守的工程解释 |
| Algorithm 2-6/7 | Q/K 互逆变换，V/O 的 U/U⁻¹ 变换 | `transform_attention` | `test_qwen_structural.py`、Prompt 回归 | A | 权重存储方向与论文公式已换算 |
| 5.2.3 | τkv 与 τgroup 的 GQA head 置换 | `make_attention_key`、`transform_attention` | GQA 单测、Qwen 0.5B 真实转换 | A | Qwen2.5-0.5B 为 GQA，已覆盖目标架构 |
| 5.2.3 | MHA/MQA 直接适配 | 通用 head 分组代码 | 无独立 MHA/MQA 模型工件 | B | 代码路径可表达，尚未单独运行论文模型 |
| 5.2.3 | MLA 低秩 Q/K 与解耦 RoPE 变换 | `transforms/deepseek_v3.py`、ToyMLA | `test_toy_models.py`、`test_deepseek_v3_adapter.py` | N/A | 0.5B Qwen 不含 MLA；仅做了 Toy/适配器验证 |
| 5.2.4 | Dense FFN 的 Gate/Up 同置换与缩放、Down 逆变换 | `transform_mlp_scaled` | `test_qwen_structural.py`、真实 checkpoint | A | SwiGLU 配对关系保持 |
| 5.2.4 | MoE Router 列置换、专家顺序同步 | `toy_architecture.py`、`deepseek_v3.py` | ToyMoE 与 DeepSeek adapter 单测 | N/A | 0.5B Qwen 为 Dense；没有 0.5B Router |
| 5.2.5 | 融合原 RMSNorm 对角权重到相邻线性层 | `transform_input_projection`、`transform_head` | `test_projection_identities` | A | input/post/final norm 均进入转换 |
| 5.2.5 | κ=E[‖xP‖/‖x‖] | `frobenius_norm_ratio_proxy` | `test_rms_kappa_includes_expanded_dimension_correction` | C | 当前仅实现二阶矩代理，不等于一般矩阵下的精确期望；工程候选使用适配 RMS 分母维度的 `analytic_rms_kappa` |
| 5.2.5 | 混淆 RMSNorm 权重为 κ·1 | `convert_qwen2_modules` | 真实 checkpoint、RMS 校准工件 | A | 每层可覆盖 κ，支持校准 |
| 5.3 | 客户端 tokenize→τ→detokenize，服务端再次 tokenize | 未按论文文本接口实现 | 无文本乱码 round-trip 工件 | C | 当前直接传置换后的 Token ID，避免 tokenizer 二次编码改变序列 |
| 5.3 | 服务端混淆模型生成混淆响应 | `serving/hf_runtime.py`、`serving/app.py` | API、流式 API、Prompt 回归 | A | 服务端不持有 τ⁻¹ |
| 5.3 | 客户端使用秘密映射恢复响应 | `TokenKey.decode_ids/decode_stream` | SDK 与 API 测试 | A | 支持逐 Token 流式恢复 |
| 5.3 Remark | 在线先执行指数机制等 Token perturbation | 无 | 无 | D | 论文把它写为可选兼容能力；当前未实现 |
| 5.3 | 自回归把混淆输出作为下一 Token | `PrivateHFRuntime.iter_token_ids` | generate 与真实 Prompt 回归 | A | Decode 全程保持混淆词表坐标 |
| 5.3 | KV Cache 与推理框架兼容 | 自定义 Qwen 模型、HF runtime、vLLM plugin | cache 单测、HF 工件、vLLM 预检 | C | HF v14 已运行；vLLM 插件和早期功能模型曾运行，v14 尚未完成完整 vLLM 回归 |

## 4. 论文 Section 5.4、Section 6：分析项审计

| 论文位置 | 分析要求 | 当前证据 | 状态 | 审计结论 |
|---|---|---|---|---|
| 5.4 Embedding/Head | 验证 Token 与 logits 的协变关系 | `test_embedding_and_head_token_covariance` | A | 无噪声代数关系已验证 |
| 5.4 Attention | QK score 近似与 V/O 输出协变 | `test_qwen_structural.py`、Prompt 回归 | B | 局部等价和端到端输出已测；没有逐层 eC 统计表 |
| 5.4 FFN | FFN 输出等于明文输出乘 Pdown | FFN 结构测试、Prompt 回归 | B | 变换等价已测；没有独立逐层误差工件 |
| 5.4 RMSNorm | RMSNorm 近似误差 enorm_C | RMS κ 测试与校准工件 | B | κ 和模型结果已测；未逐层报告论文符号下的误差上界 |
| 5.4 Putting Together | 组合各模块得到整模型协变 | 20 Prompt、五基准、forward/generate/cache | A | 端到端运行已完成 |
| 5.4 | 计算 Lipschitz 常数和 eAloePri_C 上界 | 无 | D | 当前只测经验误差，没有实现论文给出的解析上界计算器 |
| Section 6 | RmDP、RsmDP 与 Theorem 4 的隐私预算计算 | 无 | D | 论文为理论推导；仓库尚无 ε、奇异值和距离输入的计算/验证工具 |

## 5. 论文攻击逐项审计

| 论文攻击 | 论文协议 | 当前实现与工件 | 状态 | 差距 |
|---|---|---|---|---|
| Direct weight matching | 直接比较明文与混淆权重 | `run_direct_attack.py`、`direct-full-*.json` | A | 已运行 |
| VMA RowSort | 对 Y=Z₁XZ₂ 行排序后匹配 | `rowsort_nearest*`、`run_vma_rowsort.py` | A | RowSort 算子和分层投票已实现 |
| VMA Table 9：WeWh | 多层权重组合并投票 | `run_vma_pupa.py`、三 seed grid | A | 已在 PUPA Token/PII 上正式统计 |
| VMA Table 9：WeWquery(WeWkey)ᵀ | Q/K 组合 | `run_vma_rowsort.py` | B | 有实现和单次工件，未进入三 seed PUPA 主表 |
| VMA Table 9：WeWgate | Gate 组合 | `run_vma_pupa.py`、三 seed grid | A | 已在三 seed 中统计 |
| VMA Table 9：WeWup | Up 组合 | `run_vma_rowsort.py` | B | 有实现，未进入三 seed PUPA 主表 |
| VMA Table 9：WdownWh | Down/Head 组合 | `run_vma_rowsort.py` | B | 有实现，未进入三 seed PUPA 主表 |
| VMA Table 9：WeWrouter | MoE Router 组合 | 无 0.5B 路径 | N/A | 0.5B 不含 Router |
| Gate-IA | 多层 Gate 加权 Embedding 均值不变量 | `run_gate_ia.py`、`gate-ia-*.json` | A | 已运行 |
| Attn-IA | Q/K RoPE block 数学不变量 | `run_attn_ia.py`、`attn-ia-*.json` | A | 已运行 |
| ISA | 根据 hidden state loss 优化输入 | `run_isa_hidden_state.py`、`isa-hidden-state-20x100.json` | A | 20 prompts、125 tokens、100 steps，TTRSR=0 |
| NN | 最近邻恢复 Embedding | `cosine_nearest_sample`、privacy suite | B | 有实现和旧工件，未按论文 Table 2 协议单独形成正式 0.5B 表 |
| IMA | 2 层、8 head 反演模型 | `train_ima_smoke.py`、`ima-independent-8192x2000.json` | B | 架构、8192 Token、2000 step 已运行；训练语料与论文未完全对齐 |
| TFMA | 三种先验知识、Top-10/Top-100 | `run_tfma_curve.py`、`tfma-distribution-aware-curve.json` | C | 攻击与曲线已实现；未使用 CCI3/MedDialog/Huatuo26M-Lite 三组论文协议 |
| SDA | recurrence encoding 与 Transformer 解码 | `recurrence.py`、`train_sda_smoke.py`、`sda-mmlu-10000x2000.json` | C | 算法链路已运行；数据使用 MMLU 文本，不是论文医疗语料，未形成论文 BLEU-4 对照 |
| Known plaintext | 观测明密文对恢复映射 | `run_known_plaintext_curve.py`、对应 JSON | A | 属于项目补充攻击，不是论文 Table 2–5 的核心行 |

## 6. 论文实验逐项审计

| 论文实验 | 当前 0.5B 情况 | 状态 | 后续动作 |
|---|---|---|---|
| SST2 | 未运行 | D | 增加 SST2 明文/混淆成对评测 |
| MMLU | 14,042 题已运行 | A | 保存逐题配对正确性以计算差值区间 |
| C-Eval | 1,346 题已运行 | A | 保存逐题配对正确性 |
| HumanEval | 164 题已运行 | A | 保留沙箱执行日志与 pass@1 |
| PIQA | 1,838 题已运行 | A | 保存逐题配对正确性 |
| IFEval | 541 prompts / 834 instructions 已运行 | A | 同时保留 prompt/instruction strict |
| PUPA VMA | 3432 Token occurrence、664 PII 单元、三 seed | A | 把 Table 9 其余 Dense 组合加入同一三 seed 协议 |
| CCI3→Huatuo26M TFMA/SDA | 未按该数据组合运行 | D | 获取语料并按 Table 5 运行 |
| MedDialog→Huatuo26M TFMA/SDA | 未按该数据组合运行 | D | 获取语料并按 Table 5 运行 |
| Huatuo26M→Huatuo26M TFMA/SDA | 未按该数据组合运行 | D | 划分不重叠子集后运行 |
| BLEU-4 | 主结果未计算 | D | 对 VMA/SDA 恢复文本计算 BLEU-4 |
| CosSim | 主结果未计算 | D | 固定与论文一致的句向量模型后计算 |
| Top-k | TFMA 曲线已有局部结果 | C | 输出论文 Top-10、Top-100 表 |
| 默认生成参数 0.65/20/0.95 | 正式评测器与 Prompt 回归未统一使用该采样协议 | C | 增加论文采样协议的独立复现实验，不覆盖确定性验收 |
| max-seq-len=8192、BF16 | 0.5B 正式候选主要为 FP32，本地精度工件 dtype 不完全统一 | C | 生成统一 BF16/FP32 对照工件 |
| 默认隐私参数 1.0/0.2/0.3/128/8/1e3 | v13 已转换并运行 Prompt/VMA；五基准使用低噪声 v11 | C | 在默认参数下补全五基准，或搜索满足联合门禁的 0.5B 参数 |
| Table 4 ISA 消融 | 已有 ISA 与若干机制工件，未形成 Noise→KeyMat→Head&BlockPerm 三行同协议表 | C | 按 Table 4 固定数据和 seed 重跑三档 |
| 离线转换时间 Table 6 | 0.5B 有转换流程，未形成稳定重复时间表 | C | 独立重复并报告均值/区间 |
| 在线 TTFT/TPOT Table 7 | HF eager、20 串行请求已运行 | C | 按论文 17 输入、100 输出、并发 1/4 和 vLLM 重跑 |
| Figure 5：h=128/256/384/512 | 未在 0.5B 形成完整 h 曲线 | D | 固定模型与请求分布，运行四个 h |

## 7. 工程兼容性审计

| 项目 | 证据 | 状态 | 结论 |
|---|---|---|---|
| 自定义 HF 模型注册 | `models/modeling_aloepri_qwen2.py` | A | save/reload/forward/generate/cache 已测 |
| Checkpoint 分片 | 转换器与 streaming 转换脚本 | A | 支持 shard 写入 |
| `.partial`、断点续跑、原子提交 | `conversion/vocab_checkpoint.py`、streaming 脚本 | A | 有恢复单测 |
| SHA-256 与 manifest | `manifest.py`、`conversion/verify.py` | A | 有秘密字段扫描与校验 |
| HF 私有服务 | `serving/app.py`、`hf_runtime.py` | A | 非流式与 SSE 流式测试通过 |
| vLLM 注册 | `serving/vllm_plugin.py`、`vllm_qwen2.py` | C | 插件存在并有早期功能工件；正式候选未完成最终回归 |
| SGLang | 无 | D | 论文宣称兼容，当前仓库未做 SGLang 接入 |
| P/D disaggregation | 无 | D | 0.5B 本机未验证 |
| 多节点/异构 xPU | 无 0.5B 对应运行 | N/A | 不属于本机 0.5B 验证范围 |

## 8. 必须补齐的优先顺序

| 优先级 | 项目 | 完成条件 |
|---:|---|---|
| P0 | 默认论文参数下的五项精度 | 使用 1.0/0.2/0.3/128/8/1e3，统一 dtype，生成五项完整结果 |
| P0 | 完整 Dense Table 9 VMA 三 seed | WeWh、QK、Gate、Up、Down/Head 使用相同 PUPA 样本、候选集和 seed |
| P0 | RMSNorm 双 κ 对照 | 论文原 κ 与扩维 RMS κ 在相同 checkpoint、dtype、Prompt 和五基准上对照 |
| P1 | TFMA/SDA Table 5 | 三种论文数据先验，输出 Top-10、Top-100、BLEU-4 |
| P1 | Table 4 ISA 消融 | Noise、Noise+KeyMat、完整 Attention 混淆三档同协议运行 |
| P1 | 论文在线文本接口 round-trip | 验证 τ 后 detokenize/re-tokenize 是否逐 Token 保持；不通过则正式记录 Token ID API 差异 |
| P1 | vLLM 正式候选性能 | 17 Token 输入、100 Token 输出、并发 1/4，报告 TTFT/TPOT |
| P2 | SST2、CosSim、BLEU-4 | 补齐论文全部评价指标 |
| P2 | RmDP 与误差上界工具 | 输入权重奇异值、α、σ、P 和层常数，输出论文 ε 与 eC 账本 |

## 9. 本次审计运行记录

```powershell
Set-Location E:\AloePri
uv run python scripts/build_implementation_coverage.py
uv run pytest -q
```

本次测试结果：`62 passed, 1 skipped`。跳过项为需要显式设置 `ALOEPRI_RUN_MODEL_TESTS=1` 的真实 0.5B API 集成测试；真实 0.5B 的转换、精度、隐私与性能结果另有 `artifacts/` 工件。

## 10. 最终结论

当前仓库已经实现论文在 Dense Qwen2.5-0.5B 上的主模型变换和在线推理闭环。不能标为“论文每一步均已完成”的部分集中在四类：论文文字接口、RMSNorm/BlockPerm 的解释差异、完整攻击协议、论文实验环境与数据集。MoE、MLA、多节点和异构 xPU 属于 0.5B 架构之外，不计入 0.5B 模型功能缺失，但必须与 0.5B 结果分开陈述。
