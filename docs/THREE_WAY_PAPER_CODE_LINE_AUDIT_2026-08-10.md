# AloePri 三方逐行复现审计

审计日期：2026-08-10

本地代码：`eaa5da9fe8d49023583b60eb755f4995a86c9d91`

对方代码：`sheng1feng/Aloepri@60e8ea3cc04353b7a0058e9c86d67461c7d25763`

目标模型：`Qwen2.5-0.5B-Instruct`

## 1. 结论

### 1.1 功能层结论

当前 v31 已经完成并实测的内容是：

- 全词表 `tau` / `inverse_tau`；
- Embedding、LM Head 噪声和词表同步变换；
- 论文 Algorithm 1 的 `d -> d+2h` 矩形 `P/Q`，并修正了论文中 `C/D` 的维度错误；
- Qwen 稠密 Attention 的 Q/K/V/O、GQA、head/group permutation、`beta=8` BlockPerm；
- Qwen 实际 RoPE 布局、KV Cache、prefill/decode；
- FFN gate/up/down 的中间维置换和缩放；
- Residual 坐标一致性；
- 经过数学修正的 exact-metric RMSNorm；
- HF、vLLM、SGLang 三后端 32-token greedy 一致性；
- 客户端 Chat Template、本地置换、服务端私有 checkpoint、SSE 逐 token 恢复；
- 在线/离线密钥拆分、manifest、SHA-256 和服务器包扫描。

这说明“在 Qwen2.5-0.5B 上运行一条真实的客户端到混淆模型再到客户端回答的链路”已经成立。

当前不能写成“论文 v2 功能 100% 原样复现”，原因有六项：

1. v31 的 RMSNorm 是 `G = Q Q^T` 的全矩阵算子，不是论文写的标量 `kappa` 方案。
2. v31 为保证 `beta=8` BlockPerm 与 RoPE 的函数一致性，运行时使用 `B^T R(t) B`，不是原始 Qwen RoPE 算子。
3. 论文 v2 的序列级 M1 只实现了小词表精确枚举；Qwen 在线模式是长度一机制逐 token 组合。
4. Attention-IA 使用修正后的代理，因为论文公式维度不成立；不能计作论文原攻击。
5. IMA、ISA、TFMA、SDA 尚未按论文数据规模和训练协议完成。
6. Qwen2.5-0.5B 不含 MoE 和 MLA，不能用该 checkpoint 验证这两类结构。

### 1.2 性能层结论

最新 v31 没有完成精度、隐私和延迟的正式验收。当前工件只支持功能结论，不支持下列结论：

- 各精度任务下降不超过 3.5 个百分点；
- TTRSR、PIIRSR、BLEU-4、CosSim 达到 PPT 阈值；
- TTFT、TPOT 劣化不超过 15%；
- vLLM/SGLang 在标准模型类、标准算子、Tensor Parallel 下无需修改即可运行。

### 1.3 对方仓库结论

对方仓库不是甲方 PDF 对应的完整参考实现：

- 对方附带论文是 arXiv v1，甲方 PDF 是 arXiv v2；
- v2 新增 RmDP 和 M1，对方源码没有相应实现；
- 对方仓库未提交 `artifacts/` 和 `outputs/`，README 中的完成声明不能由当前 clone 重新计算；
- 对方 IMA 比我们的 IMA smoke 更接近论文；
- 对方 ISA、TFMA、SDA 仍是代理协议；
- 对方 IA 只有计划模板，没有实际 IA；
- 对方 BlockPerm 后仍直接调用普通 RoPE，没有处理跨频率块置换，代码层面不如我们的 v31 修正；
- 对方 Stage-K correctness 的通过条件是生成完全匹配率大于 0，而不是等于 1。

## 2. 审计输入与可复查身份

| 输入 | 身份 | SHA-256 / Commit |
|---|---|---|
| 甲方论文 | `01_client_technical_route.pdf`，28 页，arXiv:2603.01499v2，2026-03-30 | `A5FD16662E1AC2FEB55952940EDD0FDF9D4007DA22BD2F0EACB5F5EA41DC4748` |
| 甲方 PPT | `02_client_project_brief.pptx`，7 页 | 以仓库原文件为准 |
| 本地实现 | 当前 Git commit | `eaa5da9fe8d49023583b60eb755f4995a86c9d91` |
| 对方仓库 | `tmp/upstream_aloepri` | `60e8ea3cc04353b7a0058e9c86d67461c7d25763` |
| 对方附带论文 | 31 页，arXiv:2603.01499v1，2026-03-02 | `FC91199CC848F00B88422CAD45F450C33A6814240F3E41BF2CC55E30A5E9CDE9` |
| 最新本地 checkpoint | `data/packages/qwen05b-product-v31-blockperm8` | manifest `a43414eeedcaff94eb81481df9c397a5e483f78ece5f517e0dc557c0511b3ed0` |
| 最新本地在线密钥 | `data/keys/qwen05b-product-v31-blockperm8-online` | manifest `fe973adee236fcb902291870194508c394b2e557e608fac3fdbd813eb0f18c2c` |

## 3. v1 与 v2 论文差异

| 内容 | 对方仓库的 v1 | 甲方给定的 v2 | 对复现的影响 |
|---|---|---|---|
| 标题 | Collaborative Obfuscation | Covariant Obfuscation | 不是同一个最终稿 |
| 安全分析 | PAC Privacy、互信息、静态攻击成功率上界 | RmDP、排列度量、M1、分段隐私预算 | 对方安全实现不能覆盖 v2 |
| 数据扰动 | 仅在 remark 中说明可结合 token perturbation | 明确给出序列级指数机制 M1 | v2 需要新增在线 M1 |
| 定理编号 | v1 有额外的隐私优势定理 | v2 重排 composition theorem，并新增 RmDP 定理 | 文档引用必须按 v2 重做 |
| 实验章节 | Section 6 | Section 7 | 对方文档引用多处针对旧稿 |
| Attack 附录 | Appendix F | Appendix D | 代码注释需按 v2 重新核对 |

对方代码中搜索 `rmdp`、`epsilon1`、`epsilon2`、`exponential mechanism` 和 M1 没有命中。我们的 v2 路径位于：

- `src/aloepri/privacy/rmdp.py:58-116`：小词表序列级 M1 精确枚举；
- `src/aloepri/privacy/rmdp.py:128-160`：Qwen 可执行的长度一机制逐 token 组合；
- `src/aloepri/privacy/rmdp.py:163-212`：v2 Theorem 4 的预算计算器。

## 4. 状态定义

| 状态 | 含义 |
|---|---|
| `EXACT` | 与甲方 v2 的可执行公式一致 |
| `CORRECTED` | 论文公式存在明确错误，代码使用已推导的修正式 |
| `SUBSTITUTE` | 为产品可运行采用工程替代，不是论文原路径 |
| `PROFILE` | 方法已实现，但 v31 使用了不同参数或运行配置 |
| `PROXY` | 只实现了同类攻击或缩小实验，不能填论文原表 |
| `N/A-0.5B` | Qwen2.5-0.5B 不具备该结构 |
| `NOT-RUN-v31` | 有工具或旧结果，但没有与 v31 checkpoint 绑定的正式工件 |

这些状态不相加为“完成率”。逐行账本当前覆盖 25 个本地核心文件、5,108 条可审计行，分类计数为：

| 分类 | 行数 |
|---|---:|
| `PAPER_EXACT` | 413 |
| `PAPER_CORRECTED` | 481 |
| `ENGINEERING_SUBSTITUTE` | 940 |
| `PAPER_UNDERSPECIFIED` | 850 |
| `PROFILE_DEPENDENT` | 251 |
| `PROXY_NOT_PAPER_EXACT` | 606 |
| `NUMERICAL_STABILIZATION` | 2 |
| `VERIFICATION` | 1,565 |

逐物理行源码、符号、状态、论文章节和测试文件见 `docs/PAPER_CODE_LINE_BY_LINE_AUDIT_0.5B.html`。这里的 5,108/5,108 表示“每行已分类”，不表示“正确率 100%”。

## 5. 论文完整流程的三方追踪

| 步骤 | 论文/PPT 要求 | 我们的代码 | 对方代码 | 当前判定 |
|---|---|---|---|---|
| 1 | 客户端保留明文和秘密映射 | `client/sdk.py:122-167` | 仅离线脚本加载 `client_secret.pt` | 我们更完整 |
| 2 | 本地 Chat Template 和分词 | `client/sdk.py:248-261` | `model_loader.py:60-82` | 双方有 |
| 3 | 可选 M1 扰动 | `privacy/rmdp.py:128-160` | 无 | 我们为 `SUBSTITUTE` |
| 4 | 使用 `tau` 混淆 token | `transforms/vocab.py:41-45` | `transforms.py:8-10` | 双方有 |
| 5 | 混淆 Embedding | `paper_qwen2.py:57-59,92-109` | `keymat_embed_head.py:25-39,92-119` | 双方有；对方不扰动特殊 token |
| 6 | Algorithm 1 生成 `P/Q` | `paper_key_matrix.py:97-169` | `keymat.py:82-145` | 双方均修正论文维度 |
| 7 | Attention Algorithm 2 | `qwen_structural.py:43-145,238-296` | `stage_h_attention_static.py:114-187` | 我们更接近 v2，均非字面原式 |
| 8 | RoPE、GQA、BlockPerm | `qwen_structural.py:22-218`；`modeling_aloepri_qwen2.py:115-242` | `stage_h_attention_static.py:145-177,258-283` | 对方缺同步 RoPE 修正 |
| 9 | FFN gate/up/down | `qwen_structural.py:298-323` | `obfuscate_ffn.py:57-109`；`stage_g_ffn.py:68-112` | 我们离线融合更完整 |
| 10 | RMSNorm 与 Residual | `paper_qwen2.py:99-170`；`modeling_aloepri_qwen2.py:20-46` | `stage_g_norm.py:9-45` | 双方 exact-metric 都偏离论文标量方案 |
| 11 | MoE router/expert | `transforms/deepseek_v3.py:110-123`，未用于 Qwen | 无 Qwen 实测 | `N/A-0.5B` |
| 12 | MLA | `transforms/deepseek_v3.py:37-108`，未用于 Qwen | 无 Qwen 实测 | `N/A-0.5B` |
| 13 | 服务端只持私有模型 | `packaging.py:124-261` | release 同时给 server/client 目录，无服务器包扫描 | 我们更完整 |
| 14 | 私有推理和 KV Cache | `hf_runtime.py:65-116` | 多数生成循环每步重算完整前缀 | 我们已实测 cache |
| 15 | 客户端恢复输出 | `client/sdk.py:297-320,465-483` | `transforms.py:13-20` 和推理脚本 | 双方有；我们有网络闭环 |
| 16 | vLLM/SGLang | 自定义模型插件 | 仅 HF 兼容 checkpoint 路线；无 SGLang | 我们功能更完整但不满足“无需修改” |
| 17 | 精度实验 | 工具存在，v31 未跑 | 仓库未提交结果工件 | 双方均未形成可复算证据 |
| 18 | VMA/IA/ISA/IMA/TFMA/SDA | 部分正式、部分代理 | 部分正式、部分代理、IA 未实现 | 均未完整复现 v2 |
| 19 | TTFT/TPOT/吞吐 | 工具存在，v31 未跑 ABBA | 无当前工件 | 均未验收 |
| 20 | RmDP 证明与预算 | v2 计算器 + 替代 M1 | 无 | 我们部分完成 |

## 6. 逐文件、逐行关键结论

### 6.1 词表置换与特殊 token

论文要求：对词表索引应用秘密排列 `tau`，Embedding 行和 Head 输出同步排列，客户端用 `tau^-1` 恢复。

我们的实现：

- `src/aloepri/transforms/vocab.py:13-23` 检查一维整数双射；
- `vocab.py:26-32` 生成逆置换；
- `vocab.py:35-38` 使用 `inverse_tau` 重排权重行，使 private row 对应正确 plain row；
- `vocab.py:41-52` 完成输入和输出 token 映射；
- `scripts/convert_paper_qwen2_checkpoint.py:91-96,320-322` 同步改写 EOS/BOS/PAD 等配置 token；
- `client/sdk.py:75-81` 在加载在线 key 时验证 `inverse_tau[tau]`。

对方实现：

- `tmp/upstream_aloepri/src/key_manager.py:9-12` 明确排除所有 special tokens；
- `key_manager.py:20-27` 只置换 movable IDs；
- `transforms.py:8-20` 提供输入、输出和 logits 映射。

差距：

- 我们符合论文的全词表置换，并处理了生成配置中的特殊 token。
- 对方保留特殊 token 明文位置，降低了映射秘密的覆盖范围；这是工程稳定性选择，不是论文全排列。
- 我们的 `encoded_text` 是可逆 token-ID 文本封装，不是自然语言乱码。它解决了论文/PPT 的 detokenize/re-tokenize 不可逆问题，状态为 `CORRECTED/SUBSTITUTE`。

### 6.2 Embedding、LM Head 与噪声

我们的实现：

- `paper_noise.py:19-37` 使用总体标准差 `std(unbiased=False)`；
- `paper_noise.py:27-28` 采样同形状高斯噪声并计算 `W + alpha E`；
- `paper_qwen2.py:57-63` 分别实现 `Pi W_e* P` 和 `Q W_h* Pi^T` 在 PyTorch 行存储下的等价形式；
- `paper_qwen2.py:92-97` Embedding 和 Head 使用独立 seed；
- `paper_qwen2.py:172-174` Head 融合 final RMSNorm 权重后写入 checkpoint。

对方实现：

- `keymat_embed_head.py:25-55` 方法与论文噪声形式一致；
- 但 `keymat_embed_head.py:19-22,34-38,51-55` 只对 movable rows 计算标准差并加噪；
- `aloepri/layers/embeddings.py:30-44` 为了复用旧函数构造临时 `MockModel`；
- `embeddings.py:41` 固定 `alpha_h=0.0`，再在 Head 路径单独处理。

差距：

- 我们的方法更接近论文全矩阵噪声。
- v31 参数不是论文默认值：`alpha_e=0.01` 对比论文 `1.0`，`alpha_h=0.002` 对比论文 `0.2`。
- 参数可校准不构成功能错误，但 v31 的精度和隐私不能直接对比论文默认参数表。

### 6.3 Algorithm 1：矩形 `P/Q`

论文目标为 `P in R^(d x (d+2h))`、`Q in R^((d+2h) x d)`、`P Q = I_d`。

我们的实现：

- `paper_key_matrix.py:97-114` 检查 `h` 为正偶数并创建随机源；
- `paper_key_matrix.py:115-126` 构造 `B = U + lambda V`、`E=E1E2`、`F=F1F2`；
- `paper_key_matrix.py:127-139` 从 `null(F^T)` 和 `null(E)` 构造维度正确的 `C/D`；
- `paper_key_matrix.py:140-151` 构造 `P/Q` 并计算约束误差；
- `paper_key_matrix.py:195-238` 共享 `B/E/F/Z`、为 Head/Q/K/V/Gate/Up 生成独立兼容 `D`。

对方实现：

- `keymat.py:82-112` 构造同样的 `B/E/F/Z`；
- `keymat.py:115-124` 生成维度正确的 `C/D/P/Q`；
- `keymat.py:149-166` 验证 `P Q`；
- 但主线把一个 `KeyMatTransform` 传给全部层和全部组件，例如 `stage_h.py:91-225`。

差距：

- 双方都没有照抄论文错误维度，均使用了必要修正。
- 我们至少按组件区分兼容逆矩阵；对方主线大范围复用同一个 inverse，混淆多样性更低。
- 论文没有公开 `D` 的系数分布和 key 复用计划；双方相关随机分布都只能标为 `PAPER_UNDERSPECIFIED`。

### 6.4 Attention、GQA、RoPE 与 BlockPerm

我们的实现：

- `qwen_structural.py:22-40` 在 Qwen split-half RoPE 坐标中构造可交换二维旋转；
- `qwen_structural.py:43-145` 生成 Q/K 缩放、GQA 同步排列、V/O 可逆矩阵；
- `qwen_structural.py:148-197` 修正论文 BlockPerm 的漏更新、边界和未使用 `gamma`；
- `qwen_structural.py:200-218` 区分论文频率和 Qwen 实际频率；
- `qwen_structural.py:238-296` 把 Q/K/V/O 变换离线写入权重；
- `modeling_aloepri_qwen2.py:115-157` 计算同步 RoPE `B^T R(t) B`；
- `modeling_aloepri_qwen2.py:160-242` 把该算子用于 prefill、decode 和 cache。

对方实现：

- `stage_h_attention_static.py:114-187` 构造 intra-head 和 inter-head 变换；
- `stage_h_attention_static.py:164-167` 把变换写入 Q/K/V/O；
- `stage_h_attention_static.py:253-256` 只为 recorder 恢复 Q/K/V；
- `stage_h_attention_static.py:258-263` 对已经 BlockPerm 的 Q/K 直接调用普通 `apply_rotary_pos_emb`；
- `stage_h_attention_static.py:265-283` 随后进入标准 cache 和 attention。

关键问题：

若 `B` 把不同 RoPE 频率块互换，一般有 `R(t) B != B R(t)`。离线权重应用 `B` 后，运行时必须使用 `B^T R(t) B`，或者把 RoPE 频率表同步置换。对方代码在 `258-263` 没有做这一步。因此对方的 `beta>1` BlockPerm 不能由这些行证明函数保持。

我们的修正保证函数闭环，但引出新的工程差距：

- HF、vLLM、SGLang 都需要自定义 Attention；
- 这不满足 PPT 的“RoPE 保留、现有 CUDA 算子无需修改”；
- `src/aloepri/serving/vllm_qwen2.py:132-135` 明确拒绝 BlockPerm 下 `tensor_parallel_size != 1`。

### 6.5 FFN

我们的实现：

- `qwen_structural.py:298-323` 对 gate/up 使用同一中间维置换；
- up 和 down 使用互逆缩放；
- 所有置换和缩放在 checkpoint 转换时完成，服务端 FFN forward 不需要额外恢复步骤。

注意：代码变量中的 `scales` 与论文 `H` 采用互逆命名约定。`up/scales` 与 `down*scales` 对应论文 `up*H` 与 `H^-1*down`，其中代码的 `scales = H^-1`。这不是新方法，但文档必须说明，否则逐式比较会误判方向。

对方实现：

- `obfuscate_ffn.py:57-74` 提供运行时 permutation/scale/inverse；
- `obfuscate_ffn.py:96-106` 在 forward 中恢复明文坐标、执行原投影、再混淆；
- `stage_g_ffn.py:68-74` 生成融合 P/Q 的基础权重；
- `stage_g_ffn.py:106-112` 仍在 forward 执行置换、缩放和逆缩放。

差距：

- 对方早期路径依赖包装层和运行时变换，不是论文所称的完全离线权重改写。
- 我们的 FFN 更接近“服务端只加载混淆权重”。
- 双方都没有在 Qwen0.5B 上验证 MoE router 和 expert permutation。

### 6.6 RMSNorm 与 Residual

论文写法：用标量 `kappa = E[||xP||/||x||]`，把 RMSNorm 权重替换为 `kappa * 1`，再将原 norm 权重融合入相邻线性层。

问题：一般矩形、非等距 `P` 不满足对所有输入 `||xP|| = kappa ||x||`。标量只能是分布平均近似，不能保证逐输入函数等价。

我们的实现：

- `paper_qwen2.py:22-35` 保留论文 kappa 计算和近似指标；
- v31 在 `paper_qwen2.py:99-105` 存储 `G=Q Q^T`；
- `modeling_aloepri_qwen2.py:20-46` 用二次型恢复明文 RMS；
- `paper_qwen2.py:119-123,167-170` 把 norm weight 设为 1，并将原权重融合到投影/Head。

对方实现：

- `stage_g_norm.py:24-29` 同样存储 inverse-derived metric；
- `stage_g_norm.py:32-45` 同样执行全矩阵二次型；
- 参数 `kappa` 在 `stage_g_norm.py:14,20` 保存，但 forward 没有使用；
- `stage_j_standard_bridge.py:110-117` 提供 `ones`、`kappa_fused`、`metric_diag_sqrt` 三种标准形状近似；
- `stage_j_standard_bridge.py:35-36` 自己声明 bridge 尚未证明与 buffered redesign 等价。

差距：

- 双方 exact-metric 路线在数学正确性上优于论文标量近似。
- 双方 exact-metric 都需要非标准 RMSNorm，并把 `Q Q^T` 的派生信息交给服务器。
- 我们的服务器包扫描只把该矩阵列为 warning，见 `packaging.py:189-210`；其额外泄露尚未形成论文式安全分析。
- 全矩阵二次型对每个 token 增加与隐藏维度平方相关的计算，必须实测是否满足 15% 性能门禁。

### 6.7 Checkpoint、HF、vLLM、SGLang

我们的实现：

- 最新公式核对覆盖 316/316 个张量；
- HF cache 从 36 推进到 37；
- HF 与 vLLM 32/32 token 一致；
- SGLang 与 vLLM 32/32 token 一致；
- 服务器包扫描无直接 `tau/P/Q/seed` 泄露。

代码事实：

- `modeling_aloepri_qwen2.py:301-303` 注册自定义 HF 模型；
- `serving/vllm_plugin.py:11-...` 注册自定义 vLLM 模型；
- `serving/vllm_qwen2.py:24-59` 是自定义 metric RMSNorm；
- `serving/vllm_qwen2.py:62-202` 是自定义 Attention；
- `serving/sglang_models/aloepri_qwen2.py:15-90` 同样替换 RMSNorm 和 Attention。

对方实现：

- Stage-I/J/K 主要导出 HF 可见 checkpoint；
- `stage_j_standard_bridge.py:28-37` 明确 bridge 未证明等价；
- `stage_k_release.py:19-35` 的 default/reference 实际指向同一 source；
- `stage_k_correctness.py:193-198` 只要求生成 ID 和文本完全匹配率大于 0 即判 pass。

差距：

- 我们的后端功能证据更强，但不是“无需修改现有 vLLM/SGLang”。
- 对方的 release 命名为 paper-consistent，但其标准 bridge manifest 明确说未证明等价，且 pass 阈值过低。
- PPT 要求的 Tensor Parallel、P/D 分离、多节点、异构 xPU 均未在双方 0.5B 路线上验证。

### 6.8 客户端、服务端与安全边界

我们的实现：

- `client/sdk.py:248-261` 本地 Chat Template；
- `client/sdk.py:178-203` 只构造 private IDs 或可逆 private-ID 文本；
- `client/sdk.py:297-320` 非流式恢复；
- `client/sdk.py:429-483` 流式逐 token 恢复；
- `serving/app.py:45-55` Bearer Token 和 Content-Length 限制；
- `serving/app.py:61-78` token-ID 主接口；
- `serving/app.py:119-180` SSE；
- `packaging.py:41-96` 在线/离线密钥拆分；
- `packaging.py:124-218` 服务器包泄露扫描。

仍存在的代码缺口：

1. `client/sdk.py:136-147` 接受任意 `base_url`，远程 `http://` 没有在 SDK 强制拒绝。
2. `cli.py:234` 只要求远程服务配置 Bearer Token，没有强制 HTTPS。
3. `client/sdk.py:441` 把流式 M1 返回值赋给 `_privacy` 后丢弃，流式调用方拿不到隐私账本。
4. SDK 没有实现“发送前展示预计 token 改变率并允许取消”的交互；CLI 需要单独检查和补齐。
5. SSE 断流、重连、序号缺失和重复 token 的客户端恢复策略没有形成最终验收工件。

对方实现：

- `stage_k_release.py:122-128` 只写静态 deployment contract；
- `infer_stage_k_release.py` 在一个进程内同时加载 server checkpoint 和 client secret；
- 没有 FastAPI、SSE、Bearer Token、TLS 约束、日志约束、key manifest 或服务器包扫描。

结论：产品化链路我们明显领先，但远程 TLS 和流式隐私账本仍未完成。

### 6.9 RmDP 与 M1

我们的实现分成两条：

- `rmdp.py:58-116`：枚举词表所有排列，得到论文序列级 M1 的精确小词表 oracle，复杂度为词表大小阶乘；
- `rmdp.py:128-160`：对每个 token 运行长度一指数机制，然后逐 token 组合；
- `rmdp.py:163-212`：使用 v2 公式计算 `epsilon_e`、`epsilon_h`、`epsilon_2` 和分段 `epsilon`。

不到位的地方：

- Qwen 151,936 词表无法枚举序列级 M1；
- token-wise 组合不是论文序列分布的等价高效采样器；
- 论文的排列距离对 token 重数不同的序列不连通，见 `paper_errata/E17_...`；
- 产品隐私账本只记录 `epsilon1` 和改变率，没有给出长序列组合后的严格 v2 RmDP 预算。

对方 v1 代码没有该模块，因此不能用于补齐 v2。

## 7. 攻击实现逐项对照

| 攻击 | 论文 v2 协议 | 我们的代码 | 对方代码 | 判定 |
|---|---|---|---|---|
| Direct match | 直接匹配明文/混淆权重 | `mapping.py:25-43` | VMA direct source | 双方有基础实现 |
| NN/VMA | Table 9 六类矩阵组合、跨层投票、PUPA | `run_vma_pupa.py:387-624` | `security_qwen/vma.py:412-565` | 我们更接近 Table 9；v31 未正式重跑 |
| Gate-IA | 多层 gate 均值 invariant | `mapping.py:191-199`；`run_gate_ia.py:38-77` | `ia.py:7-30` 只有 planned template | 我们已实现，v31 未重跑 |
| Attn-IA | 论文 inverse-Gram 公式 | `mapping.py:202-221` 明确为 proxy | 对方未实现 | 论文公式本身维度错误 |
| ISA | 从中间状态优化输入 embedding | `run_isa_hidden_state.py:108-167` | `security_qwen/isa.py:315-480` 使用 ridge 回归 | 我们更像优化流程，但因扩维使用 Gram proxy；双方非精确 |
| IMA | 2-layer、8-head Qwen2 inverter，公开数据训练 | `train_ima_smoke.py` 小型 smoke | `ima.py:125-153,598-790` | 对方明显更接近论文 |
| TFMA | CCI3/MedDialog/Huatuo 三种先验设置 | `run_tfma_curve.py:39-87` 同 prompt mixture | `tfma.py:101-129,231-305` 三类合成语料 | 双方都未使用论文三数据集 |
| SDA | 训练 Transformer 恢复文本，报告 BLEU-4 | `train_sda_smoke.py` 小 recurrence decoder | `sda.py:66-229` unigram+bigram signature | 双方都是代理，均非论文训练协议 |
| Known plaintext | 观测量—恢复率曲线 | `mapping.py:56-72` 和 privacy suite | 无正式 v2 曲线 | 我们有基础工具，v31 未重跑 |

### 7.1 我们 VMA 的优点和不足

优点：

- `run_vma_pupa.py:203-209` 直接加载 `Columbia-NLP/PUPA:pupa_tnb:train`；
- `run_vma_pupa.py:387-398` 绑定数据集、候选范围、层和 hash；
- `run_vma_pupa.py:425-563` 实现 `We Wh`、`We Wgate`、`We Wup`、`Wdown Wh`、`We Wq (We Wk)^T`；
- `run_vma_pupa.py:584-602` 跨来源和跨层投票；
- `run_vma_pupa.py:619-624` 标记缓存是否具备正式绑定。

不足：

- 默认候选规模是 1,024/4,096，而计划目标是 16,384；
- 默认层是 7 层，不是全部 24 层；
- 最新 v31 没有该脚本的绑定输出；
- 没有 5 个独立 key 的均值、标准差和置信区间。

### 7.2 对方 VMA 的问题

- `vma.py:305-315` 使用 64-bin sorted quantile signature，不是论文 RowSort 全行；
- `vma.py:412-424` 默认只评估 256 token、4,096 candidates、3 层；
- `vma.py:480-496` 使用 q/k/v/gate/up 的单投影来源，不等于论文 Table 9 的六个矩阵乘积；
- `vma.py:522-557` 自己把协议命名为 `gate1_minimal` 和 `completed_minimal_baseline`。

因此对方 VMA 不能替代我们的 PUPA/Table-9 实现，但可借鉴其 source attribution 和 layer ablation 报告结构。

### 7.3 IMA 可借鉴部分

对方 `ima.py:125-153` 确实创建 2-layer、8-head、8 KV-head 的 Qwen 配置；`ima.py:598-790` 有公开文档切窗、训练/验证/测试、AdamW、MSE 和独立测试。这部分比我们的 `train_ima_smoke.py` 更接近论文，值得移植。

仍需修正：

- 默认 `baseline_model_dir="model/Qwen2.5-0.5B-Instruct"` 不是有效仓库 ID，也不是本地可移植路径；
- `AutoConfig.from_pretrained` 没有 `local_files_only=True`；
- 定向测试因此发起网络请求并失败；
- 公开语料使用仓库本地 docs，不是论文公开但未明确给出的训练语料；
- 仍没有与我们 v31 checkpoint/key 绑定的运行结果。

## 8. 最新 v31 证据边界

直接搜索 `qwen05b-product-v31-blockperm8` 只找到五个绑定文件：

1. `formula.json`
2. `layerwise.json`
3. `generation.json`
4. `private_api.json`
5. `functional_evidence_index.json`

已实测数值：

| 项目 | v31 结果 |
|---|---:|
| 公式张量覆盖 | 316/316 |
| 公式失败张量 | 0 |
| 公式漏检 | 0 |
| `P Q` FP64 相对误差 | `2.2734e-14` |
| private prefill cache | 36 |
| decode cache | 37 |
| 16-token greedy 恢复 | 完全相同 |
| HF/vLLM 32-token | 完全相同 |
| vLLM/SGLang 32-token | 完全相同 |
| API | HTTP 200 |
| SSE 与非流式 | 相同 |
| 错误 key | HTTP 400 |
| 明文模型峰值显存 | 2,115,368,448 bytes |
| 私有模型峰值显存 | 3,692,555,776 bytes |
| 中文回答 | “矩阵乘法是一种将一个矩阵与另一个矩阵相乘的运算，其” |

不能从这些文件推出：

- MMLU/C-Eval/PIQA/IFEval/HumanEval 下降；
- TTRSR/PIIRSR/BLEU-4/CosSim；
- TTFT/TPOT p50/p95/p99；
- 2,048 或 8,192 context 的稳定性；
- 5-key 置信区间；
- 远程 TLS 抓包结论。

## 9. 论文、PPT、v31 结果同表

本表只填 v31 已测值。论文没有 Qwen2.5-0.5B 原始表，因此不把 14B/671B 数值伪装为 0.5B 对照。

| 指标 | 论文方法/公开数据 | PPT 目标 | v31 实测 | 差距 |
|---|---|---|---|---|
| 扩维 | `d -> d+2h`，默认 `h=128` | 完成模型混淆 | `896 -> 1152` | 无功能差距 |
| `P Q` | `I_d` | 可恢复 | 相对误差 `2.2734e-14` | 无功能差距 |
| BlockPerm | `beta=8, gamma=1000` | Attention 混淆 | `beta=8`，24 层 | 采用修正 RoPE |
| 噪声 | `alpha_e=1, alpha_h=0.2` | 精度下降小于等于 3.5pp | `0.01, 0.002` | 参数不同；精度未测 |
| dtype | BF16 | 可部署 | 权重 FP32、Attention FP64 | 配置不同 |
| 上下文 | 8,192 | 未单列 | 功能证据 128；server 上限 2,048 | 未覆盖论文长度 |
| TTRSR | 论文 Qwen2.5-14B AloePri/VMA 为 13.51%；DS-V3.1 为 4.80% | 合格小于等于 15%，优秀小于等于 5% | 未测 v31 | 待测 |
| PIIRSR | DS-V3.1 为 1.12% | 小于等于 3% | 未测 v31 | 待测 |
| BLEU-4 | DS-V3.1 为 0.40 | 小于等于 2.5 | 未测 v31 | 待测 |
| CosSim | Qwen2.5-14B VMA 为 0.31 | 小于等于 0.5 | 未测 v31 | 待测 |
| TTFT/TPOT | 论文报告部分模型接近明文 | 劣化小于等于 15% | 未跑 v31 ABBA | 待测 |
| 问答闭环 | 客户端混淆、服务端推理、客户端恢复 | 服务端不见明文 prompt | 中文问答和 API/SSE 已跑通 | 远程 TLS 未测 |

## 10. 本地代码目前复现不到位的清单

### P0：影响“论文功能完成”的项目

1. **论文标量 RMSNorm 未成立。** 当前采用 exact-metric 修正，功能正确但架构和算子发生变化。
2. **标准 RoPE 无法承载非平凡 BlockPerm。** 当前自定义同步 RoPE 正确，但不符合“无需修改算子”。
3. **序列级 M1 没有 Qwen 规模精确采样器。** 当前是 token-wise 组合。
4. **Attn-IA 不是论文原式。** 原式维度错误，需要论文修正式或明确取消该 exact gate。
5. **IMA 未达到论文 2-layer/8-head 训练协议。** 可移植对方实现后重跑。
6. **ISA 因 `d -> d+2h` 使用 Gram objective。** 不是论文 same-dimension MSE。
7. **TFMA 未使用 CCI3、MedDialog、Huatuo26M-Lite。**
8. **SDA 未使用论文规模 Transformer 恢复模型和语料。**
9. **MoE/MLA 在 Qwen0.5B 上不可测。** 只能列为结构不适用，不能列“已实现并验收”。

### P1：影响“PPT 产品目标”的项目

1. SDK 未强制远程 HTTPS。
2. 流式 RmDP 隐私账本被丢弃。
3. 自定义 vLLM BlockPerm 只支持 TP=1。
4. 没有 P/D disaggregation、多节点和异构 xPU 验证。
5. exact-metric 的额外在线计算是否小于 15% 未测。
6. server package 含 `Q Q^T`；其安全影响未完成实验。
7. 最终 `release/qwen05b-product/` 目录尚不存在。

### P2：影响“科研结果可复算”的项目

1. v31 没有完整精度表和 bootstrap 95% CI。
2. v31 没有完整攻击表和 5-key 统计。
3. v31 没有 ABBA TTFT/TPOT/吞吐实验。
4. v31 没有 2,048/8,192 context 回归。
5. 当前参数是 product 参数，不是 paper-default 参数。

## 11. 对方代码值得移植与不能移植的部分

### 可以移植

| 模块 | 对方行号 | 用途 |
|---|---|---|
| Paper-like IMA 配置 | `security_qwen/ima.py:125-153` | 补齐 2-layer/8-head inverter |
| IMA 切窗和数据拆分 | `ima.py:162-204` | 建立公开语料 train/val/test |
| IMA 训练循环 | `ima.py:598-790` | 替换我们的 smoke |
| VMA source attribution | `vma.py:625-655` | 分解每类权重来源贡献 |
| VMA layer ablation | `vma.py:656-...` | 报告层数—恢复率关系 |
| 安全结果 schema | `security_qwen/schema.py` | 可参考字段设计，不直接复制路径 |

### 不能直接移植

| 模块 | 原因 |
|---|---|
| v1 安全分析 | 甲方是 v2 RmDP |
| BlockPerm + 普通 RoPE | 跨频率块置换不保持函数 |
| Stage-J standard bridge | manifest 自己声明未证明等价 |
| Stage-K pass 判定 | `match_rate > 0` 阈值不合格 |
| minimal VMA | 不等于 Table 9 六组合协议 |
| ridge ISA | 不等于论文 embedding optimization |
| 合成 TFMA | 不等于三医疗数据集 |
| bigram SDA | 不等于论文 Transformer 恢复训练 |
| `client_secret.pt` | pickle 格式、无 manifest，不满足产品密钥规范 |

## 12. 测试复核

### 我们的仓库

远程 CI 已通过：

```text
https://github.com/Science-openmycode/SafeLLM/actions/runs/31362047080
```

当前逐行账本重建命令：

```powershell
cd E:\AloePri
uv run python scripts/build_paper_line_audit.py
```

输出：25 个文件，5,108 条可审计行，5,108 条已分类。

### 对方仓库

完整 `pytest -q` 在 64 秒上限内未结束。随后运行核心定向测试：

```powershell
cd E:\AloePri\tmp\upstream_aloepri
uv run pytest -q tests/test_keymat.py tests/test_attention_keys.py `
  tests/test_security_qwen_ima.py tests/test_security_qwen_isa.py `
  tests/test_security_qwen_frequency.py
```

结果：18 passed，1 failed。失败位置：

```text
tests/test_security_qwen_ima.py:90
src/security_qwen/ima.py:133
```

失败原因：默认模型路径 `model/Qwen2.5-0.5B-Instruct` 被当作 Hugging Face repo ID，请求返回 401。该测试没有隔离网络，也没有使用仓库内可用的本地模型路径。

## 13. 复查命令

```powershell
cd E:\AloePri

# 固定双方版本
git rev-parse HEAD
git -C tmp/upstream_aloepri rev-parse HEAD

# 检查两版论文
Get-FileHash 01_client_technical_route.pdf -Algorithm SHA256
Get-FileHash 'tmp/upstream_aloepri/docs/Towards Privacy-Preserving LLM Inference via Collaborative Obfuscation (Technical Report).pdf' -Algorithm SHA256

# 重建本地逐行审计
uv run python scripts/build_paper_line_audit.py

# 重算 v31 功能索引
uv run python scripts/build_v31_functional_evidence_index.py

# 搜索 v31 绑定工件
rg -l 'qwen05b-product-v31-blockperm8' artifacts

# 检查对方是否实现 v2 RmDP/M1
rg -n -i 'rmdp|epsilon1|epsilon2|exponential mechanism' `
  tmp/upstream_aloepri --glob '!docs/**' --glob '*.py'

# 对方核心定向测试
cd E:\AloePri\tmp\upstream_aloepri
uv run pytest -q tests/test_keymat.py tests/test_attention_keys.py `
  tests/test_security_qwen_ima.py tests/test_security_qwen_isa.py `
  tests/test_security_qwen_frequency.py
```

## 14. 最终判定表

| 判定对象 | 结果 |
|---|---|
| Qwen0.5B 真实问答闭环 | 已完成 |
| Qwen 稠密模型核心混淆变换 | 已完成，但含论文修正 |
| 无噪声/低噪声函数链路 | 已完成最新功能门禁 |
| 论文原样 RMSNorm | 未完成；原式不能逐输入等价 |
| 非平凡 BlockPerm | 已完成修正版；需要自定义 RoPE |
| 论文 v2 序列级 M1 | 仅小词表精确；Qwen 路径未原样完成 |
| v2 全部攻击协议 | 未完成 |
| PPT 精度阈值 | v31 未测试 |
| PPT 隐私阈值 | v31 未测试 |
| PPT 性能阈值 | v31 未测试 |
| “现有 vLLM/SGLang 无需修改” | 未满足 |
| 对方仓库可作为 v2 官方基准 | 不可以 |
| 对方仓库可借鉴部分 | IMA、VMA 报告结构 |
