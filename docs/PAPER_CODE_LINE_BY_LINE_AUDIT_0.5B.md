---
title: "AloePri 论文实现逐行审计与人工验收手册"
subtitle: "Qwen2.5-0.5B · arXiv:2603.01499v2 · v18"
date: "2026-08-08"
lang: zh-CN
---

# 1. 审计对象和结论

本次审计只检查 `Qwen2.5-0.5B`。论文基线是
`01_client_technical_route.pdf`（arXiv:2603.01499v2），需求基线是
`02_client_project_brief.pptx`。当前重新生成的实现为：

| 项目 | 路径或数值 |
|---|---|
| 原模型 | `data/models/qwen2.5-0.5b` |
| 私有模型 | `data/checkpoints/qwen2.5-0.5b-paper-v18-audited-bf16` |
| 客户端密钥 | `data/keys/dev-qwen05b-paper-v18-audited-bf16` |
| 原隐藏维 | $d=896$ |
| 扩展宽度 | $h=128$ |
| 私有隐藏维 | $D=d+2h=1152$ |
| 论文数值参数 | $\lambda=0.3$，$\alpha_e=1.0$，$\alpha_h=0.2$，$\beta=8$，$\gamma=1000$ |
| 存储精度 | BF16 |
| 逐张量公式反算 | 291/291 通过；缺少公式覆盖 0 |
| Algorithm 1 块结构 | 通过 |
| Algorithm 2 密钥结构 | 24/24 层通过 |
| 真实模型功能保持 | 未通过：prefill top-1 与 greedy generation 明显偏离 |

结论分成两件事：

1. v18 checkpoint 的每个张量都能由原权重和基础/派生密钥重建；验证器还单独
   检查 Algorithm 1 块结构及 Algorithm 2 的置换、GQA、Q/K、Value 和 FFN
   结构约束。
2. 论文默认噪声和扩维参数应用到 Qwen2.5-0.5B 后没有保持原模型行为，因此当前 0.5B 结果不能作为可用私有推理模型交付。

# 2. 论文要求实现的完整流程

## 2.1 可信客户端

客户端保存词表置换 $\tau$ 和逆置换 $\tau^{-1}$。明文 prompt 先由原始
tokenizer 编码：

$$
x = \operatorname{Tokenizer}(\text{prompt}),
\qquad
\widetilde{x}_i = \tau(x_i).
$$

服务端返回私有 token $\widetilde{y}_i$ 后，客户端逐 token 恢复：

$$
y_i = \tau^{-1}(\widetilde{y}_i),
\qquad
\text{answer}=\operatorname{Decode}(y).
$$

论文文字流程在 $\widetilde{x}$ 后增加一次 detokenize，并要求服务端重新
tokenize。该步骤隐含条件

$$
\operatorname{Encode}(\operatorname{Decode}(\widetilde{x}))=\widetilde{x}.
$$

Qwen tokenizer 不满足这个条件。固定密钥的随机实验共测 1000 个序列，失败
208 个；长度 1、2、4、8、16 的失败数分别为 2、11、28、61、106。实际反例的
中间文本由 Unicode 码点 `U+0E41 U+0E21 U+0E49 U+0E27 U+0E48 U+0E32` 组成：

$$
[140688]\xrightarrow{\operatorname{Decode}}\text{上述六个 Unicode 码点}
\xrightarrow{\operatorname{Encode}}[124840,124379,64741].
$$

因此主接口发送 token ID，不发送中间乱码文本。该修改不改变 $\tau$、模型权重
或输出恢复公式，具体记录在勘误 E12。

## 2.2 离线词表和权重变换

定义 $\tau(i)$ 为明文 token $i$ 在私有词表中的位置。Embedding 和 LM Head
按同一置换重排行：

$$
\widetilde{E}_{\tau(i),:}=(E_{i,:}+\alpha_e N^E_{i,:})P,
$$

$$
\widetilde{W}^{\mathrm{head}}_{\tau(i),:}
=\left(W^{\mathrm{head}}_{i,:}+\alpha_h N^H_{i,:}\right)
\operatorname{Diag}(g_f)Q_h^{\mathsf T}.
$$

噪声按完整张量的总体标准差采样：

$$
N^E_{ij}\sim\mathcal N(0,\sigma_E^2),
\qquad
N^H_{ij}\sim\mathcal N(0,\sigma_H^2).
$$

其中 $\sigma_E=\operatorname{Std}(E)$，
$\sigma_H=\operatorname{Std}(W^{\mathrm{head}})$。

## 2.3 Algorithm 1：矩形 P/Q 扩维

取 $D=d+2h$。论文构造共享初始化状态：

$$
B=U+\lambda V,
\qquad
E=E_1E_2,
\qquad
F=F_1F_2,
$$

并选取 $C,D_j$ 满足

$$
CF=0,
\qquad
ED_j=0.
$$

随后定义

$$
P=[B\;C\;E]Z\in\mathbb R^{d\times D},
$$

$$
Q_j=Z^{\mathsf T}
\begin{bmatrix}
B^{-1}\\F\\D_j
\end{bmatrix}
\in\mathbb R^{D\times d}.
$$

于是每个分支均满足

$$
PQ_j=[B\;C\;E]
\begin{bmatrix}
B^{-1}\\F\\D_j
\end{bmatrix}
=I+CF+ED_j=I.
$$

v18 对 Head、Attention Q/K/V、FFN Gate/Up 使用同一次论文
`INIT=(B,B^{-1},E,F,Z)`。固定 $C$ 构造唯一的 $P$，再为六个角色分别采样
$D_j$。论文未给出 $D_j$ 的概率分布；代码固定使用 $1/\sqrt d$ 尺度的高斯
系数，再投影到 $\ker(E)$。密钥文件保存 $B,B^{-1},E,F,Z$，人工核验器可直接
检查每个 $Q_j$ 是否确实保持论文块形状。

## 2.4 RMSNorm 与线性层

Hugging Face `nn.Linear` 使用 $y=xW^{\mathsf T}$。对私有残差
$\widetilde{x}=xP$，输入投影在存储方向写为

$$
\widetilde{W}_{\mathrm{in}}
=W_{\mathrm{in}}\operatorname{Diag}(g)Q_j^{\mathsf T}.
$$

运行时：

$$
\widetilde{x}\widetilde{W}_{\mathrm{in}}^{\mathsf T}
=xPQ_j\operatorname{Diag}(g)W_{\mathrm{in}}^{\mathsf T}
=x\operatorname{Diag}(g)W_{\mathrm{in}}^{\mathsf T}.
$$

输出投影在存储方向写为

$$
\widetilde{W}_{\mathrm{out}}=P^{\mathsf T}W_{\mathrm{out}}.
$$

论文给出的 $\kappa$ 是 L2 范数比，但 Qwen RMSNorm 使用均方根。维度从
$d$ 变为 $D$ 后必须包含

$$
\frac{\operatorname{RMS}(xP)}{\operatorname{RMS}(x)}
=\sqrt{\frac dD}\frac{\lVert xP\rVert_2}{\lVert x\rVert_2}.
$$

v18 使用固定 20 条校准样本得到逐层 RMS 比，并将原 RMSNorm 权重融合进下游
Q/K/V 或 Gate/Up 权重；私有 RMSNorm 自身只保留标量 $\kappa$。

## 2.5 Attention、GQA、RoPE 和 KV Cache

每个 Q/K head 使用在 RoPE 二维子空间内可交换的变换。Q/K 必须使用同侧块
置换 $Z_b$：

$$
q'=qZ_b,
\qquad
k'=kZ_b,
\qquad
q'k'^{\mathsf T}=qZ_bZ_b^{\mathsf T}k^{\mathsf T}=qk^{\mathsf T}.
$$

论文打印的 K 侧 $Z_b^{\mathsf T}$ 一般会产生 $Z_b^2$，不能保持内积；v18
采用上述代数可逆修正并记录为 E06。Qwen2.5-0.5B 有 GQA，因此每个 KV head
对应一组 Q heads；代码以 `group_size=num_attention_heads/num_key_value_heads`
选择共享映射。Value 采用可逆 $U_{vo}$，O 投影乘 $U_{vo}^{-1}$ 消去变换。

prefill 后的 K/V cache 保持私有坐标，decode 输入也必须先执行 $\tau$。核验器
用同一明文 token $x_{t+1}$ 和对应私有 token $\tau(x_{t+1})$ 分别续接两边
cache，以避免把“模型已选出不同 token”和“cache 实现错误”混为一谈。

## 2.6 FFN

Qwen 的 SwiGLU 为

$$
\operatorname{FFN}(x)
=\left[\operatorname{SiLU}(xW_g^{\mathsf T})
\odot(xW_u^{\mathsf T})\right]W_d^{\mathsf T}.
$$

对中间维执行同一置换 $\pi$，并使用非零对角缩放 $S$：Gate 只置换，Up
除以 $S$，Down 乘以 $S$，从而乘积中的尺度抵消。v18 在
$[0.5,2.0]$ 上使用对数均匀正数，保证 $S^{-1}$ 存在；论文只写随机缩放，未给
分布，并且写 $s_i\in\mathbb R$ 却没有排除 0，该问题记录为 E13。

## 2.7 在线服务端

服务端只加载私有 checkpoint 和公开的 `model_id/key_id`，不加载
$\tau^{-1}$。请求顺序是：

1. 校验 `model_id` 和 `key_id`；
2. 接收 `input_ids=\tau(x)`；
3. 私有模型执行 prefill；
4. 使用私有 KV Cache 逐 token decode；
5. 非流式返回完整私有 token 列表，或用 SSE 逐 token 返回；
6. 客户端执行 $\tau^{-1}$ 后由原 tokenizer 解码。

# 3. 论文到代码的逐项对应

| 论文步骤 | 代码入口 | 关键测试或证据 | 审计判定 |
|---|---|---|---|
| 词表置换与逆置换 | `transforms/vocab.py` | vocab、special-token tests | 论文精确 |
| Embedding/Head 高斯噪声 | `transforms/paper_noise.py` | noise test | 论文精确 |
| Algorithm 1 的 P/Q | `transforms/paper_key_matrix.py` | key-matrix test；反算 JSON | 按 E01/E02 修正维度 |
| 六个独立 $Q_j$ | inverse-family 函数 | 六个 $PQ_j$、块重建 | 共享 Init；$D_j$ 分布未给 |
| Embedding/Head/层投影 | `conversion/paper_qwen2.py` | conversion test | 存储方向已核对 |
| Q/K/V/O、GQA、RoPE | `transforms/qwen_structural.py` | structural test | E03--E06、E10 修正 |
| FFN permutation/scaling | 同上 | structural test | 代数精确；分布未给 |
| RMSNorm | `conversion/paper_qwen2.py` | conversion test；反算 | 按 E07 修正维度 |
| 自定义 HF 架构 | `models/` | model test | 工程适配 |
| checkpoint 和密钥保存 | converter script | manifest、SHA-256、反算 | 已实现 |
| 客户端 | `client/sdk.py` | client test | 置换精确；ID 协议按 E12 修正 |
| 服务端/API/SSE | `serving/` | API test、真实模型 test | 工程适配 |
| VMA/Gate-IA | `attacks/mapping.py`、脚本 | attack tests | 原语实现；实验细节未给全 |
| Attn-IA | `run_attn_ia.py` | E09 | 仅代理，不算复现 |
| ISA/IMA/TFMA/SDA | 对应运行脚本 | 各代理测试 | 协议不完整，不算精确复现 |

# 4. 本轮逐行审计

机器可读配置位于 `configs/audit/paper_line_audit_0.5b.json`。生成器使用 Python
AST 找出每条物理行所在的最小函数或类，并给每条非空、非纯注释行附加状态、
论文位置、测试和说明。文件 SHA-256 与源代码一起写入 JSON 和 HTML，任何代码
变化都会改变哈希。

| 分类 | 行数 | 含义 |
|---|---:|---|
| `PAPER_EXACT` | 440 | 论文公式和维度可以唯一确定，代码按该式实现 |
| `PAPER_CORRECTED` | 467 | 论文原式存在可证明错误，代码采用对应勘误中的最小修正 |
| `ENGINEERING_SUBSTITUTE` | 310 | 模型注册、HTTP、SSE 等论文未规定的软件适配 |
| `PAPER_UNDERSPECIFIED` | 850 | 论文给出方向但没有给出足够参数或完整实验协议 |
| `PROXY_NOT_PAPER_EXACT` | 606 | 明确标记的诊断或攻击代理，不能算论文原实验复现 |
| `VERIFICATION` | 1092 | 转换、反算、证据生成和检查代码 |
| 合计 | 3765 | 23 个核心文件全部分类 |

“3765/3765 已分类”只表示审计账本覆盖全部范围，不表示正确率 100%。人工阅读
入口是 `docs/PAPER_CODE_LINE_BY_LINE_AUDIT_0.5B.html`；页面可按状态过滤，并显示
行号、函数、原始代码、论文位置、测试路径和文件 SHA-256。原始账本为
`artifacts/audit/paper-line-audit-0.5b.json`。

# 5. 逐张量人工反算

执行：

```powershell
uv run python scripts/verify_paper_formula_checkpoint.py `
  --source data/models/qwen2.5-0.5b `
  --private data/checkpoints/qwen2.5-0.5b-paper-v18-audited-bf16 `
  --key-dir data/keys/dev-qwen05b-paper-v18-audited-bf16 `
  --out artifacts/audit/paper-formula-checkpoint-v18.json
```

该命令不是只比较几个抽样层，而是检查私有 safetensors 中全部 291 个张量，
并独立检查密钥包的 Algorithm 2 结构约束：

- `embed_tokens.weight`：噪声、词表逆行索引、右乘 $P$；
- 每层 RMSNorm：逐层 $\kappa$；
- 每层 Q/K/V weight 和 bias：RMSNorm 融合、各自 $Q_j$、GQA head 顺序、
  RoPE-compatible 变换；
- 每层 O：Value 逆变换、head 顺序、左乘 $P^{\mathsf T}$；
- 每层 Gate/Up/Down：各自 $Q_j$、FFN 置换和缩放；
- final norm 与 LM Head：$\kappa$、噪声、Head $Q_j$、词表置换；
- Q/K/FFN order 是否为双射，Q order 是否按 GQA 分组对齐；
- 每个 RoPE block order 是否为双射；
- `q_maps/k_maps` 是否能由 `rope_maps`、正缩放和 block map 重建；
- $Q_{map}K_{map}^{\mathsf T}$ 是否为单位阵，Value map 是否可逆；
- FFN scales 是否为正且落在 checkpoint 配置区间。

通过条件为：

```text
overall_pass = true
algorithm1_pass = true
algorithm2_pass = true
tensor_check_count = 291
failed_tensors = []
missing_formula_coverage = []
```

v18 实测满足全部条件。Algorithm 1 的浮点残差为：

| 检查 | 最大绝对误差 |
|---|---:|
| $BB^{-1}-I$ | $6.25\times10^{-15}$ |
| $ZZ^{\mathsf T}-I$ | $8.88\times10^{-16}$ |
| $CF$ | $2.69\times10^{-17}$ |
| $P-[B\;C\;E]Z$ | $8.33\times10^{-17}$ |
| 六个 $ED_j$ 中最大值 | $2.96\times10^{-17}$ |
| 六个 $Q_j$ 块重建中最大值 | $8.33\times10^{-17}$ |

Algorithm 2 的 24 层结构检查也必须全部为 `pass=true`；其中
$Q_{map}K_{map}^{\mathsf T}-I$ 的全层最大绝对误差会直接记录在 JSON，不使用
checkpoint 目标权重反推这一约束。

| Algorithm 2 独立检查 | v18 全 24 层最坏值 |
|---|---:|
| $Q_{map}K_{map}^{\mathsf T}-I$ 最大绝对误差 | $8.13\times10^{-8}$ |
| Q map 从基础变量重建的最大误差 | $1.49\times10^{-7}$ |
| K map 从基础变量重建的最大误差 | $1.68\times10^{-7}$ |
| RoPE pair-block 非法位置最大值 | $0$ |
| RoPE map 与三个位置旋转的最大交换误差 | $1.11\times10^{-16}$ |
| RoPE map 按 seed 重建的最大误差 | $2.98\times10^{-8}$ |
| Value map 最小奇异值（全层最小） | $5.72\times10^{-4}$ |
| Value 高斯 map 按 seed 重建的最大误差 | $2.70\times10^{-8}$ |
| 置换、GQA 对齐、缩放范围 | 24/24 层通过 |

公式 JSON 的 `formal_run_binding=true`，绑定 1 个源权重 shard、私有模型 manifest
及 8 个私有文件、密钥文件、验证脚本 SHA-256 和 Python/PyTorch/CUDA/GPU 环境；
`data_files` 另绑定源/私有 `config.json`、`key.json` 和 RMS 校准 JSON。

# 6. 真实 0.5B 运行检查

执行：

```powershell
uv run python scripts/verify_paper_qwen2_checkpoint.py `
  --source data/models/qwen2.5-0.5b `
  --private data/checkpoints/qwen2.5-0.5b-paper-v18-audited-bf16 `
  --key-dir data/keys/dev-qwen05b-paper-v18-audited-bf16 `
  --dtype bfloat16 `
  --max-new-tokens 16 `
  --out artifacts/verification/paper-qwen05b-v18-audited-bf16.json
```

检查用同一条中文 chat prompt，原模型输入为 $x$，私有模型输入为 $\tau(x)$；
logits 按 $\tau^{-1}$ 恢复顺序。模型分阶段加载，未同时驻留 GPU。

| 指标 | v18 实测 | 含义 |
|---|---:|---|
| prefill cache 长度（明文/私有） | 36 / 36 | prompt 长度一致 |
| 加入同一对应 token 后 cache 长度 | 37 | decode cache 可继续增长 |
| decode 输入是否为 $\tau(x_{t+1})$ | 是 | 已排除输入 token 不对应的混杂因素 |
| prefill logits 最大绝对误差 | 25.03125 | 未保持 logits |
| prefill logits 平均绝对误差 | 2.631565 | 未保持 logits |
| decode logits 最大绝对误差 | 19.65625 | 对应输入下仍未保持 logits |
| decode logits 平均绝对误差 | 3.621766 | 对应输入下仍未保持 logits |
| prefill 全位置 top-1 一致率 | 22.22%（8/36） | 不是估计比例，不给置信区间 |
| 下一 token 是否一致 | 否 | prompt 结束处已分叉 |
| 16-token greedy IDs 是否一致 | 否 | 功能保持失败 |
| 原模型阶段峰值显存 | 1,137,571,840 bytes | CUDA allocated peak |
| 私有模型阶段峰值显存 | 1,774,570,496 bytes | CUDA allocated peak；未同时加载两模型 |

当前 cache 证据只确认 prefill/decode 路径可执行、长度从 36 增长到 37，并用对应
token 隔离 decode 输入；尚未逐层证明明文和私有 K/V cache 张量之间的坐标变换
关系，也未比较 cache/no-cache logits，因此不把该项写成“KV Cache 数值等价”。

私有输出经 $\tau^{-1}$ 恢复后的新增文本是重复的
`) - ( ) - ( - ( - ( - ( - ( - (`，不是原模型回答。

逐张量通过但输出未保持的直接含义是：代码忠实执行了当前选定的论文公式和默认
参数，而这些变换在该 0.5B checkpoint 上引入的数值/噪声扰动足以改变预测。
这项单模型实验不能证明“失败由参数量小导致”；证明参数量因果关系需要固定所有
变换和数据，再增加更大模型对照。

# 7. 一键人工验收

完整命令：

```powershell
Set-Location E:\AloePri
.\scripts\run_human_verification_0.5b.ps1
```

只检查源代码、逐行账本和普通测试，不加载真实模型：

```powershell
.\scripts\run_human_verification_0.5b.ps1 -SkipModel
```

| 阶段 | 人类看到的输出 | 通过条件 |
|---|---|---|
| 逐行账本 | HTML、JSON、23 个文件哈希 | 3765 条可审计行都有分类 |
| Ruff | 终端输出 | exit code 0 |
| mypy | 终端输出 | exit code 0 |
| pytest | 测试计数 | 失败 0；默认真实模型测试会 skip |
| tokenizer round-trip | JSON | 结果用于验证 E12；预期 `paper_text_transport_safe=false` |
| 291 张量反算 | JSON | `overall_pass=true` 且缺失覆盖为空 |
| HF 实模 | JSON | 报告误差、top-1、cache 长度、greedy 和显存，不隐藏失败 |
| 真实 API | pytest | HTTP 200、SSE/非流式 token 一致、错误 key=400、恢复文本非空 |

HF 实模 JSON 和公式反算 JSON 保存模型 shard、密钥、脚本和运行环境来源；
逐行账本另存每个受审文件的 SHA-256。复核者先看 provenance 和文件哈希，再看
指标，避免把旧 checkpoint 的结果当成 v18 结果。

本轮实际执行结果：Ruff 通过；审计范围内 15 个 Python 文件 mypy 通过；普通
测试为 89 passed、1 skipped，skip 项是需要显式加载实模的 API 测试；设置上述
环境变量后单独执行该测试为 1 passed。若执行 `mypy src scripts` 检查全部历史
脚本，当前会得到 146 个既有错误，主要来自缺少 vLLM、第三方库无类型桩以及旧
评测脚本的类型标注。本报告不把全仓 mypy 写成通过。

# 8. 独立子 Agent 审核

独立审核 Agent 在不接受主实现结论的前提下重新检查了论文、核心代码、测试和
旧证据。其结论如下：

| 审核项 | 独立结论 | 本轮处理 |
|---|---|---|
| 新 Algorithm 1/Attention/FFN/RMS 代码 | 未发现新的矩阵方向错误 | 用 291 张量反算补充实证 |
| 旧 v15 数据 | 来自旧 inverse family，不能证明 v18 | v15 与 v18 完全分表，不复用结果 |
| 真实 API test | 只验证 HTTP/stream/非空，不验证模型精度 | 精度和 logits 由独立 HF 核验器检查 |
| vLLM | 没有修复候选的实跑证据 | 不列为已完成 |
| SGLang | 未实现 | 不列为已完成 |
| 文本在线接口 | 不等价于 token-ID 接口 | E12 给出可运行反例 |
| RmDP 总体定理 | 依赖可选在线指数机制 $M_1$ | 当前未实现，不宣称完整 RmDP 系统 |
| 攻击实验 | 多项仅代理或协议欠定 | HTML 逐行标记，不放入论文精确结果 |
| 0.5B 失败归因 | 不能证明由参数量导致 | 本报告不作参数量因果结论 |
| 291 张量反算的独立性 | 原版未检查 Algorithm 2 派生密钥结构 | 增加 24 层置换、GQA、Q/K、Value、FFN 约束检查 |
| 公式 JSON 来源绑定 | 原版没有 shard、脚本和运行环境绑定 | 增加完整 `run_provenance` |
| Algorithm 1 INIT 文字 | 报告误把 $C$ 列入 INIT | 改为 $(B,B^{-1},E,F,Z)$，$C$ 单列 |
| KV Cache 证据 | 只证明路径和长度 | 明确限制，不宣称 cache 张量数值等价 |

独立 Agent 在上述修复后进行了第三次复核，确认 RoPE pair-block/交换性、按 seed
重建 Value、Algorithm 2 门禁、实际输入 provenance 和 key manifest 均已闭合，
未发现新的 P0/P1 问题；复核过程未修改文件。

# 9. 论文错误与未给信息

每个确定错误有单独的公式推导文档和 PDF，索引位于
`docs/paper_errata/README.md`，PDF 位于 `output/pdf/paper_errata/`。

| 编号 | 内容 | 判定 |
|---|---|---|
| E01 | Algorithm 1 的 $C$ 把行写成列 | 维度错误 |
| E02 | Algorithm 1 的 $D$ 把列写成行 | 维度错误 |
| E03 | BlockPerm 声明 $\gamma$ 但采样式未使用 | 伪代码错误 |
| E04 | BlockPerm 循环未更新 $t$ | 伪代码错误 |
| E05 | 自然补上更新后仍漏最后一个 block | 边界错误 |
| E06 | Q/K 两侧分别用 $Z$ 和 $Z^{\mathsf T}$ 产生 $Z^2$ | 代数错误 |
| E07 | RMSNorm 的 $\kappa$ 少维度因子 | 归一化错误 |
| E08 | Algorithm 2 是否共享一次 Init | 实现歧义 |
| E09 | Attn-IA inverse-Gram 乘法维度不成立 | 维度错误 |
| E10 | RoPE 频率指数与 Qwen 实际实现相差两倍 | 适配错误 |
| E11 | Summation Composition 使用未定义 $\psi_X\vert_{C_2}$ | 符号错误 |
| E12 | detokenize/tokenize 不保持任意混淆 token 序列 | 接口假设错误 |
| E13 | $s_i\in\mathbb R$ 未排除 0，却使用 $H^{-1}$ | 定义域缺项 |
| E14 | 误差上界连乘下标为 $j$，被乘项写成 $M_i$ | 下标错误 |

论文仍未给出并直接影响复现唯一性的内容包括：六个 $D_j$ 的分布、FFN scaling
分布、完整随机种子、checkpoint revision、RMSNorm 校准语料、VMA 候选空间和
投票细节、Gate-IA/ISA/IMA/TFMA/SDA 的完整训练或搜索协议、性能服务器型号和
vLLM 启动参数。代码对这些位置全部显式标为 `PAPER_UNDERSPECIFIED` 或
`PROXY_NOT_PAPER_EXACT`。

# 10. 当前可以确认与不能确认的内容

| 内容 | 当前证据 | 结论 |
|---|---|---|
| v18 是否按仓库所列论文公式生成 | 291/291 逐张量反算 | 可以确认 |
| P/Q 是否来自论文共享 Init 块结构 | 保存并反算 $B,B^{-1},E,F,Z,D_j$ | 可以确认 |
| 0.5B checkpoint 能否由 HF 加载 | 真实 CUDA 加载 | 可以确认 |
| prefill/decode/KV Cache 是否能执行 | 真实 CUDA 运行 | 可以确认 |
| 默认论文参数是否保持 0.5B 输出 | top-1、greedy、恢复文本 | 不能保持 |
| 是否因模型只有 0.5B 而失败 | 只有一个参数规模 | 不能确认 |
| v18 是否达到 MMLU/C-Eval/PIQA/IFEval/HumanEval 目标 | v18 未跑完整基准 | 未测试，不写结果 |
| v18 是否达到全部隐私攻击目标 | v18 未重跑完整攻击 | 未测试，不写结果 |
| vLLM/SGLang 生产可用 | 无 v18 实跑证据 | 不能确认 |
| 论文完整 RmDP 机制 | 在线指数机制 $M_1$ 未实现 | 未完成 |
