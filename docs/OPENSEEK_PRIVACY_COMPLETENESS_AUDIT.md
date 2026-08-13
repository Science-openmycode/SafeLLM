# OpenSeek-Small-v1-SFT 隐私功能完整性严格审核

审核对象固定为：

- 源模型：`BAAI/OpenSeek-Small-v1-SFT`
- 源 revision：`1515c184e6fe4a91e6061be513a79d607e8787cb`
- 私有模型：`data/packages/openseek-small-v1-sft-paper-complete`
- 在线密钥：`data/keys/openseek-small-v1-sft-paper-complete-online`
- 离线主密钥：`data/keys/openseek-small-v1-sft-paper-complete-offline`
- 固定配置：`configs/product/openseek_small_v1_sft_paper_complete.yaml`
- 总证据索引：`artifacts/openseek-small-v1-sft-paper-complete/evidence-index.json`

## 1. 审核结论

甲方基线重新读取并固定为：

- `01_client_technical_route.pdf`，SHA-256 `a5fd16662e1ac2feb55952940edd0fdf9d4007da22bd2f0eacb5f5ea41dc4748`
- `02_client_project_brief.pptx`，SHA-256 `244b6fcd3b48f75d8a77e84dd16df5456c1993ec840c0c719cb148a46360b337`

| 审核层 | 结论 | 证据 |
|---|---|---|
| 适用于该 checkpoint 的论文/PPT 隐私机制 | PASS | 16 项机制检查全部为 `true` |
| 源权重转换覆盖 | PASS | 源张量 83 个，83 个均发生变换；私有包另含 1 个 RoPE 运行时张量 |
| 在线隐私边界 | PASS | 明文、明文 token 序列和在线密钥均未进入模型服务器或服务日志 |
| 密钥拆分与服务端包净化 | PASS | 服务器包不含 `tau`、`inverse_tau`、P/Q、转换种子和原模型权重 |
| 真实端到端问答 | PASS | Chat Template、分词、置换、私有生成、逆置换和本地解码实际执行成功 |
| 可执行攻击路径 | PASS（冒烟规模） | 1 个密钥、64 候选的 VMA、MLA-IA、直接匹配和 known-plaintext 均可执行 |
| 完整统计攻击验收 | 尚未完成 | 尚需全词表、至少 5 个独立密钥和论文指定语料；不得用 64 候选冒烟代替 |
| MTP | 不适用 | 上游配置声明 `nextn=1`，但发布 checkpoint 的 MTP 张量数为 0 |

这里的“功能完整”指每一项适用隐私机制都有真实权重变换、运行时支持、密钥材料、自动审计和可人工复算证据。它不等于已经证明任意攻击都不能恢复信息；后者必须由完整攻击实验给出统计结论。

### 1.1 与甲方技术路线的严格对齐判定

| 技术路线原要求 | 判定 | 当前实现或差异 |
|---|---|---|
| 离线模型混淆，在线只做轻量 token 处理 | 完全对齐 | 转换在客户端离线完成；在线客户端只执行 Chat Template、分词、可选 M1、`tau`、`inverse_tau` 和解码 |
| 词表置换同时保护 prompt 与 response | 完全对齐 | Embedding、LM Head、BOS/EOS 和在线输入输出使用同一 `tau` |
| Embedding/Head 高斯噪声隐藏词表关系 | 完全对齐 | `alpha_e=1.0`、`alpha_h=0.2`，按权重标准差采样；固定种子可复算 |
| Algorithm 1 的 `d -> d+2h` 两侧矩阵掩蔽 | 修正后对齐 | 保持 `h=128, lambda=0.3` 和 `PQ=I`；按 E01/E02/E08 修正原算法维度和独立右逆问题 |
| Attention 的 Q/K/V/O、RoPE block、head 变换 | 修正后对齐 | Q/K 使用互逆映射；V/O 对消；Q/K/运行时 RoPE 顺序同步；按 E03–E06、E10、E13 修正原式 |
| MLA 低秩矩阵使用额外可逆变换 | 原文欠规定，已完成架构适配 | 原文没有给 MLA 的逐张量公式；实现覆盖 q low-rank、kv latent、NoPE、decoupled RoPE、V/O 和 Cache，但不能称为逐式复刻未公开公式 |
| Dense FFN、Shared Experts、Routed Experts 和 Router | 完全对齐 | Gate/Up/Down 同步置换与互逆缩放；专家整体置换；Router 行归一化和同序重排 |
| RMSNorm 使用论文 `kappa` 并融合原 Norm 权重 | 按原文实现但原文仅近似 | 当前使用 `paper_kappa`；E07/E15 已证明一般矩形 P 下不是严格等价，精度必须实测 |
| 在线先 detokenize 私有 token，再由服务器重新 tokenize | 有意不按字面实现 | 当前 Qwen v31 的200条真实Chat输入有88条round-trip失败；按 E12 改为token-ID-only接口，才能保证服务器实际收到 `tau(x)` |
| 可选 token 扰动与 RmDP 预算 | 可执行范围内对齐 | 全词表单 token 指数机制可运行；小词表可枚举原序列机制；一般长序列精确采样不可行且 E17 指出原距离并非总是有限 |
| honest-but-curious 服务器只能观察私有模型、输入和输出 | 完全对齐 | 抓包、日志和 server package 检查通过；不扩张为恶意服务器下的密码学绝对安全声明 |
| vLLM/SGLang 无需修改直接运行 | OpenSeek 尚未对齐 | 当前 OpenSeek `paper-complete` 使用自定义扩维 DeepSeek-V3 HF runtime；OpenSeek 的 vLLM/SGLang 适配尚未完成 |
| VMA、IA、ISA、IMA、NN、TFMA、SDA 安全验证 | 攻击代码具备，OpenSeek 完整实验未对齐 | 当前只完成单 key、64 候选攻击冒烟；全词表、至少 5 key 和指定语料尚未运行 |
| 精度下降、TTFT、TPOT 和完整隐私指标 | 尚未对齐 | 真实问答链路已通过，但当前回答有重复；完整精度、隐私和性能验收不能由功能审计替代 |

因此，严格结论不是“整条技术路线已经全部验收”，而是：**OpenSeek 当前权重中能够实现的核心隐私变换均已实现；其中论文错误部分采用有独立推导的修正式，MLA 按原文原则完成架构适配。尚未完成的是 OpenSeek 的 vLLM/SGLang 工程兼容以及完整统计实验，它们分别属于推理框架适配和性能验收，不是缺少某个权重隐私变换。**

## 2. 客户端到服务端的完整隐私流程

设原 token 序列为 $x$，词表置换为 $\tau$，逆置换为 $\tau^{-1}$。可信客户端执行：

$$
x=\operatorname{Tokenizer}(\operatorname{ChatTemplate}(m)),
\qquad
\widetilde{x}_i=\tau(x_i).
$$

客户端只发送 $\widetilde{x}$。服务器加载经过权重混淆的模型 $\widetilde f$ 并返回私有 token：

$$
\widetilde y=\widetilde f(\widetilde x).
$$

客户端逐 token 恢复并解码：

$$
y_i=\tau^{-1}(\widetilde y_i),
\qquad
\text{answer}=\operatorname{Tokenizer}^{-1}(y).
$$

可选 M1 模式在 $\tau$ 之前逐 token 执行指数机制；默认产品模式只执行确定性词表置换。服务器接口始终只接受 `input_ids`，不提供明文 prompt 字段，也不在服务器端重新分词。

## 3. 逐项要求、公式、代码和实测证据

| 模块 | 实现要求与公式 | 主要代码 | 自动证据 | 当前结果 |
|---|---|---|---|---|
| 词表置换 | $\widetilde x=\tau(x)$，$x=\tau^{-1}(\widetilde x)$；Embedding 行、Head 输出行及特殊 token 同步重排 | `src/aloepri/keys/generate.py`、`src/aloepri/transforms/vocab.py`、`deepseek_streaming.py` | `vocabulary_roundtrip`、`special_tokens_mapped` | PASS |
| Embedding 噪声 | $E^\star=E+\alpha_e\sigma_E\Xi_e$，$\widetilde E=\Pi E^\star P$ | `src/aloepri/transforms/paper_noise.py`、`deepseek_streaming.py` | 8 行按固定随机种子重算，BF16 误差不超过 1 ULP | PASS，最大误差 `0.0001220703125` |
| LM Head 噪声 | $H^\star=H+\alpha_h\sigma_H\Xi_h$，再融合 Final RMSNorm、右逆和词表置换 | 同上 | 8 行重算，BF16 误差不超过 1 ULP | PASS，最大误差 `0.000244140625` |
| P/Q 扩维 | $P\in\mathbb R^{1280\times1536}$，$Q_j\in\mathbb R^{1536\times1280}$，$PQ_j=I$ | `src/aloepri/transforms/paper_key_matrix.py`、`deepseek_streaming.py` | 7 个兼容右逆逐个复算 | PASS，最大相对误差 `3.265e-14` |
| Residual 坐标 | 所有残差输出统一回到 $\widetilde x=xP$，输入投影使用相应 $Q_j$ | `deepseek_streaming.py` 的 `transform_input_projection` / `transform_output_projection` 调用 | 所有相关张量形状、名称和变换覆盖 | PASS |
| RMSNorm | 当前 checkpoint 忠实采用论文标量 $\kappa$ 路线，并把原 Norm 权重融合进后续输入投影 | `paper_qwen2.py`、`modeling_aloepri_deepseek_v3.py` | private Norm 宽度为 1536；运行时确认原生 RMSNorm 路径 | PASS（论文路线）；数学限制见第 5 节 |
| MLA 低秩坐标 | q 低秩、kv latent、NoPE、RoPE、value、head 均有独立置换/映射 | `src/aloepri/transforms/deepseek.py` | 6 层逐层验证 head、kv latent、NoPE、RoPE、value key 均非平凡 | PASS |
| Q/K 缩放 | 对每个映射使用互逆缩放，使 $A_qA_k^T=I$，保持注意力点积 | `deepseek.py` | NoPE 和 RoPE 映射逐层复算互逆误差 | PASS，误差门限 `1e-10` |
| RoPE BlockPerm | 以 $\beta=8,\gamma=1000$ 生成 pair order；Q/K 和运行时频率使用同一顺序 | `qwen_structural.py`、`deepseek.py`、`modeling_aloepri_deepseek_v3.py` | 离线 6 层 order、配置 order、服务器 tensor 三方逐元素一致 | PASS |
| V/O 映射 | V 使用条件数受限高斯 $U_{vo}$，O 使用 $U_{vo}^{-T}$ 对消 | `deepseek.py` | 每层检查非正交性和 `cond(Uvo)<=100` | PASS |
| Dense/Shared FFN | Gate、Up、Down 使用同一中间维置换；Up/Down 使用互逆缩放；输入/输出接 P/Q | `deepseek.py`、`deepseek_streaming.py` | 6 层 FFN order 合法且缩放非 1 | PASS |
| Routed Experts | 专家整体置换；每个专家内部独立 FFN 置换和缩放 | 同上 | 第 1–5 层 expert order、每专家 order/scales 检查 | PASS |
| Router | 专家行按相同 expert order 重排；按论文描述执行行归一化；输入接私有坐标 | `deepseek_streaming.py` | 第 1–5 层从原权重、order、Norm、Q 逐字节重算 | PASS |
| KV Cache / 生成 | 私有 checkpoint 使用自定义 DeepSeek-V3 注册，RoPE order 在 prefill/decode 共用 | `modeling_aloepri_deepseek_v3.py`、`serving/hf_runtime.py` | 真实加载 84/84 张量并完成自回归生成 | PASS |
| M1 / RmDP | 单 token 指数机制先于 $\tau$；$\epsilon_1$ 只在客户端使用 | `src/aloepri/privacy/rmdp.py` | 64 token 可执行检查、范围检查和改变率记录 | PASS（逐 token 产品实现） |
| 在线边界 | 请求只含私有 IDs；响应只含私有 IDs；日志只留元数据 | `src/aloepri/serving/app.py`、`verify_product_privacy_boundary.py` | 唯一 marker、明文 token、错误 key/model/token 和包扫描 | PASS |
| 密钥与工件 | online 只含 $\tau,\tau^{-1}$；offline 保存 P/Q 和层密钥；server 只含私有权重 | `deepseek_streaming.py`、`packaging.py` | 文件集合、字节数、SHA-256 和禁用秘密扫描 | PASS |

## 4. 当前固定参数

```yaml
expansion_h: 128
lambda: 0.3
alpha_e: 1.0
alpha_h: 0.2
block_beta: 8
sampling_gamma: 1000.0
qk_scale_min: 0.5
qk_scale_max: 2.0
uvo_condition_max: 100.0
ffn_scale_min: 0.5
ffn_scale_max: 2.0
rms_mode: paper_kappa
router_normalize: true
dtype: bfloat16
```

私有隐藏宽度为 `1280 + 2*128 = 1536`。Embedding 和 LM Head 已解除权重绑定，确保两处可以使用论文要求的独立噪声。

## 5. 论文公式中必须明确记录的限制和修正

实现没有静默改变论文。已确认的问题分别保存在 `docs/paper_errata/`：

1. Algorithm 1 的 C/D 零空间维度、多个独立右逆和矩阵方向问题：E01、E02、E08。
2. BlockPerm 的 `gamma`、循环更新、边界和 Q/K 对消式问题：E03–E06。
3. RMSNorm 标量 $\kappa$ 对一般矩形 $P$ 不是严格函数等价：E07、E15。当前 OpenSeek checkpoint 选择论文标量路线，因此“机制忠实”通过，但不能把它写成无条件精确等价。
4. RoPE 频率指数和 Attention 缩放可逆性问题：E10、E13。
5. 服务端 tokenizer round-trip 会破坏任意 token 置换：E12；产品改用 token-ID-only 协议。
6. RoPE BlockPerm 一般不保持原函数：E16；当前实现确保 Q/K/频率同步，保留论文机制，但不宣称噪声后 token 必须与原模型逐个相同。
7. 论文序列置换距离并非对全部 token 序列有限：E17；产品只把逐 token M1 预算作为逐 token 机制报告。
8. Router 行归一化会改变路由 logit 的尺度关系。代码按论文描述实现并逐公式验证，但这不是严格函数保持变换；效果必须由精度实验评价。
9. 同步 RoPE 需要服务器运行时知道 pair order，因此服务器包额外保存派生顺序张量。它不是词表置换、P/Q、随机种子或逆密钥，但包扫描会显式列出而不是隐藏。

## 6. 实测结果

### 6.1 机制与边界

| 指标 | 实测 |
|---|---:|
| 审计检查 | 16/16 为真 |
| 源张量转换 | 83/83 |
| P/Q 最大相对误差 | `3.265e-14` |
| 服务端秘密扫描 | 0 个禁用秘密 |
| 明文 marker 出现在传输/日志 | 否 |
| 明文 token 序列出现在传输/日志 | 否 |
| 传输 IDs 是否等于 $\tau(x)$ | 是 |
| 错 key / 错 model / 越界 token | 全部安全失败 |

### 6.2 真实问答

固定问题：`你好，请用一句话介绍你自己。`

恢复回答：`你好，我很高兴 to be to be`

| 指标 | 实测 |
|---|---:|
| 输入 token | 25 |
| 输出 token | 8 |
| CPU BF16 TTFT | 1275.674 ms |
| CPU BF16 TPOT | 82.143 ms |

这证明产品链路可以真实问答，但该回答后半段发生重复，说明当前噪声和 Router 参数仍需做效用校准。功能完整性不以回答风格替代，产品质量也不能只由“能生成”判定。

### 6.3 64 候选攻击冒烟

| 攻击 | 恢复率 | 随机基线 |
|---|---:|---:|
| 直接权重匹配 | 0/64 | 不适用；残差宽度已从 1280 变为 1536 |
| Embedding–Head VMA | 7/64 = 10.9375% | 1.5625% |
| Embedding–Gate VMA | 7/64 = 10.9375% | 1.5625% |
| MLA Attention-IA self-score | 2/64 = 3.125% | 1.5625% |

known-plaintext 在观测 1、8、32、64 个不同映射后，恢复的全词表比例分别为 `1/151851`、`8/151851`、`32/151851`、`64/151851`。这是攻击程序可执行性检查，不是完整论文攻击结果。

## 7. 人工复核命令

在 `E:\AloePri` 执行：

```powershell
.\.venv\Scripts\python.exe scripts\audit_openseek_paper_complete.py `
  --source data\models\openseek-small-v1-sft-transformers `
  --private data\packages\openseek-small-v1-sft-paper-complete `
  --offline-key-dir data\keys\openseek-small-v1-sft-paper-complete-offline `
  --online-key-dir data\keys\openseek-small-v1-sft-paper-complete-online `
  --out artifacts\openseek-small-v1-sft-paper-complete\privacy-completeness-audit.json

.\.venv\Scripts\python.exe scripts\verify_product_privacy_boundary.py `
  --server http://127.0.0.1:8011 `
  --online-key-dir data\keys\openseek-small-v1-sft-paper-complete-online `
  --tokenizer data\models\openseek-small-v1-sft-transformers `
  --server-package data\packages\openseek-small-v1-sft-paper-complete `
  --server-log artifacts\openseek-small-v1-sft-paper-complete\server.stdout.log `
  --server-log artifacts\openseek-small-v1-sft-paper-complete\server.stderr.log `
  --out artifacts\openseek-small-v1-sft-paper-complete\product-privacy-boundary.json

.\.venv\Scripts\python.exe scripts\run_openseek_privacy_attack_smoke.py `
  --source data\models\openseek-small-v1-sft-transformers `
  --private data\packages\openseek-small-v1-sft-paper-complete `
  --online-key-dir data\keys\openseek-small-v1-sft-paper-complete-online `
  --sample-size 64 `
  --out artifacts\openseek-small-v1-sft-paper-complete\privacy-attack-smoke-64.json
```

任何工件路径、字节数或 SHA-256 变化后，必须重新运行审计和证据索引；旧 JSON 不得继续作为当前 checkpoint 的结论。

## 8. 尚需执行的性能验收

代码层没有待补的适用隐私模块。发布前仍需运行：

- 至少 5 个独立 key 的全词表 VMA、IA、IMA、ISA、TFMA、SDA 和 known-plaintext；
- 完整精度集及配对 bootstrap 95% 置信区间；
- 当前参数的长上下文、KV Cache、TTFT、TPOT、吞吐和显存统计；
- 对 Embedding/Head 噪声、Router 归一化和 FFN/QK 缩放做独立校准，消除当前回答重复；
- 若未来得到真实 MTP 权重，再新增 MTP 投影、MTP Transformer block、MTP Head 和推测解码的专门转换与验收。

这些项目决定“隐私和效用性能是否达标”，不再属于“代码里有没有实现该隐私功能”的问题。
