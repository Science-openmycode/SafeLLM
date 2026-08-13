# Qwen2.5-0.5B AloePri 论文、PPT 与代码复核报告

复核日期：2026-08-13
复核对象：当前工作区 `E:\AloePri` 及 Qwen2.5-0.5B 工件
代码基线：`codex/aloepri-baseline`，起始提交 `9f7af0e008bc05909b55361a4feed6a9c2e4a072`

## 1. 复核基线

| 基线文件 | SHA-256 | 用途 |
|---|---|---|
| `01_client_technical_route.pdf` | `a5fd16662e1ac2feb55952940edd0fdf9d4007da22bd2f0eacb5f5ea41dc4748` | AloePri 论文 arXiv:2603.01499v2，共 28 页 |
| `02_client_project_brief.pptx` | `244b6fcd3b48f75d8a77e84dd16df5456c1993ec840c0c719cb148a46360b337` | 甲方功能、架构和性能目标，共 7 页 |
| `.codex_review/arxiv_v2_source/src/chap/` | 文件级哈希见逐行审计 JSON | 论文 LaTeX 原文、算法、证明和实验表 |
| `artifacts/acceptance/qwen05b-v47-reaudit-20260813.json` | 由当前配置重新生成 | v47 正式验收结论，不沿用旧 checkpoint 结论 |
| `artifacts/audit/paper-line-audit-v47-20260813.json` | 文件内包含 31 个源码文件的 SHA-256 | 论文相关代码逐行分类账本 |

本次逐行审计范围为 31 个核心文件、6,081 条非空且非纯注释的物理行。6,081 条均有分类；这个数字只说明没有遗漏审计范围内的代码行，不表示代码正确率为 100%。

## 2. 论文要求的完整数据流

设明文模型为 $f_\theta$，输入 token 序列为 $x$，模型输出为 $y=f_\theta(x)$。论文的目标不是只修改输入，而是同时变换输入、权重和输出：

$$
\widetilde f_{\phi_\Theta(\theta)}\left(\phi_X(x)\right)
\approx \phi_Y\left(f_\theta(x)\right).
$$

对离散 token，客户端持有词表置换 $\tau$ 和逆置换 $\tau^{-1}$：

$$
\widetilde x=\tau(x),\qquad
\widetilde y=\widetilde f(\widetilde x),\qquad
\widehat y=\tau^{-1}(\widetilde y).
$$

真实客户端和服务端应按以下顺序工作。

| 顺序 | 执行位置 | 论文/PPT要求 | 本工程实际行为 |
|---:|---|---|---|
| 1 | 客户端 | 接收真实 prompt | 浏览器或 CLI 接收真实中文/英文问题 |
| 2 | 客户端 | 使用原模型 tokenizer | 应用 Qwen chat template，再得到明文 token IDs |
| 3 | 客户端 | 可选 M1/RmDP 扰动 | `rmdp` 模式逐 token 执行指数机制；默认展示模式不扰动 |
| 4 | 客户端 | 执行 $\tau$ | 每个明文 ID 映射为私有 ID；长度不变 |
| 5 | 网络 | 发送混淆输入 | 正式模型接口发送私有 ID，不发送 prompt 和明文 ID |
| 6 | 服务端 | 使用混淆 checkpoint | Embedding、Attention、FFN、RMSNorm、Residual 和 LM Head 均在私有坐标中计算 |
| 7 | 服务端 | 输出混淆 token | 返回私有输出 ID；服务端不执行 $\tau^{-1}$ |
| 8 | 客户端 | 执行 $\tau^{-1}$ | 私有输出 ID 恢复为原词表 ID |
| 9 | 客户端 | tokenizer 解码 | 只在客户端得到最终自然语言回答 |

PPT 第 5 页把步骤 4 后的 ID 再解码成“乱码文本”、由服务端重新分词。一般 tokenizer 不满足 `encode(decode(ids)) == ids`，这个路径不能保证可逆。生产接口因此使用 token ID；展示页面仍会把私有 ID 解码成可视字符，但真正发给模型服务的权威数据始终是私有 ID。

## 3. 离线模型变换与实现位置

### 3.1 词表、Embedding 和 LM Head

论文使用同一个置换矩阵 $\Pi$ 同步处理输入词表、Embedding 行和 LM Head 输出：

$$
W_e^\star=W_e+\alpha_e\mathcal E_e,
\qquad
W_h^\star=W_h+\alpha_h\mathcal E_h,
$$

$$
\widetilde W_e=\Pi W_e^\star P_e,
\qquad
\widetilde W_h=Q_hW_h^\star\Pi^T.
$$

| 检查项 | 代码 | 结论 |
|---|---|---|
| `tau[plain_id] -> private_id` 与逆置换 | `src/aloepri/transforms/vocab.py`、`src/aloepri/client/sdk.py` | 已实现并通过全词表往返检查 |
| Embedding/Head 高斯噪声按权重总体标准差缩放 | `src/aloepri/transforms/paper_noise.py` | 与论文给出的噪声形式一致 |
| Embedding 与 LM Head 使用同一词表置换 | `src/aloepri/conversion/paper_qwen2.py` | 已实现 |
| 特殊 token 的私有 ID | online key 的 `key.json` | 已实现，服务端 EOS 判断使用私有 EOS |

### 3.2 Algorithm 1 的 P/Q 扩维

论文令原隐藏维度为 $d$，扩维后为 $D=d+2h$：

$$
P\in\mathbb R^{d\times D},
\qquad
Q\in\mathbb R^{D\times d},
\qquad
PQ=I_d.
$$

实现保留论文的矩形扩维路线，没有换成自行发明的可逆方阵。`src/aloepri/transforms/paper_key_matrix.py` 生成共享 INIT、独立 $D_j$ 的兼容右逆族；`scripts/verify_paper_formula_checkpoint.py` 从 checkpoint 反算 $PQ$、所有层张量形状和 Algorithm 2 组合关系。

论文 Algorithm 1 中 $C$、$D$ 的零空间维数描述存在不一致，原式不能直接得到打印形状。生产代码采用满足打印形状且能证明 $PQ=I_d$ 的修正式。推导见 `docs/paper_errata/E01_*`、`E02_*` 及对应 PDF。

### 3.3 Attention、GQA、RoPE 与 KV Cache

论文对 Q/K/V/O 使用旋转、缩放、head permutation、block permutation 和 value/output 可逆映射。Qwen2.5-0.5B 还要求 14 个 Q heads 与 2 个 KV heads 的 GQA 关系保持同步。

| 检查项 | 代码位置 | 结论 |
|---|---|---|
| Q/K 旋转与互逆缩放 | `src/aloepri/transforms/qwen_structural.py` | 已实现 |
| Q/K 同步 head/block permutation | 同上 | 已实现；采用可保持内积的同侧置换修正式 |
| Qwen `rotate_half` 实际坐标和频率配对 | 同上、`src/aloepri/models/modeling_aloepri_qwen2.py` | 已实现 |
| GQA 的 Q head 到 KV head 分组 | 同上 | 已实现并有结构测试 |
| V/O 可逆映射 | `paper_qwen2.py`、`qwen_structural.py` | 已实现并限制条件数 |
| Prefill/decode 私有坐标与 KV Cache 推进 | `modeling_aloepri_qwen2.py`、`hf_runtime.py` | HF 当前 checkpoint 已通过功能验证 |

论文打印的 `q'=qZ, k'=kZ^T` 通常产生 $qZ^2k^T$，不能保证注意力分数不变；生产代码使用 $q'=qZ, k'=kZ$。论文还允许跨 RoPE 频率任意 block permutation，但这不与位置相关旋转交换；生产代码把置换限制为保持 RoPE 频率对同步。对应推导为 E06、E10、E13、E16。

### 3.4 FFN

Qwen 的 SwiGLU 为：

$$
\operatorname{FFN}(x)=
\left(\operatorname{SiLU}(xW_{gate})\odot xW_{up}\right)W_{down}.
$$

工程同步变换 gate、up 和 down 的中间维置换，并成对使用论文的缩放与逆缩放。实现位于 `src/aloepri/transforms/qwen_structural.py` 和 `src/aloepri/conversion/paper_qwen2.py`。这部分适用于 Qwen2.5-0.5B 的稠密 FFN；MoE router 和 expert permutation 不属于该模型结构，不计入这个 checkpoint 的函数缺失。

### 3.5 RMSNorm 与 Residual

论文用一个期望标量 $\kappa$ 近似扩维后的范数变化。对一般矩形 $P$，单一标量不能使所有输入都保持 RMSNorm 函数等价。本工程的正式 v47 checkpoint 使用精确度量：若私有状态 $z=xP$ 且 $PQ=I$，则

$$
\operatorname{RMS}(x)^2
=\frac{1}{d}zQQ^Tz^T
=\frac{1}{d}\lVert zF\rVert_2^2,
\qquad FF^T=QQ^T.
$$

第二种稳定因子形式避免 FP32 直接计算 Gram 二次型时发生正负消去。实现位于 `src/aloepri/models/modeling_aloepri_qwen2.py`；转换/升级脚本为 `scripts/upgrade_exact_rms_stable_factor.py`。Residual 两条支路保持同一个私有坐标系。该修正不是论文打印的标量近似，原因和证明见 E07、E15 以及 `docs/QWEN05B_EXACT_RMS_STABLE_FACTOR.md`。

## 4. 在线产品、安全边界和展示页面

### 4.1 已实现组件

| 组件 | 位置 | 本次复核 |
|---|---|---|
| token-ID FastAPI 服务、SSE | `src/aloepri/serving/app.py` | 已实现；本次补充实际 body 长度检查，不能靠伪造 `Content-Length` 绕过配置门限 |
| HF 私有模型运行时 | `src/aloepri/serving/hf_runtime.py` | 已实现，真实 0.5B 问答通过 |
| 客户端 online key 与逆置换 | `src/aloepri/client/sdk.py` | 已实现 |
| 离线 master key / online key / server package 拆分 | `src/aloepri/packaging.py` | 代码已实现；当前 v47 正式 server package 尚未生成，因此验收仍失败 |
| Bearer Token、远程 HTTP 拒绝、日志字段限制 | serving/client 模块 | 已实现；远程部署仍必须由 TLS 反向代理终止 HTTPS |
| 本地展示页面 | `src/aloepri/demo/` | 本次新增并完成真实 checkpoint 闭环 |

### 4.2 展示页面显示的数据

页面绑定 `127.0.0.1:7860`，把浏览器和本地 gateway 作为可信客户端。它显示：

1. 用户输入的真实 prompt；
2. chat template 后的明文 token IDs；
3. 实际发送给模型服务的私有 token IDs；
4. 私有输入 ID 用原 tokenizer 解码后的可视字符；
5. 模型服务实际返回的私有输出 IDs 和可视字符；
6. $\tau^{-1}$ 恢复后的 token IDs 与最终回答；
7. 本次输入前 96 个 token 的 `plain_id -> private_id` 映射样本；
8. request ID、model/key ID、TTFT、TPOT 和端到端时间。

展示推理使用真实流式链路。模型服务生成一个私有 token 后立即发送 SSE 事件；可信 gateway 检查 request ID 和连续序号、执行一次 $\tau^{-1}$，再把累计私有文本和累计恢复文本发送给浏览器。浏览器按真实到达顺序更新私有 token，同时以 22 ms 的字符间隔呈现新恢复的字符。字符动画只平滑显示单个 token 内可能包含的多个字符，不会预先持有完整回答。

页面不会向模型服务发送 prompt，也不会展示或导出完整 `tau`/`inverse_tau`。浏览器到本地 gateway 之间含明文，因为它们共同属于可信客户端；如果把 7860 端口公开到远程网络，这个边界假设就不成立。

### 4.3 本次真实展示闭环结果

运行对象为低噪声展示 checkpoint `qwen05b-product-v31-blockperm8`，不是 v47 正式安全验收 checkpoint。

| 项目 | 本次值 |
|---|---:|
| 输入 | `请用一句话解释矩阵乘法。` |
| 明文输入 token 数 | 37 |
| 私有输入 token 数 | 37 |
| 两组 ID 是否不同 | 是 |
| 私有输出 token 数 | 23 |
| 恢复输出 token 数 | 23 |
| 回答 | `矩阵乘法是一种数学运算，用于将一个矩阵与另一个矩阵相乘，以生成一个新的矩阵。` |
| TTFT | 282.55 ms |
| TPOT | 90.57 ms/token |
| gateway 端到端 | 2,335.08 ms |

该结果证明当前机器可以实打实输入问题、让混淆 checkpoint 生成私有 token、再在客户端恢复回答。它不证明 v31 达到论文隐私指标；v31 的作用是稳定产品演示。

流式改造后再次使用同一问题实测：共收到 23 个严格连续的 token 事件；首个 token 在请求后约 1,304.62 ms 到达，第二个在约 1,393.66 ms 到达，流在约 3,679.65 ms 完成。模型内部本次 TTFT 为 341.11 ms、TPOT 为 106.68 ms/token。不同计时起点包含浏览器/gateway 分词和传输，因此外部首事件时间不等于模型内部 TTFT。

## 5. 论文/PPT 功能逐项结论

| 论文或 PPT 项目 | Qwen0.5B 状态 | 说明 |
|---|---|---|
| 词表置换和逆置换 | 已实现 | 全词表往返通过 |
| Embedding/Head 同步置换 | 已实现 | 公式反算通过 |
| Embedding/Head 高斯噪声 | 已实现 | 噪声参数可配；不要求加噪后 token 与明文逐字相同 |
| $d\to d+2h\to d$ P/Q | 已修正实现 | 保留论文架构，修正 Algorithm 1 维数问题 |
| Attention Q/K/V/O 变换 | 已修正实现 | 修正 block permutation 取消关系 |
| GQA、RoPE、head/block permutation | 已修正实现 | 按 Qwen 实际布局限制为保持函数的同步置换 |
| 稠密 FFN | 已实现 | gate/up/down 同步 |
| RMSNorm、Residual | 已修正实现 | v47 使用 exact-metric stable factor |
| checkpoint 转换、保存、加载 | 已实现 | 支持 safetensors/manifest；v47 sanitized server package 尚缺 |
| HF forward/generate/KV Cache | 已实现 | 当前 HF 功能验证通过 |
| vLLM | 有实现，当前验收失败 | v47 HF/vLLM 32-token 序列不一致 |
| SGLang | 有实现，尚未形成 v47 正式证据 | 当前验收为 NOT_TESTED |
| CUDA/任意框架“直接兼容” | 未按 PPT 字面满足 | 扩维和精确 RMS 需要自定义模型适配器，不是零代码直接加载 |
| CLI/SDK/API/SSE | 已实现 | 正式模型接口只使用私有 token IDs |
| M1/RmDP | 论文计算器 + 工程替代 | 精确小词表 oracle 已实现；大词表采用逐 token 组合，不冒充论文未给出的长序列采样器 |
| VMA/权重匹配/IA/IMA/ISA | 已实现并有当前工件 | 部分指标未过门限，见下一节 |
| TFMA/SDA | 算法已实现，正式协议未完成 | 当前 corpus 标记为 `formal_corpus_complete=false` |
| MoE、MLA、MTP | 对 Qwen0.5B 不适用 | 这个 checkpoint 没有 router、experts、MLA 或 MTP 参数，不能伪造为 0.5B 实测 |

结论：Qwen2.5-0.5B 架构实际具备的模型变换已经有对应实现；其中不能成立的论文打印公式均采用单独推导过的修正式。不能把这个结论写成“论文全部字面公式 100% 原样实现”，因为原文存在相互矛盾的公式，也不能把尚未通过的攻击、跨后端和性能实验写成已经完成。

## 6. v47 当前正式验收结果

v47 正式目标为 `data/packages/qwen05b-candidate-v47-stable-factor`。重新绑定当前工件后，验收器结论为 `NO-GO`：21 项中 6 PASS、9 FAIL、6 NOT_TESTED。

### 6.1 已通过

| 检查 | 当前结果 |
|---|---:|
| 词表往返、Algorithm 1、Algorithm 2 | PASS |
| 检查层数 | 24/24 |
| 公式、运行时、功能验证汇总 | PASS |
| 直接权重匹配 Top-1 | 0.0000 |
| Attention-IA Top-1 | 0.00390625 |
| IMA TTRSR | 0.00001 |
| ISA-Attention TTRSR | 0.0000 |

### 6.2 未通过或证据不足

| 检查 | 当前实测/状态 | 门限或原因 |
|---|---:|---|
| VMA TTRSR | 30.7868% | 要求不超过 15% |
| Gate-IA Top-1 | 19.1406% | 要求不超过 15% |
| TFMA Top-10 | 1.9742% | 数值过门限，但正式三语料协议未完成，因此 FAIL |
| SDA TTRSR / BLEU-4 | 11.8416% / 0.1991 | 数值过门限，但正式三语料协议未完成，因此 FAIL |
| Known-plaintext TTRSR | 17.1910% | 要求不超过 15% |
| MMLU | -4.7785 个百分点 | 超过 3.5；且明文/私有 dtype 绑定不一致 |
| C-Eval | -18.5736 个百分点 | 超过 3.5；且 dtype 绑定不一致 |
| PIQA | -7.4538 个百分点 | 超过 3.5；且 dtype 绑定不一致 |
| HF/vLLM | 32-token 序列不一致 | FAIL |
| IFEval、HumanEval | NOT_TESTED | 当前 stable-factor checkpoint 无完整绑定结果 |
| vLLM/SGLang | NOT_TESTED | 无当前正式工件 |
| 产品 100 问、抓包隐私、性能 | NOT_TESTED | 无当前正式工件 |
| v47 sanitized server package | 不存在 | `data/server-packages/qwen05b-candidate-v47-stable-factor` 尚未生成 |

MMLU、C-Eval、PIQA 的差值可用于发现质量问题，但由于 `dtype_match=false`，不能作为严格同条件最终比较。它们当前既没有通过点估计门限，也没有通过配对 95% CI 下界门限。

## 7. 与论文原始结果和 PPT 目标的对比

只列已经获得当前数值的项目；未测试项目不填入此表。

| 指标 | 论文 Qwen3-14B 原始数据 | PPT/项目目标 | 本工程 Qwen2.5-0.5B v47 | 是否达到本项目门限 |
|---|---:|---:|---:|---|
| MMLU 绝对下降 | -3.07 pp | 不超过 3.5 pp | -4.7785 pp | 否；且 dtype 绑定不一致 |
| C-Eval 绝对下降 | -0.23 pp | 不超过 3.5 pp | -18.5736 pp | 否；且 dtype 绑定不一致 |
| PIQA 绝对变化 | +0.54 pp | 下降不超过 3.5 pp | -7.4538 pp | 否；且 dtype 绑定不一致 |
| VMA TTRSR | 25.05% | 本工程门限不超过 15% | 30.7868% | 否 |
| VMA PIIRSR | 1.62% | 不超过 3% | 1.0542% | 是 |
| VMA BLEU-4 | 1.72 | 本工程门限不超过 2.5 | 0.3701 | 是 |

论文结果来自 Qwen3-14B，当前结果来自 Qwen2.5-0.5B，二者只能并列报告，不能把差异只归因于参数量。要证明参数量是原因，仍需相同代码、相同数据、相同 dtype 和多个模型规模的受控实验。

## 8. 论文错误与工程修正记录

已建立 17 组一事一文档的 Markdown/PDF 推导，位于 `docs/paper_errata/` 和 `output/pdf/paper_errata/`：

| 编号范围 | 内容 |
|---|---|
| E01–E02 | Algorithm 1 的 C/D 零空间维数与打印形状 |
| E03–E05 | BlockPerm 的 gamma 未使用、循环变量不更新、边界错误 |
| E06 | Attention block permutation 的取消方向 |
| E07、E15 | RMSNorm 的维度缩放与标量 $\kappa$ 不能精确等价 |
| E08–E10、E13、E16 | Attention key 独立性、IA 维度、RoPE 频率指数、缩放可逆性、跨频率置换 |
| E11 | summation deobfuscation 符号 |
| E12 | 在线 tokenizer 文本往返不可保证 |
| E14 | 误差上界乘积索引 |
| E17 | 论文 token-sequence permutation distance 不是所有序列上的有限度量 |

生产代码只保留推导后可执行的修正式；逐行审计把这些代码标为 `PAPER_CORRECTED`，不会标为 `PAPER_EXACT`。

## 9. 展示页面运行命令

终端 1：启动真实混淆模型服务。

```powershell
cd E:\AloePri
uv run aloepri serve --config configs/product/qwen05b_v31_blockperm8.yaml
```

终端 2：启动可信本地展示客户端。

```powershell
cd E:\AloePri
uv run aloepri demo `
  --server http://127.0.0.1:8000 `
  --key-dir data/keys/qwen05b-product-v31-blockperm8-online `
  --tokenizer data/models/qwen2.5-0.5b `
  --port 7860
```

浏览器打开：`http://127.0.0.1:7860`

页面中的私有输入/输出 ID 是模型请求和响应的真实值。私有“文本”只是为了展示，把这些 ID 交给原 tokenizer 得到的可视结果；算法和传输均以 ID 为准。

## 10. 人工检查方法

1. 输入一个带唯一标记的问题，例如 `ALOEPRI-LOCAL-20260813：用一句话解释矩阵乘法`。
2. 确认“明文提示词”显示原问题，“混淆后的提示词”显示不同字符和不同 token IDs。
3. 确认“混淆后的回答”先出现私有输出 IDs，随后“最终回答”出现正常文本。
4. 展开映射表，逐行检查页面显示的 `private_id` 是否等于 online key 中的 `tau[plain_id]`；页面只展示本次输入样本，不展示完整密钥。
5. 在浏览器 Network 中查看 `/api/generate`：这是浏览器到可信本地 gateway 的请求，因此含 prompt。
6. 在模型服务端抓取 `/v1/private/generate`：请求体应只包含 `model_id`、`key_id`、私有 `input_ids` 和生成参数，不应包含 prompt 或明文 IDs。
7. 检查模型服务日志：只应出现 request/model/key ID、token 数和耗时，不应出现正文或完整 token 流。

## 11. 最终判断

当前工程已经具备可用的 Qwen2.5-0.5B 本地产品演示：真实问题在可信客户端分词和置换，混淆模型生成私有回答，客户端逆置换并恢复自然语言。论文在该稠密模型上适用的核心权重变换均有代码；原文不能成立的部分有逐项推导和修正实现。

当前工程尚不能发布为“v47 已满足论文和 PPT 全部性能目标”。正式 v47 状态仍是 `NO-GO`，主要剩余项是高噪声精度、VMA/Gate-IA/known-plaintext 门限、TFMA/SDA 正式语料协议、HF/vLLM/SGLang 一致性、IFEval/HumanEval、产品问题集、抓包和性能证据，以及 sanitized server package。展示 checkpoint v31 与正式验收 checkpoint v47 必须继续分开报告。
