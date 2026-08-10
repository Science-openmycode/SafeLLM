# AloePri 论文复现所缺信息

核对版本：arXiv:2603.01499v2，核对日期：2026-08-06。

本文件只记录论文未给出或无法唯一确定的复现参数。可由公式直接判定的错误见
`docs/paper_errata/` 下 E01–E10。

## 1. 论文已经给出的信息

| 项目 | 论文给出的内容 |
|---|---|
| 生成参数 | temperature 0.65、top-k 20、top-p 0.95、max-seq-len 8192、BF16 |
| 默认隐私参数 | `alpha_e=1.0`、`alpha_h=0.2`、`lambda=0.3`、`h=128`、`beta=8`、`gamma=1000` |
| 服务框架 | vLLM 0.9.1、CUDA 12.4 |
| 客户端硬件 | 2 × Intel Xeon Platinum 8457C，合计 96 核 |
| IMA 结构 | Qwen2 架构、2 层、8 个 Attention head |
| VMA 基本规则 | 对行内元素排序，逐层恢复后投票 |
| 数据集名称 | SST2、MMLU、C-Eval、HumanEval、PIQA、IFEval、PUPA、CCI3、Huatuo26M-Lite、MedDialog |

## 2. 模型与软件版本

| 缺项 | 对复现的影响 | 本仓库处理 |
|---|---|---|
| 没有 checkpoint 仓库名、commit、revision 和文件哈希 | 无法确认表中模型的准确权重版本 | 0.5B 固定本地模型目录，并在 manifest 中记录 SHA-256 |
| 没有 Transformers、PyTorch、lm-eval-harness、数据集 revision | 分词、模板和评分可能随版本变化 | 锁定本地环境；结果 JSON 保存库版本 |
| 没有服务器 GPU 型号、数量、节点数和互联 | 无法重算论文 TTFT、TPOT 和转换时间 | 本机结果只与同一 RTX 3060 Laptop GPU 基线比较 |
| 没有公开代码仓库地址 | 无法核对作者实现对公式错误和歧义的实际处理 | 依据论文源码独立实现；截至核对日未在论文页和 ByteDance 公共仓库列表中找到 AloePri 代码 |

公开入口：[arXiv:2603.01499](https://arxiv.org/abs/2603.01499)，
[ByteDance 公共仓库列表](https://github.com/orgs/bytedance/repositories)。

## 3. 离线模型变换

| 缺项 | 无法唯一确定的内容 | 本仓库固定值 |
|---|---|---|
| 全部随机种子 | 词表置换、P/Q、噪声、head permutation、block permutation、FFN permutation 的具体样本 | 每个 checkpoint 在 `key.json` 中记录 seed；密钥张量单独保存 |
| Algorithm 1 的共享范围 | Q/K/V/O、Embedding、Head、FFN 是共用一次 `Init`，还是每个矩阵重新 `Init` | v18 固定共用一次 `Init=(B,C,E,F,Z)` 和同一个 P；六个分支分别采样满足 `ED_j=0` 的 `D_j`，严格构造 `Q_j=Z^T[B^{-1};F;D_j]`。`D_j` 的采样分布论文未给出 |
| RMSNorm 的期望估计协议 | 使用哪个语料、多少 token、每层还是全局、均值还是中位数 | 20 条固定 prompt、822 个 token 向量/层、BF16、逐层样本均值和标准误 |
| 噪声实际采样细节 | 随机生成器、精度、先加噪还是先 cast、权重标准差的统计范围 | FP64/FP32 中按完整张量标准差采样，再按 checkpoint dtype 保存 |
| Uvo 的数值处理 | 是否拒绝病态高斯矩阵、阈值是多少 | 论文参数分支不加条件数筛选；工程分支记录条件数上限 100 |
| FFN 缩放采样分布 | 论文只说随机 scaling，未给区间和分布 | 对角缩放从 `[0.5,2.0]` 均匀采样 |

## 4. 准确率评测

| 缺项 | 对结果的影响 | 本仓库协议 |
|---|---|---|
| MMLU、C-Eval 的 few-shot 数和 prompt 模板 | 绝对分数不可直接比较 | 本地固定 0-shot YAML；明文和混淆模型共用逐样本文件 |
| IFEval 的 evaluator revision | strict/loose 规则可能变化 | 保存逐 prompt、逐 instruction 布尔结果 |
| HumanEval 的生成条数、停止词、执行器版本 | pass@1 会变化 | 每题 1 个确定性生成，隔离执行测试；逐题保存通过/失败 |
| 随机采样与重复次数 | temperature 0.65 下单次结果方差未知 | 准确率主表采用确定性评分；差值用 10,000 次配对 bootstrap |
| 论文表格没有置信区间 | 无法判断小幅差值是否属于抽样波动 | 同时报告差值和 95% 配对置信区间 |

## 5. 隐私攻击与指标

| 缺项 | 论文中缺少的参数 |
|---|---|
| PUPA 预处理 | split、字段清洗、是否去重、PII 单元切分、特殊 token 处理 |
| VMA 候选空间 | 全词表还是闭集候选；每个 token 的 decoy 数量；距离归一化；投票平局规则 |
| VMA 组合 | Table 9 列出 6 组权重，但没有说明主表使用单组最强值、组合投票值还是平均值 |
| TTRSR | 分母是 token occurrence、unique token、输入文本 token，还是仅 PII token |
| PIIRSR | 单元完全匹配、token 比例、字符串匹配和规范化规则均未给出 |
| BLEU-4 | tokenizer、大小写、平滑方法、0–1 或 0–100 量纲未给出 |
| CosSim | 句向量模型、模型 revision、pooling 和归一化方法未给出 |
| Gate-IA | 候选空间、距离、层选择和投票方法未给出 |
| Attn-IA | 论文公式维度不成立；没有可执行替代公式 |
| ISA | 目标层、优化步数、学习率、初始化、损失函数和候选 token 搜索方法未给出 |
| IMA | 公共训练集、训练样本量、步数、batch size、优化器、目标函数和独立密钥采样协议未给出 |
| NN | 使用哪一层表示、距离、候选范围和 top-k 规则未给出 |
| TFMA | 三组公开/私有语料的 split、样本量、tokenizer 和频率并列处理未给出 |
| SDA | Transformer 宽度、训练步数、batch size、学习率、最大长度和 BLEU 计算方法未给出 |

本仓库的每个攻击 JSON 必须记录 `candidate_scope`、样本数、层、距离、训练规模和
`paper_exact`。只有字段完整且协议与论文文字一致的工件进入主结果表；代理实验和闭集实验单列。

## 6. 性能评测

论文给出了 R1-Distill-14B、TP=4、平均 17-token 输入、100-token 输出和不同并发，
但没有给出以下内容：

- GPU 型号、显存、CPU、内存、节点互联和功率状态；
- vLLM 启动参数、KV Cache dtype、GPU memory utilization、max-num-seqs；
- 每档并发的请求数、warm-up 次数和重复次数；
- TTFT/TPOT 的均值、分位数或误差区间；
- tokenizer 时间和客户端置换时间是否计入 TTFT；
- 离线转换计时是否包含模型下载、对象存储读写和 checksum。

本仓库性能表固定同一设备、同一 prompt 顺序、20 次请求、每次生成 100 token，分别报告
p50、p95、p99，并用配对 bootstrap 给出相对劣化的 95% 上界。
