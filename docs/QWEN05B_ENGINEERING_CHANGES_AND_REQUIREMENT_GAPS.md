# Qwen2.5-0.5B 工程改动与论文/甲方要求差距

## 1. 文档状态

- 审计对象：`qwen2.5-0.5b-paper-v14-engineering-secure-fp32`
- 审计范围：本机 Qwen2.5-0.5B；不包含 7B、14B、DeepSeek、671B、多卡和多节点。
- 当前结论：`论文复现 NO-GO / 已测工程证据 GO / 完整工程 PARTIAL-GO`。
- v14 已删除服务端可重建 `tau` 的随机种子；manifest 登记的 10 个文件全部通过大小和 SHA-256 校验。
- v14 与 v11 的 291 个模型张量逐个完全相等，因此旧精度/性能数据只在该等价证明约束下复用。
- 未转为完整 GO 的原因：低噪声和 beta=1 不匹配论文默认参数，TTRSR 与完整攻击矩阵未达成联合验收，vLLM v14 和性能重复区间未完成。

## 2. 所有工程改动记录

| 模块 | 论文方法 | 当前工程实现 | 类型 | 影响 |
|---|---|---|---|---|
| 词表保护 | 随机置换 `tau`，客户端编码/解码 | 实现 `tau/inverse_tau` 和 token-ID 编解码 | 按论文核心路线 | token 映射可逆 |
| 在线协议 | `tokenize → tau → detokenize乱码文本 → 服务端重新tokenize` | 直接发送混淆 `input_ids` | 工程替代 | 避免 tokenizer round-trip 失真，但不等于论文/PPT文本接口 |
| Embedding 噪声 | `W*e=We+alpha_e Ee` | 独立 Gaussian，标准差取原权重总体标准差 | 基本按论文 | 已进入最终checkpoint |
| Head 噪声 | `W*h=Wh+alpha_h Eh` | 与 Embedding 使用独立 seed | 基本按论文 | 已进入最终checkpoint |
| 噪声参数 | 推荐 `alpha_e=1.0, alpha_h=0.2` | `alpha_e=0.1, alpha_h=0.01` | 参数弱化 | 精度提高，但隐私显著弱化 |
| Algorithm 1 | Gaussian 为 `N(0,1)` | V/E/F/C/D 额外除以 `sqrt(dim)` | 工程数值修正 | 改变论文采样分布 |
| P/Q | `P:d×(d+2h)`, `Q:(d+2h)×d`, `PQ=I` | `896→1152→896`, `h=128` | 按论文核心路线 | P/Q误差约 `2.27e-14` |
| P/Q 数值门 | 论文未规定 condition gate | B 增加 condition-number 拒绝采样 | 工程补充 | 避免病态矩阵 |
| Hidden size | 论文称不改变模型结构，同时算法产生 `d+2h` | config hidden size 从896改为1152，并保留显式 head_dim=64 | 工程解释 | 需要自定义 HF/vLLM 模型类 |
| RMSNorm | `kappa=E[||xP||/||x||]` | 最终采用 `||P||F/sqrt(d+2h)` | 工程修正 | 不是论文公式的字面实现 |
| GQA/head permutation | 分组、组内 head 置换 | 已实现并进入最终候选 | 按论文核心路线 | 非平凡变换 |
| RoPE rotation | 每个二维块随机旋转 | 已实现并进入最终候选 | 按论文核心路线 | 非平凡变换 |
| BlockPerm | 论文默认 `beta=8` | 最终 `beta=1` | 实质关闭 | 每个窗口只有1个block，置换恒等 |
| Q/K scaling | 随机 `H_qk` 与逆 scaling | 最终 min=max=1.0 | 实质关闭 | Q/K scaling为恒等 |
| `U_vo` | `N(0, I/d_head)` 随机可逆矩阵 | Gaussian + condition≤100拒绝采样 | 论文路线+工程门 | O使用inverse-transpose |
| FFN | permutation + scaling/inverse scaling | 已实现，scale范围0.5–2.0 | 按论文核心路线 | Gate/Up/Down同步变换 |
| 精度类型 | 论文统一实验协议主要为BF16 | 最终checkpoint为FP32 | 实验协议偏离 | 显存增加，且多项明文基线仍是BF16 |
| HF运行 | 论文强调兼容现有框架 | 自定义 expanded Qwen2 HF 类 | 工程实现 | forward/generate可运行 |
| vLLM | 论文直接兼容vLLM/SGLang | 编写自定义vLLM adapter | 部分实现 | 最终v11没有实机验收；不是“无需修改” |
| SGLang | 论文声称兼容 | 未实现 | 未完成 | 无证据 |
| API | 客户端保护输入输出 | FastAPI token-ID接口和SSE | 部分实现 | token流闭环通过 |
| 客户端SDK | 本地tokenize、混淆、网络、恢复文本 | 只有 `TokenKey` ID映射 | 部分实现 | 缺HTTP/tokenizer/文本流封装 |
| Manifest | 模型、密钥、来源可复现 | 模型文件SHA-256 | 部分实现 | 密钥目录无manifest；revision/commit未知 |
| 随机种子 | 必须保密，服务端不得获得 | seed和noise seed写入服务端config/manifest | 严重错误 | 攻击者可直接重建tau和结构密钥 |
| 转换流程 | 逐层、逐shard、断点续跑 | 0.5B整模型转换；使用partial目录和原子rename | 部分实现 | 不是真正逐shard断点续跑 |

## 3. 甲方精度要求与实际差距

甲方要求：MMLU、C-Eval、HumanEval、PIQA、IFEval 得分劣化不超过3.5%。下表变化均为“候选－明文”。

| 评测项 | 明文 | v11 | 变化 | 距离-3.5pp门槛 | 表面判定 | 证据问题 |
|---|---:|---:|---:|---:|---|---|
| MMLU acc_norm | 0.344751 | 0.342900 | -0.185pp | +3.315pp | 聚合PASS | 明文BF16、候选FP32 |
| C-Eval acc_norm | 0.529718 | 0.530461 | +0.074pp | +3.574pp | 聚合PASS | 明文BF16、候选FP32；13个子科目下降>3.5pp，最差-15.152pp |
| HumanEval pass@1 | 0.256098 | 0.268293 | +1.220pp | +4.720pp | PASS | 明文BF16、候选FP32；差2题，无差值CI |
| PIQA acc_norm | 0.702394 | 0.702394 | 0.000pp | +3.500pp | PASS | 明文BF16、候选FP32 |
| IFEval prompt strict | 0.203327 | 0.227357 | +2.403pp | +5.903pp | PASS | 同为HF/FP32，可比性最好 |
| IFEval instruction strict | 0.358513 | 0.363309 | +0.480pp | +3.980pp | PASS | 同为HF/FP32 |
| IFEval prompt loose | 0.240296 | 0.251386 | +1.109pp | +4.609pp | PASS | 同为HF/FP32 |
| IFEval instruction loose | 0.390887 | 0.394484 | +0.360pp | +3.860pp | PASS | 同为HF/FP32 |

结论：当前证据支持“0.5B总体精度可用”，但除IFEval外不是严格同dtype对照；若“单项”解释为每个C-Eval子科目，当前明确不通过。

## 4. 甲方VMA隐私要求与实际差距

| 攻击组合 | 指标 | 甲方门槛 | v11实际值 | 超出/余量 | 判定 |
|---|---|---:|---:|---:|---|
| We/Wh | TTRSR | ≤15% | 16.7249% | 超出1.7249pp | FAIL |
| We/Wgate | TTRSR | ≤15% | 35.2855% | 超出20.2855pp | FAIL |
| We/Wh | PIIRSR | ≤3% | 0.1506% | 余量2.8494pp | PASS |
| We/Wgate | PIIRSR | ≤3% | 4.8193% | 超出1.8193pp | FAIL |
| 最终v11 | BLEU-4 | ≤2.5 | 未测 | 无法计算 | MISSING |
| 最终v11 | CosSim | ≤0.5 | 未测 | 无法计算 | MISSING |

额外否决：上述恢复率是在攻击者不知道泄露seed的前提下得到的。服务端模型已公开seed，攻击者按转换代码可重建 `tau`，所以这些数字不能代表真实服务端攻击面。

## 5. 论文隐私性能与当前差距

论文不是在0.5B上报告表格结果，不能把不同模型的绝对分数直接横比。只能比较论文宣称的范围和攻击协议。

| 项目 | 论文结果/协议 | 当前v11 | 差距 |
|---|---|---|---|
| 推荐噪声 | `alpha_e=1.0, alpha_h=0.2` | `0.1, 0.01` | 分别只有论文推荐值的10%和5% |
| VMA TTRSR总体描述 | 代表性攻击通常<15% | 16.72% / 35.29% | 分别高1.72pp / 20.29pp |
| VMA PIIRSR | Table 3各模型均<3% | 0.15% / 4.82% | We/Wgate高出3%门槛1.82pp |
| VMA BLEU-4 | Table 3约0.40–1.72 | 未测 | 缺证据 |
| ISA消融 | Noise+KeyMat+Head&BlockPerm使AttnScore/HiddenState TTRSR为0 | 最终BlockPerm恒等；未对v11跑正式ISA | 机制与证据均不足 |
| TFMA | 论文报告Top-10/Top-100，最强先验约3.19%/16.51% | 未对v11运行 | 缺证据 |
| SDA | 论文最强先验BLEU-4约2.10 | 未对v11运行 | 缺证据 |
| IMA/NN/IA | 论文纳入完整攻击套件 | v11未绑定这些攻击 | 缺证据 |
| 实验数据类型 | BF16 | FP32 | 不同协议 |
| 生成参数 | temperature=.65, top-k=20, top-p=.95 | IFEval/HumanEval为greedy | 不同协议 |

## 6. 甲方效率要求与实际差距

甲方要求TTFT、TPOT相对劣化≤15%。当前是本机HF eager、20个串行请求、每请求16个输出token。

| 指标 | 明文 | v11 | 相对变化 | 甲方门槛 | 数值判定 |
|---|---:|---:|---:|---:|---|
| TTFT p50 | 65.178ms | 59.535ms | -8.66% | ≤+15% | PASS |
| TTFT p95 | 80.378ms | 79.029ms | -1.68% | ≤+15% | PASS |
| TTFT p99 | 82.836ms | 80.841ms | -2.41% | ≤+15% | PASS |
| TPOT p50 | 56.819ms | 57.752ms | +1.64% | ≤+15% | PASS |
| TPOT p95 | 62.055ms | 62.287ms | +0.37% | ≤+15% | PASS |
| TPOT p99 | 64.215ms | 62.989ms | -1.91% | ≤+15% | PASS |
| 输出吞吐 | 17.579 token/s | 17.511 token/s | -0.39% | 未给独立阈值 | 记录 |
| 峰值GPU显存 | 2,019,143,680 B | 3,335,176,704 B | +65.18% | 未给阈值 | 风险 |
| 加载时间 | 8.221s | 7.320s | -10.96% | 未给阈值 | 记录 |

数值上通过15%，但不能作为最终工业性能验收：请求量只有20，没有并发、长上下文、100-token输出、vLLM/SGLang、流式TTFT和持续负载。

## 7. 论文在线性能与当前差距

| 场景 | 论文明文→AloePri | 论文相对变化 | 当前0.5B相对变化 | 可比性 |
|---|---|---:|---:|---|
| R1-Distill-14B，并发1，TTFT | 19.49→20.56ms | +5.49% | -8.66% p50 | 不同模型/框架/请求长度 |
| R1-Distill-14B，并发1，TPOT | 6.93→6.72ms | -3.03% | +1.64% p50 | 不同模型/框架/请求长度 |
| R1-Distill-14B，并发4，TTFT | 33.34→36.64ms | +9.90% | 未测并发 | 不可直接比较 |
| R1-Distill-14B，并发4，TPOT | 7.41→7.22ms | -2.56% | 未测并发 | 不可直接比较 |
| DeepSeek-671B，并发1，TTFT | 166.08→172.04ms | +3.59% | 不在本机范围 | 不可比较 |
| DeepSeek-671B，并发4，TTFT | 185.11→199.97ms | +8.03% | 不在本机范围 | 不可比较 |

论文使用vLLM/SGLang集群环境，并测试平均17-token输入、100-token输出和并发。当前绝对延迟明显更高是本机HF eager环境造成，不能据此判断算法比论文慢或快。

## 8. 工程加载与运行状态

| 加载项 | 当前状态 | 已验证内容 | 未验证内容 |
|---|---|---|---|
| HF FP32 checkpoint | 已加载 | 24层、810,207,360参数、forward/generate | 长时间服务稳定性 |
| KV Cache | 部分通过 | prefill长度36，decode后37 | cached decode与full decode逐logit一致性 |
| FastAPI | 已加载 | 非流式200、SSE一致、错误key=400 | TLS、鉴权、并发、真实网络部署 |
| 客户端TokenKey | 已加载 | tau/inverse_tau token恢复 | tokenizer/HTTP/文本流SDK |
| vLLM adapter | 代码存在 | 旧候选有运行产物 | 最终v11加载、生成、KV、流式、性能 |
| SGLang | 未加载 | 无 | 全部 |
| Manifest | 部分通过 | 模型目录10个文件SHA-256 | 密钥manifest、来源revision、签名 |
| 攻击工具 | 部分加载 | v11完整词表We/Wh、We/Wgate | 其余VMA组合、IA/ISA/IMA/TFMA/SDA/NN |

## 9. 修正后的验收结论

| 验收域 | 结论 |
|---|---|
| 核心算法代码 | 基本完成 |
| 0.5B HF模型可运行 | 完成 |
| 聚合精度可用性 | 初步通过，需同dtype复验 |
| 甲方VMA隐私指标 | 不通过 |
| 密钥安全边界 | 严重不通过 |
| 论文完整攻击复现 | 未完成 |
| 最终v11 vLLM/SGLang | 未完成 |
| 工业性能复现 | 未完成 |
| 当前总判定 | `NO-GO` |
