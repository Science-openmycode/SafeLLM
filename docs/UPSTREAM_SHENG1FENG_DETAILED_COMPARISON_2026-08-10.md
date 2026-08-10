# AloePri 0.5B 实现与 `sheng1feng/Aloepri` 详细对照审计

审计日期：2026-08-10

## 1. 审计输入

| 对象 | 固定输入 |
|---|---|
| 本地实现 | `Qwen2.5-0.5B-Instruct` |
| 本地当前 checkpoint | `data/packages/qwen05b-product-v31-blockperm8` |
| 本地当前配置 | `configs/product/qwen05b_v31_blockperm8.yaml` |
| 本地功能证据 | `artifacts/verification/qwen05b-product-v31-blockperm8/functional_evidence_index.json` |
| 外部仓库 | `https://github.com/sheng1feng/Aloepri` |
| 外部仓库提交 | `60e8ea3cc04353b7a0058e9c86d67461c7d25763` |
| 外部仓库标签 | `v0.1-qwen2.5-0.5b-20260407` |

外部仓库没有 `LICENSE`，本审计只分析公开实现思路，不复制其代码。

## 2. 总结论

本地实现在真实 0.5B checkpoint、全权重公式重建、非平凡 BlockPerm、精确 RMSNorm、HF/vLLM/SGLang 和客户端—服务端产品闭环上更完整。

外部仓库在攻击代码组织、paper-like IMA、参数扫描、阶段化证据目录和 Git 历史上更完整。

当前最大差距不在基础模型转换，而在以下四项：

1. v31 尚未重跑完整精度、攻击和性能验收。
2. 精确 RMSNorm 使用论文没有的在线 Gram 度量算子，需要独立证明其泄漏面和性能代价。
3. 本地 IMA/SDA 仍然是缩小协议，外部仓库的 `paper_like IMA` 数据窗口、训练/测试分离和序列逆置器更接近论文。
4. 本地项目尚未形成 Git 提交和最终 `release/` 交付包。

## 3. 核心变换对照

| 组件 | 论文要求 | 本地实现 | 外部实现 | 审计结论 |
|---|---|---|---|---|
| 词表置换 | 客户端 `tau`，返回后 `inverse_tau` | 全词表互逆、特殊 token 同步、真实 API 验证 | 已实现 | 无核心缺口 |
| Embedding/Head 噪声 | 按权重标准差注入 | 已实现独立 seed 和单次 tied-weight 处理 | 已实现 | 无核心缺口 |
| Algorithm 1 | `d -> d+2h -> d`，`P Q_j = I` | 保留 `B/E/F/Z/C/D` 结构，100 seed 单测，checkpoint 重建 | 已实现同类 KeyMat | 本地验证更严格 |
| Attention Q/K | RoPE 共变变换 | 重建 24 层 Q/K，使用 Qwen 实际频率 | 已实现 `R_qk/H_qk/Z_block` | 两者均有实现 |
| Value/O | 高斯可逆矩阵及逆变换 | 高斯采样、条件数门禁、FP64 离线转换 | 已实现变换 | 本地数值门禁更完整 |
| GQA | KV head/group 置换同步 | 已实现并持久化顺序表 | 已实现 `tau_kv/tau_group` | 无核心缺口 |
| BlockPerm | 论文默认 `beta=8` | 已修正分布/边界错误，并在运行时使用 `B^T R(t) B` | 有 BlockPerm 密钥生成，但最终完成门禁只检查 profile 字符和元数据 | 本地功能正确性更强 |
| FFN | gate/up/down 中间维置换和缩放 | 已实现并逐张重建 | 已实现 | 无核心缺口 |
| RMSNorm | 论文使用标量 `kappa` | 产品主线使用完整 `G=Q Q^T` | `kappa_fused` 或 `metric_diag_sqrt` | 本地更精确，但不是论文标准运行图 |
| Residual | 两支必须处于同一私有坐标 | 统一扩维坐标，有 coordinate graph 单测 | 通过分阶段 wrapper 实现 | 本地证据更直接 |
| KV Cache | prefill/decode 坐标一致 | HF 真实验证 36 -> 37，vLLM/SGLang 生成验证 | wrapper 有 cache 逻辑，远端不含当前工件 | 本地当前证据更强 |

## 4. 本地 RMSNorm 主线的两面性

本地 `AloePriMetricRMSNorm` 在私有坐标下计算：

```text
rms_plain^2 = x_private (Q Q^T) x_private^T / d
```

优点：

- 对一般矩形 Algorithm 1 KeyMat 是精确的；
- 无需假设 `Q Q^T` 接近标量单位阵；
- 已经在 HF、vLLM、SGLang 实际运行。

缺点：

- checkpoint 额外暴露 `aloepri_rms_metric`；
- 它是由秘密坐标矩阵推导的 Gram 矩阵，尚未针对该矩阵执行专项攻击；
- 需要三个后端的自定义 RMSNorm，不满足论文“完全复用现有标准算子”的强表述；
- 额外二次型计算和 FP32 工作精度尚未完成性能门禁。

必须保留两个明确 profile：

1. `paper-kappa`：用于检验论文原式的近似路线。
2. `exact-metric`：用于产品功能正确性，必须标注为对论文 RMSNorm 错误的修正。

不能把 `exact-metric` 直接写成“论文原样复现”。

## 5. 攻击实现对照

| 攻击 | 本地实现 | 外部实现 | 本地缺口 |
|---|---|---|---|
| 直接权重匹配 | 已实现 | 通过 VMA 目标体系组织 | v31 未重跑 |
| VMA | 五类权重组合、分层、候选规模和证据 hash | 实现了投影层、来源归因和层消融 | 应吸收“来源归因+层消融”实验设计 |
| Gate-IA | 已实现可执行特征 | 当前 `ia.py` 主要是 schema/template | v31 未重跑 |
| Attention-IA | 明确标记为维度有效代理 | 没有发现更完整的论文精确实现 | 论文公式本身维度有问题，仍需维持代理标记 |
| IMA | 两层 Transformer smoke，训练规模未达论文 | 有 public token windows、训练/测试分离、paper-like inverter | 这是最明确的实现落后项 |
| ISA | 维度不同时用 token Gram proxy | 支持 hidden state 和 attention score observable | 需对 v31 扩维路线明确定义可比 observable |
| TFMA | 有观测量曲线 | 有三种攻击知识设定 | 应增加外部仓库的三种 knowledge setting |
| SDA | 两层 recurrence decoder smoke，未达论文数据规模 | 有 bigram signature 和 BLEU 评估 | 需锁定语料、训练量和 held-out 协议 |
| known-plaintext | 已实现观测量曲线 | 未发现完整实现 | v31 未重跑 |

## 6. 部署与产品对照

| 项目 | 本地 | 外部 | 本地剩余缺口 |
|---|---|---|---|
| HF | 真实 checkpoint 和 API 实测 | 有标准工件和推理脚本 | 需长上下文和并发验证 |
| vLLM | 自定义模型、32 token 实测 | Stage I 有导出/回归脚本，对扩维和 metric RMSNorm 曾明确标记 blocked | 当前非平凡 BlockPerm 限定 TP=1 |
| SGLang | 自定义模型、32 token 实测 | 主要是文档声明的 target surface | 需 2048 上下文和多请求验证 |
| API | FastAPI、SSE、model/key 校验 | 没有同等产品 API | 需 TLS 反向代理配置 |
| SDK/CLI | Chat Template、本地历史、逐 token 恢复 | 主要是推理脚本 | 需正式安装包和版本号 |
| 密钥拆分 | online/offline/server 三分 | client secret/server 两分 | 需最终 release 包 |
| RmDP/M1 | 计算器、tokenwise M1、小词表精确 oracle | 未发现 | 需产品端到端手工验收 |

## 7. 不应从外部仓库采用的部分

1. 不采用 `generated_ids_exact_match_rate > 0` 作为 correctness 通过条件。
2. 不把 attention profile 名称中含 `block/group` 当成权重已正确变换的证明。
3. 不把 `metric_diag_sqrt` 或 `diag_friendly` KeyMat 写成论文 Algorithm 1 原样复现。
4. 不引用远端未提交的 `outputs/` 数值作为本地验收证据。
5. 不在没有许可证的情况下直接复制外部代码。

## 8. 本地实现必须补齐的工作

### P0：当前产品 checkpoint 验收

1. 固定 v31 checkpoint、online key、offline key 和 tokenizer hash。
2. 在 v31 上重跑 200 条 greedy 固定语料。
3. 在 v31 上跑完 MMLU、C-Eval、PIQA、IFEval、HumanEval 和配对置信区间。
4. 在 v31 上重跑 VMA、Gate-IA、Attention-IA proxy、IMA、ISA、TFMA、SDA、known-plaintext。
5. 使用 ABBA/BAAB 对 v31 跑 TTFT、TPOT、吞吐和显存。

### P0：精确 RMSNorm 泄漏面

1. 将 `aloepri_rms_metric` 明确加入服务端攻击者 observable。
2. 评估特征值、特征向量、条件数和不同key间的相关性。
3. 重跑 VMA/IA，比较攻击者是否利用 Gram 矩阵提升恢复率。
4. 如果攻击超标，`exact-metric` 必须保持 `NO-GO`。

### P0：攻击协议完整化

1. 根据论文独立实现 paper-like IMA：公共语料窗口、训练/验证/测试隔离、固定模型结构和 top-k 恢复。
2. 为 TFMA 增加 exact/similar/unrelated 三种先验知识。
3. 为 VMA 增加 source attribution 和 layer ablation。
4. 固定 SDA 语料、held-out 集和训练 token 量。

### P1：部署完整性

1. HF/vLLM/SGLang 实测 2048 上下文。
2. 实测连续多轮对话、多请求和中途 SSE 断开。
3. 补 TLS 反向代理和部署级限流。
4. 生成 `release/qwen05b-product/`，在干净环境重放安装与问答。

## 9. 当前阶段判定

| 阶段 | 状态 |
|---|---|
| Qwen 0.5B 核心转换 | 已完成 |
| 逐权重公式验证 | 已完成 |
| 真实问答产品闭环 | 已完成 |
| 论文原式与修正式分离 | 部分完成，RMSNorm 需继续保持双 profile |
| v31 精度验收 | 未完成 |
| v31 隐私验收 | 未完成 |
| v31 性能验收 | 未完成 |
| 最终版本化发布 | 未完成 |

因此当前结论仍为：核心功能完成，论文与产品总验收未完成。
