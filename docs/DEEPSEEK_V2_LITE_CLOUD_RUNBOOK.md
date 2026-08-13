# DeepSeek-V2-Lite-Chat 云端执行手册

## 1. 固定对象

| 项目 | 固定值 |
|---|---|
| 模型 | `deepseek-ai/DeepSeek-V2-Lite-Chat` |
| Hugging Face revision | `85864749cd611b4353ce1decdb286193298f64c7` |
| 总参数 | 16B |
| 每token激活参数 | 2.4B |
| Transformer层数 | 27 |
| Dense层 | Layer 0 |
| MoE层 | Layer 1–26 |
| Routed experts | 64/层 |
| Activated experts | 6/token |
| MLA | 启用 |
| MTP | 不具备 |
| 源权重大小 | 约31.4GB |

配置文件：`configs/models/deepseek_v2_lite_chat.yaml`。

本阶段验证训练完成模型上的词表置换、MLA和MoE。P/Q扩维、Embedding/LM Head噪声和MTP不包含在该checkpoint中；这些功能继续以Qwen2.5-0.5B正式工件为准。

## 2. 已实现变换

### 2.1 词表

客户端密钥满足：

```text
tau[plain_id] = private_id
inverse_tau[private_id] = plain_id
```

Embedding和LM Head按以下规则同步置换：

```text
W_private[tau(i), :] = W_plain[i, :]
```

`config.json`与`generation_config.json`中的BOS、EOS、PAD同步映射。服务器checkpoint不包含`tau`或`inverse_tau`。

### 2.2 MLA

每层生成：

```text
head_order             [16]
kv_latent_order        [512]
q_latent_order         不生成（该模型q_lora_rank=null）
nope_maps              [16, 128, 128]
rope_map               [64, 64]
value_maps             [16, 128, 128]
```

第`L`层使用`base_seed + 100000 * L`作为层种子，层内再划分低秩、head、RoPE、value、expert和expert-FFN子区间，避免相邻层与相邻专家复用同一随机流。

DeepSeek-V2-Lite的KV低秩链路为：

```text
hidden
→ kv_a_proj_with_mqa
→ compressed_kv[512]
→ kv_a_layernorm
→ kv_b_proj
→ per-head K_nope/V
```

设`S`为`kv_latent_order`对应的512维置换矩阵，私有低秩坐标为
`c' = S^T c`。checkpoint同步改写：

```text
KV_A_compressed' = S^T KV_A_compressed
RMS_weight'      = S^T RMS_weight
KV_B'            = KV_B S
```

置换保持`mean(c^2)`不变，RMSNorm满足
`RMSNorm(S^T c, S^T w) = S^T RMSNorm(c, w)`，因此三处改写后低秩链路严格保持函数。若目标DeepSeek配置存在`q_lora_rank`，Q侧按相同规则同步改写`q_a_proj`、`q_a_layernorm`和`q_b_proj`。

对新head `h`，旧head为 `p = head_order[h]`：

```text
Q_nope'[h] = N[p]^T Q_nope[p]
Q_rope'[h] = R^T Q_rope[p]
K_nope'[h] = N[p]^T K_nope[p]
V'[h]      = Vmap[p]^T V[p]
O'[h]      = O[p] Vmap[p]
K_rope'    = R^T K_rope
```

`R`由独立二维旋转块组成，与DeepSeek交错RoPE的每个复数对旋转可交换。Prefill和增量Decode使用同一个转换后Cache坐标。

### 2.3 MoE

每个MoE层生成：

```text
expert_order            [64]
expert_ffn_orders       [64, 1408]
expert_ffn_scales       [64, 1408]
```

Router和专家同步执行：

```text
Router'[h, :] = Router[expert_order[h], :]
```

当`n_group > 1`时，`expert_order`由整组置换和各组内部置换组成，不允许把单个专家任意跨组打散；这样group score与`topk_group`选择保持等价。DeepSeek-V2-Lite的`n_group=1`，退化为64个专家的全排列。

每个新专家内部执行：

```text
Gate' = Gate[order, :]
Up'   = Up[order, :] / scale[:, None]
Down' = Down[:, order] * scale[None, :]
```

`scale > 0`，因此SwiGLU的逐元素乘积与Down投影保持函数等价。共享专家不参与路由专家置换。

Layer 0的Dense FFN和Layer 1–26的Shared Experts不改变模块位置，但各自独立执行同样的Gate/Up同步置换、Up正缩放和Down补偿。也就是说，“不参与Router置换”不等于“FFN权重原样透传”。

## 3. 官方张量形状门禁

| 张量 | 期望形状 |
|---|---:|
| `q_proj.weight` | `[3072, 2048]` |
| `kv_a_proj_with_mqa.weight` | `[576, 2048]` |
| `kv_a_layernorm.weight` | `[512]` |
| `kv_b_proj.weight` | `[4096, 512]` |
| `o_proj.weight` | `[2048, 2048]` |
| `mlp.gate.weight` | `[64, 2048]` |
| 每专家`gate_proj.weight` | `[1408, 2048]` |
| 每专家`up_proj.weight` | `[1408, 2048]` |
| 每专家`down_proj.weight` | `[2048, 1408]` |
| Layer 0 Dense `gate/up_proj.weight` | `[10944, 2048]` |
| Layer 0 Dense `down_proj.weight` | `[2048, 10944]` |
| Shared `gate/up_proj.weight` | `[2816, 2048]` |
| Shared `down_proj.weight` | `[2048, 2816]` |
| `embed_tokens.weight` | `[102400, 2048]` |
| `lm_head.weight` | `[102400, 2048]` |

`scripts/audit_deepseek_checkpoint.py`通过safetensors header检查形状，不把完整权重载入内存。

## 4. 云主机要求

最低配置：

```text
GPU：2 × RTX 3090 24GB
推荐：3 × RTX 3090 24GB
CPU内存：>= 64GB
数据盘可用空间：>= 120GB
推荐数据盘：200GB NVMe
Linux：Ubuntu 22.04/24.04
驱动：530/535
CUDA运行时：12.1
```

120GB空间用于同时保存：

```text
官方源checkpoint                 31.4GB
逐张量断点转换checkpoint          约31.4GB
2GB标准分片checkpoint             约31.4GB
环境、密钥、缓存和实验工件         20GB以上
```

50GB数据盘不通过预检，不得开始下载。

## 5. 上传目录

在Windows本机生成不含模型、密钥和实验工件的源码包：

```powershell
uv run python scripts/build_deepseek_cloud_bundle.py
```

输出：

```text
release/AloePri-deepseek-v2-lite-cloud-source.tar.gz
```

上传后在Linux数据盘解压：

```bash
cd /data
tar -xzf AloePri-deepseek-v2-lite-cloud-source.tar.gz
```

把整个仓库上传到Linux数据盘，例如：

```bash
/data/AloePri
```

进入仓库：

```bash
cd /data/AloePri
git status --short
git rev-parse HEAD
```

不得只上传`scripts/`；转换需要`src/`、`configs/`和项目元数据。IFEval输入清单若未上传，精度脚本会下载后校验固定内容SHA-256；内容发生变化会停止。

## 6. CUDA 12.1环境

仓库本机锁文件使用PyTorch 2.9.1 CUDA 12.8。驱动530/535不能直接运行CUDA 12.8 wheel。云端必须使用独立环境：

```bash
bash scripts/cloud/bootstrap_cuda121.sh
```

该脚本创建：

```text
.venv-cloud/
PyTorch 2.5.1 + CUDA 12.1
Transformers 5.12.0
Accelerate 1.14.0
```

预检输出：

```text
artifacts/cloud/deepseek-v2-lite-preflight.json
```

以下任一项失败立即停止：

```text
CUDA不可用
可见GPU少于2张
任一GPU显存小于约24GB
总显存小于48GB
内存小于64GB
数据盘可用空间小于120GB
```

## 7. 下载、审计、转换和重分片

执行：

```bash
bash scripts/cloud/download_and_convert_deepseek_v2_lite.sh
```

顺序：

```text
固定revision断点下载
→ 源checkpoint形状审计
→ 逐张量转换
→ 每张量原子保存和SHA-256记录
→ 断点转换结果按2GB重新分片
→ 私有checkpoint再次审计
```

目录：

```text
data/models/deepseek-v2-lite-chat/
data/packages/deepseek-v2-lite-chat-mla-moe-unpacked/
data/packages/deepseek-v2-lite-chat-mla-moe/
data/keys/deepseek-v2-lite-chat-mla-moe-offline/
data/keys/deepseek-v2-lite-chat-mla-moe-online/
```

转换中断后重新执行同一脚本。脚本检测`.partial`目录并自动增加`--resume`。下列变化会拒绝续跑：

```text
源config SHA-256变化
张量名集合变化
seed变化
FFN缩放范围变化
词表置换开关变化
转换算法版本变化
固定Hugging Face revision变化
已完成文件大小或SHA-256变化
```

不得手工删除`.partial`后宣称续跑成功。

## 8. 无噪声函数等价

执行：

```bash
bash scripts/cloud/run_deepseek_v2_lite_equivalence.sh
```

脚本严格分时加载明文和私有模型，记录：

```text
32-token Prefill logits
1-token KV Cache Decode logits
32-token Greedy Generation
GPU device_map
checkpoint、key和脚本SHA-256
```

输出：

```text
artifacts/deepseek-v2-lite-chat/baseline.safetensors
artifacts/deepseek-v2-lite-chat/private.safetensors
artifacts/deepseek-v2-lite-chat/equivalence.json
```

门禁：

```text
Prefill normalized RMSE <= 0.025
Cached Decode normalized RMSE <= 0.025
Prefill逐位置Top-1一致率 >= 95%
同一Cache Decode的下一token Top-1完全相同
Greedy 32-token序列是否完全相同单独报告
不得发生CPU或磁盘offload
```

阈值来自27层BF16极小DeepSeek-V2回归。变换公式在FP32测试中仍按`2e-5`门禁；BF16写回随机正交矩阵会产生逐层舍入，不能用两层模型的`0.005`误差上限判定27层模型。最终是否保留任务能力仍由MMLU、C-Eval、PIQA、IFEval和HumanEval的配对门禁决定。

## 9. 产品问答冒烟

执行：

```bash
bash scripts/cloud/run_deepseek_v2_lite_product_smoke.sh
```

服务端使用：

```text
configs/product/deepseek_v2_lite_cloud.yaml
device=cuda-auto
dtype=bfloat16
gpu_memory_fraction=0.90
```

客户端本地加载原始tokenizer和在线密钥。服务器加载私有checkpoint，不读取在线或离线密钥。冒烟检查：

```text
健康检查成功
中文问题产生非空回答
客户端发送tau(input_ids)
客户端用inverse_tau恢复输出
服务端stdout/stderr不出现明文问题
```

工件：

```text
artifacts/deepseek-v2-lite-chat/product-smoke/client.json
artifacts/deepseek-v2-lite-chat/product-smoke/server.stdout.log
artifacts/deepseek-v2-lite-chat/product-smoke/server.stderr.log
artifacts/deepseek-v2-lite-chat/product-smoke/result.json
```

## 10. 全量精度

执行：

```bash
bash scripts/cloud/run_deepseek_v2_lite_accuracy.sh
```

脚本分时运行：

```text
MMLU
C-Eval
PIQA
IFEval 541 prompts
HumanEval 164 problems
```

明文和私有模型使用相同dtype、batch、任务和文档hash。输出位于：

```text
artifacts/deepseek-v2-lite-chat/accuracy/
```

比较工件包含配对bootstrap 95%置信区间。当前DeepSeek转换没有加入噪声，理论目标是保持评分；如果出现下降，先检查函数等价，不得通过调低门禁掩盖错误。

## 11. 一键执行

完成磁盘扩容并上传仓库后：

```bash
cd /data/AloePri
CUDA_VISIBLE_DEVICES=0,1,2 bash scripts/cloud/run_deepseek_v2_lite_all.sh
```

该命令依次执行环境预检、下载、转换、重分片、等价验证、产品问答、全量精度和最终验收。各步骤工件均原子落盘；中断后再次执行时转换与IFEval会从已验证工件续跑。

只需先验证架构闭环时，依次执行：

```bash
bash scripts/cloud/download_and_convert_deepseek_v2_lite.sh
bash scripts/cloud/run_deepseek_v2_lite_equivalence.sh
bash scripts/cloud/run_deepseek_v2_lite_product_smoke.sh
```

最终验收输出：

```text
artifacts/deepseek-v2-lite-chat/final-acceptance.json
```

该文件分别给出`architecture_validation_decision`和`deepseek_privacy_release_decision`，不得把MLA/MoE架构通过写成DeepSeek隐私产品通过。

## 12. 安全边界

服务器目录只允许：

```text
data/packages/deepseek-v2-lite-chat-mla-moe/
configs/product/deepseek_v2_lite_cloud.yaml
服务端代码
```

服务器不得复制：

```text
data/keys/deepseek-v2-lite-chat-mla-moe-online/
data/keys/deepseek-v2-lite-chat-mla-moe-offline/
明文测试prompt工件
```

在线客户端只持有：

```text
online_key.safetensors：tau、inverse_tau
key.json
manifest.json
原始tokenizer
```

离线密钥包含MLA矩阵、专家置换、专家FFN置换/缩放和词表映射。转换完成后离线归档。

## 13. 当前能力边界

| 功能 | DeepSeek-V2-Lite状态 |
|---|---|
| 训练完成模型 | 已选择，待云端下载 |
| 词表置换/逆置换 | 已实现并通过极小实模round-trip |
| MLA q/kv低秩坐标及RMSNorm同步 | 已实现并通过前向与Cache测试 |
| MLA head/nope/rope/value变换 | 已实现并通过前向与Cache测试 |
| MoE专家与Router同步置换 | 已实现 |
| 每专家FFN置换/缩放 | 已实现 |
| Dense FFN与Shared Experts内部变换 | 已实现 |
| 流式转换与断点恢复 | 已实现并模拟中断测试 |
| 2GB标准分片 | 已实现并测试 |
| 多GPU HF加载 | 已实现，待真实16B实测 |
| 产品token-ID API | 已接入，待真实16B实测 |
| P/Q隐藏维扩维 | 未对DeepSeek实现 |
| Embedding/Head噪声 | 未对DeepSeek启用 |
| MTP | 模型不具备 |
| 671B | 当前硬件不支持 |

由于DeepSeek阶段尚未应用P/Q和噪声，直接Embedding权重匹配可以恢复词表映射。因此该阶段只能证明MLA/MoE功能适配和产品数据流可运行，不能用其攻击指标替代Qwen2.5-0.5B的正式隐私验收。
