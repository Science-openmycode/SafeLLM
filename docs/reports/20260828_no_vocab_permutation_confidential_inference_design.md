# 隐变智模：取消词表置换后的完整机密推理方案

日期：2026-08-28

## 1. 要解决的问题

固定条件：

- 客户端及企业本地网关可信；
- 云服务器操作系统、容器、运维人员和普通 GPU 运行环境不可信；
- 不再使用词表置换 `tau`；
- Token ID 保持原模型编号；
- 保留现有 `P/Q` 权重改造；
- 输出必须与原模型一致；
- 云端敌手不能得到 prompt、response、Token ID、logits 或可直接匹配 Token 的 Embedding 表；
- 必须支持云端自回归和高并发。

## 2. 不可绕过的约束

对任意一步生成，原模型执行：

```text
logits = h * W_head^T
y = Sample(logits)
next_embedding = E[y]
```

取消词表置换后，`y` 就是原始词表编号。某个执行方必须知道 `y`，否则无法选取 `E[y]` 继续生成。

因此只有四类地方可以承担这一步：

1. 可信客户端或可信本地网关；
2. 经远程证明的 CPU TEE/CVM；
3. 经远程证明的机密 GPU；
4. HE/MPC 等密码学安全计算。

若普通服务器直接得到 `logits` 或执行 `argmax`，则“服务器不知道输出 Token”在逻辑上不成立。这不是实现技巧问题，而是自回归生成的数据依赖决定的。

## 3. 推荐方案：CVM + 机密 GPU 全链路推理

### 3.1 数据流

```text
可信客户端/网关
  prompt
  -> tokenizer
  -> 原始 token_ids
  -> AEAD 加密
          |
          v
经远程证明的 CVM
  -> 解密 token_ids
  -> 验证 model_id / key_id / nonce / counter
          |
          v
机密 GPU 保护域
  -> E_private[i] = E_original[i] P
  -> P/Q 私有 Transformer
  -> 私有 KV cache
  -> Final RMSNorm
  -> 无词表置换的 LM Head
  -> sampling 得到原始 token y
  -> E_private[y] 继续自循环
          |
          v
CVM
  -> 加密输出 token y
          |
          v
客户端/网关
  -> 解密
  -> tokenizer.decode(y)
```

### 3.2 离线权重变化

Embedding 不再改变词表行号：

```text
E_private[i] = E_original[i] P
```

LM Head 不再乘词表置换矩阵：

```text
W_head_private = Q_head W_head_original
```

Attention、FFN、Residual、RMSNorm、RoPE、KV cache 的 `P/Q` 适配保持现有设计。

### 3.3 密钥释放

```text
客户端取得 CVM + GPU 证明
-> 验证硬件身份、固件、驱动、启动镜像和运行时测量值
-> 验证模型 manifest 和服务策略
-> 验证通过后建立临时会话密钥
-> 才允许发送 Token 和模型解密密钥
```

每次请求使用 AEAD；AAD 至少绑定：

```text
deployment_id
model_id
key_id
session_id
sequence_number
request_direction
```

### 3.4 为什么该方案解决 LM Head 问题

LM Head 不再放入 CPU TEE，也不需要通过普通 GPU 暴露 logits。它与模型主体一起在机密 GPU 内执行，因此仍使用原来的 GPU GEMM 和 batching。服务器宿主只能观察密文、长度和公开调度信息。

现有研究在 H100 机密 GPU 上报告约 4%–8% 的吞吐损耗；该数字不能直接作为 671B 验收结果，但证明了完整 GPU TEE 路线的工程可用性。

## 4. 普通 GPU 备选：CVM + ReMO/Slalom 线性层外包

在没有机密 GPU、但存在可信 CPU CVM 时，可以对每个线性层执行：

```text
真实输入：E
临时掩码：M
发送给普通 GPU：E_hat = E + M

GPU：
O_hat = E_hat W
      = EW + MW

CVM：
O = O_hat - MW
```

为了避免 CVM 每次重新计算 `MW`，可建立恢复池：

```text
M = M_pvt M_pub
R_pub = M_pub W
MW = M_pvt R_pub
```

LM Head 同样可以外包：

```text
h_hat = h + m
GPU 返回：l_hat = h_hat W_head^T
CVM 恢复：logits = l_hat - m W_head^T
CVM 内执行 sampling 和自循环
```

这样普通 GPU 承担完整 Head 大矩阵乘法，CVM 不保存明文 Token 之外的完整模型。

### 4.1 必须增加完整性验证

Talaria 假设云端 honest-but-curious。若服务器可能故意返回错误结果，需使用 Slalom/Freivalds 型检查。对 `O = EW`，CVM 选取随机向量 `r` 并预计算 `t = Wr`：

```text
检查：O r == E t
```

重复独立检查可把错误逃逸概率降到目标安全级别。请求、恢复池编号和 mask 必须单次使用并防重放。

### 4.2 该方案的边界

Talaria 使用 `m < d` 以避免恢复池唯一暴露模型权重，但这使掩码只覆盖一个低秩子空间。论文附录明确说明存在 `d-m` 维残余空间；其攻击实验表现很好，但不能称为全方向完美隐藏。

本项目的模型权重属于客户，客户端不是敌手，因此可以放弃“向客户端隐藏模型”这一约束，使用更强的全空间 mask 或离线一次性恢复池。然而，全空间在线恢复会重新引入接近 LM Head 的计算量；一次性恢复池则消耗大量预计算和存储。它适合作为普通 GPU 兼容模式，不应代替机密 GPU 主路线。

## 5. 可信本地网关拆分模式

另一种不依赖云端 TEE 的办法是把 Embedding、Final RMSNorm、LM Head、sampling 放在可信本地网关：

```text
本地网关：token -> E[token]P
云端：P/Q Transformer body -> hP
本地网关：hP -> logits -> token -> E[token]P
```

对 DeepSeek-V3 级配置，若 `V=129280, d=7168`：

```text
Head 参数 = V*d = 926,679,040
BF16 权重约 1.85 GB
每输出 Token 约 1.85 GFLOPs
```

1000 个会话、每会话 30 Token/s：

```text
总输出速率 = 30,000 Token/s
Head 算力约 = 55.6 TFLOPs
双向隐藏向量流量约 = 30,000 * 2 * 7168 * 2 bytes
                       = 860 MB/s
                       = 6.88 Gbit/s（未含协议开销）
```

因此需要本地 GPU 和至少 10GbE，工程上可行，但云端仍可观察 `P/Q` 私有隐藏状态。其安全性依赖当前模型混淆及攻击门禁，而不是纯密码学保证，故不能作为“彻底解决”版本。

## 6. HE/MPC 路线

若不信任任何服务器硬件，只有 HE/MPC 可以从密码学上计算加密的 LM Head、argmax 和自循环。问题是：

- LM Head 输出维度等于整个词表；
- autoregressive decoding 每个 Token 都要安全比较或安全采样；
- 每层非线性、KV cache 和通信均需安全协议；
- 已有 MPC Transformer 结果仍显示数量级级别的延迟和通信开销。

这一路线适合长期研究，不适合作为当前 671B 商业产品的主方案。

## 7. 产品决策

建议增加三个明确模式：

```text
cc-full
  无词表置换
  CVM + 机密 GPU
  全模型、KV、LM Head、sampling 在保护域
  正式强保证模式

cvm-outsourced
  无词表置换
  CPU CVM + ReMO/Slalom + 普通 GPU
  实验/兼容模式

obfuscated-standard
  普通 GPU
  当前 P/Q + 词表置换
  低硬件门槛模式
```

不要把三者声明为相同安全等级。

## 8. 最终结论

彻底解决 LM Head、自循环和原始 Token 暴露问题的最现实方案是：

> 使用客户端可验证的 CVM 和机密 GPU，把 Embedding、P/Q 模型主体、KV cache、LM Head、sampling 和自循环全部放入同一受证明保护域；客户端只在证明通过后发送加密的原始 Token。词表置换可完全取消，模型输出保持不变，LM Head 继续使用 GPU，不形成 CPU 瓶颈。

若硬件只能使用 3090/4090 等普通 GPU，则不存在同时满足“无词表置换、普通 GPU、服务器为敌手、精确高性能推理”的纯软件捷径。此时只能在 ReMO/Slalom、可信本地 Head、保留词表置换或 HE/MPC 之间选择相应成本。

## 9. 后续验证工作

1. 在 Qwen2.5-0.5B 增加 `tau=identity` 的转换模式，确认无词表置换下函数等价。
2. 新建 attested-session 协议，定义证明、密钥释放、AEAD counter 和失败策略。
3. 先用软件模拟 CVM 边界验证 Token、Embedding、KV、logits 和采样不会进入宿主日志。
4. 实现 LM Head ReMO 原型，比较普通 Head、CPU Head、低秩 ReMO 和全空间一次性 pad。
5. 增加恶意返回测试和 Freivalds 完整性检查。
6. 在支持机密计算的单 GPU 环境完成 Qwen0.5B 配对测试，再决定 671B 多 GPU 拓扑。
