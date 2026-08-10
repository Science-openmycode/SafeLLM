# C01：非平凡 BlockPerm 必须同步修改运行时 RoPE

## 修改结论

当 `block_beta > 1` 且 BlockPerm 实际交换了不同频率的 RoPE 块时，不能继续调用原始 Qwen RoPE。实现必须把每个 GQA 组的运行时旋转改为

$$
R_B(t)=B^{\mathsf T}R(t)B.
$$

这不是调参，也不是增加新的隐私变换。它是使论文已经写入 Q/K 权重的 BlockPerm 能够保持 Attention 函数所必需的坐标同步。

## 修改前的实现

离线转换把行向量形式的 Query 和 Key 写成

$$
q'=qADB,\qquad k'=kAD^{-1}B,
$$

其中：

- $A$ 是每个 RoPE 二维块内的正交旋转；
- $D$ 是成对的正缩放；
- $B$ 是 RoPE 频率块置换。

修改前，运行时仍直接使用原始旋转 $R(t)$：

$$
q'R(t),\qquad k'R(t).
$$

只有当 $BR(t)=R(t)B$ 时，该写法才保持原注意力分数。Qwen 各块频率不同，非平凡 $B$ 一般不满足这个条件。

## 必须修改的证明

$A$ 和 $D$ 都以同一频率二维块为单位构造，因此与 $R(t)$ 对易。令运行时使用

$$
R_B(t)=B^{\mathsf T}R(t)B,
$$

则

$$
ADB R_B(t)
=ADB B^{\mathsf T}R(t)B
=AD R(t)B
=R(t)ADB.
$$

因此旋转后的私有坐标等于“先执行原始 RoPE，再执行私有坐标变换”：

$$
q'R_B(t)=(qR(t))ADB.
$$

由于

$$
(ADB)(AD^{-1}B)^{\mathsf T}=I,
$$

任意 Query/Key 位置组合的内积均与原模型相同。

## GQA 与 KV Cache

Qwen2.5-0.5B 使用 14 个 Query head 和 2 个 KV head，每个 KV 组对应 7 个 Query head。离线转换先重排 KV 组，再重排组内 Query head，所以运行时保存的 BlockPerm 表也必须按新的 KV 组顺序重排。

RoPE 后的 Key 以私有坐标存入 KV Cache。prefill 和 decode 都调用同一 $R_B(t)$，因此缓存不需要还原到明文坐标，也不能在 decode 时重复施加 $B$。

## 代码修改

- `src/aloepri/models/modeling_aloepri_qwen2.py`
  - `apply_synchronized_block_permuted_rope()`：按组执行 $B^{\mathsf T}R(t)B$；
  - `AloePriSynchronizedBlockPermAttention`：在 prefill/decode 中统一使用修正后的 RoPE；
  - `install_synchronized_blockperm()`：加载配置中的逐层、逐 KV 组置换表。
- `src/aloepri/transforms/qwen_structural.py`
  - 将离线密钥中的旧 KV 组顺序转换为运行时新 KV 组顺序；
  - 只在 `block_beta > 1` 时安装同步运行路径。
- `src/aloepri/models/configuration_aloepri_qwen2.py`
  - 保存 `aloepri_rope_block_orders`，保证 checkpoint 重载后执行相同算子。

## 自动检验证据

`tests/unit/test_qwen_structural.py` 包含两类门禁：

1. 反例：非平凡 $B$ 配合原始 $R(t)$ 时，Attention score 误差必须显著大于零；
2. 修正：改为 $B^{\mathsf T}R(t)B$ 后，FP64 score 在 `1e-12` 误差内恢复；
3. 整层：非平凡 BlockPerm 下，prefill logits、decode logits 与未转换模型在设定数值误差内一致；
4. 重载：配置、算子类型和 logits 在 `save_pretrained()` / `from_pretrained()` 后保持一致。

执行命令：

```powershell
uv run pytest -q tests/unit/test_qwen_structural.py tests/unit/test_aloepri_qwen2_model.py
```

真实 0.5B checkpoint：

- 配置：`configs/product/qwen05b_v31_blockperm8.yaml`；
- `block_beta=8`，24 层均保存非平凡运行时置换表；
- 公式重建：316 个 checkpoint 张量通过，失败 0，遗漏覆盖 0；
- HF GPU：prefill cache 长度 36，decode 后 37，16-token greedy 与明文模型一致；
- vLLM：32-token 私有序列与 HF 一致；
- SGLang：32-token 私有序列与 HF/vLLM 一致；
- API：普通响应与 SSE 流式响应的私有 token 序列一致。

证据文件：

- `artifacts/verification/qwen05b-product-v31-blockperm8/formula.json`
- `artifacts/verification/qwen05b-product-v31-blockperm8/generation.json`
- `artifacts/verification/qwen05b-product-v31-blockperm8/verification_summary.json`
- `artifacts/hf-vllm-greedy-v31-blockperm8-32tokens.json`
- `artifacts/sglang-smoke-v31-blockperm8-32tokens.json`

## 不修改的部分

- BlockPerm 的抽样算法、`beta`、`gamma` 和随机种子不因本修正改变；
- Q/K 的 $A$、$D$、$B$ 离线变换公式不改变；
- `block_beta=1` 时仍走原始 Qwen 注意力路径；
- 本修正不把论文未定义的新噪声或新攻击假设加入模型。
