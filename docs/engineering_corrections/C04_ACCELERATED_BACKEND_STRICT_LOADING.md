# C04：vLLM 与 SGLang 必须严格接收 BlockPerm 审计张量

## 问题

非平凡 BlockPerm checkpoint 在每层 Attention 中保存：

```text
model.layers.<L>.self_attn.aloepri_block_orders
```

该张量不是另一份密钥。它记录运行时计算

$$
R_B(t)=B^{\mathsf T}R(t)B
$$

所用的频率块顺序，使 checkpoint、配置和离线 key 可以交叉核对。

HF 模型注册了同名持久 buffer，因此可以严格加载。vLLM 和 SGLang 初版适配器只从
`config.aloepri_rope_block_orders` 构造运行时矩阵，没有在相同模块路径注册持久张量。
真实 `block_beta=8` checkpoint 首次加载时，vLLM 因发现未知权重而停止：

```text
There is no module or parameter named
layers.0.self_attn.aloepri_block_orders
```

## 修改

### vLLM

`src/aloepri/serving/vllm_qwen2.py` 在每层 `AloePriQwen2Attention` 注册同名持久
buffer。vLLM 严格 loader 必须读取 checkpoint 中的值；缺失、多余或形状不匹配都会使
加载失败。

### SGLang

SGLang 0.5.17 的旧式权重 loader 枚举参数，因此
`src/aloepri/serving/sglang_models/aloepri_qwen2.py` 把整数顺序表注册为
`requires_grad=False` 的参数。该张量不参与训练或矩阵计算，只用于严格消费和审计
checkpoint 内容。

两个后端实际执行 RoPE 时仍使用由同一顺序表生成的 $B$，计算顺序均为：

$$
q'B^{\mathsf T}\rightarrow R(t)\rightarrow B,
\qquad
k'B^{\mathsf T}\rightarrow R(t)\rightarrow B.
$$

## 独立一致性检查

`scripts/verify_paper_formula_checkpoint.py` 对每层执行三方核对：

$$
B_{\mathrm{checkpoint}}
=B_{\mathrm{config}}
=\operatorname{index\_select}(B_{\mathrm{offline}},\pi_{KV}).
$$

同时验证每个 KV head 的顺序表都是 $\{0,\ldots,d_h/2-1\}$ 的完整置换。
这些张量计入“全部 checkpoint 张量必须有公式覆盖”的门禁。

## 实测命令

```powershell
wsl -e bash /mnt/e/AloePri/scripts/wsl/run_vllm_smoke.sh
wsl -e bash /mnt/e/AloePri/scripts/wsl/run_sglang_smoke.sh
```

实测对象为 `data/packages/qwen05b-product-v31-blockperm8`，两个后端均在 RTX 3060
6GB、显存比例 75%、上下文 128、batch 1、FP32 下加载成功。两者生成的 32 个私有
token 与 HF cache 路径逐个相同。

证据：

- `artifacts/hf-vllm-greedy-v31-blockperm8-32tokens.json`
- `artifacts/sglang-smoke-v31-blockperm8-32tokens.json`
