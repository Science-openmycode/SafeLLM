# 隐变智模网关容量与模型性能损耗实测报告

日期：2026-08-21

## 1. 结论

本地词表置换本身不是性能瓶颈。512个输入Token、256个输出Token时，输入 `tau` 查表仅需0.037 ms，输出SSE解析和 `inverse_tau` 查表需0.851 ms。旧实现的主要开销是每收到一个Token就重新Detokenize全部累计Token，单请求约8.27 ms。

接入Qwen字节级增量Detokenize后，私有网关处理时间从10.67 ms降至2.31 ms。与0.86 ms的明文转发相比，单请求只增加约1.46 ms。4物理核并发实测达到1,202 req/s，1000个持续活跃会话仅产生约117 req/s，因此4C8G单机可以运行；商业部署建议使用两台4C8G做高可用。

云端Qwen2.5-0.5B复测表明，当前私有模型在RTX 3090、HF eager、BF16下，TTFT p50增加47.05%，TPOT p50增加49.61%，吞吐下降33.37%，显存增加62.58%。这些是模型侧开销，与本地网关开销是两件独立的事。

“模型越大，损耗百分比越小”只对固定扩维 `h` 造成的维度膨胀成立，不是完整模型的无条件结论。当前精确度量RMS路径包含FP64稠密计算，宽度增加后其绝对耗时上升，可能抵消扩维比例下降。必须用真实7B/14B模型才能证明完整端到端比例。

## 2. 场景与计算公式

基准场景：

- 平均输入：512 Token；
- 平均输出：256 Token；
- 云端模型生成速度：20或30 Token/s；
- CPU容量上限按60%利用率计算，保留40%给HTTP、TLS、连接管理、日志和抖动；
- “活跃会话”指正在持续生成Token的会话，不是已登录但空闲的用户。
- 生产路径关闭“私有Token文本轨迹”展示；该功能用于Demo，会额外反复解码私有文本，不应进入高并发服务。

完成请求率：

```text
R_req = N_active × r_token / T_output
```

CPU核需求：

```text
C_core = R_req × t_cpu_ms / 1000
C_vCPU,target = C_core / 0.60
```

例如1000个活跃会话、每会话30 Token/s、每回答256 Token：

```text
R_req = 1000 × 30 / 256 = 117.19 req/s
```

## 3. 本地网关实测

### 3.1 实验环境

- CPU：Intel Core i7-12700H，14核20线程；
- 内存：16 GB；
- 并发测试固定使用4个分离的物理核逻辑处理器：0、2、4、6；
- Python 3.11.15；
- Qwen2.5 Tokenizer；
- 私有词表大小：151,936；
- 每个场景先预热，再计时；
- 明文与私有路径采用同进程配对和ABBA/BAAB顺序消除漂移。

明文路径执行：JSON请求转发、明文SSE解析和序列化。

私有路径执行：Chat Template、分词、`tau`、私有SSE解析、`inverse_tau`、Detokenize和明文SSE序列化。模型推理和公网等待不计入本地CPU时间。

### 3.2 单请求CPU时间

| 路径 | 平均CPU时间 | p50 | p99 | 相比明文新增 |
|---|---:|---:|---:|---:|
| 明文转发 | 0.858 ms | 0.845 ms | 1.166 ms | — |
| 旧私有路径（累计Detokenize） | 10.666 ms | 10.613 ms | 11.907 ms | 9.807 ms |
| 新私有路径（增量Detokenize） | 2.314 ms | 2.282 ms | 2.843 ms | 1.456 ms |
| 旧私有路径＋M1，`epsilon1=10` | 11.208 ms | 11.306 ms | 13.659 ms | 10.350 ms |

明文函数几乎只做转发，因此旧私有函数看起来增加约11.4倍CPU时间。这个百分比不能直接解释成服务器要增加11.4倍：商业硬件要看绝对增加了多少CPU核。

### 3.3 私有路径组成

| 组成 | 平均时间 |
|---|---:|
| Chat Template与分词 | 0.463 ms |
| 输入 `tau` 向量查表 | 0.037 ms |
| 输出SSE JSON解析与 `inverse_tau` | 0.851 ms |
| 旧累计Detokenize | 8.268 ms |

输入置换仅占旧私有路径约0.35%。性能优化应放在增量Detokenize、JSON/SSE批处理和进程模型，不应削弱置换算法。

### 3.4 4核并发实测

每个Worker独立加载Tokenizer和在线密钥，处理512/256请求。加载时间不计入吞吐。

| Worker数 | 明文 req/s | 旧私有 req/s | 新私有 req/s | 新私有p99 |
|---:|---:|---:|---:|---:|
| 1 | 1,024 | 80 | 378 | 4.12 ms |
| 2 | 1,965 | 159 | 736 | 3.36 ms |
| 4 | 1,581 | 268 | 1,202 | 4.53 ms |

4 Worker的明文结果受笔记本异构核、功耗和后台调度影响低于2 Worker的线性值；新私有路径仍达到目标负载的10.3倍。容量计算使用更稳定的配对单请求结果，4进程结果用于验证不会在117 req/s附近饱和。

当前Python进程每Worker RSS约0.85 GB，4 Worker合计约3.39 GB。在线密钥张量本身只有2.32 MiB，内存主要来自PyTorch、Transformers和Tokenizer运行时。4C8G足够；若将网关改成无PyTorch的NumPy/Rust运行时，内存还能显著降低。

### 3.5 高并发CPU需求

以下使用新增量Detokenize路径和30 Token/s。数值只表示网关请求变换CPU，不含固定Web框架消耗。

| 持续活跃会话 | 请求率 | 明文目标vCPU | 私有目标vCPU | 混淆新增vCPU |
|---:|---:|---:|---:|---:|
| 100 | 11.72 req/s | 0.017 | 0.045 | 0.028 |
| 500 | 58.59 req/s | 0.084 | 0.226 | 0.142 |
| 1,000 | 117.19 req/s | 0.168 | 0.452 | 0.284 |
| 5,000 | 585.94 req/s | 0.838 | 2.260 | 1.422 |
| 10,000 | 1,171.88 req/s | 1.676 | 4.520 | 2.844 |

工程配置建议：

| 持续活跃会话 | 明文单机建议 | 私有单机建议 | 私有生产高可用建议 |
|---:|---|---|---|
| 100 | 2C4G | 2C4G | 2 × 2C4G |
| 500 | 2C4G | 2C4G | 2 × 2C4G |
| 1,000 | 2C4G | 4C8G | 2 × 4C8G |
| 5,000 | 4C8G | 8C16G | 2 × 8C16G |
| 10,000 | 8C16G | 16C32G | 2 × 16C32G |

这里的“用户”必须按同时生成的活跃会话计算。若平台注册用户很多、同时生成比例只有5%，应使用注册用户数乘5%后查表。

代表场景的私有协议字节数比明文增加1.94%。网关不需要GPU；优先采购高主频CPU、足够内存和稳定网络。

## 4. 云端模型复测

### 4.1 实验协议

- GPU：NVIDIA GeForce RTX 3090 24 GB；
- CUDA 12.1，PyTorch 2.5.1+cu121；
- HF eager，BF16，batch=1；
- 固定50个提示词，每次生成64 Token；
- 原模型和私有模型各4次独立进程运行；
- 顺序：baseline、private、private、baseline、private、baseline、baseline、private；
- 每个角色200个请求，总计400个请求；
- 置信区间：配对请求Bootstrap，10,000次重采样。

### 4.2 结果

| 指标 | 原模型 | 私有模型 | 损耗 | 95%置信区间 |
|---|---:|---:|---:|---:|
| TTFT p50 | 21.99 ms | 32.33 ms | +47.05% | +46.51% ～ +47.68% |
| TTFT p95 | 22.79 ms | 33.13 ms | +45.37% | +41.32% ～ +48.05% |
| TTFT p99 | 24.30 ms | 34.94 ms | +43.79% | +35.12% ～ +54.92% |
| TPOT p50 | 20.90 ms | 31.27 ms | +49.61% | +49.00% ～ +50.27% |
| TPOT p95 | 21.29 ms | 31.92 ms | +49.91% | +49.19% ～ +50.44% |
| TPOT p99 | 21.72 ms | 31.97 ms | +47.16% | +46.42% ～ +50.22% |
| 输出吞吐 | 47.86 Token/s | 31.89 Token/s | -33.37% | — |
| 峰值显存 | 0.950 GiB | 1.544 GiB | +62.58% | — |
| 模型加载 | 0.413 s | 0.489 s | +18.40% | — |

0.5B模型的原始计算很短，扩维和精确RMS的固定附加核函数占比较高，因此百分比明显。该结果超过PPT的15%性能门禁，当前0.5B checkpoint的性能状态仍是不通过。

## 5. 模型规模与损耗比例

论文扩维为：

```text
d_private = d + 2h
```

固定 `h=128` 时：

```text
维度相对增加 = 2h / d
当前Qwen实现只扩张残差面对的输入/输出维，Attention Head维与FFN中间维不变，
普通投影相对计算量近似 = (d + 2h) / d - 1
Embedding/LM Head相对计算量近似 = (d + 2h) / d - 1
```

完整型号核算还必须单独加入精确RMS的FP64 `D×d` 度量计算；详见 `20260821_compute_scaling_revised.md`。

因此仅看扩维，隐藏维度越大，相对增加越小，极限趋近于0。

RTX 3090隔离核函数实测：

| 原隐藏维 `d` | 私有维 `d+256` | 维度增加 | 原始FLOP增加 | LM Head实测增加 | 精确RMS每Token附加 |
|---:|---:|---:|---:|---:|---:|
| 896 | 1,152 | 28.57% | 38.60% | 26.65% | 3.83 ms |
| 1,536 | 1,792 | 16.67% | 26.97% | 16.17% | 4.03 ms |
| 2,048 | 2,304 | 12.50% | 22.87% | 11.93% | 4.03 ms |
| 3,584 | 3,840 | 7.14% | 17.68% | 6.99% | 6.27 ms |
| 5,120 | 5,376 | 5.00% | 15.61% | 2.11% | 12.24 ms |
| 7,168 | 7,424 | 3.57% | 14.22% | -5.02%* | 23.59 ms |

`*` 最后一项是单个LM Head核函数的调度和测量波动，不表示私有模型真的加速。

完整结论：

1. 扩维带来的显存和BF16线性层百分比通常会随 `d` 增大而下降。
2. 本地网关开销与模型参数量无直接关系，只与输入/输出Token、Tokenizer和Token速率有关。
3. 大模型通常生成更慢，因此同样活跃用户数下，本地网关收到的Token更少，网关所需CPU反而更少。
4. 当前精确度量RMS使用FP64稠密运算，绝对耗时随宽度上升。它不会自动随大模型消失。
5. MoE、MLA、量化、Tensor Parallel和推理引擎都会改变总比例，不能用0.5B的49.6%直接外推到7B、70B或671B。
6. 现有证据支持“扩维部分的百分比下降”，不支持“完整私有模型总损耗必然下降”。

## 6. 本轮代码变更

- 在线密钥只在加载时验证一次，请求热路径不再重复排序全词表；
- `tau` 输入改为张量向量查表；
- 输出 `inverse_tau` 使用常数时间标量查表；
- 新增Qwen/GPT-2字节级增量Detokenize；
- 非字节级Tokenizer保留严格累计解码回退；
- 新增本地配对微基准、4进程并发基准、M1基准；
- 云端基准支持预先生成的明文/私有Token对，不向模型服务器上传完整在线密钥；
- 新增ABBA+BAAB 8轮远端自动测试脚本；
- 新增隐藏维扩展和精确RMS隔离核函数测试。

## 7. 证据与复现命令

关键工件：

- `artifacts/performance/20260821-gateway-capacity/gateway-overhead-512x256-incremental.json`
- `artifacts/performance/20260821-gateway-capacity/gateway-concurrency-4-physical-cores-incremental.json`
- `artifacts/performance/20260821-3090-qwen05b-expanded/full-50x64/balanced-comparison.json`
- `artifacts/performance/20260821-3090-qwen05b-expanded/expansion-scaling-3090.json`

本地网关：

```powershell
uv run python scripts/benchmark_gateway_overhead.py `
  --tokenizer data/models/qwen2.5-0.5b-instruct `
  --key-dir data/private/yinbian-model-keys/full `
  --input-tokens 512 --output-tokens 256 --runs 100 `
  --m1-epsilon1 10 `
  --out artifacts/performance/20260821-gateway-capacity/gateway-overhead.json

uv run python scripts/benchmark_gateway_concurrency.py `
  --tokenizer data/models/qwen2.5-0.5b-instruct `
  --key-dir data/private/yinbian-model-keys/full `
  --input-tokens 512 --output-tokens 256 `
  --workers 1 2 4 --requests-per-worker 300 --cpu-cores 4 `
  --out artifacts/performance/20260821-gateway-capacity/gateway-concurrency.json
```

代码位置：

- `src/aloepri/client/sdk.py`
- `src/aloepri/client/streaming_decode.py`
- `scripts/benchmark_gateway_overhead.py`
- `scripts/benchmark_gateway_concurrency.py`
- `scripts/benchmark_hf.py`
- `scripts/compare_performance.py`
- `scripts/benchmark_expansion_scaling.py`
- `scripts/cloud/run_qwen05b_expanded_remote.sh`
