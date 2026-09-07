# 隐变智模并发容量、算力需求与模型推理损耗评估报告

**报告版本：** 1.0（外发版）  
**报告日期：** 2026-08-24  
**评估对象：** 隐变智模可信混淆网关与云端私有模型  
**实测模型：** Qwen2.5-0.5B-Instruct  

---

## 1. 执行摘要

隐变智模的在线开销由两个彼此独立的部分组成：

1. **可信网关开销：** 在企业侧完成Chat Template、Tokenizer、输入Token置换、输出Token逆置换和增量文本恢复，主要消耗CPU。
2. **云端私有模型开销：** 私有模型因隐藏维扩展和精确RMS计算产生额外GPU计算、显存和推理时间。

本轮结论如下。

| 结论项 | 结果 |
|---|---:|
| 典型请求 | 512输入Token＋256输出Token |
| 普通网关计算量 | 3,120周期/总Token |
| 私有网关计算量 | 8,299周期/总Token |
| 隐私方案新增 | 5,179周期/总Token |
| 相同业务量所需网关算力 | 普通方案的2.66倍，新增约166% |
| 4 Worker私有网关实测吞吐 | 90.01万输入＋输出Token/s |
| 4 Worker、60%安全负载 | 约54.0万输入＋输出Token/s |
| 在线密钥大小 | 2.32 MiB |
| 网关是否需要GPU | 不需要 |
| 0.5B私有模型TTFT p50 | +47.05% |
| 0.5B私有模型TPOT p50 | +49.61% |
| 0.5B私有模型输出吞吐 | -33.37% |
| 0.5B私有模型峰值显存 | +62.58% |

对于商业部署，最重要的容量公式是：

```text
所需私有网关有效算力(Gcycle/s)
= 业务总Token/s × 8,298.511 ÷ 10^9 ÷ 目标利用率
```

例如，峰值业务为100万输入＋输出Token/s、网关最高使用70%时，需要：

```text
100万 × 8,298.511 ÷ 10^9 ÷ 0.70
= 11.855 Gcycle/s有效计算能力
```

---

## 2. 系统边界与数据流

```text
用户明文问题
    ↓
企业侧可信网关
Chat Template → Tokenizer → tau置换
    ↓ 仅发送私有Token ID
云端私有模型
私有Token推理与SSE返回
    ↓
企业侧可信网关
inverse_tau → 增量Detokenize
    ↓
用户获得明文回答
```

本报告分别计算：

```text
A. 企业侧网关：Token处理需要多少CPU类有效算力
B. 云端模型：私有模型相对原模型增加多少GPU推理开销
```

两者不能混算。网关算力与模型参数量没有直接关系；云端模型算力与模型结构、隐藏维、层数、推理后端和GPU有关。

---

## 3. 指标定义

| 指标 | 含义 |
|---|---|
| 输入Token | 一次请求经Tokenizer处理后的输入序列长度 |
| 输出Token | 模型本轮生成的Token数量 |
| 总Token | 输入Token＋输出Token，用于网关业务量核算 |
| Token/s | 网关每秒完成的输入＋输出Token，不是模型生成速度 |
| 周期/Token | 当前实现处理一个总Token消耗的CPU时钟周期 |
| Gcycle/s | 每秒十亿个CPU周期，用作本报告的计算能力单位 |
| 活跃会话 | 当前正在生成回答的会话，不是注册用户或空闲连接数 |
| TTFT | 从请求开始到收到第一个输出Token的时间 |
| TPOT | 首Token之后，每生成一个Token的平均时间 |
| p50/p95/p99 | 50%、95%、99%的请求不超过该数值 |

本地网关包含的真实处理步骤为：

- Chat Template；
- Tokenizer分词；
- 输入Token的 `tau` 置换；
- 私有SSE响应解析；
- 输出Token的 `inverse_tau` 逆置换；
- 增量Detokenize；
- 明文SSE重新封装。

不包含云端模型推理和公网等待时间。

---

## 4. 本地混淆网关实测

### 4.1 实验环境

| 项目 | 配置 |
|---|---|
| CPU | Intel Core i7-12700H |
| 内存 | 16 GB |
| Python | 3.11.15 |
| Tokenizer | Qwen2.5 Tokenizer |
| 词表规模 | 151,936 |
| 并行测试 | 1、2、4个独立Worker |
| 典型请求 | 512输入Token、256输出Token |
| 计时方式 | 热路径预热后配对计时；模型等待时间排除 |

### 4.2 不同请求长度的单请求时间

| 输入/输出Token | 普通网关 | 私有网关 | 隐私新增 | 私有/普通 |
|---:|---:|---:|---:|---:|
| 128 / 128 | 0.448 ms | 1.215 ms | 0.766 ms | 2.71倍 |
| 512 / 256 | 0.901 ms | 2.491 ms | 1.590 ms | 2.76倍 |
| 2,048 / 512 | 1.740 ms | 5.425 ms | 3.686 ms | 3.12倍 |

在128～2,048输入Token、128～512输出Token的运营区间内，私有网关时间拟合为：

```text
私有处理时间(ms)
= 0.2982
+ 0.000835 × 输入Token数
+ 0.006727 × 输出Token数
```

拟合优度 `R²=0.9972`。输出Token成本更高，是因为每个输出Token均需完成SSE解析、逆置换和增量文本恢复。

### 4.3 典型请求的绝对计算量

典型512输入、256输出请求共有768个总Token。

| 路径 | 周期/请求 | 周期/总Token | 相对普通网关 |
|---|---:|---:|---:|
| 普通网关 | 239.59万 | 3,120 | 1.00倍 |
| 私有网关 | 637.33万 | 8,299 | 2.66倍 |
| 隐私新增 | 397.73万 | 5,179 | +166.0% |

因此，对相同的输入＋输出Token业务量，当前Python私有网关需要普通网关约2.66倍的CPU类有效计算能力。

独立的4 Worker并发吞吐测试得到2.62倍算力倍率，与周期法的2.66倍相差约1.5%。本报告容量规划采用更保守的2.66倍。

这里不是说整套系统成本增加166%。云端GPU通常仍是主要成本；166%仅表示可信网关热路径的CPU计算增量。

### 4.4 真实并行吞吐

| Worker | 普通网关 | 私有网关 | 私有总Token吞吐 | 私有p99 |
|---:|---:|---:|---:|---:|
| 1 | 1,001 req/s | 372 req/s | 28.54万Token/s | 4.85 ms |
| 2 | 1,779 req/s | 696 req/s | 53.48万Token/s | 4.44 ms |
| 4 | 3,071 req/s | 1,172 req/s | 90.01万Token/s | 4.46 ms |

4 Worker并行效率约为单Worker线性值的78.8%。下降来自进程调度、内存访问和当前Python/PyTorch/Transformers运行时，并不是置换表计算变复杂。

### 4.5 网关内存

| 项目 | 实测占用 |
|---|---:|
| 在线置换密钥 | 2.32 MiB |
| 单Worker RSS | 约0.79 GiB |
| 4 Worker合计RSS | 约3.16 GiB |

内存主要来自Python、PyTorch、Transformers和Tokenizer。在线密钥本身很小。当前实现推荐4 Worker节点至少配置8 GiB内存；若改为无PyTorch的Rust或轻量C++网关，内存和周期/Token仍有较大下降空间。

---

## 5. 从Token业务量换算所需算力

### 5.1 主公式

设业务峰值为 (B) 个输入＋输出Token/s，则：

```text
普通网关实际计算量(Gcycle/s) = B × 3,119.724 ÷ 10^9
私有网关实际计算量(Gcycle/s) = B × 8,298.511 ÷ 10^9
隐私新增计算量(Gcycle/s)     = B × 5,178.787 ÷ 10^9
```

考虑目标最高利用率 (u)：

```text
规划算力 = 实际计算量 ÷ u
```

### 5.2 算力需求表

| 峰值总Token/s | 普通网关 | 私有网关 | 隐私新增 | 私有规划算力（70%负载） |
|---:|---:|---:|---:|---:|
| 1万 | 0.031 Gcycle/s | 0.083 Gcycle/s | 0.052 Gcycle/s | 0.119 Gcycle/s |
| 10万 | 0.312 Gcycle/s | 0.830 Gcycle/s | 0.518 Gcycle/s | 1.186 Gcycle/s |
| 100万 | 3.120 Gcycle/s | 8.299 Gcycle/s | 5.179 Gcycle/s | 11.855 Gcycle/s |
| 1,000万 | 31.197 Gcycle/s | 82.985 Gcycle/s | 51.788 Gcycle/s | 118.550 Gcycle/s |
| 1亿 | 311.972 Gcycle/s | 829.851 Gcycle/s | 517.879 Gcycle/s | 1,185.502 Gcycle/s |

### 5.3 参考硬件映射

当前参考机4 Worker的实测上限为90.01万Token/s。按60%安全负载，单个参考计算单元按54.0万Token/s规划。

| 峰值总Token/s | 参考4-Worker计算单元 | 当前Python实现的参考资源 |
|---:|---:|---|
| 10万 | 1个 | 4 vCPU / 8 GiB |
| 50万 | 1个 | 4 vCPU / 8 GiB |
| 100万 | 2个 | 约8 vCPU / 16 GiB，或2台4C8G |
| 500万 | 10个 | 建议多节点部署，约40 vCPU级参考算力 |
| 1,000万 | 19个 | 建议多节点部署，约76 vCPU级参考算力 |

这张表只把当前参考机性能映射为可理解的部署规模。正式采购应在目标CPU上执行同一基准，以实测Token/s或周期/Token替换参考值。

### 5.4 网络带宽

典型私有请求平均传输约11,799字节。包含20%的TLS、TCP和流量波动余量：

```text
带宽(Mbps)
= 总Token/s ÷ 768 × 11,799 × 8 × 1.20 ÷ 10^6
```

| 峰值总Token/s | 典型私有协议带宽 |
|---:|---:|
| 10万 | 14.7 Mbps |
| 100万 | 147.5 Mbps |
| 500万 | 737.4 Mbps |
| 1,000万 | 1.47 Gbps |

超过约500万总Token/s时，千兆网络会先于CPU成为明显约束，建议采用10GbE并拆分多个网关节点。

---

## 6. 从活跃会话换算业务负载

只有业务给出“同时生成会话数”时，才需要使用模型单会话生成速度。

设：

```text
N = 同时生成会话数
r = 每个会话平均输出速度，Token/s
Lout = 平均回答长度
Lin = 平均输入长度
```

则：

```text
完成请求率 = N × r ÷ Lout
网关总Token/s = 完成请求率 × (Lin + Lout)
```

以512输入、256输出为例：

| 活跃会话 | 单会话速度 | 完成请求率 | 网关总Token/s | 私有规划算力（70%） |
|---:|---:|---:|---:|---:|
| 1,000 | 10 Token/s | 39.1 req/s | 3.0万 | 0.356 Gcycle/s |
| 1,000 | 30 Token/s | 117.2 req/s | 9.0万 | 1.067 Gcycle/s |
| 10,000 | 10 Token/s | 390.6 req/s | 30.0万 | 3.556 Gcycle/s |
| 10,000 | 30 Token/s | 1,171.9 req/s | 90.0万 | 10.669 Gcycle/s |
| 50,000 | 30 Token/s | 5,859.4 req/s | 450.0万 | 53.347 Gcycle/s |

“并发用户”必须指正在生成Token的活跃会话。平台注册用户、登录用户和长连接数量不能直接代入该公式。

---

## 7. 云端私有模型推理损耗实测

### 7.1 测试协议

| 项目 | 配置 |
|---|---|
| GPU | NVIDIA GeForce RTX 3090 24 GB |
| 模型 | Qwen2.5-0.5B-Instruct |
| 后端 | Hugging Face eager |
| 精度 | BF16 |
| Batch | 1 |
| 样本 | 50个固定Prompt，每次生成64 Token |
| 运行次数 | 原模型4次＋私有模型4次 |
| 请求数 | 每种模型200次，共400次 |
| 顺序控制 | ABBA＋BAAB独立进程运行 |
| 置信区间 | 配对请求Bootstrap，10,000次重采样 |

### 7.2 实测结果

| 指标 | 原模型 | 私有模型 | 损耗 | 95%置信区间 |
|---|---:|---:|---:|---:|
| TTFT p50 | 21.99 ms | 32.33 ms | +47.05% | +46.51%～+47.68% |
| TTFT p95 | 22.79 ms | 33.13 ms | +45.37% | +41.32%～+48.05% |
| TTFT p99 | 24.30 ms | 34.94 ms | +43.79% | +35.12%～+54.92% |
| TPOT p50 | 20.90 ms | 31.27 ms | +49.61% | +49.00%～+50.27% |
| TPOT p95 | 21.29 ms | 31.92 ms | +49.91% | +49.19%～+50.44% |
| TPOT p99 | 21.72 ms | 31.97 ms | +47.16% | +46.42%～+50.22% |
| 输出吞吐 | 47.86 Token/s | 31.89 Token/s | -33.37% | — |
| 峰值显存 | 0.950 GiB | 1.544 GiB | +62.58% | — |
| 模型加载 | 0.413 s | 0.489 s | +18.40% | — |

当前0.5B产品checkpoint没有达到“TTFT和TPOT损耗不超过15%”的目标。主要原因是0.5B原模型计算量很小，而固定扩维和精确RMS附加计算占比较高。

---

## 8. 模型越大，私有推理损耗是否越小

论文扩维关系为：

```text
D = d + 2h
```

当前使用 `h=128`，因此 `D=d+256`。只看残差坐标扩维，比例为：

```text
扩维比例 = 256 / d
```

隐藏维越大，这部分百分比必然下降。但当前实现还包含每层FP64精确RMS，所以完整运行时间不一定单调下降。

| Qwen2.5规模 | d→D | 私有权重增加 | 普通投影FLOP增加 | 加精确RMS后的原始FLOP增加 |
|---:|---:|---:|---:|---:|
| 0.5B | 896→1,152 | +65.37% | +27.35% | +37.15% |
| 1.5B | 1,536→1,792 | +35.43% | +16.20% | +26.09% |
| 3B | 2,048→2,304 | +24.80% | +12.20% | +23.10% |
| 7B | 3,584→3,840 | +8.25% | +7.04% | +17.98% |
| 14B | 5,120→5,376 | +6.14% | +4.91% | +23.66% |
| 32B | 5,120→5,376 | +5.51% | +4.95% | +15.93% |
| 72B | 8,192→8,448 | +3.70% | +3.10% | +18.54% |

可直接采用的结论是：

- 私有权重增幅总体随隐藏维增大而下降；
- 普通扩维投影的相对计算增量总体下降；
- 完整推理时间是否下降，不能只由参数量判断；
- 14B与72B的原始FLOP增量并不低于相邻较小型号，原因是层数、FFN宽度和精确RMS占比不同；
- 目前只有0.5B完成原始/私有完整checkpoint配对实测，其他规模是结构核算，不是运行时间实测。

---

## 9. 容量规划建议

### 9.1 小型验证或内部业务

```text
峰值 ≤ 50万总Token/s
当前实现参考：4 vCPU / 8 GiB
网络：千兆
节点：建议2台主备或双活
```

### 9.2 中型生产业务

```text
峰值约100万总Token/s
所需私有规划算力：11.855 Gcycle/s
当前实现参考：8 vCPU / 16 GiB或2×4C8G
网络：千兆可用，建议预留双网口或10GbE升级能力
节点：至少2台，按N+1部署
```

### 9.3 大型高并发业务

```text
峰值500万～1,000万总Token/s
私有规划算力：59.3～118.6 Gcycle/s
网络：10GbE
部署：无状态网关水平扩展
会话状态：外置或按一致性哈希固定到网关
密钥：本地安全加载，不进入浏览器和云端模型服务器
```

商业上线前，在目标CPU和目标Tokenizer上运行相同基准，得到该硬件真实的Token/s、周期/Token、p99和RSS，再确定最终服务器数量。

---

## 10. 成本换算

本地网关计算成本可以按Token业务量计算：

```text
每10亿总Token的私有计算周期
= 10^9 × 8,298.511
= 8,298.511 Gcycle
```

若供应商给出的有效计算能力价格为 `P 元/(Gcycle/s·小时)`，持续业务量为 `B Token/s`：

```text
每小时私有网关计算成本
= B × 8,298.511 ÷ 10^9 ÷ 目标利用率 × P
```

实际服务器报价通常按实例小时或月计费，因此落地时采用：

```text
月费用 = 节点数量 × 实例月价
隐私增量月费用 = 私有网关总价 - 普通网关总价
```

云端GPU模型费用需独立计算：

```text
GPU月费用
= GPU单价 × GPU数量 × 运行小时
```

不能把网关CPU增量和云端GPU推理损耗相加成一个百分比。

---

## 11. 结论

1. 当前私有网关典型负载需要 **8,299周期/总Token**，比普通网关新增 **5,179周期/总Token**。
2. 峰值100万总Token/s时，按70%利用率应配置至少 **11.855 Gcycle/s** 的有效网关计算能力。
3. 当前Python实现4 Worker实测达到 **90.01万总Token/s**，按60%安全负载规划为 **54.0万Token/s**。
4. Token置换不是主要资源来源；主要开销来自Tokenizer、SSE解析、增量Detokenize和Python运行时。
5. 可信网关不需要GPU，优先选择高主频CPU、充足内存和稳定网络。
6. Qwen2.5-0.5B私有模型实测TTFT增加47.05%、TPOT增加49.61%，尚未达到15%目标。
7. 大模型能够稀释固定扩维的权重和普通投影开销，但完整推理损耗是否下降仍需真实7B及以上checkpoint配对测试。

---

## 附录A：复现工件

本地网关：

- `artifacts/performance/20260821-gateway-capacity/gateway-overhead-grid-20cell-v2.json`
- `artifacts/performance/20260821-gateway-capacity/gateway-cycles-v1.json`
- `artifacts/performance/20260821-gateway-capacity/gateway-concurrency-512x256-v2.json`
- `artifacts/performance/20260821-gateway-capacity/gateway-machine-capacity-v2.json`
- `artifacts/performance/20260821-gateway-capacity/compute-scaling-revised-v2.json`

云端模型：

- `artifacts/performance/20260821-3090-qwen05b-expanded/full-50x64/balanced-comparison.json`
- `artifacts/performance/20260821-3090-qwen05b-expanded/expansion-scaling-3090.json`

## 附录B：复现命令

```powershell
uv run python scripts/benchmark_gateway_cycles.py `
  --tokenizer data/models/qwen2.5-0.5b-instruct `
  --key-dir data/private/yinbian-model-keys/full `
  --input-tokens 128 512 2048 `
  --output-tokens 128 256 512 `
  --samples 100 --batch-size 20 `
  --out artifacts/performance/20260821-gateway-capacity/gateway-cycles-v1.json

uv run python scripts/benchmark_gateway_concurrency.py `
  --tokenizer data/models/qwen2.5-0.5b-instruct `
  --key-dir data/private/yinbian-model-keys/full `
  --input-tokens 512 --output-tokens 256 `
  --workers 1 2 4 --requests-per-worker 300 --cpu-cores 4 `
  --out artifacts/performance/20260821-gateway-capacity/gateway-concurrency-512x256-v2.json

uv run python scripts/analyze_compute_scaling.py `
  --gateway-grid artifacts/performance/20260821-gateway-capacity/gateway-overhead-grid-20cell-v2.json `
  --gateway-concurrency artifacts/performance/20260821-gateway-capacity/gateway-concurrency-512x256-v2.json `
  --expansion-h 128 --context-tokens 512 `
  --out artifacts/performance/20260821-gateway-capacity/compute-scaling-revised-v2.json `
  --csv artifacts/performance/20260821-gateway-capacity/model-scaling-revised-v2.csv
```
