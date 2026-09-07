# 隐变智模扩展性能指标调研与现有结果

日期：2026-08-21

## 1. 推荐采用的指标体系

单独报告“并发数”或“性能下降百分比”不够。正式产品应使用以下四层指标。

| 层级 | 核心指标 | 回答的问题 |
|---|---|---|
| 本地混淆计算 | ms/请求、Token/s/core、核时/百万请求、Scaling efficiency | 本地网关需要多少CPU、扩容是否线性 |
| 用户体验 | TTFT、TPOT/ITL、端到端延迟、Inter-chunk jitter、p95/p99 | 用户多久看到首字、流式输出是否卡顿 |
| 云端资源与费用 | Token/s/GPU、Token/s/GiB、GPU时/百万Token、显存放大、J/Token | 私有模型多花多少GPU、显存、电和钱 |
| SLO有效容量 | Goodput、Queue time、KV Cache占用、错误率、恢复率 | 高负载下真正有多少请求仍满足服务标准 |

NVIDIA AIPerf/GenAI-Perf使用TTFT、ITL、请求延迟、每用户和聚合Token吞吐；GenAI-Perf把Goodput定义为满足指定SLO的完成请求吞吐。vLLM还暴露队列时间、KV Cache使用率、Token计数和Prefix Cache命中率。MLPerf在服务器场景中使用延迟约束下的吞吐，并要求能耗使用整机墙上功率测量。

## 2. 当前已经算出的新增指标

### 2.1 本地网关尾延迟和网络放大

| 输入/输出Token | 平均私有处理 | p95 | p99 | p99/p50 | 网络字节增加 |
|---:|---:|---:|---:|---:|---:|
| 128/128 | 1.205 ms | 1.575 ms | 1.705 ms | 1.47× | 0.56% |
| 512/256 | 2.342 ms | 2.872 ms | 3.142 ms | 1.37× | 1.94% |
| 2,048/512 | 5.430 ms | 6.116 ms | 6.608 ms | 1.21× | 3.41% |

结论：协议字节增长很小，本地主要成本是CPU处理和当前Python运行时，而不是网络传输量。短请求的绝对时间最低，但固定调度抖动占比较大，所以p99/p50反而更高。

### 2.2 多核扩展效率和内存效率

固定512输入、256输出：

| Worker | 请求/s | 线性扩展效率 | RSS | 请求/s/GiB | p99处理延迟 |
|---:|---:|---:|---:|---:|---:|
| 1 | 377.7 | 100.0% | 0.790 GiB | 478.3 | 4.12 ms |
| 2 | 735.6 | 97.4% | 1.580 GiB | 465.5 | 3.36 ms |
| 4 | 1,202.5 | 79.6% | 3.157 GiB | 380.9 | 4.53 ms |

当前4 Worker已经出现效率下降。商业版本应优先采用轻量进程、共享Tokenizer只读数据或Rust/NumPy实现，而不是无限增加当前Python Worker。

### 2.3 M1的独立计算成本

旧M1基准混入了累计Detokenize，已经作废。修正为增量Detokenize后的同路径实测：

```text
仅词表置换：       2.288 ms/请求
词表置换 + M1：    2.419 ms/请求
M1新增：            0.131 ms/请求
M1计算增幅：        5.74%
```

在 `epsilon1=10`、词表151,936时，理论Token改变率为87.34%。这说明M1计算开销不大，但回答精度和可用性必须独立验收，不能因计算便宜就默认开启。

### 2.4 云端单位Token费用

Qwen2.5-0.5B，RTX 3090，固定50个提示词、64输出Token、4轮原始＋4轮私有：

| 指标 | 原始模型 | 私有模型 | 私有损耗 |
|---|---:|---:|---:|
| 输出吞吐 | 47.861 Token/s | 31.891 Token/s | -33.37% |
| 每100万输出Token所需GPU时 | 5.804 | 8.710 | +50.08% |
| 峰值显存 | 0.950 GiB | 1.544 GiB | +62.58% |
| Token/s/GiB | 50.40 | 20.66 | -59.02% |
| 模型加载时间 | 0.420 s | 0.488 s | +16.09% |

若GPU小时价格为 `P` 元：

```text
原始模型每100万输出TokenGPU费用 = 5.804 × P
私有模型每100万输出TokenGPU费用 = 8.710 × P
私有增量费用                     = 2.906 × P
```

例如只为了理解公式，若 `P=1元/GPU小时`，则分别为5.80元、8.71元和新增2.91元。实际报价必须使用部署当天的GPU价格。

### 2.5 15%产品门禁逐提示词通过率

对50个提示词分别取4轮原始均值和4轮私有均值，再要求私有时间不超过对应原始时间的115%：

| 门禁 | 通过提示词 | 通过率 |
|---|---:|---:|
| TTFT不超过+15% | 0/50 | 0% |
| TPOT不超过+15% | 0/50 | 0% |
| TTFT和TPOT同时通过 | 0/50 | 0% |

这不是标准开放负载Goodput，因为当前测试是顺序请求，没有注入固定到达率；它是当前产品15%相对门禁的逐提示词通过率。正式Goodput必须在开放负载下测量“每秒完成且满足绝对TTFT/TPOT SLO的请求数”。

## 3. 下一轮最值得增加的指标

### P0：必须增加

| 指标 | 计算方法 | 原因 |
|---|---|---|
| Open-loop Goodput | 固定到达率压测，统计同时满足TTFT和TPOT SLO的请求/s | 避免闭环压测隐藏排队问题 |
| Queue time p50/p95/p99 | 服务接收请求到开始Prefill的时间 | 判断慢在排队还是模型计算 |
| Gateway-added TTFT/ITL | 同一请求在网关入口、云端入口、云端出口、网关出口打点 | 分离本地混淆与云端模型损耗 |
| Saturation knee | 逐级提高RPS，寻找Goodput开始下降的位置 | 给出真实最大安全容量 |
| Error/retry/disconnect rate | 错误请求数、重试数、SSE中断数/总请求 | 吞吐高但错误多不能商用 |

### P1：大模型和多卡必须增加

| 指标 | 计算方法 | 原因 |
|---|---|---|
| Output Token/s/GPU | 聚合输出Token ÷ 时间 ÷ GPU数 | 跨不同GPU数量比较 |
| KV Cache bytes/token | 峰值KV增量 ÷ 活跃缓存Token | 推算最大Batch和上下文 |
| KV Cache使用率 | 运行时已用Cache块/总块 | 判断OOM和排队拐点 |
| Prefix Cache命中率 | 命中Cache块/查询Cache块 | 多轮对话可能显著影响TTFT |
| 通信占比 | NCCL时间/端到端模型时间 | 判断TP/PP/EP扩展损耗 |
| Scaling efficiency/GPU | N卡吞吐/(N×单卡吞吐) | 判断加卡是否划算 |

### P2：商业成本建议增加

| 指标 | 计算方法 | 原因 |
|---|---|---|
| Wh/百万Token | 整机墙上能耗/输出Token×1,000,000 | 估算电费和能效 |
| 元/百万有效Token | 总费用/满足SLO的输出Token×1,000,000 | 把失败和慢请求计入成本 |
| 元/百万请求 | 网关＋GPU＋网络＋存储总费用/请求 | 便于商务报价 |
| 利用率加权成本 | 节点小时费用/实际Goodput | 防止只按理论峰值报价 |

## 4. 不建议作为单独结论的指标

- 只写“并发数”，不写输入/输出长度、到达率和Token速度；
- 只写平均延迟，不写p95/p99；
- 只写FLOP增幅，不区分BF16、FP32和FP64；
- 只写Token/s，不写GPU数、显存、Batch、上下文和SLO；
- 只写最大吞吐，不写Goodput和错误率；
- 用TDP代替实际整机能耗。

## 5. 工件与来源

计算结果：

- `artifacts/performance/20260821-gateway-capacity/extended-performance-metrics.json`
- `artifacts/performance/20260821-gateway-capacity/gateway-overhead-512x256-m1-incremental-corrected.json`

调研来源摘要：

- `sources/research_llm_serving_metrics_20260821.md`

主要一手资料：

- NVIDIA AIPerf Metrics Reference: https://docs.nvidia.com/aiperf/reference/ai-perf-metrics-reference
- NVIDIA GenAI-Perf Goodput: https://docs.nvidia.com/deeplearning/triton-inference-server/archives/triton-inference-server-2500/user-guide/docs/perf_analyzer/genai-perf/docs/goodput.html
- vLLM Production Metrics: https://docs.vllm.ai/en/v0.22.0/design/metrics/
- TensorRT-LLM Performance Overview: https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/developer-guide/perf-overview.md
- MLPerf Inference Datacenter: https://mlcommons.org/benchmarks/inference-datacenter/
- DistServe: https://arxiv.org/abs/2401.09670
