# 隐变智模：本地混淆算力与云端私有模型推理损耗

日期：2026-08-21  
测试机：Intel Core i7-12700H，14核20线程，16 GB内存，Windows，Python 3.11  
本地模型：Qwen2.5 Tokenizer；本地网关不加载大模型权重

## 1. 先给结论

系统里有两项完全不同的成本：

| 成本 | 在哪里发生 | 由什么决定 | 是否随大模型参数量变化 |
|---|---|---|---|
| 本地混淆与恢复 | 用户侧/离线隐私网关CPU | 请求数、输入Token、输出Token、Tokenizer、是否启用M1 | **不随0.5B、7B、72B变化** |
| 私有模型推理 | 云端GPU | 模型宽度、层数、`h`、上下文、Batch、后端和GPU | **会变化** |

本机对当前Python产品路径的实测结论：

- 典型请求为512个输入Token、256个输出Token时，4个工作进程可处理私有请求约`1,172 req/s`；按60%负载留出突发余量后，设计容量为约`703 req/s`。
- 同一台机器、同一负载下，普通路径为`3,071 req/s`。因此当前私有路径处理相同流量需要普通路径约`2.62倍`的CPU运行容量，即CPU容量增加约`162%`。
- 百分比高，是因为普通路径只做JSON/SSE转发，本身非常轻；实际增加为每请求约`1.88 ms`。
- 典型请求下，一个CPU执行核原始处理约`22.50万总Token/s`，按60%利用率规划为`13.50万总Token/s/核`；每处理100万总Token实际消耗`4.44核秒`。
- 4个工作进程的安全Token流量约为：输入`36.0万 token/s`、输出`18.0万 token/s`、合计`54.0万 token/s`。
- 以当前CPU单核性能为参照，`4C8G`可承载约`700 req/s`，`8C16G`约`1,400 req/s`，`16C32G`约`2,800 req/s`。4C8G来自实测，以上更大规格是同构分片换算，正式采购前应在目标Linux服务器复测。

云端GPU侧目前只有Qwen2.5-0.5B完成了同机成对实测。RTX 3090、HF eager、BF16、Batch 1下，私有模型TPOT p50增加`49.61%`，输出吞吐下降`33.37%`。论文默认`h=128`是固定扩维量，理论上的结构相对开销会随模型宽度增加而下降，但这不等于已经实测证明大模型运行时一定按同一比例下降。

## 2. 本地混淆：具体计算了什么

普通路径：

```text
接收明文请求 → JSON/SSE转发 → 返回明文Token
```

私有路径：

```text
Chat Template
→ Tokenizer
→ input_ids执行tau查表
→ 发送私有Token ID
→ 接收私有输出Token
→ output_ids执行inverse_tau查表
→ 增量Detokenize
→ 向用户流式返回文本
```

本地网关只处理Token ID和文本，不做Transformer矩阵乘法。因此本地负载使用三个业务量描述：

```text
R    = 每秒请求数
Tin  = 每秒输入Token数
Tout = 每秒输出Token数
```

大模型参数量不进入这个公式。同一Tokenizer、同一请求流量下，将云端模型从0.5B换成72B，本地每个Token仍然只做相同次数的置换、逆置换和解码。

### 2.1 “Token/算力/时间”的计量单位

本地网关不是矩阵乘法负载，主要是Tokenizer、数组查表、JSON/SSE和UTF-8处理。因此不使用GPU TFLOPS，而使用CPU服务最直接的两个单位：

```text
Token吞吐      = Token/s/CPU核
单位计算成本   = CPU核秒/100万Token
```

两者互为倒数：

```text
核秒/100万Token = 1,000,000 / (Token/s/核)
```

对任意总Token流量`T`和可用CPU核数`C`：

```text
所需忙碌CPU核数 = T / 实测Token吞吐率
处理时间(s) = Token数量 × 每Token核秒 / CPU核数
持续服务规划核数 = 所需忙碌CPU核数 / 0.60
```

### 2.2 典型512输入/256输出的直接关系

这个请求一共处理768个Token，输入与输出比例为2:1。4进程稳定实测换算如下：

| 路径 | 总Token/s/核 | 每100万Token核秒 | 4核处理100万Token耗时 | 相同流量CPU倍数 |
|---|---:|---:|---:|---:|
| 普通路径 | 589,567 | 1.696 | 0.424 s | 1.00× |
| 完整私有路径 | 225,015 | 4.444 | 1.111 s | 2.62× |
| 隐私方案净增加 | — | 2.748 | 0.687 s | +162% |

为了维持60%CPU利用率，完整私有路径按下列数值规划：

```text
每核安全总Token吞吐 = 225,015 × 60% = 135,009 Token/s/核
每百万Token规划算力 = 4.444 / 60% = 7.407 核秒
```

因此可以直接使用：

```text
所需CPU核数 = 总Token/s / 135,009
```

| 持续总Token流量 | 60%利用率下需要CPU核 | 合理机器档位 |
|---:|---:|---|
| 10万 Token/s | 0.74核 | 4C8G |
| 50万 Token/s | 3.70核 | 4C8G专用或8C16G混部 |
| 100万 Token/s | 7.41核 | 8C16G |
| 500万 Token/s | 37.03核 | 48C96G或两台24C |
| 1,000万 Token/s | 74.07核 | 96C192G或多机分片 |

如果业务只统计大模型输出Token，在512/256比例下：

```text
总Token/s = 3 × 输出Token/s
所需CPU核数 = 输出Token/s / 45,003
```

例如云端集群合计输出10万Token/s，本地同时约有20万输入Token/s，总流量30万Token/s，隐私网关需要：

```text
300,000 / 135,009 = 2.22个CPU核
```

实际应选择4C8G，以保留系统和突发余量。

### 2.3 纯置换成本与完整网关成本

同一个512输入/256输出请求的单进程组件实测：

| 组件 | 每请求时间 | 折算 |
|---|---:|---:|
| Chat Template与Tokenizer | 0.531 ms | 1.037微秒/输入Token |
| 输入`tau`向量置换 | 0.0477 ms | 0.093微秒/输入Token，约1,073万输入Token/s/核 |
| 输出SSE JSON解析 | 0.301 ms | 1.177微秒/输出Token |
| 输出`inverse_tau`逐Token恢复 | 0.493 ms | 1.924微秒/输出Token，约52.0万输出Token/s/核 |
| 产品增量Detokenize | 0.093 ms | 0.364微秒/输出Token |

只计算`tau + inverse_tau`：

```text
每请求纯置换时间 = 0.0477 + 0.4925 = 0.5402 ms
纯置换混合吞吐   = 768 / 0.0005402 = 142.2万总Token/s/核
纯置换单位成本   = 0.703核秒/100万总Token
```

完整私有路径还要进行Tokenizer、SSE解析、增量解码、请求和响应序列化以及进程调度，所以并发实测为`4.444核秒/100万Token`。这两个数字不能混用：

- `0.703核秒/百万Token`回答“置换本身消耗多少”；
- `4.444核秒/百万Token`回答“整套可用本地隐私服务消耗多少”。

### 2.4 输入和输出不能简单当成同一种Token

20组长度实验的正交拟合表明，当前完整私有路径的边际成本约为：

```text
输入Token：1.785微秒/Token
输出Token：7.748微秒/Token
```

输出Token更贵，是因为每个输出Token都要单独完成SSE事件解析、`inverse_tau`、UTF-8增量解码和流式响应序列化。输入Token可以一次向量化处理。因此最完整的CPU关系是：

```text
忙碌CPU核数
≈ 1.785e-6 × 每秒输入Token
 + 7.748e-6 × 每秒输出Token
 + 请求级固定开销
```

工程采购优先使用前面的实测混合吞吐；该边际公式用于输入输出比例明显偏离2:1时进行修正。

## 3. 本机20组长度实验

测试范围：输入Token为128、512、2,048、4,096；输出Token为64、128、256、512、1,024，共20组，每组100次。

本轮v2测试先在计时区外验证增量解码与完整Tokenizer解码完全一致，正式计时区不再重复执行仅用于断言的完整`tokenizer.decode()`；旧工件不用于本报告容量结论。

在该测试范围内，当前Python实现拟合得到：

```text
普通路径时间(ms/请求)
= -0.32088 + 0.0002000 × 输入Token + 0.0038918 × 输出Token
R² = 0.9456

私有路径时间(ms/请求)
= -0.78480 + 0.0017846 × 输入Token + 0.0077478 × 输出Token
R² = 0.9442

新增时间(ms/请求)
= -0.46392 + 0.0015846 × 输入Token + 0.0038559 × 输出Token
```

拟合截距受Python启动抖动和长序列缓存行为影响，仅用于本次128–4,096输入、64–1,024输出的区间估算；实际容量优先使用下表的并发实测值。

任意业务流量的CPU核需求可按下式计算：

```text
忙碌CPU核数 = 每秒请求数 × 每请求处理时间(ms) / 1000

规划CPU核数 = 忙碌CPU核数 / 目标利用率
```

本报告的目标利用率取60%，即保留40%给流量突发、系统调度和日志。

## 4. 三种请求长度的并发实测

每种负载均使用4个独立工作进程，每档300次/进程。

| 负载 | 输入/输出Token | 普通路径req/s | 私有路径req/s | 同流量CPU容量倍数 | 新增平均时间 | 私有p99 | 60%安全req/s | 60%安全总Token/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 短 | 128 / 128 | 6,303 | 2,476 | 2.55× | 0.883 ms | 2.503 ms | 1,486 | 380,292 |
| 典型 | 512 / 256 | 3,071 | 1,172 | 2.62× | 1.877 ms | 4.457 ms | 703 | 540,036 |
| 长 | 2,048 / 512 | 1,674 | 520 | 3.22× | 4.649 ms | 8.836 ms | 312 | 798,783 |

长请求的`req/s`更低但`Token/s`更高，是因为一次请求固定的JSON、SSE和函数调用成本被更多Token摊薄。采购机器时必须同时给出请求长度，不能只说“需要支持100并发”。

典型请求的4进程资源：

```text
私有路径原始吞吐       1,172 req/s
私有路径输入吞吐     600,040 token/s
私有路径输出吞吐     300,020 token/s
私有路径总吞吐       900,060 token/s
4进程稳定RSS             3.16 GiB
4进程扩展效率             78.84%
```

## 5. 服务器规格换算

以下按典型负载512输入/256输出、60%利用率计算：

| 服务器 | 安全req/s | 输入Token/s | 输出Token/s | 总Token/s | 运行时RSS估算 | 证据性质 |
|---|---:|---:|---:|---:|---:|---|
| 4C8G | 703 | 360,024 | 180,012 | 540,036 | 3.16 GiB | 本机4进程实测 |
| 8C16G | 1,406 | 720,048 | 360,024 | 1,080,071 | 6.32 GiB | 2个4进程分片换算 |
| 16C32G | 2,813 | 1,440,095 | 720,048 | 2,160,143 | 12.64 GiB | 4个分片换算 |
| 32C64G | 5,625 | 2,880,190 | 1,440,095 | 4,320,286 | 25.28 GiB | 8个分片换算 |
| 64C128G | 11,251 | 5,760,381 | 2,880,190 | 8,640,571 | 50.57 GiB | 16个分片换算 |

这里的“C”按当前测试机的一个工作进程/一个逻辑CPU执行单元换算。服务端Xeon/EPYC与i7-12700H单核性能不同，所以这张表用于初选规格，不用于最终SLA签字。

按典型业务请求率反推采购规格：

| 目标请求率 | 普通路径所需CPU等效核 | 私有路径所需CPU等效核 | 推荐私有网关 |
|---:|---:|---:|---|
| 100 req/s | 0.22 | 0.57 | 4C8G，可同时运行管理服务 |
| 1,000 req/s | 2.17 | 5.69 | 8C16G |
| 5,000 req/s | 10.86 | 28.44 | 32C64G |
| 10,000 req/s | 21.71 | 56.89 | 64C128G |

计算示例，1,000 req/s：

```text
私有路径每核原始容量 = 1,172 / 4 = 292.99 req/s
按60%运行的每核容量 = 292.99 × 0.60 = 175.79 req/s
需要CPU等效核       = 1,000 / 175.79 = 5.69
因此选择8C16G
```

## 6. M1的额外本地成本

固定输出256 Token、`epsilon1=10`，M1在现有确定性置换产品路径上增加：

| 输入Token | 不启用M1 | 启用M1 | M1新增 | 相对当前私有路径 |
|---:|---:|---:|---:|---:|
| 128 | 2.610 ms | 2.848 ms | 0.238 ms | +9.11% |
| 512 | 3.170 ms | 3.319 ms | 0.150 ms | +4.73% |
| 2,048 | 4.457 ms | 4.722 ms | 0.265 ms | +5.95% |

M1仍然只属于本地网关。它与模型参数量无关，但受输入Token数、词表大小、`epsilon1`和采样实现影响。

## 7. 云端私有模型推理损耗

行业推理服务通常分别报告TTFT、TPOT/ITL、请求吞吐、Token吞吐、并发、显存和满足SLO的Goodput；不能把本地网关毫秒数并入模型TTFT后再声称是“模型损耗”。指标定义参考[NVIDIA AIPerf](https://docs.nvidia.com/aiperf/reference/ai-perf-metrics-reference)、[TensorRT-LLM性能说明](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/developer-guide/perf-overview.md)和[MLPerf Server场景](https://mlcommons.org/benchmarks/inference-datacenter/)。

当前完成的正式同机比较：Qwen2.5-0.5B、RTX 3090、BF16、HF eager、Batch 1、50条提示词、每角色4轮、共200个配对请求。

| 指标 | 原始模型 | 私有模型 | 损耗 |
|---|---:|---:|---:|
| TTFT p50 | 21.987 ms | 32.333 ms | +47.05%，95% CI `[46.51%, 47.68%]` |
| TPOT p50 | 20.899 ms | 31.268 ms | +49.61%，95% CI `[49.00%, 50.27%]` |
| 输出吞吐 | 47.861 token/s | 31.891 token/s | -33.37% |
| 峰值GPU显存 | 0.950 GiB | 1.544 GiB | +62.58% |
| 模型加载时间 | 0.420 s | 0.488 s | +16.09% |

这组数据只说明当前0.5B完整运行时的实际表现，不应替代7B、14B或72B实测。

## 8. 模型越大，损耗比例是否越小

[Aloe论文](https://arxiv.org/abs/2603.01499)将隐藏宽度从`d`扩展为：

```text
D = d + 2h
```

论文默认`h=128`，并在14B上比较过`h=128/256/384/512`。论文没有要求`h`随参数量同比增长。固定`h=128`时，单纯宽度增加比例为：

```text
(D-d)/d = 2h/d = 256/d
```

因此模型隐藏宽度越大，固定扩展256维所占比例越小。按照各Qwen2.5官方配置、上下文512 Token计算：

| 模型 | 隐藏宽度d | 扩维后D | 权重字节结构增幅 | 结构投影FLOPs增幅 |
|---|---:|---:|---:|---:|
| 0.5B | 896 | 1,152 | +65.37% | +27.35% |
| 1.5B | 1,536 | 1,792 | +35.43% | +16.20% |
| 3B | 2,048 | 2,304 | +24.80% | +12.20% |
| 7B | 3,584 | 3,840 | +8.25% | +7.04% |
| 14B | 5,120 | 5,376 | +6.14% | +4.91% |
| 32B | 5,120 | 5,376 | +5.51% | +4.95% |
| 72B | 8,192 | 8,448 | +3.70% | +3.10% |

可以成立的结论是：**固定`h=128`时，结构投影的相对开销总体随模型宽度增大而下降。**

还不能直接写成“72B实际推理只损失3.10%”，原因是当前完整实现还包含精确RMS坐标处理、不同dtype核函数、显存访问和自定义模型调度。0.5B实测约50%的TPOT损耗也说明运行时损耗不等于纸面FLOPs。后续应在同一后端和同一GPU类型上至少实测0.5B、3B、7B三个点，再拟合真实运行时曲线。

## 9. 工件与复算命令

主要工件：

```text
artifacts/performance/20260821-gateway-capacity/
├── gateway-overhead-grid-20cell-v2.json
├── gateway-component-512x256-v2.json
├── gateway-concurrency-128x128-v2.json
├── gateway-concurrency-512x256-v2.json
├── gateway-concurrency-2048x512-v2.json
├── gateway-m1-input-scaling-v2.json
├── compute-scaling-v2.json
├── gateway-machine-capacity-v2.json
└── gateway-machine-capacity-typical-v2.csv

artifacts/performance/20260821-3090-qwen05b-expanded/full-50x64/
├── run-01-baseline-abba-1.json ... run-08-candidate-baab-2.json
└── balanced-comparison.json
```

重新生成容量表：

```powershell
uv run python scripts/analyze_gateway_machine_capacity.py `
  --short artifacts/performance/20260821-gateway-capacity/gateway-concurrency-128x128-v2.json `
  --typical artifacts/performance/20260821-gateway-capacity/gateway-concurrency-512x256-v2.json `
  --long artifacts/performance/20260821-gateway-capacity/gateway-concurrency-2048x512-v2.json `
  --utilization 0.60 `
  --out artifacts/performance/20260821-gateway-capacity/gateway-machine-capacity-v2.json `
  --csv artifacts/performance/20260821-gateway-capacity/gateway-machine-capacity-typical-v2.csv
```

指标口径还参考了[OpenTelemetry HTTP服务指标](https://opentelemetry.io/docs/specs/semconv/http/http-metrics/)和[NVIDIA Goodput说明](https://docs.nvidia.com/aiperf/tutorials/metrics-analysis/benchmark-goodput-with-ai-perf)。
