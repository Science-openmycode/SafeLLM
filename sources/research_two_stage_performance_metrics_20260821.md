# 两段式性能核算的一手资料（2026-08-21）

本文件保存“本地隐私网关”和“云端模型推理”必须分开测量的外部依据。

## 本地HTTP/网关服务

- OpenTelemetry HTTP Metrics：<https://opentelemetry.io/docs/specs/semconv/http/http-metrics/>
  - 推荐记录 `http.server.request.duration`、`http.server.active_requests`、请求体大小和响应体大小。
  - 这些指标描述HTTP服务本身，不包含模型参数量这一维度。
- Hugging Face Tokenizers Pipeline：<https://huggingface.co/docs/tokenizers/main/pipeline>
  - Tokenizer输入处理由 normalization、pre-tokenization、model、post-processing组成。
- Hugging Face Tokenizers Quicktour：<https://huggingface.co/docs/tokenizers/v0.13.0/en/quicktour>
  - 批量输入应使用 `encode_batch`，说明网关容量会受批处理方式和Tokenizer实现影响。

## 云端LLM推理

- NVIDIA AIPerf Metrics Reference：<https://docs.nvidia.com/aiperf/reference/ai-perf-metrics-reference>
  - TTFT包含网络、排队、Prefill和首Token生成。
  - ITL/TPOT表示稳态生成阶段相邻Token的平均时间。
  - Output Token Throughput是所有并发请求的聚合输出Token数除以测试时长。
  - Request Throughput是完成请求数除以测试时长。
- NVIDIA GenAI-Perf：<https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_analyzer/genai-perf/README.html>
  - 正式报告包含输入/输出长度、请求吞吐、输出Token吞吐、请求延迟及GPU遥测。
- NVIDIA TensorRT-LLM Performance Overview：<https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/developer-guide/perf-overview.md>
  - 最大吞吐用 output tokens/s/GPU，且结果与模型、GPU、Batch、并行方式和输入/输出长度绑定。
- NVIDIA AIPerf Goodput：<https://docs.nvidia.com/aiperf/tutorials/metrics-analysis/benchmark-goodput-with-ai-perf>
  - Goodput定义为满足给定SLO的完成请求/s。
- MLPerf Inference Datacenter：<https://mlcommons.org/benchmarks/inference-datacenter/>
  - Server场景在标准请求到达模式和延迟约束下报告吞吐。

## 本项目采用的边界

```text
本地隐私网关：明文输入 -> 本地分词 -> tau/M1 -> 私有Token请求
                私有Token流 -> inverse_tau -> 增量解码 -> 明文输出

云端模型推理：私有Token请求 -> 私有checkpoint Prefill/Decode -> 私有Token流
```

两段通过网络连接，但性能账本必须独立：本地网关用请求率和Token流量核算；云端推理用模型、后端、GPU、Batch、上下文及SLO核算。
