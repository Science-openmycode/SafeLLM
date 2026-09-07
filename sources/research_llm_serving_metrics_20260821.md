# LLM serving and gateway metric research — 2026-08-21

## Primary sources

1. NVIDIA AIPerf metrics reference
   - URL: https://docs.nvidia.com/aiperf/reference/ai-perf-metrics-reference
   - Relevant definitions: TTFT, TTST, decode duration, ITL, inter-chunk latency, per-user output throughput, prefill throughput, record distributions and aggregate metrics.

2. NVIDIA GenAI-Perf documentation
   - URL: https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_analyzer/genai-perf/README.html
   - Relevant definitions: output token throughput, request throughput, input/output sequence length, request latency, TTFT and inter-token latency.

3. NVIDIA GenAI-Perf goodput guide
   - URL: https://docs.nvidia.com/deeplearning/triton-inference-server/archives/triton-inference-server-2500/user-guide/docs/perf_analyzer/genai-perf/docs/goodput.html
   - Relevant definition: completed requests per second that satisfy specified latency SLOs.

4. vLLM production metrics
   - URL: https://docs.vllm.ai/en/v0.22.0/design/metrics/
   - Relevant metrics: KV-cache usage, prompt/generation tokens, request queue time and prefix-cache hit rate.

5. NVIDIA TensorRT-LLM performance overview
   - URL: https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/developer-guide/perf-overview.md
   - Relevant normalization: output tokens/s/GPU, per-user output tokens/s, TTFT, TPOT, request throughput and total token throughput.

6. MLCommons MLPerf Inference Datacenter
   - URL: https://mlcommons.org/benchmarks/inference-datacenter/
   - Relevant methodology: server and offline scenarios, latency-constrained throughput, and full-system wall-power measurement.

7. DistServe paper
   - URL: https://arxiv.org/abs/2401.09670
   - Relevant methodology: separate TTFT and TPOT SLOs and compare serving systems by goodput under SLO attainment.

## Metrics selected for Yinbian

- Local gateway: service time by input/output token, requests/s/core, tokens/s/core, core-hours/million requests, multi-core scaling efficiency, p95/p99 tail amplification, protocol byte amplification, RSS/worker, requests/s/GiB, M1 incremental overhead.
- Private model: TTFT, ITL/TPOT, request latency, per-user tokens/s, aggregate output tokens/s/GPU, goodput under the 15% product gate, weight and GPU-memory amplification, tokens/s/GiB, GPU-hours/million output tokens.
- Production-only future metrics: queue time, KV-cache usage, prefix-cache hit rate, inter-chunk jitter, error/retry rate, wall energy per million tokens, cost per SLO-qualified million tokens.
