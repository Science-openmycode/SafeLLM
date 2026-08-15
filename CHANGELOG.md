# Changelog

## Unreleased - 2026-08-15

- Added the local-first product workflow: a model can be downloaded, converted,
  verified and retained without a server, then uploaded and deployed later without
  repeating the download, key generation or conversion.
- Promoted the pinned OpenSeek-Small-v1-SFT checkpoint through the product
  conversion and deployment gate, including its normalized fused-expert layout.
- Hardened Ubuntu native deployment with isolated Python startup, pinned CUDA 12.1
  PyTorch packages, dependency verification, GPU/disk preflight, progress reporting
  and retry-safe reuse of already uploaded model files.
- Fixed resumable SFTP lifecycle handling and remote-capacity accounting so a retry
  checks only bytes which are still absent on the server.
- Fixed desktop task recovery, immutable per-job resume plans, local-only deployment
  selection and duplicate deployment operations.
- Fixed OpenSeek chat startup by preserving a self-contained tokenizer from the
  converted package instead of loading repository-specific tokenizer code from the
  source checkpoint. Existing deployment records are repaired on selection.
- Verification: Ruff and Mypy pass; 344 tests pass and the opt-in real-model
  integration test remains skipped unless explicitly enabled.

- Added the OpenSeek paper-complete conversion path for every privacy mechanism
  applicable to the published checkpoint: vocabulary/noise/P-Q expansion,
  residual and RMS coordinates, MLA, reciprocal Q/K scaling, synchronized RoPE
  BlockPerm, Gaussian Uvo, dense/shared/routed FFNs, experts and router.
- Added the custom expanded-width `aloepri_deepseek_v3` runtime and registered it
  in the HF service and DeepSeek-compatible attack launchers.
- Added strict checkpoint completeness, product transport/log boundary, bounded
  attack, real-chat and SHA-256 evidence-index tools. The final mechanism audit
  transforms all 83 source tensors and passes all 17 applicable checks.
- Added the evidence-bound standalone project report covering the paper/PPT
  workflow, Qwen v31/v47 status, OpenSeek coverage, tokenizer round-trip
  counterexamples, runnable commands, and remaining acceptance gates.
- Clarified that the live web path uses one deterministic vocabulary permutation;
  the exact sequence-level M1 is a small-vocabulary formula oracle, while the
  scalable SDK/CLI tokenwise M1 remains an explicitly labelled engineering
  substitute rather than a paper-exact sampler.
- Fixed paper IMA initialization so its two-layer Qwen2 attack backbone remains
  independent of a DeepSeek/MoE target checkpoint.
- Added a localhost-only FastAPI demonstration client that shows the real prompt,
  private input IDs/text, received private output IDs/text, recovered answer, token
  mapping samples, and latency from an actual Qwen2.5-0.5B checkpoint.
- Added a real SSE chain from the model server through the trusted demo gateway;
  every private output token now appears on arrival while the recovered answer is
  rendered character by character instead of replaying a completed response.
- Fixed the demo typewriter completion race triggered by an initial special token
  with empty decoded text; the prompt and submit button now recover after every
  response and support consecutive questions without reloading the page.
- Reworked the local demo into a conversation-first, neutral interface with a
  bottom composer, prompt suggestions, Enter-to-send, and a remembered privacy
  trace switch. Obfuscated input/output, token mappings, and metrics are hidden
  by default and can be revealed without changing the inference request.
- Added trusted-client multi-turn context: the browser submits prior user and
  assistant messages to the local gateway, which applies one chat template and
  obfuscates the entire retained context before contacting the model server.
  Old complete turns are pruned locally when the 1,800-token demo budget is hit.
- Added `/privacy`, an interactive explanation page with offline package
  separation, an online three-lane architecture diagram, selectable protection
  stages, formulas, and a client/network/server visibility matrix.
- Added a live privacy laboratory to `/privacy`: visitors can enter a real
  prompt and watch the current tokenizer, vocabulary permutation, server
  request payload, private output stream, inverse permutation, and recovered
  answer populate the architecture step by step.
- Added `aloepri demo` and an end-to-end-tested two-process demo path without
  changing the token-ID-only model-server protocol.
- Rebuilt the paper line audit for 31 files and 6,081 auditable physical lines;
  parent-symbol classifications now propagate to methods and nested symbols.
- Regenerated the v47 acceptance report against the stable-factor checkpoint;
  its current formal decision remains `NO-GO` (6 PASS, 9 FAIL, 6 NOT_TESTED).
- Hardened request-size enforcement by checking the received body even when the
  caller omits or falsifies `Content-Length`.
- Added `docs/QWEN05B_PAPER_PPT_REAUDIT_2026-08-13.md` with the paper/PPT/code
  mapping, current measurements, corrections, gaps, and human inspection steps.

## v0.3.1 - Qwen0.5B stable exact-metric RMS

- Replaced the cancellation-prone FP32 `z(QQ^T)z^T` runtime path with the
  algebraically identical stable factor form `||zF||^2`, where `FF^T=QQ^T`.
- Added deterministic checkpoint upgrade, manifest binding, formula coverage,
  and a nullspace regression test for the new FP64 RMS factor.
- Updated HF, vLLM, and SGLang adapters to load and evaluate the same factor.
- Reproduced the former long-decode NaN and verified the repaired 0.5B
  checkpoint completes that sample without NaN or invalid token IDs.

## 0.3.0 - 2026-08-11

### Added

- 增加DeepSeek-V2/V3通用适配器，覆盖q/kv低秩坐标、低秩RMSNorm、MLA head/nope/RoPE/value、Dense FFN、Shared Experts、Router和Routed Experts。
- 增加DeepSeek-V2-Lite-Chat固定revision下载、源文件收据校验、逐张量断点转换、2GB重分片和在线/离线密钥拆分。
- 增加多GPU BF16 Prefill、KV Cache Decode、自由生成、MMLU、C-Eval、PIQA、IFEval和HumanEval脚本。
- 增加DeepSeek架构验收器、产品问答冒烟、云主机预检、一键执行脚本和源码上传包生成器。

### Changed

- `inspect-package`现在校验服务端manifest的完整文件集合、字节数、SHA-256、重复项和路径越界；未登记文件或被篡改权重直接拒绝。
- v47的IFEval明文基线改为由当前脚本现场生成的FP32/SDPA工件，不再复用旧v5/BF16结果。
- v47精度验收新增checkpoint、key、脚本、dtype、样本数和上下游工件绑定；不匹配证据不能进入最终结论。
- HF运行时增加`cuda-auto`模型并行、逐GPU显存上限、禁止CPU/磁盘offload以及跨GPU自回归token回送。
- MoE专家置换在`n_group > 1`时改为整组置换加组内置换，保持group-limited routing等价。
- DeepSeek转换密钥按层划分不重叠随机流，并把Dense/Shared/Routed三类FFN全部纳入转换。
- 项目Torch依赖改为`>=2.4,<3`；本地锁文件继续使用CUDA 12.8，云端脚本固定安装PyTorch 2.5.1 CUDA 12.1。

### Verification

- 全仓`ruff`通过。
- `mypy src`检查59个源码文件通过。
- 全仓测试为182 passed、1个需显式开启真实0.5B权重的集成测试跳过。
- DeepSeek-V2/V3极小真实类已通过FP32前向、Prefill、Cache Decode、q_lora有/无、Attention bias、Dense/Shared/Routed FFN、流式转换中断恢复和词表往返测试。

### Cloud scope

- 固定目标为`deepseek-ai/DeepSeek-V2-Lite-Chat@85864749cd611b4353ce1decdb286193298f64c7`。
- 云端真实16B结果由`artifacts/deepseek-v2-lite-chat/final-acceptance.json`生成；本地版本不预填云端数值。
- DeepSeek阶段验收MLA/MoE架构适配；Qwen2.5-0.5B仍是P/Q、噪声、攻击和产品隐私的正式目标。

## 0.2.0 - 2026-08-11

### Added

- 将 Qwen2.5-0.5B 的正式产品目标固定为 v47，并增加单一验收配置。
- 增加目标密钥隔离的 Gate-IA、Attention-IA、IMA、ISA、TFMA、SDA 攻击与独立评分器。
- 增加真实频率攻击语料下载器，数据集 revision、切分规则和文件 SHA-256 写入 manifest。
- 增加私有 token 观测工件，工件只保存服务端可见的混淆 token。
- 增加 v47 验收报告生成器；证据缺失、字段缺失、hash 失配均返回 `NO-GO`。

### Changed

- Gate-IA 使用代数等价的结合律计算，避免构造词表大小乘 FFN 中间维度的临时矩阵。
- Attention-IA 使用维度成立的 leverage 特征和 Qwen 实际 RoPE 配对。
- IMA 训练数据只允许来自非目标转换密钥，训练、目标攻击和目标密钥评分分进程执行。
- TFMA 不再在候选搜索函数中读取 `inverse_tau`。
- SDA 不再把训练、目标攻击和评分混在一个进程，也不允许真值逆置换回退。

### Compatibility

- 历史 v15-v38 配置、脚本和工件保留用于复核；v47 验收器不会读取其结果。
- `run_gate_ia.py`、`run_tfma_curve.py`、`train_sda_smoke.py` 保留为历史实验入口；正式入口
  分别为 `run_gate_ia_isolated.py`、`run_tfma_isolated.py` 和
  `train_sda_transformer.py`/`run_sda_target_attack.py`。
