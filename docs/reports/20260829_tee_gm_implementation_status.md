# 隐变智模 TEE 国密模式实施状态

日期：2026-08-29  
基线分支：`codex/aloepri-baseline`  
基线提交：`a350f719b4d206ec2723420488b2dfa79cf3aa6b`

## 已完成

| 工作项 | 实现位置 | 本轮证据 |
|---|---|---|
| 模式组合与旧配置默认值 | `src/aloepri/tee/config.py` | 单元测试通过 |
| Qwen TEE split 转换 | `scripts/convert_paper_qwen2_checkpoint.py` | 真实0.5B三包工件 |
| Embedding/Head边界拆分 | `src/aloepri/tee/boundary.py` | greedy对齐 |
| `inputs_embeds`主体与KV Cache | `src/aloepri/serving/tee_runtime.py` | API/SSE测试 |
| Local TEE Head | `src/aloepri/tee/boundary.py` | 38个真实生成Token一致 |
| 掩码外包Head与Freivalds | `src/aloepri/secure_head/` | 单元、恶意返回和真实单Token测试 |
| 软件模拟证明状态机 | `src/aloepri/tee/attestation.py`、`service.py` | 测试通过，明确非硬件 |
| Intel DCAP verifier适配边界 | `src/aloepri/tee/attestation.py` | 外部验证器契约测试 |
| RFC8998原生客户端编排 | `src/aloepri/tee/gm_client_entry.py` | TLS版本、套件和公钥Pin测试 |
| SM2/SM3/SM4 helper契约 | `src/aloepri/tee/gm.py` | 签名/验签/分块加密接口 |
| 包哈希、签名与秘密扫描 | `src/aloepri/tee/package.py` | 真实server包扫描通过 |
| CLI与桌面模式选择 | `src/aloepri/cli.py`、`product_cli.py`、`desktop/` | 桌面/API回归通过 |
| 本地TEE部署与问答 | `src/aloepri/product/local_deployment.py` | 独立进程中文问答成功 |

## 实测工件

```text
artifacts/tee/qwen05b-tee-sim-v2-body
artifacts/tee/qwen05b-tee-sim-v2-boundary
artifacts/tee/qwen05b-tee-sim-v2-offline-raw
artifacts/tee/qwen05b-tee-sim-v2-verification.json
artifacts/tee/qwen05b-tee-sim-v2-verification-200x32.json
artifacts/tee/qwen05b-tee-sim-masked-head-verification-v2.json
```

真实0.5B软件模拟结果：

| 指标 | 结果 |
|---|---:|
| 验证Prompt | 200 |
| 生成Token | 6,351 |
| Local Head greedy一致率 | 100%（6,351/6,351 Token；200/200 Prompt） |
| 逐层Logits NRMSE | `3.5e-6`至`5.3e-6` |
| Masked Head真实权重闭环 | 1 Token，认证后一致 |
| TEE split TTFT p50/p95/p99 | 97.14 / 104.89 / 109.45 ms |
| TEE split TPOT p50/p95/p99 | 65.11 / 67.36 / 69.52 ms |
| server package检查 | 7文件、318张量、0失败 |
| 全仓库自动化测试 | 393 passed，1 skipped |
| Ruff | PASS |
| Mypy | PASS |

200条使用固定清单 `configs/eval/gate1_prompts_200.json`，每条最多生成32 Token。

## 尚需真实硬件执行

| 门禁 | 当前状态 | 完成条件 |
|---|---|---|
| Intel TDX Guest | 未执行 | `/dev/tdx_guest`和非Debug TD |
| DCAP Quote/PCCS | 未执行 | QVL验证Quote、TCB与Collateral |
| MRTD/RTMR白名单 | 未执行 | 固定Guest镜像测量并发布白名单 |
| RFC8998线上互操作 | 未执行 | Tongsuo两端实际握手与抓包 |
| TDX私有页/GPU共享缓冲 | 未执行 | TDX服务器上的宿主篡改实验 |
| TDX三模式性能表 | 未执行 | 同一TDX硬件三种Head实测 |

因此当前准确状态是：

```text
TEE_GM_SOFTWARE_SIM_PASS
TEE_GM_TDX_ATTESTED_PASS = NOT_YET_PHYSICALLY_EXECUTED
TEE_GM_MASKED_HEAD_PASS = SOFTWARE_SIM_ONLY
```

## 复现命令

```powershell
uv run python -m aloepri.serving.tee_native_entry `
  --model artifacts/tee/qwen05b-tee-sim-v2-body `
  --tee-boundary artifacts/tee/qwen05b-tee-sim-v2-boundary `
  --host 127.0.0.1 --port 18131 --device cuda --head-mode local

uv run yinbian chat-tee `
  --server http://127.0.0.1:18131 `
  --tee-backend software_sim `
  --tokenizer data/models/qwen2.5-0.5b-instruct `
  --model-id qwen2.5-0.5b-instruct-tee-sim-v2 `
  --model-version 7ae557604adf67be50417f59c2c2f167def9a775 `
  --key-id key-tee-sim-v2-20260829 `
  --vocab-size 151936 `
  --prompt "请只回答：北京" `
  --max-new-tokens 8
```

本轮实测输出：`北京`。
