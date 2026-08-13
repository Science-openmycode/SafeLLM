# OpenSeek-Small-v1-SFT 本地构建与运行手册

## 固定对象

| 项目 | 值 |
|---|---|
| Hugging Face 仓库 | `BAAI/OpenSeek-Small-v1-SFT` |
| revision | `1515c184e6fe4a91e6061be513a79d607e8787cb` |
| 架构 | DeepSeek-V3，6 层，MLA，64 routed experts，2 shared experts，Top-6 |
| MTP | 上游配置声明 1 层，但 checkpoint 没有 MTP 张量，因此当前不适用 |
| 正式配置 | `configs/product/openseek_small_v1_sft_paper_complete.yaml` |

## 目录

```text
data/models/openseek-small-v1-sft/                         固定 revision 原文件
data/models/openseek-small-v1-sft-transformers/            标准 Transformers checkpoint
data/packages/openseek-small-v1-sft-paper-complete/        只部署到模型服务器
data/keys/openseek-small-v1-sft-paper-complete-online/     只放可信客户端
data/keys/openseek-small-v1-sft-paper-complete-offline/    转换后离线归档
artifacts/openseek-small-v1-sft-paper-complete/             当前 checkpoint 证据
```

禁止把 online/offline key 复制进 server package。服务器只能读取 `data/packages/...paper-complete`。

## 从零构建

```powershell
cd E:\AloePri
uv sync --frozen

.\.venv\Scripts\python.exe scripts\download_hf_repo.py `
  --repo BAAI/OpenSeek-Small-v1-SFT `
  --revision 1515c184e6fe4a91e6061be513a79d607e8787cb `
  --out data\models\openseek-small-v1-sft `
  --allow '*' --chunk-size-mb 128 --workers 4 --retries 20

.\.venv\Scripts\python.exe scripts\normalize_openseek_checkpoint.py `
  --source data\models\openseek-small-v1-sft `
  --output data\models\openseek-small-v1-sft-transformers `
  --max-shard-size-gib 0.5

.\.venv\Scripts\python.exe scripts\convert_deepseek_streaming.py `
  --source data\models\openseek-small-v1-sft-transformers `
  --output data\packages\openseek-small-v1-sft-paper-complete `
  --key-dir data\keys\openseek-small-v1-sft-paper-complete-offline `
  --online-key-dir data\keys\openseek-small-v1-sft-paper-complete-online `
  --seed 20260813 `
  --vocab-permutation `
  --paper-complete `
  --expansion-h 128 --lambda 0.3 `
  --alpha-e 1.0 --alpha-h 0.2 `
  --block-beta 8 --sampling-gamma 1000 `
  --qk-scale-min 0.5 --qk-scale-max 2.0 `
  --uvo-condition-max 100 `
  --ffn-scale-min 0.5 --ffn-scale-max 2.0 `
  --blockperm-mode paper-distribution-boundary-corrected `
  --rms-mode paper_kappa `
  --router-normalize `
  --expected-source-revision 1515c184e6fe4a91e6061be513a79d607e8787cb
```

转换器逐 tensor 写入 `.partial` 目录，每个文件原子落盘并记录 SHA-256。中断后使用同一命令加 `--resume`；任一参数改变都会拒绝续跑。

## 启动模型服务

CPU：

```powershell
$env:OMP_NUM_THREADS='8'
$env:MKL_NUM_THREADS='8'
.\.venv\Scripts\python.exe scripts\serve_private.py `
  --model data\packages\openseek-small-v1-sft-paper-complete `
  --host 127.0.0.1 --port 8011 --device cpu --dtype bfloat16
```

有足够显存的 CUDA 主机：

```powershell
.\.venv\Scripts\python.exe scripts\serve_private.py `
  --model data\packages\openseek-small-v1-sft-paper-complete `
  --host 127.0.0.1 --port 8011 --device cuda --dtype bfloat16
```

模型服务不要直接监听公网。远程使用时在外层配置 TLS 和 Bearer Token；模型端口保持内网或 localhost。

## 真实本地问答

```powershell
.\.venv\Scripts\python.exe scripts\run_openseek_local_smoke.py `
  --model data\packages\openseek-small-v1-sft-paper-complete `
  --online-key data\keys\openseek-small-v1-sft-paper-complete-online\online_key.safetensors `
  --tokenizer data\models\openseek-small-v1-sft-transformers `
  --device cpu --dtype bfloat16 `
  --prompt "你好，请用一句话介绍你自己。" `
  --out artifacts\openseek-small-v1-sft-paper-complete\real-chat-smoke.json
```

该命令实际执行 Chat Template、分词、`tau`、私有 checkpoint 生成、`inverse_tau` 和本地解码，并保存私有输入、私有输出、恢复 token、TTFT 和 TPOT。

## 必跑验收

```powershell
.\.venv\Scripts\python.exe scripts\audit_openseek_paper_complete.py `
  --source data\models\openseek-small-v1-sft-transformers `
  --private data\packages\openseek-small-v1-sft-paper-complete `
  --online-key-dir data\keys\openseek-small-v1-sft-paper-complete-online `
  --out artifacts\openseek-small-v1-sft-paper-complete\privacy-completeness-audit.json

.\.venv\Scripts\python.exe scripts\run_openseek_privacy_attack_smoke.py `
  --source data\models\openseek-small-v1-sft-transformers `
  --private data\packages\openseek-small-v1-sft-paper-complete `
  --offline-key-dir data\keys\openseek-small-v1-sft-paper-complete-offline `
  --online-key-dir data\keys\openseek-small-v1-sft-paper-complete-online `
  --sample-size 64 `
  --out artifacts\openseek-small-v1-sft-paper-complete\privacy-attack-smoke-64.json

.\.venv\Scripts\python.exe scripts\build_openseek_privacy_evidence_index.py `
  --package data\packages\openseek-small-v1-sft-paper-complete `
  --online-key-dir data\keys\openseek-small-v1-sft-paper-complete-online `
  --offline-key-dir data\keys\openseek-small-v1-sft-paper-complete-offline `
  --evidence-dir artifacts\openseek-small-v1-sft-paper-complete `
  --out artifacts\openseek-small-v1-sft-paper-complete\evidence-index.json
```

完整的公式—代码—工件逐项审核见 `docs/OPENSEEK_PRIVACY_COMPLETENESS_AUDIT.md`。
