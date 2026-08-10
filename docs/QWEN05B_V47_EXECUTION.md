# Qwen2.5-0.5B v47 执行手册

## 1. 工作目录和环境

在 Windows PowerShell 中执行：

```powershell
Set-Location E:\AloePri
uv sync --frozen --extra eval
uv run python -c "import torch; print(torch.cuda.get_device_name(0)); print(torch.cuda.mem_get_info())"
```

正式模型、密钥和输出目录：

```text
原模型          data/models/qwen2.5-0.5b
私有模型        data/packages/qwen05b-candidate-v47-best-single
完整离线密钥    data/keys/dev-qwen05b-candidate-v47-best-single
客户端在线密钥  data/keys/qwen05b-candidate-v47-best-single-online
服务器模型包    data/packages/qwen05b-candidate-v47-best-single
正式证据        artifacts/{verification,privacy,accuracy,product,performance}/v47
```

## 2. 转换、公式与运行时

```powershell
uv run aloepri convert --config configs/product/qwen05b_v47_best_single_candidate.yaml
uv run aloepri verify --config configs/product/qwen05b_v47_best_single_candidate.yaml
uv run aloepri inspect-package --server-package data/packages/qwen05b-candidate-v47-best-single
```

转换使用 `.partial`、分片写入、原子重命名和 manifest SHA-256。`verify` 重新检查 24 层
Algorithm 1/2、Embedding/Head、Attention、FFN、RMSNorm、Residual、词表往返、prefill、
decode 和 KV Cache。

## 3. 真实语料和私有观测

```powershell
uv run python scripts/prepare_frequency_attack_corpora.py `
  --out-dir data/eval/privacy/frequency --max-documents 10000

uv run python scripts/build_private_token_observations.py `
  --tokenizer data/models/qwen2.5-0.5b `
  --target-key-dir data/keys/qwen05b-candidate-v47-best-single-online `
  --corpus data/eval/privacy/frequency/huatuo_observed.jsonl `
  --out artifacts/privacy/v47/private-observations.json
```

语料准备器固定 Hugging Face revision。Huatuo 文本按 `SHA256(text) mod 100` 做 80/20
确定性切分，prior 和 observed 之间按文本 hash 去重。

CCI3 当前需要先在 Hugging Face 接受数据集条款并设置 `HF_TOKEN`。无令牌时只能用
`--skip cci3` 做下载器冒烟；manifest 会写入 `formal_corpus_complete=false`，该工件不能用于
正式 TFMA/SDA 验收。

MedDialog 仍以旧 Hugging Face dataset script 发布，而本项目锁定的 `datasets==5.0.0` 已
移除 dataset script 执行。先从数据集卡给出的 Google Drive 地址下载 `processed.zh` train
文件，再给准备器增加 `--meddialog-processed-zh-file <文件路径>`。使用
`--skip meddialog` 同样只允许做本机冒烟。

## 4. Direct Match 与 VMA

```powershell
uv run python scripts/run_direct_weight_match_isolated.py `
  --original data/models/qwen2.5-0.5b `
  --private data/packages/qwen05b-candidate-v47-best-single `
  --sample 512 --out artifacts/privacy/v47/direct-weight-predictions.json

uv run python scripts/score_mapping_predictions.py `
  --predictions artifacts/privacy/v47/direct-weight-predictions.json `
  --target-key-dir data/keys/dev-qwen05b-candidate-v47-best-single `
  --out artifacts/privacy/v47/direct-weight-score.json

uv run python scripts/build_vma_candidate_observations.py `
  --original data/models/qwen2.5-0.5b `
  --target-key-dir data/keys/dev-qwen05b-candidate-v47-best-single `
  --candidate-sizes 16384 `
  --out artifacts/privacy/v47/vma-candidates-c16384.json

$layers = 0..23
uv run python scripts/run_vma_pupa.py `
  --original data/models/qwen2.5-0.5b `
  --private data/packages/qwen05b-candidate-v47-best-single `
  --candidate-observations artifacts/privacy/v47/vma-candidates-c16384.json `
  --candidate-sizes 16384 --layers $layers `
  --combinations We_Wh We_Wq_We_WkT We_Wgate We_Wup Wdown_Wh `
  --prediction-cache-dir artifacts/privacy/cache/v47-isolated-c16384 `
  --out artifacts/privacy/v47/vma-predictions.json

uv run python scripts/score_vma_predictions.py `
  --original data/models/qwen2.5-0.5b `
  --predictions artifacts/privacy/v47/vma-predictions.json `
  --target-key-dir data/keys/dev-qwen05b-candidate-v47-best-single `
  --out artifacts/privacy/v47/vma-score.json
```

候选准备器读取密钥后只输出打乱的私有候选集合；攻击器知道明文候选集合和私有候选集合，
但不知道二者映射。五种组合分别跨 24 层投票，评分器取各组合最强恢复结果做安全门禁。

## 5. Gate-IA 与 Attention-IA

```powershell
uv run python scripts/run_gate_ia_isolated.py `
  --original data/models/qwen2.5-0.5b `
  --private data/packages/qwen05b-candidate-v47-best-single `
  --sample 512 --candidate-size 0 `
  --out artifacts/privacy/v47/gate-ia-predictions.json

uv run python scripts/score_mapping_predictions.py `
  --predictions artifacts/privacy/v47/gate-ia-predictions.json `
  --target-key-dir data/keys/dev-qwen05b-candidate-v47-best-single `
  --out artifacts/privacy/v47/gate-ia-score.json

uv run python scripts/run_attn_ia_corrected.py `
  --original data/models/qwen2.5-0.5b `
  --private data/packages/qwen05b-candidate-v47-best-single `
  --sample 512 --candidate-size 0 `
  --out artifacts/privacy/v47/attention-ia-predictions.json

uv run python scripts/score_mapping_predictions.py `
  --predictions artifacts/privacy/v47/attention-ia-predictions.json `
  --target-key-dir data/keys/dev-qwen05b-candidate-v47-best-single `
  --out artifacts/privacy/v47/attention-ia-score.json
```

`candidate-size=0` 表示全词表。非零前缀候选只用于调试，不能作为正式验收工件。

## 6. IMA

IMA 训练 checkpoint 必须使用不同结构种子。当前独立训练 key 为 v37：

```powershell
uv run aloepri convert `
  --config configs/product/qwen05b_v37_seed20260001_ae065_ah06_blockperm8.yaml

uv run python scripts/build_ima_training_pairs.py `
  --original data/models/qwen2.5-0.5b `
  --target-private data/packages/qwen05b-candidate-v47-best-single `
  --target-key-dir data/keys/dev-qwen05b-candidate-v47-best-single `
  --training-private data/packages/qwen05b-candidate-v37-seed20260001-ae065-ah06-blockperm8 `
  --training-key-dir data/keys/dev-qwen05b-candidate-v37-seed20260001-ae065-ah06-blockperm8 `
  --corpus data/eval/privacy/frequency/huatuo_prior.jsonl `
  --sequence-length 32 --train-sequences 128 --val-sequences 16 `
  --out artifacts/privacy/v47/ima-pairs.safetensors

uv run python scripts/train_ima_inverter.py `
  --original data/models/qwen2.5-0.5b `
  --pairs artifacts/privacy/v47/ima-pairs.safetensors `
  --epochs 2 --batch-size 4 --gpu-memory-fraction 0.70 `
  --out-dir artifacts/privacy/v47/ima-inverter

uv run python scripts/run_ima_target_attack.py `
  --original data/models/qwen2.5-0.5b `
  --private data/packages/qwen05b-candidate-v47-best-single `
  --inverter artifacts/privacy/v47/ima-inverter `
  --private-token-ids artifacts/privacy/v47/private-observations.json `
  --out artifacts/privacy/v47/ima-predictions.json

uv run python scripts/score_token_inversion_predictions.py `
  --predictions artifacts/privacy/v47/ima-predictions.json `
  --target-key-dir data/keys/dev-qwen05b-candidate-v47-best-single `
  --out artifacts/privacy/v47/ima-score.json
```

训练对构建器会比较绝对路径并拒绝把 v47 checkpoint 或 key 放进训练集合。

## 7. ISA

```powershell
uv run python scripts/run_isa_attention_score.py `
  --original data/models/qwen2.5-0.5b `
  --private data/packages/qwen05b-candidate-v47-best-single `
  --private-token-ids artifacts/privacy/v47/private-observations.json `
  --layer 12 --steps 100 --learning-rate 0.05 `
  --max-sequences 20 --max-length 16 `
  --out artifacts/privacy/v47/isa-attention-predictions.json

uv run python scripts/score_token_inversion_predictions.py `
  --predictions artifacts/privacy/v47/isa-attention-predictions.json `
  --target-key-dir data/keys/dev-qwen05b-candidate-v47-best-single `
  --out artifacts/privacy/v47/isa-attention-score.json
```

Attention ISA 对 Transformers 返回的 post-softmax attention 权重做论文形式的输入优化。
论文没有明确 E10 的 score 指 pre-softmax 还是 post-softmax，因此结果 JSON 固定记录该选择。
Expanded Hidden ISA 另有 `run_isa_hidden_state_isolated.py`，其 token-Gram 目标只作为诊断，
不写入 Attention ISA 的正式结果。

## 8. TFMA

```powershell
uv run python scripts/run_tfma_isolated.py `
  --tokenizer data/models/qwen2.5-0.5b `
  --prior-corpus data/eval/privacy/frequency/cci3_prior.jsonl `
  --prior-corpus data/eval/privacy/frequency/meddialog_prior.jsonl `
  --private-observations artifacts/privacy/v47/private-observations.json `
  --corpus-manifest data/eval/privacy/frequency/manifest.json `
  --topk 100 --out artifacts/privacy/v47/tfma-predictions.json

uv run python scripts/score_tfma_predictions.py `
  --predictions artifacts/privacy/v47/tfma-predictions.json `
  --target-key-dir data/keys/dev-qwen05b-candidate-v47-best-single `
  --out artifacts/privacy/v47/tfma-score.json
```

## 9. SDA

```powershell
uv run python scripts/train_sda_transformer.py `
  --tokenizer data/models/qwen2.5-0.5b `
  --train-corpus data/eval/privacy/frequency/huatuo_prior.jsonl `
  --corpus-manifest data/eval/privacy/frequency/manifest.json `
  --max-sequences 10000 --max-length 128 --steps 3000 --batch-size 32 `
  --out-dir artifacts/privacy/v47/sda-decoder

uv run python scripts/run_sda_target_attack.py `
  --decoder-dir artifacts/privacy/v47/sda-decoder `
  --private-observations artifacts/privacy/v47/private-observations.json `
  --out artifacts/privacy/v47/sda-predictions.json

uv run python scripts/score_sda_predictions.py `
  --predictions artifacts/privacy/v47/sda-predictions.json `
  --target-key-dir data/keys/dev-qwen05b-candidate-v47-best-single `
  --out artifacts/privacy/v47/sda-score.json
```

## 10. Known-plaintext

```powershell
uv run python scripts/build_known_plaintext_observations.py `
  --tokenizer data/models/qwen2.5-0.5b `
  --target-key-dir data/keys/qwen05b-candidate-v47-best-single-online `
  --corpus data/eval/privacy/frequency/huatuo_observed.jsonl `
  --known-pairs 1000 --test-tokens 100000 `
  --out artifacts/privacy/v47/known-plaintext-observations.json

uv run python scripts/run_known_plaintext_isolated.py `
  --observations artifacts/privacy/v47/known-plaintext-observations.json `
  --out artifacts/privacy/v47/known-plaintext-predictions.json

uv run python scripts/score_token_inversion_predictions.py `
  --predictions artifacts/privacy/v47/known-plaintext-predictions.json `
  --target-key-dir data/keys/dev-qwen05b-candidate-v47-best-single `
  --out artifacts/privacy/v47/known-plaintext-score.json
```

改变 `--known-pairs` 为 `10, 100, 1000, 10000` 分别运行即可得到观测量增长曲线；每个点
使用相同 `seed` 和测试流，只有已知映射前缀长度变化。

## 11. 代码检查和统一验收

```powershell
uv run ruff check src scripts tests
uv run mypy src
uv run pytest -q
uv run python scripts/build_qwen05b_v47_acceptance.py `
  --config configs/acceptance/qwen05b_v47.yaml `
  --out artifacts/acceptance/qwen05b-v47-current.json
```

验收器只读取配置列出的 v47 工件。路径不存在、字段缺失、密钥隔离标志缺失、checkpoint
manifest 失败或指标越界，最终结论都是 `NO-GO`。
