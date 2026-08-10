# AloePri 工程实施 Runbook v2.1

内部执行文档。命令默认在 Ubuntu 22.04/24.04、仓库根目录 `/opt/aloepri` 执行。

---

## 1. 固定技术栈

| 项目 | 固定值 | 来源 |
|---|---|---|
| 基础模型 | `Qwen/Qwen2.5-0.5B-Instruct` | `https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct` |
| 规模回归 | `Qwen/Qwen2.5-7B-Instruct`、`Qwen/Qwen2.5-14B-Instruct` | Hugging Face Qwen 组织 |
| DeepSeek 适配参考 | `deepseek-ai/DeepSeek-V3` | `https://github.com/deepseek-ai/DeepSeek-V3` |
| Python | `3.11` | `uv` 管理 |
| PyTorch | 项目锁文件固定；CUDA wheel 按机器驱动选择 | `https://pytorch.org/get-started/locally/` |
| Transformers | 首次建仓锁定后禁止自动升级 | `https://github.com/huggingface/transformers` |
| vLLM | 独立环境锁定；不得与 HF 环境共用 | `https://github.com/vllm-project/vllm` |
| 权重格式 | `safetensors` | `https://github.com/huggingface/safetensors` |
| API | FastAPI + Uvicorn | `https://github.com/fastapi/fastapi` |
| 评测 | `lm-evaluation-harness`、`evalplus` | 官方 Git 仓库 |
| 依赖锁定 | `uv.lock` | `https://docs.astral.sh/uv/` |

不把任何库的 `main` 分支用于验收。Day 1 解析出可运行版本后提交 `uv.lock`、镜像 digest、GPU 驱动和 CUDA 信息，后续只能通过变更单升级。

## 2. 机器和目录

### 2.1 本地开发机

Windows PowerShell（管理员）：

```powershell
wsl --install -d Ubuntu-22.04
wsl --update
wsl --shutdown
```

Ubuntu：

```bash
sudo apt-get update
sudo apt-get install -y git git-lfs build-essential cmake ninja-build jq curl aria2 \
  libopenmpi-dev openssh-server rsync tmux htop nvtop
git lfs install
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
nvidia-smi
```

成功条件：`nvidia-smi` 能看到 GPU；`uv --version`、`git lfs version` 返回 0。

### 2.2 固定路径

```bash
sudo mkdir -p /opt/aloepri /data/models /data/checkpoints /data/evidence /data/cache/huggingface
sudo chown -R "$USER":"$USER" /opt/aloepri /data
export HF_HOME=/data/cache/huggingface
export TRANSFORMERS_CACHE=/data/cache/huggingface/hub
export ALOEPRI_ROOT=/opt/aloepri
```

将三行 `export` 写入 `~/.bashrc`。模型、转换结果和证据不得放在仓库内。

## 3. 建仓和依赖

### 3.1 创建仓库

```bash
cd /opt/aloepri
git init
uv init --lib --python 3.11
uv add torch transformers accelerate safetensors tokenizers numpy scipy pydantic \
  pyyaml typer rich fastapi 'uvicorn[standard]' httpx orjson cryptography psutil
uv add --dev pytest pytest-xdist pytest-cov hypothesis ruff mypy
uv lock
uv sync --frozen
```

vLLM 单独建环境：

```bash
uv venv .venv-vllm --python 3.11
UV_PROJECT_ENVIRONMENT=.venv-vllm uv pip install vllm
.venv-vllm/bin/python -c 'import vllm; print(vllm.__version__)'
```

### 3.2 仓库结构

```text
/opt/aloepri/
├── pyproject.toml
├── uv.lock
├── configs/
│   ├── models/qwen05b.yaml
│   ├── models/qwen7b.yaml
│   ├── models/qwen14b.yaml
│   ├── transform/paper.yaml
│   ├── transform/square_fallback.yaml
│   ├── serving/hf.yaml
│   ├── serving/vllm.yaml
│   └── eval/{smoke,accuracy,privacy,performance}.yaml
├── src/aloepri/
│   ├── config.py
│   ├── manifest.py
│   ├── keys/{schema.py,generate.py,store.py}
│   ├── transforms/{vocab.py,noise.py,pq.py,linear.py,norm.py,rope.py,attention.py,ffn.py}
│   ├── models/{qwen2.py,toy_transformer.py,toy_moe.py,toy_mla.py,deepseek.py}
│   ├── conversion/{planner.py,reader.py,writer.py,convert.py,resume.py,verify.py}
│   ├── serving/{protocol.py,hf_server.py,vllm_model.py,vllm_server.py}
│   ├── client/{tokenizer.py,sdk.py,stream.py}
│   ├── attacks/{direct.py,vma.py,ia.py,tfma.py,sda.py,known_plaintext.py}
│   └── eval/{accuracy.py,privacy.py,performance.py,compare.py}
├── scripts/{download_model.py,audit_model.py,generate_key.py,convert_model.py,serve_hf.py,client_smoke.py,run_eval.py}
├── tests/{unit,integration,equivalence,serving,attacks,performance}/
├── deploy/{Dockerfile.hf,Dockerfile.vllm,compose.yaml,slurm,systemd}/
└── docs/{tensor_ledger.md,decisions,runbooks}/
```

创建目录：

```bash
mkdir -p configs/{models,transform,serving,eval} src/aloepri/{keys,transforms,models,conversion,serving,client,attacks,eval} \
  scripts tests/{unit,integration,equivalence,serving,attacks,performance} deploy/{slurm,systemd} docs/{decisions,runbooks}
find src/aloepri -type d -exec touch {}/__init__.py \;
```

## 4. 配置文件

### 4.1 `configs/models/qwen05b.yaml`

```yaml
model_id: Qwen/Qwen2.5-0.5B-Instruct
revision: main                 # 下载后替换为实际 commit SHA
source_dir: /data/models/qwen2.5-0.5b
output_dir: /data/checkpoints/qwen2.5-0.5b-obfuscated
dtype: bfloat16
trust_remote_code: false
max_position_embeddings: 32768
```

### 4.2 `configs/transform/paper.yaml`

```yaml
schema_version: 1
seed: 20260803
key_id: dev-qwen05b-001
vocab:
  enabled: true
  permutation: random
embedding_noise:
  enabled: true
  distribution: gaussian
  std: 0.001
linear:
  mode: paper_expand
  expansion_h: 64
  condition_number_max: 100.0
  inverse_residual_max: 1.0e-5
attention:
  permute_heads: true
  permute_blocks: true
  transform_q: true
  transform_k: true
  transform_v: true
  transform_o: true
rope:
  apply_before_coordinate_transform: true
checkpoint:
  shard_size_gb: 4
  resume: true
  hash: sha256
```

### 4.3 `configs/transform/square_fallback.yaml`

与上面相同，只改：

```yaml
linear:
  mode: square_orthogonal
  expansion_h: 0
  condition_number_max: 10.0
  inverse_residual_max: 1.0e-6
```

### 4.4 服务配置

`configs/serving/hf.yaml`：

```yaml
host: 0.0.0.0
port: 8000
model_dir: /data/checkpoints/qwen2.5-0.5b-obfuscated
manifest: /data/checkpoints/qwen2.5-0.5b-obfuscated/aloepri_manifest.json
key_registry: /etc/aloepri/key_registry.json
log_token_ids: false
log_prompts: false
max_input_tokens: 4096
max_new_tokens: 512
```

## 5. 密钥和 manifest

### 5.1 密钥结构

`src/aloepri/keys/schema.py` 定义以下 Pydantic 类型：

```python
class LayerKey(BaseModel):
    name: str
    p_path: str | None = None
    q_path: str | None = None
    head_perm_path: str | None = None
    block_perm_path: str | None = None

class ModelKey(BaseModel):
    schema_version: int
    key_id: str
    model_id: str
    source_revision: str
    tau_path: str
    inverse_tau_path: str
    layers: list[LayerKey]
```

`generate.py` 必须使用 `numpy.random.Generator(PCG64(seed))`；生产密钥不接受外部 seed，改用 `secrets.token_bytes(32)`。`tau` 必须是 `[vocab_size]` 的双射，`inverse_tau[tau] == arange(vocab_size)`。

### 5.2 密钥存放

```text
/data/keys/<key_id>/key.json
/data/keys/<key_id>/tau.safetensors
/data/keys/<key_id>/layer-0000.safetensors
```

权限：

```bash
sudo install -d -m 0700 -o "$USER" -g "$USER" /data/keys
chmod -R go-rwx /data/keys
```

服务端只部署 `key_id -> model_dir` 注册表，不部署 `tau`、`inverse_tau` 或矩阵密钥。

### 5.3 Manifest

`src/aloepri/manifest.py` 输出 `/data/checkpoints/<model>/aloepri_manifest.json`：

```json
{
  "schema_version": 1,
  "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
  "source_revision": "<commit-sha>",
  "key_id": "dev-qwen05b-001",
  "transform_mode": "paper_expand",
  "dtype": "bfloat16",
  "files": [{"path": "model-00001-of-00002.safetensors", "sha256": "...", "bytes": 0}],
  "created_at": "RFC3339 UTC",
  "tool_git_commit": "<commit-sha>"
}
```

## 6. 逐模块实现

### 6.1 词表置换

实现位置：`src/aloepri/transforms/vocab.py`。

函数：

```python
def permute_embedding(weight, tau):
    # 新 token i 使用原 token inverse_tau[i] 的行
    return weight.index_select(0, inverse_permutation(tau))

def permute_lm_head(weight, tau):
    return weight.index_select(0, inverse_permutation(tau))

def encode_private(input_ids, tau):
    return tau[input_ids]

def decode_private(output_ids, inverse_tau):
    return inverse_tau[output_ids]
```

必须单测：双射、逆变换、特殊 token、全词表遍历、tied/untied LM Head。若 `tie_word_embeddings=true`，转换后重新共享同一 Parameter；不得保存两份独立权重。

### 6.2 Embedding/Head 噪声

实现位置：`transforms/noise.py`。噪声在 FP32 中生成和相加，再转回目标 dtype。Embedding 和 LM Head 的噪声必须满足方案中的抵消关系；若不能证明抵消，默认关闭，不允许以“精度可接受”替代等价性证明。

接口：

```python
def add_embedding_noise(embedding, lm_head, generator, std, tied):
    ...
```

输出统计写入 manifest：均值、标准差、最大绝对值、seed ID，不写 seed 原值。

### 6.3 P/Q 矩阵

实现位置：`transforms/pq.py`。

1. 在 FP64 中生成候选矩阵
2. 用 QR/SVD 构造满秩矩阵
3. 计算 `cond(P)`、`cond(Q)` 和逆残差
4. 超阈值重新采样
5. 保存 FP32 主副本；转换时按权重 dtype 投影

```python
@dataclass
class TransformPair:
    left: Tensor
    right: Tensor
    left_inverse: Tensor
    right_inverse: Tensor
    condition_number: float

def make_square(dim: int, generator, cond_max: float) -> TransformPair: ...
def make_expanded(dim: int, h: int, generator, cond_max: float) -> TransformPair: ...
def verify_pair(pair, atol: float, rtol: float) -> None: ...
```

扩维分支的张量形状必须先写入 `docs/tensor_ledger.md`。任何残差加法两侧最后一维必须相同。Day 14 若 Toy 两层模型无法完成 forward、KV cache 和保存重载，主分支切换到 `square_orthogonal`。

### 6.4 Linear 权重变换规则

实现位置：`transforms/linear.py`。对 `y = x W^T + b`，输入坐标变换为 `x' = x A`、输出坐标变换为 `y' = y B` 时，统一由一个函数推导权重，不在各模块手写矩阵乘法：

```python
def transform_linear(weight, bias, input_map, output_map):
    # 先用 FP64/FP32 参考实现确认约定，再优化
    new_weight = output_map.T @ weight @ torch.linalg.inv(input_map).T
    new_bias = None if bias is None else bias @ output_map
    return new_weight, new_bias
```

实际代码必须用 `solve`，禁止显式 `inv`。上述代码仅用于定义坐标约定。单测随机输入比较原路径再变换与变换后路径。

### 6.5 RMSNorm 和 Residual

实现位置：`transforms/norm.py`、`models/qwen2.py`。

任意一般可逆矩阵都不会自动与 RMSNorm 对易。工程顺序：

1. 优先使用置换、符号翻转、分块正交等保持范数的变换
2. 在每个 residual stream 建立明确坐标域 ID
3. residual 两支相加前断言域 ID 相同
4. 一般矩阵变换只能放在可被相邻 Linear 吸收的位置

实现 `CoordinateTensor(tensor, domain_id)` 仅用于 debug 模式；生产模式关掉包装器。

### 6.6 RoPE、GQA 和 Attention

实现位置：`transforms/rope.py`、`transforms/attention.py`。

从 `transformers.models.qwen2.modeling_qwen2` 复制最小必要调用结构到自定义模型，不复制整个库。处理顺序固定：

```text
hidden -> q_proj/k_proj/v_proj -> reshape heads -> RoPE(q,k)
-> repeat_kv(GQA) -> attention -> merge heads -> o_proj -> residual
```

规则：

- Q head 置换长度为 `num_attention_heads`
- K/V head 置换长度为 `num_key_value_heads`
- 每组 Q head 必须跟随所属 KV head 一起移动
- RoPE 的偶/奇维成对移动，不允许单维置换破坏旋转对
- prefill 与 decode 使用同一置换规则
- KV cache 保存的是变换后 K/V；读取时不得再次变换

测试输入长度：1、2、127、128、129、1024；测试 batch：1、2；必须覆盖 cache on/off。

### 6.7 FFN

实现位置：`transforms/ffn.py`。Qwen SwiGLU：

```text
down_proj(silu(gate_proj(x)) * up_proj(x))
```

`gate_proj` 与 `up_proj` 的中间维置换必须相同；`down_proj` 输入使用同一逆置换。若对中间维加尺度，gate/up/down 三处同步吸收。单测同时比较 gate、up、乘积、down 输出。

### 6.8 Toy MoE

实现位置：`models/toy_moe.py`。

配置：`hidden=64`、`intermediate=128`、`experts=8`、`top_k=2`。实现专家置换 `pi_e`：router logits 的列按同一置换重排，expert ModuleList 同步重排；每个专家内部复用 FFN 变换。测试 router top-k ID 逆映射后完全一致。

### 6.9 Toy MLA 和 DeepSeek 适配

实现位置：`models/toy_mla.py`、`models/deepseek.py`。先读取目标模型 `config.json` 和官方 modeling 文件，生成 `docs/tensor_ledger_deepseek.md`。逐项记录 q/k 低秩压缩维、解耦 RoPE 维、value 维、cache 结构。Toy MLA 全部通过后才写 DeepSeek adapter。

禁止把 Qwen Attention 转换器直接套到 MLA。适配器通过注册表选择：

```python
ADAPTERS = {
    "qwen2": Qwen2Adapter,
    "deepseek_v3": DeepSeekV3Adapter,
}
```

## 7. Checkpoint 转换器

### 7.1 下载原模型

`scripts/download_model.py` 使用 `huggingface_hub.snapshot_download`，必须传 `revision`、`local_dir`，仅允许 `*.json`、`*.safetensors`、tokenizer 文件。

```bash
uv run python scripts/download_model.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --revision <HF_COMMIT_SHA> \
  --out /data/models/qwen2.5-0.5b
```

### 7.2 转换流程

实现位置：`conversion/convert.py`。

1. `reader.py` 读取 `model.safetensors.index.json`
2. `planner.py` 建立参数名到变换规则的静态计划
3. 一次只加载一个源 shard
4. 张量转 FP32/FP64 执行变换
5. 写入 `<target>.partial`
6. `fsync` 后计算 SHA-256
7. 原子重命名为 `.safetensors`
8. 将完成状态写入 `conversion_state.json`
9. 所有 shard 完成后写 index、config、manifest

执行：

```bash
uv run python scripts/generate_key.py \
  --model-config configs/models/qwen05b.yaml \
  --transform-config configs/transform/paper.yaml \
  --out /data/keys/dev-qwen05b-001

uv run python scripts/convert_model.py \
  --model-config configs/models/qwen05b.yaml \
  --transform-config configs/transform/paper.yaml \
  --key-dir /data/keys/dev-qwen05b-001 \
  --state /data/checkpoints/qwen05b-conversion-state.json \
  --resume
```

重跑时校验已完成 shard 的 size 和 SHA-256；不一致则隔离该 shard 并重做，不删除其他 shard。

## 8. HF 自定义模型

实现 `src/aloepri/models/qwen2.py`：

- `AloePriQwen2Config(Qwen2Config)` 增加 `aloepri_manifest`
- `AloePriQwen2ForCausalLM(Qwen2ForCausalLM)` 只替换需要改变的模块
- `from_pretrained` 加载时校验 `model_id`、`key_id`、shape、hash
- forward 返回标准 `CausalLMOutputWithPast`
- generate 不修改 Transformers 公共接口

注册：

```python
AutoConfig.register("aloepri_qwen2", AloePriQwen2Config)
AutoModelForCausalLM.register(AloePriQwen2Config, AloePriQwen2ForCausalLM)
```

首先执行 eager attention；与基线一致后再开启 SDPA，最后才测试 FlashAttention。

## 9. API、客户端和流式协议

### 9.1 请求协议

`serving/protocol.py`：

```python
class GenerateRequest(BaseModel):
    model_id: str
    key_id: str
    input_ids: list[int]
    max_new_tokens: int = Field(ge=1, le=2048)
    temperature: float = Field(ge=0.0, le=2.0)
    top_p: float = Field(gt=0.0, le=1.0)
    seed: int | None = None
```

`POST /v1/private/generate` 返回 JSON；`POST /v1/private/generate/stream` 返回 SSE，每个事件只含 `request_id`、`sequence_no`、`output_id`、时间戳。服务端拒绝不匹配的 `model_id/key_id`。

### 9.2 客户端

`client/sdk.py`：原文在客户端 tokenizer 编码，然后 `tau[input_ids]`，服务端返回的每个 token 用 `inverse_tau` 恢复，再交给原 tokenizer decode。服务端永远不调用原 tokenizer。

```bash
uv run python scripts/serve_hf.py --config configs/serving/hf.yaml
uv run python scripts/client_smoke.py \
  --url http://127.0.0.1:8000 \
  --key-dir /data/keys/dev-qwen05b-001 \
  --prompt '用一句话解释矩阵乘法'
```

日志过滤器必须删除 `prompt`、`input_ids`、`output_ids`、Authorization 和密钥路径。只记录 token 数、耗时、状态码、model_id、key_id、request_id。

## 10. vLLM 接入

实现位置：`serving/vllm_model.py`。目标不是修改 vLLM site-packages，而是实现其自定义模型接口并在进程启动前注册。先使用 HF 权重加载器验证所有参数名，再适配 tensor-parallel 分片。

```python
from vllm import ModelRegistry
ModelRegistry.register_model(
    "AloePriQwen2ForCausalLM",
    "aloepri.serving.vllm_model:AloePriQwen2ForCausalLM",
)
```

启动：

```bash
export PYTHONPATH=/opt/aloepri/src
.venv-vllm/bin/python -m aloepri.serving.vllm_server \
  --model /data/checkpoints/qwen2.5-0.5b-obfuscated \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --port 8001
```

若 vLLM 与 HF 首 token 不一致，固定 greedy、关闭 prefix cache、speculative decode 和 chunked prefill，先比较单 token eager forward。

## 11. 测试命令和门禁

### 11.1 每次提交

```bash
uv run ruff check src tests scripts
uv run mypy src/aloepri
uv run pytest -q tests/unit tests/equivalence --maxfail=1
```

### 11.2 无噪声等价

`tests/equivalence/test_qwen05b.py` 保存以下逐层数据：embedding、每层 norm、Q/K/V、attention output、FFN output、final norm、logits。比较 FP32、BF16；greedy 生成至少 100 个 prompt，每个 64 token。

```bash
uv run pytest -q tests/equivalence/test_qwen05b.py --basetemp=/data/evidence/pytest
```

通过条件：

- FP32 每层误差符合配置阈值
- BF16 误差不随层单调爆炸
- greedy token 序列完全一致；若设计允许数值误差导致分叉，必须单列并由甲方批准，不能并入通过项
- cache on/off 输出一致
- 保存重载后结果一致

### 11.3 0.5B 冻结

```bash
uv run pytest -q tests --junitxml=/data/evidence/qwen05b/junit.xml
uv run python scripts/run_eval.py --config configs/eval/smoke.yaml --out /data/evidence/qwen05b
git tag -s qwen05b-full-function-v1 -m '0.5B full function freeze'
```

只有全部测试通过和 tag 已签名，才能修改 `configs/models/qwen7b.yaml` 开始 7B。

## 12. 精度、攻击和性能工具

### 12.1 精度

`eval/accuracy.py` 只做适配和结果归一化，实际任务调用：

- MMLU/PIQA/IFEval：`lm-evaluation-harness`
- C-Eval：固定数据版本和本地 evaluator
- HumanEval：`evalplus`

每次运行保存 config、数据 commit、原始预测、原始分、bootstrap 95% CI、环境清单。明文与混淆模型使用同一 prompt、seed、batch 和解码参数。

### 12.2 攻击

每个文件实现统一接口：

```python
class Attack(Protocol):
    def run(self, view: ServerView, budget: AttackBudget) -> AttackResult: ...
```

`ServerView` 明确给攻击者权重、架构、混淆 tokenizer、查询次数和已知明文对。不得让攻击代码读取 `/data/keys`。攻击进程使用独立 Unix 用户和只读模型目录执行。

```bash
sudo -u aloepri-attacker uv run python scripts/run_eval.py \
  --config configs/eval/privacy.yaml \
  --out /data/evidence/privacy/qwen05b
```

### 12.3 性能

先预热 20 次，正式至少 1000 请求；保存每请求 TTFT、TPOT、总时延、输入/输出 token、状态。明文和混淆模型轮换运行，避免温度和后台负载偏差。

```bash
uv run python scripts/run_eval.py \
  --config configs/eval/performance.yaml \
  --out /data/evidence/performance/qwen05b
```

报告 p50/p95/p99、吞吐、峰值 GPU 显存、加载时间和转换时间。

## 13. 7B/14B 和多卡

7B/14B 不新增算法。复制 0.5B 配置，只改模型 ID、revision、目录和资源参数。先转换单 shard smoke，再全量转换。

单节点：

```bash
nvidia-smi topo -m
export NCCL_DEBUG=INFO
export NCCL_P2P_LEVEL=NVL
.venv-vllm/bin/python -m aloepri.serving.vllm_server \
  --model /data/checkpoints/qwen2.5-14b-obfuscated \
  --tensor-parallel-size 4 \
  --dtype bfloat16 \
  --max-model-len 8192
```

TP 数必须整除 Attention head 数和 KV head 分片要求；不满足时不得通过隐式 padding 继续，改 PP 或选择不同 GPU 数。

## 14. 多节点 671B

### 14.1 云实例筛选

只接受：节点内 NVLink/NVSwitch；节点间 InfiniBand/RDMA；共享对象存储；镜像可固定 digest；支持至少两个 8-GPU 节点。开机后先运行：

```bash
nvidia-smi
nvidia-smi topo -m
ibstat
ibv_devinfo
ip -br addr
df -h /data
```

### 14.2 NCCL 冒烟

每节点拉取同一版本 `nccl-tests`，执行 `all_reduce_perf`。把完整日志、拓扑、网卡名、NCCL 环境变量保存到 `/data/evidence/cluster/preflight/`。带宽不稳定或跨节点错误时禁止开始 TB 级转换。

### 14.3 启动变量

```bash
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=<RDMA_INTERFACE>
export NCCL_IB_HCA=<IB_DEVICE>
export GLOO_SOCKET_IFNAME=<CONTROL_INTERFACE>
export HF_HOME=/data/cache/huggingface
```

不要照抄网卡名；用 `ip -br addr`、`ibdev2netdev` 的实测值填写。TP 放在节点内，跨节点优先 PP。671B 只执行已在 Toy MLA/DeepSeek 小配置通过的 adapter、转换器和服务代码。

### 14.4 Go/No-Go

以下任一未满足即 No-Go：

- 0.5B、7B/14B 冻结 tag 存在且测试全绿
- Toy MoE、Toy MLA 和 DeepSeek 小配置通过
- 目标模型 config 和权重 revision 固定
- 权重、输出、临时文件、证据所需存储容量核算完成
- 两节点 NCCL 连续三次通过
- 转换断点续跑在故障注入下通过
- 预算和实例时段已批准

## 15. 故障处理

| 现象 | 首查位置 | 处理 |
|---|---|---|
| 首层即不一致 | `vocab.py`、Embedding 行索引 | 用全词表 one-hot 测试 `tau/inverse_tau` |
| RoPE 后不一致 | `rope.py` | 检查偶奇维配对和应用顺序 |
| decode 才不一致 | `attention.py` cache 分支 | 比较每步 cache shape、位置 ID、是否重复变换 |
| FFN 不一致 | `ffn.py` | 比较 gate/up 中间排列和 down 逆排列 |
| BF16 发散 | `pq.py` | 查条件数、动态范围；先 FP32 定位 |
| HF 对、vLLM 错 | `vllm_model.py` | 查参数映射、TP shard、cache layout、kernel |
| OOM | 转换日志或服务指标 | 降 batch/context；逐 shard；CPU/NVMe offload；不得改 4-bit 作为同一验收 |
| 多节点卡死 | NCCL 日志 | 查网卡、IB、端口、防火墙、拓扑和 TP/PP 配置 |

## 16. 证据目录

```text
/data/evidence/<run_id>/
├── config/
├── environment/{git.txt,uv.lock,nvidia-smi.txt,pip-freeze.txt}
├── conversion/{state.json,manifest.json,hashes.txt,logs/}
├── correctness/{layer_errors.json,generation.json,junit.xml}
├── accuracy/{raw,summary.json}
├── privacy/{raw,summary.json,curves/}
├── performance/{requests.parquet,summary.json,nvidia-smi.csv}
└── cluster/{topology.txt,nccl.log,launch.txt}
```

每个命令生成唯一 `run_id`；任何汇总值必须能追溯到原始记录。验收包只从该目录生成，不从终端截图手工抄数。

## 17. 实施顺序

1. Day 1-3：完成第 2-5 节，锁定依赖和 Qwen commit
2. Day 4-7：完成词表、客户端、HF 基线闭环
3. Day 8-14：完成噪声、P/Q、Toy Transformer；执行扩维决策门
4. Day 15-21：完成 Attention、RoPE、GQA、FFN、Norm、Residual、KV cache
5. Day 22-25：完成转换器、HF、vLLM 和 API
6. Day 26-30：完成攻击、精度、性能；冻结 0.5B
7. Day 31-35：7B/14B 转换、多卡和全量回归
8. Day 36-38：Toy MoE、Toy MLA、DeepSeek adapter、671B Go/No-Go
9. Day 39-43：671B 转换、加载、核心实验和恢复
10. Day 44-45：迁移演练、证据核查、交付

每一步只以对应测试、文件和日志完成为结束条件，不以“代码已写”作为完成。

## 18. 外部文档

- Qwen2 Transformers 实现：`https://huggingface.co/docs/transformers/model_doc/qwen2`
- Qwen2.5 模型：`https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct`
- Transformers：`https://github.com/huggingface/transformers`
- vLLM 自定义模型：`https://docs.vllm.ai/en/latest/contributing/model/registration/`
- vLLM 分布式推理：`https://docs.vllm.ai/en/latest/serving/distributed_serving/`
- safetensors：`https://github.com/huggingface/safetensors`
- uv：`https://docs.astral.sh/uv/`
- PyTorch：`https://pytorch.org/get-started/locally/`
- lm-evaluation-harness：`https://github.com/EleutherAI/lm-evaluation-harness`
- EvalPlus：`https://github.com/evalplus/evalplus`
- NCCL tests：`https://github.com/NVIDIA/nccl-tests`
- DeepSeek-V3：`https://github.com/deepseek-ai/DeepSeek-V3`
