# RTX 3060转换手册

## 固定资源参数

```yaml
device: cuda:0
gpu_memory_budget_gib: 4.5
minimum_free_gpu_gib: 1.2
host_memory_budget_gib: 11
tile_mib: 256
minimum_tile_mib: 32
temporary_disk_gib: 50
oom_policy: halve_tile_and_retry
```

`ResourceBudgetMonitor`在执行前检查空闲显存，并持续采样RSS；转换结果记录CPU和GPU峰值。CUDA OOM按`256→128→64→32MiB`降低tile，32MiB仍OOM时当前tile回退CPU。

## 生成plan

```powershell
uv run aloepri models inspect --model D:\models\model
uv run aloepri doctor --model D:\models\model
uv run aloepri plan `
  --model D:\models\model `
  --output artifacts\plans\model.yaml `
  --destination D:\private-models\model-private
```

打开YAML确认：source路径、adapter、fingerprint、output、资源预算、seed和密钥策略。

## 执行

```powershell
$env:ALOEPRI_OFFLINE_KEY_PASSWORD = "<从密码管理器读取>"
uv run aloepri convert --plan artifacts\plans\model.yaml
uv run aloepri jobs status <job-id>
```

`--schedule-only`只登记任务，不执行转换：

```powershell
uv run aloepri convert --plan artifacts\plans\model.yaml --schedule-only
```

## I/O规则

- 源Safetensors只读取8字节header长度、JSON header和目标张量range；
- FP8源路径按256MiB读取，不调用源权重`get_tensor()`；
- 每个输出先写`.partial`并计算SHA-256；
- 完成后合并成不超过2GiB的`model-xxxxx-of-yyyyy.safetensors`；
- 单张量超过2GiB时单独成片；
- SQLite记录tensor/tile完成状态；
- Resume先校验现有分片SHA-256，参数变化不能复用旧断点。

## 空间说明

50GiB是额外临时空间，不包含源模型、目标模型和上传前缓存。完整DeepSeek-V3需要约689GB源空间和相近目标空间；若要做到“本地只保留50GiB”，必须配置S3目标并在每个已验证分片上传后删除本地已上传分片，真实S3删除策略仍属于后续云验收。
