# 671B 集群预检

## 必需资源

- Slurm 中同一作业独占至少 2 个节点，每节点 8 张 H100 80GB、H200 141GB 或等效 GPU。
- 节点内 NVLink/NVSwitch；节点间 InfiniBand/RDMA；所有节点可访问同一仓库和对象存储。
- 每节点至少 512 GB RAM、2 TB 本地 NVMe；共享/对象存储可用空间至少 3 TB。
- CUDA、NCCL、PyTorch 版本在所有节点一致；仓库虚拟环境已经安装。

## 执行

```bash
cd /shared/AloePri
source .venv/bin/activate
export ALOEPRI_ROOT=/shared/AloePri
export NCCL_SOCKET_IFNAME=ib0
mkdir -p artifacts/cluster
sbatch deploy/cluster/slurm_671b_preflight.sbatch
```

作业必须生成：

- `artifacts/cluster/slurm-<job_id>.out`
- `artifacts/cluster/cluster-preflight.json`

只有 `cluster-preflight.json` 中 `pass=true`，且 16 个 rank 全部出现，才能开始 671B
权重下载、转换和推理。失败时保留 Slurm 输出，不得降低节点数绕过门禁。

当前仓库只完成了 DeepSeek-V3 单层 MLA/MoE 等价适配器和缩小模型验证，尚未完成
671B TB 级 checkpoint 转换器。因此预检通过不等于允许直接修改 671B 权重；必须先补齐
逐 shard 的 DeepSeek-V3 转换、键文件分片、断点恢复和全索引校验。
