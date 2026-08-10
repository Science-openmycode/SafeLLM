#!/usr/bin/env bash
#SBATCH --job-name=aloepri-671b-preflight
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=00:30:00
#SBATCH --exclusive
#SBATCH --output=artifacts/cluster/slurm-%j.out

set -euo pipefail

: "${ALOEPRI_ROOT:?set ALOEPRI_ROOT to the repository directory on shared storage}"
cd "$ALOEPRI_ROOT"
mkdir -p artifacts/cluster

export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^lo,docker0}"
export OMP_NUM_THREADS=8

nodes=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
head_node="${nodes[0]}"
head_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address | awk '{print $1}')
export head_ip

srun --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" --ntasks-per-node=1 \
  bash -lc 'torchrun \
    --nnodes="$SLURM_NNODES" \
    --nproc-per-node=8 \
    --node-rank="$SLURM_NODEID" \
    --master-addr="$head_ip" \
    --master-port=29500 \
    scripts/cluster_671b_probe.py \
    --out-dir artifacts/cluster \
    --min-world-size 16 \
    --min-gpu-gib 75 \
    --all-reduce-mib 512'
