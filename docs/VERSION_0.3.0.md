# AloePri 0.3.0

## 目标

`0.3.0`增加训练完成的`DeepSeek-V2-Lite-Chat`云端架构验证路径。Qwen2.5-0.5B v47正式验收继续独立运行，二者不共用结果。

## 转换覆盖

| 模块 | 代码 | 本地证据 |
|---|---|---|
| 词表、Embedding、LM Head | `src/aloepri/conversion/deepseek_streaming.py` | 私有token输入与logits逆置换等价 |
| q/kv低秩坐标与RMSNorm | `src/aloepri/transforms/deepseek.py` | q_lora存在/不存在两条路径前向通过 |
| MLA Q/K/RoPE/V/O | `src/aloepri/transforms/deepseek.py` | Prefill与Cache Decode通过 |
| Dense FFN | `transform_deepseek_moe()` | Layer 0权重实际改变且前向保持 |
| Shared Experts | `transform_deepseek_moe()` | Shared权重实际改变且前向保持 |
| Routed Experts | `transform_deepseek_moe()` | Router、专家顺序和每专家FFN同步 |
| Group-limited routing | `make_deepseek_key()` | 2组极小V3前向保持 |
| 断点转换 | `convert_deepseek_checkpoint()` | 模拟第6个张量中断后续跑通过 |
| 重分片 | `repack_checkpoint()` | 多个单张量文件重组并逐张量比对 |

## 固定云端入口

Windows生成上传包：

```powershell
uv run python scripts/build_deepseek_cloud_bundle.py
```

Linux一键执行：

```bash
cd /data/AloePri
CUDA_VISIBLE_DEVICES=0,1,2 bash scripts/cloud/run_deepseek_v2_lite_all.sh
```

一键入口依次运行：

```text
硬件和磁盘预检
→ 固定revision下载与逐文件SHA-256收据
→ 源张量形状审计
→ 逐张量MLA/MoE转换
→ 在线/离线密钥拆分
→ 2GB标准重分片
→ manifest与服务端密钥泄漏检查
→ BF16 Prefill/Cache/生成对比
→ 中文产品问答
→ MMLU/C-Eval/PIQA/IFEval/HumanEval
→ 最终架构验收
```

## 输出

| 输出 | 路径 |
|---|---|
| 上传包 | `release/AloePri-deepseek-v2-lite-cloud-source.tar.gz` |
| 云端私有模型 | `data/packages/deepseek-v2-lite-chat-mla-moe` |
| 在线密钥 | `data/keys/deepseek-v2-lite-chat-mla-moe-online` |
| 离线主密钥 | `data/keys/deepseek-v2-lite-chat-mla-moe-offline` |
| 架构验收 | `artifacts/deepseek-v2-lite-chat/final-acceptance.json` |

执行步骤和故障恢复见`docs/DEEPSEEK_V2_LITE_CLOUD_RUNBOOK.md`。
