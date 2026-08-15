# 隐变智模 1.1：模型族适配与一键部署

## 1. 用户流程

图形界面执行以下完整链路：

```text
选择模型族和具体模型
→ 固定 Hugging Face commit
→ 检查许可证与本地磁盘/内存
→ 检查 SSH Host Key、Ubuntu、驱动、远端磁盘和全部 GPU
→ 必要时自动安装 Python 基础环境
→ 下载并校验 checkpoint
→ 按结构指纹重新检查全部张量
→ 生成本地密钥并转换私有权重
→ 扫描服务器包中的秘密材料
→ 断点上传并在远端复算 SHA-256
→ 自动安装固定版本推理依赖
→ 使用全部可用 GPU 启动候选版本
→ 认证的私有生成接口完成一次真实 Token decode 后切换 active 版本
→ 在“已部署模型”中打开对话
```

用户只需要填写服务器，例如：

```text
ssh -p 51838 root@gpu.example.com
```

密码可由 Windows 当前用户凭据库加密保存。模型置换密钥和离线主密钥不会上传。

## 2. 命令行等价入口

```powershell
yinbian models list
yinbian servers add --name gpu --host gpu.example.com --port 51838 --auth-type password
yinbian servers check <server-id> --trust-host-key
yinbian plan create --model qwen2.5-0.5b-instruct --mode direct-deploy --device auto --server <server-id>
yinbian convert --plan <plan.yaml> --accept-license
yinbian deploy create --job <job-id> --server <server-id> --confirm
yinbian tunnel open <deployment-id>
yinbian chat stream --deployment <deployment-id> --prompt "你好"
```

`yinbian` 和两个桌面程序调用同一转换器、状态库、SSH 管理器和推理服务。

## 3. 已实现的架构族

| 架构族 | 已处理结构 | 转换路径 | 当前产品边界 |
|---|---|---|---|
| Qwen2/Qwen2.5 | GQA、RoPE、RMSNorm、Dense SwiGLU | Qwen 私有运行时 | 仅0.5B固定版本完成一键部署验收；其余可转换 |
| Qwen3 Dense | GQA、Q/K Norm、RoPE、Dense SwiGLU | Qwen3 专用同步变换 | 不把 Qwen3 MoE 当作 Dense |
| DeepSeek-V2/V2.5 | MLA、Dense/Shared/Routed Experts、Router | DeepSeek 流式转换器 | 无 MTP 的 V2 路径 |
| DeepSeek-V3 | MLA、MoE、FP8 128×128、MTP | DeepSeek-V3 完整路径 | 真实 671B 仍取决于本地磁盘和云端 GPU |
| GLM Dense | GQA、partial RoPE、fused Gate/Up | 旧 GLM 规范化后进入 Qwen 私有运行时 | GLM 0414/Z1 增加分支后置 RMSNorm，当前明确拒绝转换 |
| GLM4-MoE | GQA、Q/K Norm、partial RoPE、MoE、FP8、MTP | GLM 专用流式转换器 | HF 主模型问答不启用 MTP 加速 |
| Kimi-K2 | DeepSeek-like MLA、384 Experts、FP8 | 零拷贝规范化后进入 DeepSeek-V3 路径 | 纯文本 |
| Kimi-K2.6 | 61 层 MLA/MoE、384 Experts、group INT4 | 前缀虚拟映射、INT4 解码、DeepSeek-V3 路径 | 完整文本骨干；图片输入未开放 |

每个适配器只依赖 `config.json`、Safetensors 索引、张量名称、形状和 dtype。
下载和转换过程不设置 `trust_remote_code=true`。

## 4. Kimi-K2.6 INT4 规则

官方 `compressed-tensors` 格式中，每个有符号 INT4 值先加 8，再把连续 8 个值
按低位到高位装入一个 INT32。解码后按最后一维每 32 个值共享一个
`weight_scale`：

```text
q = unpack_int32(weight_packed) - 8
W[:, 32g:32(g+1)] = q[:, 32g:32(g+1)] * weight_scale[:, g]
```

转换器要求 `weight_packed`、`weight_scale` 和 `weight_shape` 三者同时存在。视觉塔和
投影器不会被误送入 DeepSeek 文本转换器。

## 5. 换服务器时的处理

部署器不再假设所有机器都有 systemd、Docker 或完全相同的 Python 环境。每台新机器
都会重新执行：

1. Host Key 固定；
2. Ubuntu 22.04、x86-64、root/非交互 sudo 检查；
3. NVIDIA 驱动 525+ 检查；
4. 所有 GPU 数量、总显存和空闲显存汇总；
5. 远端磁盘与模型包大小比较；
6. 缺少 Python 时自动安装；
7. 固定依赖安装与 CUDA 可用性验证；
8. 候选端口启动、日志采集、真实私有 Token 生成检查和失败回滚。

推理依赖固定为 PyTorch 2.5.1 CUDA 12.1 路径，并使用兼容 Ubuntu 22.04 Python 3.10
的 NumPy 2.2.6。远端安装完成后写版本 marker，重复部署不会重复安装。

## 6. 验证结果

当前代码、真实Qwen2.5-0.5B GPU往返和54个固定版本目录审计结果见
[`MODEL_SUPPORT_AUDIT_2026-08-15.md`](MODEL_SUPPORT_AUDIT_2026-08-15.md)。

Kimi-K2.6 使用官方配置和 208,550 个官方索引张量做静态覆盖检查，文本骨干预期
69,915 个规范化张量，检查结果为 missing=0、unknown=0。该结果证明转换计划覆盖，
不代表 595 GB checkpoint 已在本机物理转换，也不代表视觉输入链路已经完成。
