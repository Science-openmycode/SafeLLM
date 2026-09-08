# 隐变智模 1.1.0 版本说明

- 新增 Qwen3 Dense 专用 Q/K Norm 变换和多尺寸目录。
- 新增 GLM Dense 融合 SwiGLU 规范化。
- 新增 GLM4-MoE 的 FP8、专家路由、部分 RoPE 和 MTP 权重转换。
- 新增 Kimi-K2 零拷贝规范化及 Kimi-K2.6 官方 group INT4 文本骨干转换。
- 模型目录扩展到 54 个固定版本：Qwen2/2.5 7 个、Qwen3 Dense 6 个、DeepSeek 17 个、GLM 16 个、Kimi 8 个。
- 部署界面使用“模型族 → 参数量与版本”两级选择，模型目录按族折叠；后台进度刷新不再重建模型控件。
- Qwen2.5、Qwen3、DeepSeek、GLM、Kimi 在界面中按族分组。
- 直接部署会自动预检并补齐服务器基础环境。
- 部署前按私有包大小检查远端磁盘和全部 GPU 总空闲显存。
- 原生 HF 服务自动使用多 GPU，禁止静默 CPU/磁盘卸载。
- 部署成功门禁从进程健康检查升级为真实私有 Token 单步生成；不能生成的候选版本不会切换为 active。
- 区分“架构可转换”和“检查点已部署验收”：仅 Qwen2.5-0.5B 固定版本开放一键部署，其他模型不得绕过目录门禁调用 HF 部署器。
- 修复 Qwen3 tied embedding、逐层 RoPE buffer、GLM4-MoE 全局共享 MTP embedding/head、OpenSeek 缺失 MTP 声明、Kimi packed INT4 等权重清单问题。
- GLM 0414/Z1 因额外的 Attention/MLP 分支后置 RMSNorm 与当前轻量变换不等价，改为明确拒绝转换，防止生成错误模型。
- 新增 `YINBIAN_CACHE_DIR`，允许把模型下载缓存独立放到非系统盘。
- 新增 `YINBIAN_DATA_DIR`，统一保存原始模型、改造模型、缓存和验收证据，源码仓库不再作为新数据的默认目录。
- TEE、国密、证明、包校验、可信边界和掩码 Head 模块采用职责化名称；旧导入路径继续兼容。
- Windows 安装器、Python 包和文档版本统一为 `1.1.0`。
- 修正 Ubuntu 22.04 Python 3.10 环境的 NumPy 版本兼容问题。
- 当前测试与目录审计结果见 `docs/MODEL_SUPPORT_AUDIT_2026-08-15.md`。

限制：Kimi-K2.6 当前只开放文本问答；真实超大模型转换与多 GPU 物理加载仍需在具备
相应磁盘、内存和显存的机器上执行，代码不会把静态覆盖结果写成物理验收通过。
