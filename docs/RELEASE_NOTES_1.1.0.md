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
- 修正 Ubuntu 22.04 Python 3.10 环境的 NumPy 版本兼容问题。
- 默认自动化结果：324 passed，1 skipped；真实 Qwen2.5-0.5B 私有 API 往返另行 1 passed；Ruff 和 Mypy 通过。

限制：Kimi-K2.6 当前只开放文本问答；真实超大模型转换与多 GPU 物理加载仍需在具备
相应磁盘、内存和显存的机器上执行，代码不会把静态覆盖结果写成物理验收通过。
