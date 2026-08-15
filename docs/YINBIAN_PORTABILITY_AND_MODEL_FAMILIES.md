# 隐变智模：服务器可移植性与模型族支持

## 换服务器时的执行流程

已经完成改造的模型不需要重新下载或重新改造：

```text
选择已有健康部署
→ 选择“复制到另一台服务器”
→ 检查目标服务器
→ 从本地私有模型包断点上传
→ 逐文件SHA-256校验
→ 安装固定版本原生Python运行环境
→ 启动候选服务
→ 健康检查
→ 登记为新的独立部署
```

迁移前必须保留本地私有模型包、在线密钥DPAPI记录和原始Tokenizer目录。迁移不会生成新置换，也不会改变`model_id`或`key_id`。

## 新服务器门禁

| 项目 | Qwen2.5-0.5B最低门禁 |
|---|---:|
| 操作系统 | Ubuntu 22.04 x86-64 |
| NVIDIA驱动 | 525或更高 |
| GPU总显存 | 6 GiB |
| GPU空闲显存 | 4 GiB |
| 主机内存 | 8 GiB |
| 根文件系统空闲 | 8 GiB |
| 权限 | root或非交互sudo |
| 网络 | 可访问PyPI和PyTorch CUDA 12.1轮子源 |

运行依赖采用精确版本，不再使用浮动范围。SSH普通命令5分钟超时，软件安装30分钟超时，推理依赖安装60分钟超时。

## 模型族支持矩阵

| 架构族 | 代表模型 | 识别 | 转换 | 产品部署 | 关键差异 |
|---|---|---:|---:|---:|---|
| Qwen2/Qwen2.5 | Qwen2.5-0.5B-Instruct | 是 | 是 | 0.5B已验收 | Dense GQA、RMSNorm、RoPE |
| DeepSeek-V2 | DeepSeek-V2-Lite-Chat | 是 | 是 | 待验收 | MLA、MoE、低秩Q/KV |
| DeepSeek-V3 | OpenSeek、DeepSeek-V3 | 是 | 是 | OpenSeek仅转换验收 | MLA、MoE、FP8、MTP |
| GLM Dense | GLM-4-9B-Chat-HF | 是 | 否 | 否 | 融合`gate_up_proj`和GLM运行时 |
| GLM MoE | GLM-4.7-FP8 | 是 | 否 | 否 | Q/K Norm、partial RoPE、MoE、MTP、独立FP8格式 |
| Qwen3 Dense | Qwen3-8B | 是 | 否 | 否 | Attention内新增Q/K Norm |
| Kimi-K2 | Kimi-K2.6 | 是 | 否 | 否 | 多模态外壳、DeepSeek式文本骨干、打包INT4专家 |

“识别”表示配置和张量边界能够被正确归类，不表示权重可以安全改造。“转换”必须有该族的数学变换、张量编解码、私有运行时和测试；目录不会把尚未具备这些条件的模型显示成可执行。

## 当前代码入口

```text
src/aloepri/catalog/registry.py       模型目录和族分组
src/aloepri/catalog/fingerprint.py    配置结构指纹
src/aloepri/adapters/families.py      各族张量覆盖规则
src/aloepri/conversion/executor.py    按适配器分发转换器
src/aloepri/product/pipeline.py       下载、检查、转换、上传流水线
src/aloepri/cloud/ssh.py              服务器预检和运行环境安装
src/aloepri/cloud/hf_deployment.py    上传、启动、健康和回滚
src/aloepri/desktop/deploy_api.py     桌面API和跨服务器迁移
```
