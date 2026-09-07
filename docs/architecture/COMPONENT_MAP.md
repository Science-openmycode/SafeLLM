# 隐变智模代码部件地图

本文档是阅读项目的入口。代码按职责分层；模型、密钥、转换结果和运行证据不属于源码。

## 一条请求经过哪些部件

```text
桌面部署/对话界面
  → product：任务、状态、路径与流程编排
  → catalog/adapters：识别模型族并选择架构适配器
  → conversion/transforms：生成私有模型与边界工件
  → cloud/serving：部署并运行模型服务
  → client/tee/secure_head：发送私有请求并恢复结果
```

## 源码目录

| 目录 | 单一职责 | 主要入口 |
|---|---|---|
| `src/aloepri/desktop` | Windows 部署端和对话端 API、页面壳 | `deploy.py`、`chat.py` |
| `src/aloepri/product` | 产品任务、状态库、路径、部署编排 | `pipeline.py`、`state.py`、`paths.py` |
| `src/aloepri/catalog` | 模型清单、版本固定、下载和结构检查 | `registry.py`、`inspect.py` |
| `src/aloepri/adapters` | 不同模型架构的权重映射规则 | `registry.py` 与各族适配器 |
| `src/aloepri/conversion` | 转换执行、分片、断点与工件生成 | `executor.py`、`paper_qwen2.py` |
| `src/aloepri/transforms` | P/Q、Attention、FFN、RMSNorm 等张量变换 | 各架构变换模块 |
| `src/aloepri/keys` | 在线/离线密钥、凭据保险库与便携备份 | `vault.py`、`portable.py` |
| `src/aloepri/cloud` | SSH 连接、远端预检、上传与部署 | `ssh.py`、`hf_deployment.py` |
| `src/aloepri/serving` | Hugging Face、vLLM、SGLang 与 TEE 推理入口 | `hf_runtime.py`、`tee_runtime.py` |
| `src/aloepri/client` | Token 置换模式的请求和流式恢复 | `private_client.py` |
| `src/aloepri/tee` | 国密、证明、可信边界和 TEE 包校验 | 见下表 |
| `src/aloepri/secure_head` | LM Head 掩码外包与一次性掩码池 | `masked_outsource_engine.py` |
| `src/aloepri/eval`、`attacks` | 正确性、性能和攻击评估 | 独立评估入口 |

## TEE 模块命名

| 清晰模块名 | 内容 |
|---|---|
| `tee/gm_cryptography.py` | Tongsuo 能力检查与 SM2/SM3/SM4 原生助手接口 |
| `tee/attestation.py` | Quote 度量、TDX 预检和 DCAP 适配 |
| `tee/attestation_models.py` | 证明与配置请求/响应数据结构 |
| `tee/attestation_service.py` | 证明状态机和部署身份 |
| `tee/package_integrity.py` | TEE/服务器包哈希、签名和秘密扫描 |
| `tee/trusted_boundary.py` | Embedding、隐藏状态恢复、本地 Head 和采样 |
| `tee/software_crypto.py` | 明确标记为开发用途的软件加密传输 |
| `tee/execution_trace.py` | 原理演示使用的运行证据，不参与模型数学 |
| `secure_head/masked_outsource_engine.py` | 掩码 Head 外包、校验、候选认证和回退 |
| `secure_head/mask_pool.py` | 一次性掩码生命周期 |

旧的 `tee/gm.py`、`tee/package.py`、`tee/protocol.py`、`tee/service.py`、
`tee/boundary.py` 和 `secure_head/outsourced.py` 只保留兼容导入。新代码不得继续向这些
文件增加实现。

## 兼容性原则

- 旧配置缺少安全字段时仍进入 `permutation + in_model`。
- 旧模块路径继续导出同一个类或函数对象。
- 已有数据库、模型记录、密钥包和部署记录不原地迁移。
- 新命名只改变代码组织，不改变模型计算、API 字段和生成结果。
- 任何结构调整都必须通过旧模式回归、TEE 回归和旧导入路径测试。
