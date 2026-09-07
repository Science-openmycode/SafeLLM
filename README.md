# 隐变智模 1.1

## 多架构与一键部署

当前统一检查/转换入口覆盖以下架构族：

- Qwen2/Qwen2.5：0.5B、1.5B、3B、7B、14B、32B、72B；
- Qwen3 Dense：0.6B、1.7B、4B、8B、14B、32B，包含 Q/K Norm；
- DeepSeek-V2/V2.5 与 DeepSeek-V3：MLA、MoE、FP8、MTP；
- GLM Dense 与 GLM4-MoE：融合 SwiGLU、Q/K Norm、部分 RoPE、专家路由、FP8、MTP；GLM 0414/Z1 的分支后置 RMSNorm 已识别但暂不转换；
- Kimi-K2 与 Kimi-K2.6 文本骨干：MLA、384 专家、FP8 或官方 group-wise INT4。

桌面端按模型族分组显示。架构兼容不等于检查点已经完成部署验收：当前只有
`Qwen2.5-0.5B-Instruct@7ae5576` 开放“改造、上传并自动部署”，其他固定版本只开放
其已验证到的检查或转换阶段。选择直接部署后，系统会先检查 SSH
主机、Host Key、Ubuntu、驱动、磁盘、内存和全部 GPU，再按需安装基础运行环境；
通过后才下载、转换和上传。部署服务只监听远端 `127.0.0.1`，对话程序通过 SSH
隧道访问。候选服务只有在认证的私有生成接口实际完成一次 Token decode 后才会标记
健康，不再仅凭 HTTP 进程存活判定成功；`/readyz` 也提供同等的进程内生成探针。

Kimi-K2 Thinking 的官方固定版本使用 packed INT4 专家权重，转换器会使用对应 scale
和 shape 解码；Kimi-K2.6 当前完成的是完整文本骨干私有化，不开放图片输入。未知结构、缺失权重、
缺失 FP8 scale、缺失 INT4 scale/shape 或资源不足都会在执行阶段拒绝，不能以“同族”
名义跳过检查。详细边界见
[模型族与一键部署说明](docs/YINBIAN_1.1_MODEL_FAMILIES_AND_ONE_CLICK.md)。

隐变智模在 Windows 本地生成密钥和改造模型权重，再通过 SSH 把私有模型部署到 Ubuntu GPU 服务器。问答时，本地程序完成 Chat Template、分词、Token 置换和回答恢复；服务器只接收私有 Token ID。

## 1.1 入口

```powershell
yinbian models recommend
yinbian models download --model qwen2.5-0.5b-instruct --metadata-only
yinbian models download --model qwen2.5-0.5b-instruct --accept-license
yinbian servers add --name gpu --host 203.0.113.10 --auth-type private_key --private-key C:\Keys\id_ed25519
yinbian servers check <server-id> --trust-host-key
yinbian plan create --model qwen2.5-0.5b-instruct --mode direct-deploy --device auto --server <server-id>
yinbian convert --plan artifacts\plans\qwen05b.yaml --accept-license
yinbian tunnel open <deployment-id>
yinbian chat stream --deployment <deployment-id> --prompt "介绍一下隐变智模"
```

桌面入口为 `隐变智模部署.exe` 和 `隐变智模对话.exe`。两个程序与 `yinbian.exe` 使用同一状态库。安装、部署和故障处理见 [1.0 安装与运行手册](docs/YINBIAN_1.0_INSTALLATION.md)。

`aloepri` 命令和内部 `src/aloepri` 包名保留一个兼容周期。旧模型、旧密钥和科研证据无需改名。

模型缓存可单独放到非系统盘：

```powershell
[Environment]::SetEnvironmentVariable("YINBIAN_CACHE_DIR", "E:\YinbianCache", "User")
```

该设置不移动状态库、密钥或聊天记录，只改变以后下载的可再生成缓存位置。

## 1.1 发布状态

当前仓库包含 1.1 代码与未签名 Windows 开发安装包。自动化测试、固定版本目录审计和真实 Qwen2.5-0.5B GPU 私有 API 往返结果见[2026-08-15模型与部署审计](docs/MODEL_SUPPORT_AUDIT_2026-08-15.md)。正式外发仍需要完成干净 Windows 10/11 安装、固定 digest 运行镜像、代码签名和发布证据归档。未签名构建只能内部测试。

---

## 0.5.0 兼容记录

AloePri把“模型识别、权重私有化改造、密钥拆分、上传部署和本地问答”统一为一个可恢复任务流。0.5.0支持Qwen2/Qwen2.5、DeepSeek-V2/V2.5和DeepSeek-V3架构族；云端部分使用接口级Mock，不代表真实云集群或671B物理执行。

## 当前状态

| 项目 | 状态 |
|---|---|
| 版本 | `0.5.0` |
| 发布状态 | `CODE_COMPLETE_MOCK_CLOUD_PASS` |
| 自动化测试 | `257 collected`：`256 passed, 1 skipped`；另有当前Qwen/OpenSeek真实checkpoint前向证据 |
| Ruff / Mypy | PASS |
| 官方DeepSeek-V3索引 | 91,991张量；missing=0；unknown=0 |
| 真实Qwen2.5-0.5B | 转换完成；私有next-token经`inverse_tau`恢复为108386，与明文一致 |
| OpenSeek-Small-v1-SFT | MLA/MoE转换完成；私有next-token 123415恢复为9707，与明文一致 |
| 真实云平台 | `NOT_TESTED` |
| 真实DeepSeek-V3 671B转换 | `NOT_EXECUTED` |

## 安装与检查

```powershell
uv sync --frozen
uv run aloepri models list
uv run aloepri models recommend
uv run aloepri models inspect --model E:\models\Qwen2.5-0.5B-Instruct
uv run aloepri doctor --model E:\models\Qwen2.5-0.5B-Instruct
```

模型检查只读取`config.json`、Safetensors索引和头部，不执行模型仓库的远程代码。

## 本地计划与Mock Cloud闭环

```powershell
$env:ALOEPRI_OFFLINE_KEY_PASSWORD = "使用密码管理器生成的强密码"

uv run aloepri plan `
  --model E:\models\Qwen2.5-0.5B-Instruct `
  --output artifacts\plans\qwen05b.yaml `
  --destination artifacts\converted\qwen05b-private

uv run aloepri convert --plan artifacts\plans\qwen05b.yaml
uv run aloepri upload <job-id> --cloud-profile mock
uv run aloepri deploy <job-id> --cloud-profile mock
uv run aloepri studio --port 7861
```

浏览器打开`http://127.0.0.1:7861`。Studio只监听本地回环地址；任务状态保存到SQLite，聊天历史保存到浏览器本地存储。

## 官方DeepSeek-V3静态完整性

```powershell
uv run aloepri models inspect `
  --model-fixture tests\fixtures\deepseek-v3-official

uv run aloepri plan `
  --model-fixture tests\fixtures\deepseek-v3-official `
  --output artifacts\plans\deepseek-v3-official.yaml

uv run python scripts\audit_conversion_plan.py `
  --plan artifacts\plans\deepseek-v3-official.yaml `
  --require-layers 61 `
  --require-routed-experts 256 `
  --require-mtp-layers 1 `
  --require-fp8-block-size 128 128
```

## 文档

- [安装手册](docs/INSTALLATION_0.5.0.md)
- [模型选择手册](docs/MODEL_SELECTION_0.5.0.md)
- [RTX 3060转换手册](docs/RTX3060_CONVERSION_0.5.0.md)
- [Mock Cloud演示手册](docs/MOCK_CLOUD_DEMO_0.5.0.md)
- [架构适配器开发手册](docs/ADAPTER_DEVELOPMENT_0.5.0.md)
- [FP8与MTP实现说明](docs/FP8_MTP_IMPLEMENTATION_0.5.0.md)
- [密钥备份与换钥](docs/KEY_BACKUP_ROTATION_0.5.0.md)
- [故障恢复手册](docs/RECOVERY_0.5.0.md)
- [真实云端待验收清单](docs/REAL_CLOUD_ACCEPTANCE_TODO_0.5.0.md)
- [0.5.0验收报告](docs/ACCEPTANCE_0.5.0.md)
- [0.5.0可视化验收报告](docs/AloePri_0.5.0_Mock_Cloud_Acceptance.html)
- [版本说明](docs/VERSION_0.5.0.md)

旧Qwen、OpenSeek、论文复现、实验数据和错误推导文档仍保留在`docs/`与`artifacts/`中；0.5.0的Mock结果不得用于替代真实精度、攻击或云端性能数据。

## TEE 国密模式（增量开发）

新增 `security_mode=tee_gm`、`boundary_mode=tee_split`，旧置换模式和旧工件不迁移。
Qwen2.5-0.5B 已完成真实权重的 Embedding/Transformer/Head 拆分、软件模拟 TEE
问答、Local TEE Head 和一次性掩码外包 Head 闭环。开发启动与真实 TDX 的严格
门禁见 [TEE 国密模式说明](docs/guides/TEE_GM_MODE.md)。

当前发布证据级别为 `TEE_GM_SOFTWARE_SIM_PASS`。普通 RTX 3060/3090 不提供
Intel TDX，尚未完成的 DCAP Quote、TDX Guest 和 RFC 8998 物理互操作不得标记为
`TEE_GM_TDX_ATTESTED_PASS`。
