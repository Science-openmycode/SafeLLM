# 隐变智模 1.0 安装与运行手册

## 客户端要求

| 项目 | 要求 |
|---|---|
| 系统 | Windows 10 或 Windows 11 x64 |
| 本地磁盘 | Qwen2.5-0.5B 建议预留 12 GiB；目录可在安装后修改 |
| 内存 | 最低 16 GiB，建议 32 GiB |
| GPU | 可选；CPU 路径必须可用，NVIDIA GPU 用于加速转换 |
| 网络 | 下载模型和 SSH 上传时需要联网 |

服务器要求：Ubuntu 22.04 x86-64、NVIDIA 驱动 525 或更高版本、单张 NVIDIA GPU、root 或免交互 sudo、SSH。隐变智模不自动安装 NVIDIA 驱动。

## 安装

1. 校验安装包：

```powershell
Get-FileHash .\YinbianZhimo-1.0.0-Windows-x64-Offline.exe -Algorithm SHA256
Get-Content .\YinbianZhimo-1.0.0-Windows-x64-Offline.exe.sha256
```

2. 运行安装包。安装范围固定为 Windows 当前用户，不要求管理员权限。
3. 安装程序创建三个入口：

```text
隐变智模部署.exe
隐变智模对话.exe
yinbian.exe
```

4. 安装程序执行 `yinbian.exe --help`。自检失败时不会更新 `current.json`。

数据目录：

```text
%LOCALAPPDATA%\YinbianZhimo\
├── state\
├── chat\
├── credentials\
├── logs\
└── cache\
```

## 第一次部署

桌面程序按以下顺序操作：

```text
隐变智模部署
→ 服务器
→ 添加服务器
→ 检查连接并确认 Host Key 指纹
→ 模型
→ 选择 Qwen2.5-0.5B
→ 选择并部署
→ 查看许可证和资源检查结果
→ 选择 CPU、GPU 或自动
→ 开始任务
```

命令行执行相同操作：

```powershell
yinbian models recommend
yinbian models download --model qwen2.5-0.5b-instruct --metadata-only
yinbian models download --model qwen2.5-0.5b-instruct --accept-license
yinbian servers add `
  --name qwen-gpu `
  --host 203.0.113.10 `
  --username root `
  --auth-type private_key `
  --private-key C:\Keys\id_ed25519

yinbian servers check <server-id> --trust-host-key
yinbian doctor local --model qwen2.5-0.5b-instruct
yinbian plan create `
  --model qwen2.5-0.5b-instruct `
  --mode direct-deploy `
  --device auto `
  --server <server-id>

$env:YINBIAN_OFFLINE_KEY_PASSWORD = '<备份口令>'
yinbian convert --plan <plan.yaml> --accept-license
yinbian jobs status <job-id>
```

服务器缺少 Docker 或 NVIDIA Container Toolkit 时，先查看预检结果，再执行：

```powershell
yinbian servers bootstrap <server-id> --confirm
```

该命令会修改服务器。NVIDIA 驱动仍需服务器管理员预先安装。

## 创建部署

`direct-deploy` 计划在转换和远端 SHA-256 提交后自动创建候选部署，不需要再次上传同一模型目录。以下命令用于 `local-only` 任务或手工升级：

```powershell
yinbian deploy create `
  --job <job-id> `
  --server <server-id> `
  --server-package <private-model-dir>

yinbian deploy status <deployment-id>
yinbian deploy logs <deployment-id> --tail 200
```

服务只绑定服务器 `127.0.0.1`。不要在安全组或防火墙中开放模型端口。

## 对话

打开 `隐变智模对话.exe`，选择健康部署。程序建立 SSH 隧道后才显示输入框。

命令行：

```powershell
yinbian tunnel open <deployment-id>
yinbian chat deployments
yinbian chat stream --deployment <deployment-id> --prompt "写一段产品介绍"
yinbian tunnel status <deployment-id>
yinbian tunnel close <deployment-id>
```

会话正文只保存在本地聊天数据库，并由 Windows 当前用户 DPAPI 保护。关闭历史保存后，正文只保存在当前进程内存。

## 暂停与恢复

```powershell
yinbian jobs pause <job-id>
yinbian jobs resume <job-id>
yinbian jobs events <job-id>
```

暂停在当前安全单元结束后生效。恢复时复用已通过 SHA-256 的源分片、私有分片和远端文件。

## 升级和回滚

```powershell
yinbian deploy upgrade <deployment-id> `
  --job <new-job-id> `
  --server-package <new-private-model-dir>

yinbian deploy rollback <deployment-id>
yinbian deploy restart <deployment-id>
```

候选版本使用独立 Compose 项目和空闲端口。健康检查通过后更新 active 链接，再停止旧版本。回滚会重新启动上一个健康版本并恢复其端口。

## 卸载

卸载程序分别询问是否删除模型、密钥、服务器记录、会话历史和日志。默认全部保留。远端部署不会被卸载程序删除。
