# 代码与模型数据分离

## 目录边界

Git 仓库只保存源码、配置模板、文档和小型测试样本。以下内容必须放在仓库之外，且不会上传 GitHub：

- 原始模型权重；
- 改造后的私有模型；
- TEE 边界包和服务器主体包；
- 在线/离线密钥；
- 下载缓存、转换中间文件和运行证据；
- 本地数据库、聊天历史和服务器凭据。

推荐目录：

```text
E:\YinbianRuntime\
├── state\          # 任务与部署数据库
├── chat\           # 对话数据库
├── credentials\    # 本机保护的凭据
└── logs\

E:\YinbianData\
├── source-models\  # 原始模型
├── private-models\ # 改造后的模型
├── cache\          # 可重新生成的下载缓存
└── evidence\       # 测试与验收结果
```

## 配置

PowerShell 当前会话：

```powershell
$env:YINBIAN_HOME = "E:\YinbianRuntime"
$env:YINBIAN_DATA_DIR = "E:\YinbianData"
```

永久写入当前 Windows 用户：

```powershell
[Environment]::SetEnvironmentVariable("YINBIAN_HOME", "E:\YinbianRuntime", "User")
[Environment]::SetEnvironmentVariable("YINBIAN_DATA_DIR", "E:\YinbianData", "User")
```

`YINBIAN_CACHE_DIR` 可选；设置后只覆盖缓存目录。环境变量不设置时，程序使用用户应用数据目录，仍不会把新模型默认写进源码目录。

## 旧数据兼容

旧任务和部署在数据库中保存的是完整路径，因此整理代码不会改变其位置。已有 `data/` 和
`artifacts/` 可以继续使用；建议在没有转换或推理任务运行时，再迁移到外部数据盘。迁移前先备份，
迁移后通过软件中的模型检查重新确认路径。不要把模型权重复制回 Git 仓库。

## 提交前检查

```powershell
git status --short
git ls-files | Select-String -Pattern '\.(safetensors|bin|pt|pth)$'
git ls-files data artifacts
```

第二条命令应无模型权重输出；第三条最多只出现仓库明确保留的小型评测样本。
