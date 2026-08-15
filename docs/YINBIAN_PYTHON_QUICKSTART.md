# 隐变智模 Python 版快速使用

## 1. 首次准备

在仓库根目录打开 PowerShell：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_yinbian_python.ps1
```

脚本使用 `uv.lock` 固定依赖，并优先从 uv 缓存硬链接文件。依赖没有变化时再次执行会直接跳过，不会重复安装 PyTorch。

若已有在线演示正在使用 `.venv`，脚本不会强行修改环境或中断进程；它会先执行只读检查。需要主动更新依赖时，先关闭相关程序，再执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_yinbian_python.ps1 -Refresh
```

## 2. 打开部署程序

双击：

```text
scripts\start_yinbian_deploy_python.cmd
```

或执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_yinbian_python.ps1 deploy
```

在“服务器”页面可以直接粘贴租赁平台提供的命令：

```text
ssh -p <端口> <用户名>@<主机名>
```

再选择“密码认证”并输入密码。密码只保存在当前程序的内存中，不写入 SQLite、配置文件或日志。第一次连接会显示 SSH 主机指纹，确认后才固定该指纹。

模型必须从下拉菜单选择：

- `Qwen2.5-0.5B-Instruct`：已完成当前产品验收；
- Qwen2.5 1.5B、3B、7B、14B、32B、72B：使用同一个 Qwen2/Qwen2.5 架构适配器，执行前重新检查 checkpoint，并按页面显示的内存要求运行；
- 其他架构只在其支持阶段内开放，不会伪装成可聊天模型。

## 3. 打开对话程序

部署状态变为 `HEALTHY` 后，双击：

```text
scripts\start_yinbian_chat_python.cmd
```

或执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_yinbian_python.ps1 chat
```

## 4. 用命令行完成相同操作

```powershell
# 查看模型下拉菜单对应的目录
.\.venv\Scripts\yinbian.exe models list

# 粘贴租赁平台SSH命令；命令行不保存密码
.\.venv\Scripts\yinbian.exe servers add `
  --name "租赁GPU" `
  --ssh-command "ssh -p <端口> <用户名>@<主机名>" `
  --auth-type password

# 查看服务器ID
.\.venv\Scripts\yinbian.exe servers list

# 创建计划；模型ID来自 models list
.\.venv\Scripts\yinbian.exe plan create `
  --model qwen2.5-0.5b-instruct `
  --mode direct-deploy `
  --device auto `
  --server <server-id>
```

密码认证的连接检查会在执行时获取密码；不要把真实密码写进脚本、Git仓库或命令历史。图形界面更适合密码认证，因为它将密码保存在当前页面内存中。

## 5. 为什么新版更快

- Python开发版不解压完整离线安装包，直接复用 `.venv`；
- `setup_yinbian_python.ps1` 以 `uv.lock` 哈希判断是否需要安装，避免每次启动运行 `uv sync`；
- uv 缓存与虚拟环境同盘时使用硬链接；
- Windows安装器后续构建改用 LZ4 非solid压缩，减少对已经压缩过的 PyTorch/WebView2文件进行耗时的LZMA解压。

第一次下载CUDA版PyTorch仍然需要较长时间，这是一次性依赖下载；以后启动不会重复下载。
