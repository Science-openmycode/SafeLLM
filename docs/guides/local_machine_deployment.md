# 本机部署与对话

本机部署使用与远程部署相同的私有模型包和私有 Token-ID 协议。服务仅监听
`127.0.0.1`，不建立 SSH 隧道，也不改变任何远程部署记录。

## 图形界面

1. 启动开发版部署程序：

   ```powershell
   cd E:\AloePri
   uv run yinbian-deploy
   ```

2. 进入“转换任务”，点击“新建任务”。
3. 选择模型来源：可以从模型目录下载，也可以选择本机已经下载好的模型目录。
4. 部署方式选择“改造完成后自动部署到本机”，不需要添加云服务器。
5. 确认后，产品自动完成下载或本地读取、权重改造、私有包校验、本机服务启动、
   健康检查和私有 Token 往返检查。两个阶段的进度显示在同一张任务卡中。
6. 通过检查的模型自动进入“已部署模型”，标记为“本机”和`HEALTHY`。
7. 点击“启动对话”。对话程序会直接连接本机服务，不询问 SSH 密码。

如果创建任务时选择了“只在本地改造，稍后再部署”，完成后仍可在任务卡或
“已部署模型”页点击“部署到本机”，无需重新转换权重。

## 命令行

```powershell
# 正式部署一个已经完成转换的任务
uv run yinbian deploy local --job <job-id> --device auto

# 查看部署和实时健康状态
uv run yinbian deploy list
uv run yinbian deploy status <deployment-id>

# 停止、启动和重新启动本机服务
uv run yinbian deploy stop <deployment-id>
uv run yinbian deploy start <deployment-id>
uv run yinbian deploy restart <deployment-id>

# 查看本机运行日志
uv run yinbian deploy logs <deployment-id> --tail 200

# 使用同一个部署进行真实私有问答
uv run yinbian chat send --deployment <deployment-id> --prompt "你好"
uv run yinbian chat stream --deployment <deployment-id> --prompt "介绍一下你自己"
```

本机运行记录保存在产品状态库中；模型权重仍位于原私有模型目录。停止或删除本机
部署不会删除原始模型、私有模型或在线密钥。
