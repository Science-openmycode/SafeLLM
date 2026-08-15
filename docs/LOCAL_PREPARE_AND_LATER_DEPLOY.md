# 隐变智模：本地准备、稍后部署

## 固定流程

```text
选择模型
→ 本地磁盘与内存预检
→ 下载固定 revision 的元数据和权重
→ 校验源文件
→ 本地生成密钥
→ 本地改造并校验私有模型
→ 登记为 LOCAL_READY
→ 用户以后添加或选择服务器
→ 断点上传已经校验的私有模型
→ 远端 SHA-256 校验
→ 安装运行环境并启动服务
→ 私有 Token-ID 问答检查
```

本地改造任务不要求存在服务器记录，也不建立 SSH 连接。上传阶段只读取已经完成的私有模型，不重新下载、重新生成密钥或重新改造权重。

## 本机目录

大文件统一放在 `YINBIAN_CACHE_DIR`。当前机器已配置为：

```text
E:\YinbianCache\
├── models\<catalog-id>\       # 固定 revision 的原始模型
└── private\<catalog-id>\      # 已校验的私有模型
```

状态、凭据和聊天数据库仍保存在 `%LOCALAPPDATA%\YinbianZhimo`。C盘不保存模型权重。

## 界面操作

1. 在“转换任务”中选择“先在本地下载并改造，稍后部署”。
2. 服务器可以留空。
3. 完成后，模型会以 `LOCAL_READY` 出现在“已部署模型”页。
4. 点击“选择服务器部署”，选择目标服务器并确认。
5. 页面显示运行环境检查、上传、远端校验、启动和健康检查进度。

## 命令行等价操作

```powershell
yinbian plan create `
  --model glm-4-9b-chat-hf `
  --mode local-only `
  --device auto `
  --destination E:\YinbianCache\private\glm-4-9b-chat-hf `
  --output E:\YinbianCache\plans\glm-4-9b-chat-hf.yaml

yinbian convert `
  --plan E:\YinbianCache\plans\glm-4-9b-chat-hf.yaml `
  --accept-license

yinbian deploy create `
  --job <job-id> `
  --server <server-id>
```

`deploy create`会从已完成任务中自动找到私有模型目录；只有工件不在任务数据库中时才需要额外传入`--server-package`。

## 当前机器容量

按2026-08-15实测E盘空闲约57.99GiB计算：

| 模型 | 原始权重 | 私有模型估计 | 峰值空间（含20%余量） | 结论 |
|---|---:|---:|---:|---|
| Qwen3-8B | 15.26GiB | 17.55GiB | 约39.7GiB | 磁盘足够；当前转换器要求64GiB内存，本机不满足 |
| GLM-4-9B-Chat-HF | 17.51GiB | 20.14GiB | 约45.5GiB | 磁盘足够；当前转换器要求48GiB内存，本机不满足 |
| DeepSeek-V2-Lite-Chat | 约28.9GiB | 不低于28.9GiB | 至少约69.6GiB | 当前空间不足 |

空间和主机内存门禁在任何权重下载前重新计算。任一资源不足时任务直接拒绝，不创建大型`.partial`文件。当前电脑有约15.63GiB物理内存，因此GLM-4-9B必须先完成Qwen/GLM逐层流式转换器，或者改在至少48GiB内存的可信本地工作站转换；不能用页面文件冒充可行路径。

## 清理规则

- 默认保留原始模型和私有模型，方便重新换服务器。
- 只有私有模型校验通过后，才允许用户手动清理原始权重。
- 只有远端文件SHA-256全部通过后，才允许选择“上传后清理本地私有模型”。
- 密钥、manifest和任务数据库不随模型缓存一起清理。
