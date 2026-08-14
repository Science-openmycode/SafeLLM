# 隐变智模 1.0.0

## Breaking Changes

- 正式命令改为 `yinbian`；`aloepri` 进入一个版本周期的兼容期。
- 新状态库分离转换任务、权重分片、部署版本和聊天正文。

## New Features

- 增加 Windows 部署程序、对话程序和共享命令行入口。
- 增加固定 revision 的模型元数据下载、分片续传、SHA-256 校验和原子提交。
- 增加真实 SSH/SFTP、Ubuntu 预检、Docker HF 服务、健康检查、版本升级和回滚。
- 增加 DPAPI 凭据库、`.ybkey` 便携备份和本地加密聊天历史。
- 新转换任务完成后将在线 Token 置换密钥封装进 Windows 当前用户 DPAPI，聊天进程只在初始化时临时展开。
- 模型许可证确认绑定固定 revision 和下载到本地的许可证文件 SHA-256。
- 增加跨进程 SSH 隧道和基于健康部署的多轮流式问答。

## Fixes & Improvements

- 候选部署使用版本独立 Compose 项目和空闲端口，健康后才切换 active 链接。
- 桌面 API 增加随机会话 Cookie、Host 校验和 Origin 校验。
- 用户界面和新工件统一采用“隐变智模”名称，旧 checkpoint 保持兼容。

## Release Gate

仓库构建在完成代码签名、干净 Windows 10/11 安装和受控 Ubuntu GPU 验收前标记为开发版。Mock Cloud 结果不计入真实 SSH 部署结论。

2026-08-14 内部构建：277 项测试通过、1 项真实模型集成测试跳过；未签名安装包为 3,323,353,793 字节，SHA-256 为 `2e96741b398197967ab655ae50c782451901a826254906f3ea64bdbf5289bf1f`。该文件不是正式签名发布包。
