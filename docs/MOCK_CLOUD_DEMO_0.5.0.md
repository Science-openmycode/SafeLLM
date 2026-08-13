# Mock Cloud演示手册

## 启动Studio

```powershell
uv run aloepri studio --port 7861
```

只允许浏览器访问`http://127.0.0.1:7861`。不要把Studio直接绑定公网地址。

## UI流程

1. “模型”页输入本地模型目录并检查；
2. “转换任务”页生成plan、创建任务；
3. 任务进入`CONVERTING`后执行Mock上传；
4. 上传状态依次进入`UPLOADING→VERIFYING→READY_TO_DEPLOY`；
5. 部署生成不可变`mock-<uuid>` deployment ID；
6. “聊天”页通过本地Agent发送私有token-ID模拟流；
7. 开启“显示私有轨迹”可看私有input/output ID，不返回完整`tau`；
8. 创建新deployment后可回滚到上一个RUNNING版本。

## CLI流程

```powershell
uv run aloepri upload <job-id> --cloud-profile mock
uv run aloepri deploy <job-id> --cloud-profile mock
uv run aloepri deployment status <deployment-id>
uv run aloepri deployment rollback <deployment-id>
```

所有Mock证据必须包含：

```json
{
  "environment": "mock-cloud",
  "real_cloud_validated": false
}
```

Mock推理输出是输入和deployment ID的确定性token序列，用于验证协议、SSE、错误处理和UI，不进入精度、TTFT、TPOT或攻击结果表。
