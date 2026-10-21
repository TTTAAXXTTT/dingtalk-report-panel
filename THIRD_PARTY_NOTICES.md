# 第三方依赖

本仓库 MIT 许可证仅覆盖本项目原创代码和文档。第三方软件未随仓库复制，其原许可证继续适用。

- [DingTalk Workspace CLI](https://github.com/DingTalk-Real-AI/dingtalk-workspace-cli)：独立外部程序，`setup_dws.ps1` 从官方 Release 获取 v1.0.59，并校验下载包和可执行文件 SHA-256。授权与登录状态由 DWS 自行维护。
- FastAPI、Starlette、Uvicorn、Pydantic、HTTPX、tzdata 及传递依赖：由 Python 包索引安装。具体解析版本见 `constraints.txt`，许可信息以对应发行包为准。
- 模型接口、模型输出与聊天内容的权利不由本仓库许可证授予。
