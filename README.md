# 钉钉周报面板

## 直接下载使用

**[下载 Windows x64 便携版 v1.3.10](https://github.com/TTTAAXXTTT/dingtalk-report-panel/releases/download/v1.3.10/dingtalk-report-panel-Windows-x64-v1.3.10.zip)** · [发布说明及 SHA-256 校验](https://github.com/TTTAAXXTTT/dingtalk-report-panel/releases/tag/v1.3.10)

完整解压后双击 `04_演示模式.cmd`；真实使用先运行 `01_首次使用.cmd` 登录，再用 `02_日常启动.cmd`。包含 Python、依赖和已校验的 DWS，无需安装 Python 或 Node。


在 Windows 本机运行的手动报告工具：选择钉钉单聊或群聊、时间范围和报告本人，先预览确认，再生成 **本周已完成事项** 与 **周报（本周工作、下周计划）** 两份独立报告。

基于原项目 1.3.9 整理，包含完整 Python 后端、网页前端、DWS 适配器、证据校验引擎、存储层、四个启动入口、安装脚本与全部原有测试。MIT 开源。

## 快速开始

需要 Windows 10/11、Python 3.11+ 和 PowerShell。建议使用 Python 3.12。

```powershell
git clone https://github.com/TTTAAXXTTT/dingtalk-report-panel.git
cd dingtalk-report-panel
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt -c constraints.txt
.\start_demo.ps1 -NoBrowser
```

打开 http://127.0.0.1:8765 查看演示数据。演示模式使用本地合成会话；如果选择模型生成，仍须自行配置并测试模型。程序不会附带可用 API 密钥。

真实使用：双击 `01_首次使用.cmd` 完成依赖安装与官方 DWS 登录；之后使用 `02_日常启动.cmd`。重新登录用 `03_重新登录钉钉.cmd`，演示用 `04_演示模式.cmd`。也可以在已激活的虚拟环境中依次执行 `install.ps1`、`login_dingtalk.ps1`、`start.ps1`。

1. 刷新聊天列表，勾选会话，设置起止时间。
2. 将演示姓名“张三”改为你在钉钉中实际使用的姓名。
3. 在面板中配置模型 URL、协议、模型名和密钥，完成接口、结构化输出及本人识别测试。
4. 预览范围，明确确认后点击“确认并手动生成”。
5. 分别查看、复制或下载两份报告。

## 数据完整性与隐私

会话和消息均进行分页、唯一 ID、范围与稳定性核对；不确定或缺失时停止。每项模型结论必须关联有效来源与逐字证据，本人归属和完成状态在本地再次校验。

没有定时任务，不自动生成或发送报告。选择模型服务商后，相关消息会传输到你配置的模型接口，应由你确认该服务适合处理这些内容。最终业务报告隐藏原文与来源，任务快照和内部证据仍保存在本机。

`data/`、`outputs/`、`.tools/`、`.env`、真实聊天、报告和 DWS 登录状态均不随源码发布。模型配置与 API 密钥由面板保存在本机；详见 [安全说明](SECURITY.md)。默认仅监听回环地址；此个人面板没有为公网多人访问设计身份认证。

## 开发与验证

```powershell
python -m unittest discover -s tests -v
python tools/check_publication.py
```

完整操作与接口说明见 [USER_GUIDE.md](USER_GUIDE.md)，测试范围见 [QA.md](QA.md)，模块与交付核对见 [COMPLETENESS.md](COMPLETENESS.md)。模型环境变量见 [.env.example](.env.example)，程序不会自动加载该模板，须在 PowerShell 中设置或通过面板配置。

## 项目结构

| 路径 | 内容 |
|---|---|
| `app/main.py` | API、手动确认和任务生命周期 |
| `app/dws_adapter.py` | 官方 DWS CLI 只读采集、分页及稳定性校验 |
| `app/report_engine.py` | 本地筛选、证据和报告整理 |
| `app/llm_engine.py` | 模型传输、结构校验、重试和检查点 |
| `app/storage.py`、`llm_config_store.py` | 本机数据库、不可变产物和模型配置 |
| `app/static/` | 完整网页界面 |
| `fixtures/`、`tests/` | 合成演示数据与完整回归测试 |
| 根目录 CMD / PS1 | 安装、登录、启动和演示入口 |

本项目不属于钉钉官方产品。DWS、模型和 Python 依赖由各自项目提供，见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

