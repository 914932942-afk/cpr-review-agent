# CPR Review Agent

单视角 CPR 训练错误分析与 Agent 辅助复盘系统。支持真实视频分析、错误提示与证据、参考视频对照、助手解释与摘要、历史恢复和下次练习关注点。

**当前主线：`workspaces/m5-cascade-v1/`；启动脚本：`scripts/start.ps1`；端口：8013。** 根目录旧版 `backend/` 和历史说明供研究复现参考。

## 组员首次运行（Windows）

### 1. 前置环境和克隆

- Python **3.11**，`py -3.11 --version` 可用；Git for Windows。
- 本仓库为私有仓库，先由所有者邀请为 collaborator，接受邀请后用自己的 GitHub 账号登录 Git Credential Manager。
- 完整新视频分析需要 **NVIDIA CUDA GPU**。开发机已验证 Windows、RTX 3070 Ti、16 GB RAM；首次 CUDA PyTorch 下载为数 GB，请预留至少 15 GB 磁盘。
- 已提供冻结的前端构建。首次运行 **不需要 Node.js，请勿执行旧 bootstrap.ps1 重建前端**。

```powershell
git clone https://github.com/914932942-afk/cpr-review-agent.git
cd cpr-review-agent
```

### 2. 一次安装

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup_team.ps1 -InstallGPU
```

脚本建立 Web 环境 `.venv` 和 GPU 环境 `.venv-ml`，下载本仓库 Release 冻结资源包并校验 SHA256，恢复模型、公开视频、参考目录和演示记录，生成不含真实 key 的本地 `.env`，检查主线完整性。

有兼容 GPU Python 环境时可复用：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup_team.ps1 -MLPython 'C:\你的环境\python.exe'
```

**下载遇到 403/404 或缺少 Git 凭据：** 确认已接受仓库邀请。也可登录 GitHub，打开 [Release 资源页](https://github.com/914932942-afk/cpr-review-agent/releases/tag/team-demo-v1)，下载 `cpr-team-assets-v1.zip`，再运行（不用手动解压）：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup_team.ps1 -InstallGPU -AssetsZip 'C:\Downloads\cpr-team-assets-v1.zip'
```

资源包适用于全新 clone。遇到已有且内容不同的本地数据会停止，以保护练习记录；已有 `.env` 会保留，请自行确认 `CPR_ML_PYTHON` 指向本机 ML Python。

### 3. 启动

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\start.ps1
```

保持终端运行，浏览器打开：

| 页面 | 地址 |
|---|---|
| 实际复盘工作台 | http://127.0.0.1:8013/ |
| D01 / D02 / D03 演示入口 | http://127.0.0.1:8013/demo |
| 健康检查 | http://127.0.0.1:8013/api/health |
| API 文档 | http://127.0.0.1:8013/docs |

按 `Ctrl+C` 停止。同一工作区一次只运行一个服务。端口占用时使用 `scripts/start.ps1 -Port 8014`，浏览器地址也改为 8014。

## 演示怎么操作

1. `/demo` → **D01 完整复盘**：加载真实已保存结果，回放视频、展开证据。首次查看无需重新跑 GPU。
2. **Explain my focus / Explain** → **Make a recap**：复用已保存的视觉结果解释和整理摘要。
3. **D02 重点复核**：展示重点骨架检查与扩大范围的调用路径，新计算会显示阶段进度。
4. **D03 下一次练习**：只继承上次关注点，该视频起始没有预测；点 **Start review** 才会执行新分析。
5. **Compare references** 对照研究样片，**Export review** 导出可打印 HTML。刷新可恢复记录。

D01/D02 的初始结果随发布包提供，来源是真实已完成分析。验证现场新分析时，新建练习并上传以下现有公开视频，再点 **Start review**：

```text
workspaces/m5-cascade-v1/data/public_cpr/rgb-acceptance-v1/records/SupC04--r00--ch0/video.mp4
```

首次真实视觉计算可能需要数分钟，取决于 GPU。无需自行训练或标注。

## DeepSeek（可选，默认关闭）

**没有 API key 也能启动和完成视觉分析。** 默认 Guided review 使用按钮、表单和证据模板；自由文字会要求确认结构化任务。

需要真实文字理解时，在自己电脑的仓库根目录 `.env` 填写：

```dotenv
CPR_LLM_MODE=disabled
CPR_ML_PYTHON=C:/你自己的环境/python.exe
DEEPSEEK_API_KEY=仅在本机填写真实Key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-flash
```

```powershell
.\scripts\start.ps1 -EnableDeepSeek
```

填 key 不会自动启用外呼，必须显式加 `-EnableDeepSeek`。语言模型只处理文字目标与受限的结果组织，视频不会发送到 DeepSeek。`.env`、虚拟环境、下载缓存和本地运行数据库均被 Git 忽略；Release 不含 `.env`、key 或鉴权头。

## 五个模块

| 模块 | 负责人 | 改进内容 |
|---|---|---|
| M1 | Jinlai GE | 置信度与运动连续性约束的骨架稳定识别 |
| M2 | Wenbo Zhang | 类别专长、质量与分歧感知的 RGB—骨架融合 |
| M3 | Wenxin Yan | 来源与质量条件下的置信度校准与选择性复核 |
| M4 | Mingmin Guo | 目标覆盖与边际成本感知的 Agent 工具调度 |
| M5 | Xiaoyu Li | 质量门控与特征依赖驱动的轻量视觉级联 |

Agent 使用 12 节点 LangGraph 管理上下文、目标确认、工具执行、证据、报告和显式关注点记忆。支持白名单工具、取消、超时、结果复用与恢复。

## 代码与资源位置

```text
scripts/start.ps1                         统一启动入口
scripts/setup_team.ps1                    首次安装入口
scripts/team_assets.py                    资源下载、校验、安装
workspaces/m5-cascade-v1/
  backend/cpr_agent/                      当前 FastAPI / LangGraph / 模块接入
  frontend/src/                          当前界面源码
  frontend/dist/                         冻结且可直接使用的前端构建
  experiments/                           冻结视觉与算法代码
  data/public_cpr/weights/                模型（Release）
  data/public_cpr/rgb-acceptance-v1/records/*/video.mp4
                                         已有研究短片（Release）
  data/m5-cascade-v1/runtime/
    business.sqlite3                     业务记录
    checkpoints.sqlite3                  图状态
    media/                               上传视频
    analysis/                            工具产物与证据
data/mainline-delivery-v1/                演示目录与验收（Release）
docs/                                    工程、算法与交接说明
```

模型及演示资源约 0.9 GB，通过 Release 分发，避免大文件进入普通 Git 历史。源码、构建和资源由 SHA256 清单绑定；私有 Release 与仓库使用相同访问权限。

## 安装检查

```powershell
.\.venv\Scripts\python.exe scripts/team_assets.py --verify-only
.\scripts\start.ps1 -CheckOnly
# 启动服务后：
.\.venv\Scripts\python.exe scripts/final_acceptance.py
```

资源检查覆盖冻结文件和数据库；最终验收读取服务、演示入口和已有验收记录，不触发新推理或付费调用。已有记录验收和本机新视频推理是两项独立检查。

## 常见问题

| 问题 | 处理 |
|---|---|
| `/demo` 案例缺失 | 下载并安装配套 Release，仅 clone 源码不含数据 |
| `PIPELINE_SOURCE_CHANGED` | 检查是否修改/重建冻结工作区；用干净 clone 对照，勿直接改哈希绕过 |
| 工作区锁冲突 | 关闭旧服务再启动，不删除正在使用的锁 |
| CUDA 不可用 | 检查 NVIDIA 驱动、CUDA 版 PyTorch 和 `CPR_ML_PYTHON`；本版本新视频推理需要 GPU |
| 上传失败 | 完整 MP4/MOV/WebM；小于 50 MiB、8–900 帧、最长 45 秒、固定分辨率不超过 4K |
| DeepSeek 不可用 | 使用默认 Guided review；检查自己的 key 与配置模型可用性 |

## 范围与来源

用于 CPR 训练复盘。结论覆盖整段视频，不提供精确错误时刻、按压深度或实测 BPM；未检出不代表动作正确或通过认证。

样片与标注来源：[CPR-Coach](https://github.com/Shunli-Wang/CPR-Coach)、[公开数据集](https://huggingface.co/datasets/ShunliWang/CPR-Coach)。第三方来源清单保留；资源供本小组研究复现，软件许可证不能替代数据或模型许可。

进一步阅读：[当前交付](docs/当前主线-演示与交付.html)、[模块交接](docs/模块交接与版本冻结-v1.html)、[验收指标](docs/当前主线-验收指标.html)。[历史 README](docs/history/README-before-github.md) 保留旧版本记录。
