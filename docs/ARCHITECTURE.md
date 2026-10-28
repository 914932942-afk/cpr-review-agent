# 可选大模型适配补充

LLM有disabled/deepseek两种模式，目前disabled。N02通过ModelGateway解析目标，失败进入N03。关闭模式不构造HTTP客户端；启用模式仅向官方HTTPS接口发送文字与白名单目标上下文，返回必须符合ReviewGoal。N11仍为模板，不允许大模型改变CV结果。

N02将已解析目标/降级码保存在当前turn，表单恢复和幂等重复提交不重复解析；外部API与本地提交不构成分布式事务，因此进程恰在API响应和本地保存之间被硬终止时，不能承诺外部调用exactly-once。结构化目标始终不外呼。模型调用有短超时和连续失败熔断；调用返回后重新检查取消/服务预算，再决定是否继续，不执行取消后的CV任务。

运行开关、故障处理和上下文边界以根目录README的“大模型的两种模式”为准。下面保留之前阶段的说明。

# 当前接入补充 · 0.2

当前实现请先读根目录 README。真实模式保留12节点，并以一个 `video.analyze` 工具执行冻结v15全管线；未拆分的阶段不单列为独立工具。N04加载同clip、同视频和bundle的兼容记录；N05输出模型提示状态，不冒充M3置信度校准；N06使用确定性基线选择执行或停止，不冒充M4学习算法；N09验证并持久化真实结果；N11生成可追溯的片段级报告。N03和N08仍为真实中断等待。

以下是0.1骨架的历史实现说明，涉及“仅接受placeholder”的描述仅适用于stub模式。

---

# 骨架接线说明

## 主链

```text
浏览器 → FastAPI → Store创建Turn/Run → 单编排器 → LangGraph
N01 → N02 → N04 → N05 → N06 → N07 → N08暂停
                                           ↓
                     本地工具线程 ← 持久ToolExecution
                     写入回执 → 编排器读取DB → Command(resume)
N09 → 下一依赖N07 / 再评估N05 → N06 STOP → N10 → N11 → N12
```

没有结构化目标时：N02 → N03暂停 → 浏览器提交表单 → N02重新校验。

示范、摘要、记忆等非分析目标：N04 → N10，跳过视觉工具。`existing_only`在N06停止，不提交新工具。

## 十二节点

| 节点 | 本阶段职责 |
|---|---|
| N01 load_context | 固定会话／发布包／Clip，读取近期消息引用和显式偏好 |
| N02 resolve_goal | 校验结构化目标；文本通过禁用的ModelGateway进入表单降级 |
| N03 await_user | 保存表单等待并真实interrupt；恢复后校验、返回N02 |
| N04 prepare_context | 载入当前会话报告引用；参考目录尚未接入 |
| N05 assess_state | M3占位：全部not_evaluated，不形成真实Assessment或证据 |
| N06 decide_and_plan | M4占位固定选择＋确定性依赖规划；动作与执行分离 |
| N07 dispatch_step | 幂等写入一项ToolExecution；等待中持久任务可重建 |
| N08 await_tool | 真实interrupt；恢复信号只来自内部编排器，结果从DB读取 |
| N09 collect_result | 验归属／输入，接收明确占位回执；推进依赖或排除失败路径 |
| N10 finalize_scope | 固定回复范围；未接算法时不能完成业务分析 |
| N11 build_response | 类型化模板回复，无LLM生成、无分类claims |
| N12 commit_run | 一次事务提交终态、不可变回执和符合条件的显式记忆 |

正常路由见`graph.NORMAL_ROUTES`。节点返回`Command(goto=...)`，不再叠加静态出边，防止重复调度。全局取消／超时走N12或N10；等待节点恢复时先消费原interrupt。

## 五人入口

| 模块 | 接口槽位 | 当前接线 |
|---|---|---|
| M1 · Jinlai GE | modules/m1_skeleton.py | skeleton.predict占位工具 |
| M2 · Wenbo Zhang | modules/m2_fusion.py | fusion.predict占位工具 |
| M3 · Wenxin Yan | modules/m3_calibration.py | N05的AssessmentPort |
| M4 · Mingmin Guo | modules/m4_scheduling.py | N06的PolicyPort |
| M5 · Xiaoyu Li | modules/m5_rgb.py | rgb.predict占位工具 |

这些文件只有Protocol，不包含改进算法。`modules/contracts.py`提供未来输入／输出草案；正式签名以G0后冻结的模型、布局、标签与预处理为准。

## 契约与恢复边界

- 骨架API只接受结构化目标；owner、Clip、发布包、工具ID与执行ID不由自然语言直接控制。
- 图thread为`review:{run_id}`；新请求新Run，暂停恢复同一thread。
- 用户resume需要wait_id、interrupt_id、revision和幂等键；浏览器不能发tool_complete。
- 两个SQLite文件之间不做伪分布式事务；以业务终态对账，已提交Run不重复invoke。
- 工具提交与执行分离；已登记但未派发的任务会在重启后被发现。物理执行不承诺exactly-once。
- 取消优先于未提交报告；超时只进入一次收尾；用户表单等待有独立期限。
- 本阶段不实现生产级重试／熔断、多进程fencing、跨用户鉴权、容量管理或数据库迁移。这些属于后续可靠性阶段。

框架依据：[LangGraph interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts)、[FastAPI lifespan](https://fastapi.tiangolo.com/advanced/events/)。
