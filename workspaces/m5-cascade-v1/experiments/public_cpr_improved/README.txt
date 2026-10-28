CPR 离线识别改进实验（2026-10-21）

用途
验证“真实单视角CPR片段 → 人物关联 → 骨架 → 冻结特征 → 小型分类头 → 错误类别”的链路。
这是独立离线组件；正式网页、API、五模块占位实现和12节点Agent没有切换到本组件。
最新验收结论以 CODE/test-results/step1-improved/验证结果.html 和 validation-summary.json 为准。

实现
1. YOLOv8x-Pose负责检测与人物运动关联；保留模型原始关键点坐标，避免显示层的低置信度置零影响特征。
2. 按检测框IoU关联人物，再按持续的头部/躯干运动选择训练者；多人活动难以区分时不猜测。
3. 公开AlphaPose Halpe68权重提取骨架，取前17个人体点；严格加载全部权重。
   使用已安装torchvision的非调制DCN算子适配；完成零偏移及固定非零偏移数值测试。
   原数据集未公开其完整提取器配置，本实现不声称逐位复现原始骨架。
4. 冻结PoseC3D输出512维特征，结合271维归一化身体几何/运动特征。
5. 小型树分类头只用公开原标签拟合；阈值只在验证集选择，视频验证阶段没有调整。

数据边界
没有自行采集视频，没有新增人工标签，没有训练视觉主干。
公开CPR-Coach的ch0视角；支持标签ID为1,2,3,4,5,6,7,8,9,11,13。
深度类10/12不纳入；只有深度错误的片段没有伪装成正常样本。
应用划分为训练758、验证198、测试364。组合错误的公开原标签参与训练，区别于原论文的零样本组合测试。
原单错误测试216条保持在测试集；另保留2条补充样本作视频验证。
录像组不跨划分；没有可靠受试者身份映射，不声称跨人测试。
4段公开视频只覆盖2类错误，不足以证明11类错误、正常片段和新机位的真实视频识别质量。
未输出错误精确起止时刻、真实按压深度、实测BPM或临床合格结论。
空错误列表仅表示当前阈值下未检出支持的错误，不能解释为“CPR正确”。

运行
使用现已验证的ML环境，不要使用项目的轻量Web .venv：
C:\Users\gjj33\.conda\envs\ml\python.exe
在CODE目录执行下面的一行命令（输出文件必须是新文件）：
& 'C:\Users\gjj33\.conda\envs\ml\python.exe' .\experiments\public_cpr_improved\infer_video.py --video .\data\public_cpr\video-verification\public-C13-r12-ch0.mp4 --output .\test-results\step1-improved\manual-demo.json
可选追加 --reference-record CPR_Dataset_S0/C13_S0/r12/ch0，对已登记公开视频做原标签对照。
不传reference-record时，预测不读取原标签，亦不计算准确率；这不代表任意输入域均已验证。

输出
JSON：预测类别、未校准分数、阈值、人物关联情况、输入质量、耗时、配置指纹和使用范围。
同名NPZ：骨架、特征、分数和预测，便于后续独立验证及模块集成。
inference-manifest.json固定代码、权重、类别、特征契约和数据验证样例；不匹配即拒绝执行。
没有可用人物、人物关联不明确、视频超出帧数预算或解码不完整时不会生成错误判断。

验证
在本目录执行：
& 'C:\Users\gjj33\.conda\envs\ml\python.exe' -m unittest test_improvement test_raw_pose_adapter test_inference_contract -v
测试中的合成数组/黑帧只用于边界测试，不进入训练集。
PoseC3D加载时会提示未加载原分类头；本实现只调用已逐张量核对的冻结backbone，该分类头不参与推理。

文件位置
模型与公开数据：CODE/data/public_cpr/
当前小分类头：data/public_cpr/improvement-v2/combined-64.pt
应用划分与协议：data/public_cpr/improvement-v2/protocol.json
历史失败结果及获取脚本：test-results/step1-improved/history/ 与 reproduction-sources/
原始/改进协议的结果保留分开，不能把监督范围变化造成的指标差异全部归为算法改进。
M1—M5最终改进仍需要各自基线、消融和模块验收；本轮没有宣称五个算法模块全部完成。

公开来源
CPR-Coach：https://huggingface.co/datasets/ShunliWang/CPR-Coach
原项目：https://github.com/Shunli-Wang/CPR-Coach
AlphaPose：https://github.com/MVIG-SJTU/AlphaPose
其上游源码、修改过的兼容层、固定版本和许可证位于alpha-pose-source目录。
