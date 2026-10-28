公开标签轻量分类头验证（独立离线实验）

目的
验证“公开CPR-Coach骨架与原标签 → 冻结PoseC3D → 小分类头 → 独立测试”是否可运行。
不更改五模块或12节点Agent；不采集视频、不新增人工标签、不训练视觉主干。
它是M1的识别基线，不是已完成的算法改进，也不代表RGB、融合、校准和Agent已接入。

数据契约
来源：https://huggingface.co/datasets/ShunliWang/CPR-Coach
固定版本：b8667b8fd6a70ac63cc09fc6b763a9459b7c160a
下载ann.tar.gz及Keypoints.tar.gz，并逐项验证SHA-256。
固定单视角ch0；按原始trainlist/testlist/all_errors_testlist匹配视频记录。
公开68点骨架的前17点对应COCO人体点；确定性转换不产生新标签。
补充数据的SupC/SupDC/SupMC容器路径只作确定性别名转换；原TXT路径保留在记录清单中。
all_errors已含double_errors；读取前者作组合测试，不重复采样后者。
使用原始错误ID 1,2,3,4,5,6,7,8,9,11,13；Correct(0)编码为全零。
10/12仅深度错误样本不改写为Correct；组合样本保留原始标签并只评估当前11项。
从原trainlist每类固定取4段作验证，其余训练；官方两类测试集不参与训练、阈值选择。
没有可靠受试者ID，因此仅称原始划分的单视角实验，不能称跨人泛化验证。

文件职责
download_public.py：可续传的公开资产下载、完整性校验；不解包到任意路径。
baseline.py prepare：读取公开标签，受限反序列化骨架，保存独立记录与数据审计。
baseline.py extract：严格核对全部主干权重；主干eval且requires_grad=False；缓存512维特征。
baseline.py train：只训练线性头/小MLP；仅在验证集选阈值及候选头；输出逐类测试指标。
video_smoke.py：本地未标注视频的输入连通验证，不报告这些视频的识别准确率。
test_baseline.py：标签、划分、反序列化、指标和模型契约的独立测试。

已验证环境以environment.json为准
机器已有独立conda ml环境；Web系统的.venv没有这些视觉依赖，请勿混为一个环境。
示例解释器：C:\Users\gjj33\.conda\envs\ml\python.exe
MMAction2 1.2.0 / mmengine 0.10.7 / mmcv-lite 2.1.0。
兼容处理只作用于未使用的可选DRN/多模态导入和NumPy别名；不替换姿态网络计算。
公开PoseC3D权重仅含主干；未加载的原K400分类层从不执行。CPR小分类头单独拟合和保存。

运行（在本目录打开PowerShell；路径可通过参数指定）
1. & 'C:\Users\gjj33\.conda\envs\ml\python.exe' -X utf8 -m unittest -v test_baseline
2. & 'C:\Users\gjj33\.conda\envs\ml\python.exe' -X utf8 download_public.py
3. & 'C:\Users\gjj33\.conda\envs\ml\python.exe' -X utf8 baseline.py prepare --assets <公开资产目录> --output <输出目录>
4. & 'C:\Users\gjj33\.conda\envs\ml\python.exe' -X utf8 baseline.py extract --output <输出目录> --backbone <PoseC3D权重路径> --batch-size 4
5. & 'C:\Users\gjj33\.conda\envs\ml\python.exe' -X utf8 baseline.py train --output <输出目录>
6. & 'C:\Users\gjj33\.conda\envs\ml\python.exe' -X utf8 video_smoke.py --video <单视角短视频> --backbone <PoseC3D权重路径> --pose <YOLO-Pose权重路径> --run <输出目录>
默认资产和输出目录分别是脚本旁的assets与run；正式保留目录的实际路径见paths.json。
模型只接受已核对的权重SHA、特征配置和标签顺序。中断后可重跑；不使用未完成下载训练。

结果解释
工程连通、数据完整性、分类头训练和测试效果分别验收，不能互相替代。
head-training-results.json包含全部逐类F1/AP、正常片段误报以及组合错误测试。
所有输出是未校准分数；M3校准/拒判尚未实现，不能当作真实正确概率。
本地视频采用YOLO骨架，公开训练骨架来自AlphaPose；未标注本地视频只能验证接口，
不能证明两种提取器和本地机位的识别效果一致。第二段有少数多人检测帧，尚未验证人物跟踪。
本实验不输出精确错误起止时刻、按压深度或临床质量判定，也不把片段标签扩充成这些标签。
后续M1/M5的改进必须与本基线比较；轻量头拟合本身不算已满足五人的算法改进要求。
