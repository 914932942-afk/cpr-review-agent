第二项优化后的离线候选：几何树头 +正则化线性小分类头

当前固定公开骨架工程测试已达到原门槛；完整26段真实视频验收尚未完成。
结果以CODE/test-results/public-rgb-optimization-v4/优化验收报告.html为准。
新模型与协议位于CODE/data/public_cpr/rgb-evaluation-v4/。
候选指针位于CODE/data/public_cpr/current-offline-candidate.json。

已完成
保持训练738、验证196、测试386及26段预留录像组不变。
只训练小分类头；使用公开原标签，不采集、不新增人工标注、不训练视觉主干。
拟合与归一化只使用训练集，阈值及自动选型只使用验证集。
原树模型、失败实验、旧默认推理清单与19个系统源码文件均未修改。
同一工程测试集已多轮查看；不宣称独立盲测。

运行一个已截取CPR动作片段（在CODE目录，用ML环境）
& 'C:\Users\gjj33\.conda\envs\ml\python.exe' .\experiments\public_rgb_dense_head\infer_compact_video.py --video <视频路径> --output <新的JSON路径>
可选 --reference-record <公开录像ID>，只允许对已登记的匹配视频在预测后核对公开标签。
新候选必须用infer_compact_video.py，其加载策略支持紧凑分类头；不要把新权重直接交给旧树头专用加载器。
旧视频预处理通过独立加载适配器复用，原代码及输出语义未改。

开发复现（在本目录）
python optimize_dense.py prepare
python optimize_dense.py train
python optimize_dense.py evaluate
python optimize_dense.py export
python -m unittest test_compact test_fitted_compact -v
已冻结阶段拒绝覆盖。finalize_report.py只生成本轮结果与报告，不覆盖旧记录。

边界
当前正式网页和12节点Agent仍保留原状态，五个算法模块没有据此宣称全部实现。
1段视频冒烟通过不代表26段真实视频质量通过；预留组仍需完整验收。
输出是支持类别的未校准分数；空错误列表不是临床CPR合格结论。
仍不输出真实深度、实测按压频率或精确错误起止时间。
