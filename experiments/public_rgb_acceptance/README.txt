第一项：公开RGB验收素材准备（2026-10-21）

本步骤只做数据获取、原标签核对、媒体完整性验证和清单冻结。
不训练模型，不调整阈值，不运行错误识别，不修改现有划分或正式Agent工具。

范围
从CPR-Coach公开原标注选定ch0视角：正常动作与现有11类单错误，每类2段，共24段。
加上已取得的2段主数据集C13录像，总计26段、11,215帧。
其中新增22段；原有4段MP4按字节复用，保持之前已记录的视频指纹。
标签直接取公开ann.tar.gz，与已验证的公开骨架记录交叉核对；没有自行采集或新增人工标注。

数据获取
SupC14_Dataset.tar.gz固定版本 b8667b8fd6a70ac63cc09fc6b763a9459b7c160a。
该发布包9,683,930,812字节，采用分段续传；全部字节需匹配公开整包SHA-256。
只解出清单内的ch0 RGB图片；保留原始JPEG和逐帧哈希，不解出其他视角、光流及深度类10/12。
2段已有主集录像保留其固定Parquet版本、原标签来源和逐帧哈希；没有下载或声称核验其完整大包。

每段核对
1. 原标签、录像编号、视角和公开预期帧数。
2. 原始图片名称连续、无重名、JPEG结构与分辨率一致。
3. MP4完整解码后的帧数、分辨率、25fps时间基准与记录一致。
4. 原始JPEG、派生MP4和清单的SHA-256。
新增MP4使用H.264 CRF18，属于有损导出；精确原始JPEG仍保留，便于后续模型输入核验。
视频时间基准来自数据集公布的25fps；不将其解释为按压频率。

当前模型关系
20段录像的骨架已用于当前分类头训练，2段用于调参，另4段原测试录像的视频结果此前已查看。
因此清单中没有声称“独立、从未查看的新测试集”。本批当前用于工程回归和第二项的划分准备。
第二项需先将未来测试录像从训练和调参中排除，再拟合小分类头；所有视角按同一录像分组。
本步骤不执行这个重划分，也不重拟合分类头。
本批含正常和单错误RGB，没有新增组合错误RGB、错误精确时刻、真实按压深度或受试者身份标注。
每类仅少量样例，不能用来宣称总体准确率、跨人或跨机位泛化。

位置
数据：CODE/data/public_cpr/rgb-acceptance-v1/
计划：frozen-record-plan.json（后续不可静默改写）
数据：records/<录像编号>/images/、video.mp4、manifest.json
来源校验：source-archive-integrity.json
最终清单：collection-manifest.json
报告：CODE/test-results/public-rgb-acceptance-v1/素材验收报告.html

可复现命令（在本目录，使用现有ML Python环境）
python prepare_plan.py
python download_archive.py --workers 12 --reuse-cache <可选的旧分段缓存目录>
python build_media.py extract
python build_media.py videos --workers 2
python -m unittest test_curation -v
python finalize_collection.py
计划已存在时prepare_plan会拒绝覆盖；下载和已完成的视频导出支持恢复。
下一项应建立新的评测/推理协议，不覆盖现有离线模型的inference-manifest.json。

来源
https://huggingface.co/datasets/ShunliWang/CPR-Coach
https://github.com/Shunli-Wang/CPR-Coach
