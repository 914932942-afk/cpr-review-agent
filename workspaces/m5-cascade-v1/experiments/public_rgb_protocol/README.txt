第二项：公开RGB录像组隔离与小分类头重拟合

用途
为后续真实视频验收排除训练/调参重叠。公开数据和标签沿用第一项，不自行采集、不新增人工标签。
仅使用现有公开骨架的冻结特征，拟合一个已有固定配置的小分类头；不训练视觉主干。
本实验不是原论文零样本组合测试，不宣称独立盲测或跨人测试。

划分
把第一项全部26段录像组固定为测试，完整保留原364段测试记录。
20段移出训练、2段移出验证、4段原本为测试；得到训练738、验证196、测试386。
同一录像的所有视角必须保持同一划分，实际仍只使用ch0。
阈值只由196段验证样本选择，测试结果不得用于回调阈值或挑选模型。

位置
新协议、分类头和离线诊断清单：CODE/data/public_cpr/rgb-evaluation-v1/
报告与日志：CODE/test-results/public-rgb-protocol-v1/
最终效果结论以该目录“第二项验收报告.html”和step2-summary.json为准。
旧improvement-v2、旧inference-manifest.json、原RGB清单和正式网页/Agent未修改。
新清单只供离线诊断，未通过效果门槛时不得当作已发布能力使用。

复现（在本目录，用原ML环境，不是Web的.venv）
python rgb_protocol.py prepare
python rgb_protocol.py train
python rgb_protocol.py evaluate
python rgb_protocol.py export
python -m unittest test_protocol test_fitted_contract -v

以上阶段已完成后会拒绝覆盖。需要新实验必须另用新的--output目录并明确记录其与当前已暴露测试的关系。
合同测试及报告由本次执行保存；finalize_report.py不会覆盖既有报告。

当前边界
数据源为CPR-Coach公开原标签；支持1,2,3,4,5,6,7,8,9,11,13，深度类10/12不输出。
正常公开标签0保留，空错误预测不代表临床动作合格。
26段中含补充集全部24段可用ch0记录；全部保留为测试后，训练/验证中没有该来源录像。
来源分组分析只用于定位问题，不替换原测试分母、门槛或失败结论。
所有效果指标来自公开骨架特征，本轮未运行26段MP4的端到端识别。
本次小分类头基线不算五个模块的独立算法创新已经完成。
新模型未替换旧模型，未接入正式Agent工具。
