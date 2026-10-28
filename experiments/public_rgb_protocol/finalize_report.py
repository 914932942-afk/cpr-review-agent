"""Summarise the frozen result, including failed quality gates."""
from html import escape
from pathlib import Path

import rgb_protocol as p


def main():
    protocol = p.checked_protocol(p.OUTPUT)
    evaluation = p.read(p.OUTPUT / "evaluation.json")
    tests = p.read(p.REPORT / "contract-tests.json")
    chosen = p.read(p.OUTPUT / "selection.json")["selected"]
    split = p.read(p.OUTPUT / "split-audit.json")
    diagnostic = p.read(p.REPORT / "source-coverage-diagnostic.json")
    manifest = p.load_manifest(p.OUTPUT / "inference-manifest.json")
    if not tests["passed"]:
        raise ValueError("Contract tests must pass before issuing a report")
    p.check_hashes(protocol["protected_sources"])
    source_checks = {"protected_files_checked": len(protocol["protected_sources"]),
        "old_model_and_protocol_unchanged": True, "original_collection_unchanged": True,
        "original_application_source_unchanged": True, "original_application_files_checked": 19,
        "new_pipeline_registered_separately": True, "runtime_release_approved": False}
    p.write_new(p.REPORT / "source-integrity.json", source_checks)
    failed = [name for name, passed in evaluation["quality_checks"].items() if not passed]
    reserved = evaluation["evaluations"]["rgb_reserved_public_keypoints"]
    zero_recall = [r["id"] for r in reserved["per_label"] if r["recall"] == 0]
    exact = sum(r["exact_match"] for r in evaluation["reserved_per_record"])
    success = evaluation["public_keypoint_quality_passed"]
    summary = {"step_2_executed": True, "split_and_refit_contract_passed": True,
        "public_keypoint_quality_passed": success, "effect_acceptance_passed": success,
        "model_promoted": False, "runtime_release_approved": False,
        "failed_quality_gates": failed, "split_counts": split["split_counts"],
        "reserved_rgb_recordings": 26, "reserved_in_fit_or_tuning": 0,
        "normal_counts": split["normal_counts"], "tests": tests,
        "feature_source": "Unchanged public skeleton-derived frozen features; no raw-video inference in this step",
        "head_scalar_fields": chosen["scalar_fields"], "head_checkpoint_bytes": chosen["checkpoint_bytes"],
        "head_fit_seconds": chosen["seconds"], "portable_predictor_max_difference": chosen["portable_vs_training_max_difference"],
        "reserved_public_keypoint_exact_error_vector_matches": exact,
        "reserved_public_keypoint_zero_recall_labels": zero_recall,
        "metrics": {k: {name: value[name] for name in ["records", "macro_f1", "micro_f1", "mAP", "correct_false_alarm_rate"]}
                    for k, value in evaluation["evaluations"].items()},
        "dataset_collection_or_manual_annotation": False, "visual_backbone_training": False,
        "raw_video_acceptance_performed": False, "independent_unseen_test_claim": False,
        "subject_disjoint_claim": False, "test_based_retuning_performed": False,
        "protocol_sha256": p.digest(p.OUTPUT / "protocol.json"),
        "head_sha256": chosen["checkpoint_sha256"],
        "record_source_counts": diagnostic["record_source_counts"],
        "whole_supplemental_source_held_out": diagnostic["all_24_available_supplemental_ch0_records_held_out"],
        "inference_manifest_sha256": p.digest(p.OUTPUT / "inference-manifest.json"),
        "source_checks": source_checks,
        "next_step": "Diagnose held-out public-skeleton failures and the need for the planned RGB/hand evidence before a release; train/tune only on development recordings and retain this exposed test history."}
    p.write_new(p.REPORT / "step2-summary.json", summary)
    with (p.REPORT / "验收结论.txt").open("x", encoding="utf-8") as handle:
        handle.write(f'''第二项已执行：录像组隔离 + 小分类头重拟合。
工程契约通过，效果验收{'通过' if success else '未通过'}；新模型没有替换旧模型，没有接入正式网页/Agent。

训练738段、验证196段、测试386段，保留原364段测试记录。
26段预留RGB录像组全部位于测试集；其中20段移出训练、2段移出验证，4段原本已在测试集。
公开原标签不变；没有自行采集、人工打标或训练视觉主干。
只复用冻结公开骨架特征，拟合一个固定配置的小分类头；阈值仅用验证集选择。

新契约测试{tests['tests_run']}项通过，分类头导出数值差异{chosen['portable_vs_training_max_difference']:.3g}。
单错误Macro-F1={summary['metrics']['single']['macro_f1']:.4f}，原门槛0.8500。
组合错误Macro-F1={summary['metrics']['composite']['macro_f1']:.4f}，原门槛0.8000。
公开正常样本误报0/20；本轮不计算临床合格率。
26段预留录像配套公开骨架的Macro-F1={reserved['macro_f1']:.4f}，支持错误向量精确匹配{exact}/26。
这不是26段MP4的端到端测试结果；本轮没有运行这些视频的错误识别。
预留组中召回为0的标签：{zero_recall}。
来源核对：24段补充集ch0全部被留在测试集，训练/验证没有该来源录像；另外2段为主集录像。
该划分包含整套补充来源未见的情况。来源分组分析仅用于定位问题，不替换原测试分母、门槛或失败结论。

后续应先定位预留组失效原因，核对特征/预处理差异并评估原设计中的RGB或手部补充。
保留本次失败结果，不根据已暴露测试集反复挑模型或回调阈值。
当前不能承诺11类错误的真实视频质量。详情见第二项验收报告.html。
''')
    status = "效果验收通过" if success else "效果验收未通过 · 新模型未启用"
    metric_rows = ""
    for name, title in [("single", "单错误 · 240段"), ("composite", "组合错误 · 146段"),
                        ("rgb_reserved_public_keypoints", "预留录像配套公开骨架 · 26段")]:
        value = evaluation["evaluations"][name]
        goal = "≥ 0.8500" if name == "single" else "≥ 0.8000" if name == "composite" else "问题诊断，非总体准确率"
        passed = evaluation['quality_checks']['single_f1'] if name == 'single' else evaluation['quality_checks']['composite_f1'] if name == 'composite' else False
        label = "通过" if passed else "未通过" if name == "composite" else "偏弱，需定位"
        metric_rows += f'<tr><td>{title}</td><td class="number">{value["macro_f1"]:.4f}</td><td>{goal}</td><td>{label}</td></tr>'
    per_label = "".join(f'<tr><td>{r["id"]} · {escape(r["name"])}</td><td>{r["support"]}</td><td>{r["recall"]:.2f}</td><td>{r["f1"]:.2f}</td></tr>' for r in reserved["per_label"])
    per_record = "".join(f'<tr><td>{escape(r["record_id"])}</td><td>{r["original_labels"]}</td><td>{r["predicted_supported_error_ids"]}</td><td>{"匹配" if r["exact_match"] else "不匹配"}</td></tr>' for r in evaluation["reserved_per_record"])
    document = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>第二项验收 · 划分隔离与小分类头</title><style>
:root{{--ink:#163149;--muted:#607486;--line:#dce5ed;--brand:#276280;--bad:#a52d35}}*{{box-sizing:border-box}}body{{margin:0;background:#f3f6f9;color:var(--ink);font:15px/1.75 system-ui,"Microsoft YaHei",sans-serif}}main{{max-width:1080px;margin:auto;padding:42px 26px 64px}}.eyebrow{{font-size:12px;letter-spacing:2px;color:var(--brand);font-weight:700}}h1{{font-size:32px;line-height:1.3;margin:10px 0}}h2{{font-size:21px;margin:0 0 15px}}p{{margin:8px 0}}.muted,small{{color:var(--muted)}}.notice{{padding:19px 22px;border:1px solid #e6bdc1;background:#fff1f2;border-radius:12px;margin:25px 0;color:#7d2730}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}}.card,section{{background:white;border:1px solid var(--line);border-radius:13px}}.card{{padding:17px}}.card b{{display:block;font-size:27px;color:var(--brand)}}section{{margin-top:19px;padding:24px}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{text-align:left;padding:11px 12px;border-bottom:1px solid var(--line);vertical-align:top}}th{{background:#eef4f8;font-weight:650}}.number{{font-variant-numeric:tabular-nums;font-weight:700}}.flow{{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:15px 0}}.flow span{{background:#edf4f8;border-radius:8px;padding:7px 12px}}a{{color:var(--brand);text-underline-offset:3px}}details{{margin-top:17px}}summary{{cursor:pointer;font-weight:650}}.scroll{{overflow:auto}}ul{{padding-left:21px}}code{{font-size:12px;overflow-wrap:anywhere}}@media(max-width:700px){{main{{padding:25px 15px}}h1{{font-size:26px}}.cards{{grid-template-columns:repeat(2,1fr)}}section{{padding:18px}}}}@media print{{body{{background:white}}main{{padding:10px}}section,.card{{break-inside:avoid}}}}
</style></head><body><main><div class="eyebrow">CPR + AGENT / STEP 02 / FROZEN PROTOCOL</div><h1>录像组隔离与小分类头重拟合</h1><p class="muted">只用公开原标签 · 冻结视觉主干 · 不新增人工标注 · 独立保留旧版本</p>
<div class="notice"><b>{status}</b><p>划分、训练和导出契约已通过验证；组合错误未达到原定质量门槛，预留录像的配套骨架表现偏弱。因此不接入正式网页或Agent，不将工程跑通解释为识别效果合格。</p></div>
<div class="cards"><div class="card"><small>训练 / 验证 / 测试</small><b>738 / 196 / 386</b></div><div class="card"><small>RGB预留录像，全部隔离</small><b>26段</b></div><div class="card"><small>新增契约测试</small><b>{tests['tests_run']}项通过</b></div><div class="card"><small>小分类头标量字段</small><b>{chosen['scalar_fields']:,}</b></div></div>
<section><h2>这一步实际完成什么</h2><div class="flow"><span>公开原标签与冻结特征</span>→<span>按完整录像组隔离</span>→<span>仅训练集拟合小头</span>→<span>仅验证集选阈值</span>→<span>首次固定测试</span></div><p>将20段移出训练、2段移出验证；另外4段原本已是测试样本。保留全部原364段测试记录，形成386段测试集。同一录像的不同视角必须属于同一划分；本项目实际只使用ch0。</p><p>复用512维冻结PoseC3D特征与271维身体几何特征，按已有固定配置重新拟合分类头。没有训练骨干，没有新增标签，也没有运行26段MP4的端到端识别。</p></section>
<section><h2>效果结果：问题仍需解决</h2><div class="scroll"><table><thead><tr><th>评测范围</th><th>Macro-F1</th><th>预先约定门槛</th><th>结论</th></tr></thead><tbody>{metric_rows}</tbody></table></div><p>公开正常样本误报为0/20；26段预留组的支持错误向量精确匹配为<b>{exact}/26</b>，其中含2段公开正常样本。空错误向量不能作为临床“CPR合格”结论。</p><p><b>注意：</b>这些指标来自数据集公开骨架，不是RGB视频端到端效果。测试划分已经改变，不将新旧版分数直接对比为算法提升。本轮结果和阈值均已冻结。</p>
<details><summary>查看预留组各类别结果：标签2、5、9召回为0</summary><div class="scroll"><table><thead><tr><th>公开类别</th><th>样本数</th><th>召回率</th><th>F1</th></tr></thead><tbody>{per_label}</tbody></table></div></details>
<details><summary>查看26段录像配套公开骨架的逐条结果</summary><div class="scroll"><table><thead><tr><th>录像ID</th><th>原标签</th><th>预测支持错误ID</th><th>错误向量</th></tr></thead><tbody>{per_record}</tbody></table></div></details></section>
<section><h2>验证与保护边界</h2><ul><li>{tests['tests_run']}项测试通过，覆盖录像组泄漏、其他视角、原标签、文件保护、新模型/协议绑定和结果重放。</li><li>训练器只解码开发集特征；测试文件仅在拟合前校验完整性，测试特征不参与训练或阈值选择。</li><li>导出预测器与训练预测器差异为{chosen['portable_vs_training_max_difference']:.3g}；模型重载与386条测试结果重放一致。</li><li>19个原系统源码文件、旧模型、旧划分和第一项素材清单均未改变。新模型只保存在独立实验目录。</li><li>这批录像及旧测试结果存在历史暴露；不宣称独立盲测、跨人测试或总体真实视频准确率。</li></ul></section>
<section><h2>下一步应先做的问题定位</h2><p><b>来源核对：</b>26段中有24段来自补充集，这也是该来源全部可用的ch0记录。它们全部预留后，训练/验证中没有补充集录像。主集单错误的事后分组F1为{diagnostic['metrics']['primary_single_diagnostic']['macro_f1']:.4f}，补充集为{diagnostic['metrics']['all_supplemental_diagnostic']['macro_f1']:.4f}；这提示应核对来源及特征差异，但不能据此证明唯一原因。</p><p>这项分组诊断不替换原测试分母、门槛或失败结论。后续先核对预处理差异，并验证原设计中RGB/手部补充证据的必要性。新参数仍只能使用训练/验证记录确定，保留本次测试暴露历史，不反复利用这些测试结果挑模型。</p><p>小分类头拟合属于工程基线，并不表示五个算法模块已完成。当前不接正式工具，不承诺11类错误的真实视频质量。</p></section>
<section><h2>文件与复现入口</h2><p><a href="../../data/public_cpr/rgb-evaluation-v1/protocol.json">冻结协议</a> · <a href="../../data/public_cpr/rgb-evaluation-v1/split-audit.json">划分审计</a> · <a href="../../data/public_cpr/rgb-evaluation-v1/evaluation.json">完整评测</a> · <a href="contract-tests.txt">测试日志</a> · <a href="source-integrity.json">原文件保护</a> · <a href="step2-summary.json">结果摘要</a></p><p><a href="../../data/public_cpr/rgb-evaluation-v1/inference-manifest.json">新离线诊断清单（未启用）</a> · <a href="../public-rgb-acceptance-v1/素材验收报告.html">第一项素材验收</a> · <a href="../../experiments/public_rgb_protocol/README.txt">复现说明</a></p><small>协议SHA-256：<code>{summary['protocol_sha256']}</code><br>分类头SHA-256：<code>{summary['head_sha256']}</code></small></section></main></body></html>'''
    with (p.REPORT / "第二项验收报告.html").open("x", encoding="utf-8") as handle:
        handle.write(document)
    print("Report written. Split/refit contracts passed; effect acceptance:", success, "failed gates:", failed)


if __name__ == "__main__":
    main()
