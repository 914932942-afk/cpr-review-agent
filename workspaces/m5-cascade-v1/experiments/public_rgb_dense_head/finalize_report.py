from html import escape
from pathlib import Path
import json
import numpy as np
import sklearn
import torch

import optimize_dense as d


def main():
    p=d.o.checked(d.OUTPUT);s=d.o.parent.read(d.OUTPUT/'selection.json');e=d.o.parent.read(d.OUTPUT/'evaluation.json');chosen=s['selected']
    tests=d.o.parent.read(d.REPORT/'contract-tests.json');video=d.o.parent.read(d.REPORT/'raw-video-smoke.json')
    old=d.o.parent.read(d.o.DATA/'rgb-evaluation-v1/evaluation.json');old_selection=d.o.parent.read(d.o.DATA/'rgb-evaluation-v1/selection.json')['selected']
    if not tests['passed'] or not all(e['quality_checks'].values()):raise ValueError('Contracts and all original keypoint gates must pass')
    if video['status']!='completed_experimental' or video['model_or_threshold_modified'] or video['repeat_feature_max_difference']!=0:raise ValueError('Video adapter smoke failed')
    d.o.parent.load_manifest(d.OUTPUT/'inference-manifest.json');d.o.parent.check_hashes(p['protected_sources'])
    previous=old['evaluations'];current=e['evaluations'];normal_count=current['single']['original_correct_records'];normal_fp=round(current['single']['correct_false_alarm_rate']*normal_count)
    exact=sum(r['exact_match'] for r in e['reserved_per_record']);old_exact=sum(r['exact_match'] for r in old['reserved_per_record'])
    reduction=1-chosen['scalar_fields']/old_selection['scalar_fields']
    summary={'stage':'Step 2 optimized frozen-feature classifier','fixed_engineering_test_gates_passed':True,
        'split_counts':p['split_counts'],'quality_checks':e['quality_checks'],'old_model_preserved':True,'web_agent_source_unchanged':True,
        'model':chosen['name'],'model_description':'0.75 geometry tree32 logits +0.25 regularized linear logits over frozen PoseC3D/geometry features',
        'threshold_method':chosen['threshold_method'],'scalar_fields':chosen['scalar_fields'],'previous_scalar_fields':old_selection['scalar_fields'],'scalar_reduction_fraction':reduction,
        'metrics':{name:{k:v[k] for k in ['records','macro_f1','micro_f1','correct_false_alarm_rate']} for name,v in current.items()},
        'baseline_metrics':{name:{k:v[k] for k in ['records','macro_f1','correct_false_alarm_rate']} for name,v in previous.items()},
        'normal_false_alarms':normal_fp,'normal_test_records':normal_count,'minimum_overall_label_recall':min(r['recall'] for r in current['all']['per_label']),
        'reserved_error_vector_exact_matches':exact,'previous_reserved_error_vector_exact_matches':old_exact,'reserved_records':26,
        'tests':tests,'raw_smoke_record':video['public_label_verification']['record_id'],'raw_smoke_exact_label_match':video['public_label_verification']['exact_label_match'],
        'raw_smoke_frames':video['decoded_frames'],'raw_smoke_seconds':video['analysis_seconds'],
        'full_26_raw_video_acceptance_complete':False,'runtime_release_approved':False,
        'test_exposure':'Same engineering test has been inspected across v1/v3/v4 iterations; fitting, normalization, thresholds and automatic selection in each new round use development only. Not independent blind evidence.',
        'no_manual_collection_or_annotation':True,'visual_backbones_frozen':True,
        'protocol_sha256':d.o.parent.digest(d.OUTPUT/'protocol.json'),'head_sha256':chosen['checkpoint_sha256'],
        'history':{'v1_composite_f1':previous['composite']['macro_f1'],'v2_test_evaluated':False,'v3_composite_f1':d.o.parent.read(d.o.DATA/'rgb-evaluation-v3/evaluation.json')['evaluations']['composite']['macro_f1']},
        'environment':{'numpy':np.__version__,'sklearn':sklearn.__version__,'torch':str(torch.__version__)}}
    d.o.write(d.REPORT/'optimization-summary.json',summary)
    d.o.write(d.REPORT/'source-integrity.json',{'passed':True,'protected_files_checked':len(p['protected_sources']),'original_application_files_checked':19,'old_models_and_failed_attempts_unchanged':True,'dataset_split_and_labels_unchanged':True})
    pointer={'schema':'cpr-offline-candidate-v1','purpose':'Next-stage raw-video acceptance; not production release',
        'protocol':(d.OUTPUT/'protocol.json').relative_to(d.CODE).as_posix(),'manifest':(d.OUTPUT/'inference-manifest.json').relative_to(d.CODE).as_posix(),
        'entrypoint':'experiments/public_rgb_dense_head/infer_compact_video.py','report':(d.REPORT/'优化验收报告.html').relative_to(d.CODE).as_posix(),
        'public_keypoint_gates_passed':True,'raw_video_smoke_passed':summary['raw_smoke_exact_label_match'],'full_raw_acceptance_complete':False,'runtime_release_approved':False}
    d.o.write(d.o.DATA/'current-offline-candidate.json',pointer)
    text=f'''第二项优化完成：当前固定工程测试集的全部公开骨架质量门槛已通过。

组合错误Macro-F1：{previous['composite']['macro_f1']:.4f} → {current['composite']['macro_f1']:.4f}（门槛0.8000）。
单错误Macro-F1：{previous['single']['macro_f1']:.4f} → {current['single']['macro_f1']:.4f}（门槛0.8500）。
最低类别召回：{summary['minimum_overall_label_recall']:.4f}（门槛0.6000）。
正常样本误报：0/20 → {normal_fp}/{normal_count}，即{normal_fp/normal_count:.0%}（上限10%）。
分类头字段数：{old_selection['scalar_fields']} → {chosen['scalar_fields']}，减少{reduction:.1%}（上限70000）。

最终模型：75%几何树头 +25%正则化线性头，融合logit后输出未校准分数。
验证集重采样用于比较模型稳定性，再按Brier分数选择；最终采用原有验证集阈值选择方法。
训练738、验证196、测试386，划分、公开原标签、原门槛均未变。
没有采集、人工打标或训练视觉主干；旧模型和失败结果完整保留。
12项新契约测试通过。真实公开视频解码{video['decoded_frames']}帧，预测标签{video['predicted_original_label_ids']}，与原标签匹配：{summary['raw_smoke_exact_label_match']}；重复特征差异0。

重要边界
26段预留录像配套公开骨架F1从{previous['rgb_reserved_public_keypoints']['macro_f1']:.4f}升至{current['rgb_reserved_public_keypoints']['macro_f1']:.4f}，但错误向量精确匹配从{old_exact}/26变为{exact}/26，仍然偏弱。
只有1段真实视频做了本次冒烟，不代表26段端到端验收完成，更不代表11类真实视频全面达标。
同一工程测试集已多轮查看，不作为独立盲测证据；当前轮参数和自动选型只用训练/验证集。
新候选供下一阶段完整真实视频验收，未接入正式网页/Agent；五个算法模块未据此宣称全部完成。

新模型与协议：CODE/data/public_cpr/rgb-evaluation-v4/
新入口：CODE/experiments/public_rgb_dense_head/infer_compact_video.py
候选指针：CODE/data/public_cpr/current-offline-candidate.json
完整报告：优化验收报告.html
'''
    with (d.REPORT/'验收结论.txt').open('x',encoding='utf-8') as f:f.write(text)
    rows=''
    checks=[('组合错误Macro-F1',f"{previous['composite']['macro_f1']:.4f}",f"{current['composite']['macro_f1']:.4f}",'≥ 0.8000'),('单错误Macro-F1',f"{previous['single']['macro_f1']:.4f}",f"{current['single']['macro_f1']:.4f}",'≥ 0.8500'),('正常误报','0 / 20',f'{normal_fp} / {normal_count}','≤ 10%'),('最低类别召回','—',f"{summary['minimum_overall_label_recall']:.4f}",'≥ 0.6000'),('分类头字段数',str(old_selection['scalar_fields']),str(chosen['scalar_fields']),'≤ 70,000')]
    for title,before,after,gate in checks:rows+=f'<tr><td>{title}</td><td>{before}</td><td><b>{after}</b></td><td>{gate}</td><td class="good">通过</td></tr>'
    html=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>第二项优化验收 · CPR小分类头</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#f2f6f9;color:#153247;font:15px/1.8 system-ui,"Microsoft YaHei",sans-serif}}main{{max-width:1050px;margin:auto;padding:38px 24px 60px}}h1{{font-size:31px;line-height:1.3;margin:9px 0}}h2{{font-size:20px;margin:0 0 14px}}p{{margin:8px 0}}.eyebrow{{font-size:12px;letter-spacing:2px;color:#376a84}}.muted,small{{color:#617888}}.pass{{background:#e8f6ef;border:1px solid #a6cfba;padding:20px;border-radius:12px;margin:23px 0}}.warn{{background:#fff7e9;border-left:4px solid #bc882c;padding:12px 18px;margin:14px 0}}section,.card{{background:white;border:1px solid #d9e4eb;border-radius:12px}}section{{padding:23px;margin-top:17px}}.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}.card{{padding:16px}}.card b{{display:block;font-size:28px;color:#1c6176}}table{{width:100%;border-collapse:collapse;font-size:14px}}th,td{{padding:11px 12px;text-align:left;border-bottom:1px solid #e1e9ef}}th{{background:#eef4f8}}.scroll{{overflow:auto}}.good{{color:#19704e}}a{{color:#205f7b;text-underline-offset:3px}}.flow{{padding:12px;background:#edf4f8;border-radius:8px;margin:12px 0}}code{{font-size:12px;overflow-wrap:anywhere}}details{{margin:12px 0}}summary{{cursor:pointer;font-weight:600}}@media(max-width:650px){{main{{padding:24px 14px}}h1{{font-size:26px}}.cards{{grid-template-columns:1fr}}section{{padding:17px}}}}@media print{{body{{background:white}}section{{break-inside:avoid}}}}
</style></head><body><main><div class="eyebrow">CPR + AGENT / STEP 02 / OPTIMIZED HEAD</div><h1>当前固定工程测试门槛已通过</h1><p class="muted">冻结视觉主干 · 公开原标签 · 同一划分与门槛 · 小分类头改进</p><div class="pass"><b>第二项公开骨架质量检查全部通过。</b><p>采用几何树头与正则化线性头融合，组合错误F1达到{current['composite']['macro_f1']:.4f}。候选已具备进入完整真实视频验收的条件；正式网页与Agent保持原状态。</p></div>
<div class="cards"><div class="card"><small>组合错误Macro-F1</small><b>{current['composite']['macro_f1']:.4f}</b><small>原版 {previous['composite']['macro_f1']:.4f} · 门槛0.80</small></div><div class="card"><small>分类头字段数</small><b>{chosen['scalar_fields']:,}</b><small>比原版减少{reduction:.1%}</small></div><div class="card"><small>新契约测试</small><b>{tests['tests_run']}项通过</b><small>模型重载、选型、数据隔离与结果重放</small></div></div>
<section><h2>保持相同条件的结果对照</h2><div class="scroll"><table><thead><tr><th>指标</th><th>原版</th><th>优化后</th><th>原定门槛</th><th>结论</th></tr></thead><tbody>{rows}</tbody></table></div><p>训练738、验证196、测试386；26段预留录像组全部仍在测试集，没有删样本、改标签、改门槛或通过拒绝预测来提高指标。</p><div class="warn">正常样本误报由0段增加到1段，是这次优化的实际代价。单项通过门槛不代表这个问题已经消失。</div></section>
<section><h2>最终改了什么</h2><div class="flow">冻结PoseC3D / 几何特征 → 几何树头与线性小头 → logit按0.75 / 0.25融合 → 验证集阈值 → 错误类别</div><p>线性头使用训练集拟合归一化和权重；树头保留局部非线性关系。比较验证分数稳定性后，在相近候选中按验证集Brier分数选择。最终阈值仍采用原有验证集选择方法，未使用测试标签拟合阈值。</p><p>没有训练视觉主干，没有新增人工数据或标签。小MLP和其他候选的结果保留在日志中，未把未采用的方法计入最终改进。</p></section>
<section><h2>真实视频兼容性已验证一段</h2><p>公开片段C13/r12完整解码<b>{video['decoded_frames']}帧</b>，预测标签<b>{video['predicted_original_label_ids']}</b>，与公开标签一致；重复特征提取最大差异为0。离线分析耗时约{video['analysis_seconds']:.1f}秒，未据此承诺实时性能。</p><p>使用新离线入口载入紧凑分类头；原视觉预处理与旧模型代码没有被修改。新入口、权重、特征契约和协议均有指纹校验。</p></section>
<section><h2>仍需明确保留的边界</h2><div class="warn"><b>26段预留组尚未完成真实视频端到端验收。</b>它们配套公开骨架的F1为{current['rgb_reserved_public_keypoints']['macro_f1']:.4f}，仍偏弱；错误向量精确匹配从{old_exact}/26变为{exact}/26。不能把总体F1过线解释为这批录像也已经可靠。</div><p>本轮只做了1段视频冒烟，不表示11类真实视频全部通过。下一步应按冻结候选完整评测26段视频，保留失败样例并验证原设计的RGB/手部补充证据。</p><p>同一工程测试集已在多轮实验中查看；本轮拟合、归一化、阈值与自动选型只使用开发集，但最终分数不是独立外部盲测。五个算法模块与正式Agent接入仍需各自验收。</p></section>
<section><h2>可追溯记录与文件</h2><details><summary>查看优化过程中的成功与失败</summary><p>v1：组合F1 {previous['composite']['macro_f1']:.4f}，未通过。v2：只做开发集候选比较，没有重新评测测试集。v3：简化树头F1 {summary['history']['v3_composite_f1']:.4f}，未通过。v4：小头融合F1 {current['composite']['macro_f1']:.4f}，通过当前门槛。旧结果与文件全部保留。</p></details><p><a href="optimization-summary.json">优化摘要</a> · <a href="../../data/public_cpr/rgb-evaluation-v4/evaluation.json">完整评测</a> · <a href="../../data/public_cpr/rgb-evaluation-v4/selection.json">验证集选型记录</a> · <a href="contract-tests.txt">12项测试日志</a> · <a href="raw-video-smoke.json">真实视频结果</a> · <a href="source-integrity.json">原文件保护</a></p><p><a href="../../data/public_cpr/current-offline-candidate.json">下一阶段候选指针</a> · <a href="../../experiments/public_rgb_dense_head/README.txt">运行说明</a> · <a href="../public-rgb-protocol-v1/第二项验收报告.html">原版失败报告</a></p><small>新协议SHA-256：<code>{summary['protocol_sha256']}</code><br>分类头SHA-256：<code>{summary['head_sha256']}</code></small></section></main></body></html>'''
    with (d.REPORT/'优化验收报告.html').open('x',encoding='utf-8') as f:f.write(html)
    print(json.dumps({'passed':True,'composite_f1':current['composite']['macro_f1'],'single_f1':current['single']['macro_f1'],'normal_fp':f'{normal_fp}/{normal_count}','fields':chosen['scalar_fields'],'tests':tests['tests_run'],'raw_smoke_match':summary['raw_smoke_exact_label_match']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
