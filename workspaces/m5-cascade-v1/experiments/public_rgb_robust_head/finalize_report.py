"""Archive a rejected optimization honestly; never change the active candidate."""
from datetime import datetime, timezone
import html
import io
from pathlib import Path
import unittest

import optimize_robust as e


def main():
    p = e.o.checked(e.OUTPUT)
    selection = e.o.parent.read(e.OUTPUT/'selection.json')
    comparison = e.o.parent.read(e.OUTPUT/'promotion-comparison.json')
    if comparison['metric_promotion_eligible']:
        raise ValueError('This report writer only archives a rejected experiment')
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).parent), pattern='test_*.py')
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    tests = {'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
             'skipped': len(result.skipped), 'passed': result.wasSuccessful()}
    e.o.write(e.REPORT/'contract-tests.json', tests)
    (e.REPORT/'contract-tests.txt').write_text(stream.getvalue(), encoding='utf-8')
    if not tests['passed']:
        raise AssertionError('Archive contracts failed')
    pointer = e.o.DATA/'current-offline-candidate.json'
    if e.o.parent.digest(pointer) != p['candidate_pointer_before_sha256']:
        raise ValueError('The active candidate unexpectedly changed')
    e.o.parent.check_hashes(p['protected_sources'])
    app_hashes = e.o.parent.read(e.CODE/'test-results/public-head-validation/application-source-before.json')
    for path, expected in app_hashes.items():
        if e.o.parent.digest(e.CODE/path) != expected:
            raise ValueError('Application source changed: '+path)
    integrity = {'passed': True, 'old_protected_files_checked': len(p['protected_sources']),
                 'original_application_files_checked': len(app_hashes), 'candidate_pointer_unchanged': True,
                 'old_models_and_protocols_unchanged': True, 'split_and_public_targets_unchanged': True}
    e.o.write(e.REPORT/'source-integrity.json', integrity)
    old, new = [e.o.parent.read(folder/'evaluation.json') for folder in [e.BASE, e.OUTPUT]]
    minimum = lambda x: min(r['recall'] for r in x['evaluations']['all']['per_label'])
    exact = lambda x: sum(r['exact_match'] for r in x['reserved_per_record'])
    stats = lambda x: {k: {n: v[n] for n in ['records', 'macro_f1', 'correct_false_alarm_rate']}
                       for k, v in x['evaluations'].items()}
    summary = {'created_at': datetime.now(timezone.utc).isoformat(), 'adopted': False,
        'decision': 'Keep passing v4; v5 improves aggregate F1 but regresses reserved F1 and worst-label recall',
        'current_candidate': e.o.parent.read(pointer), 'challenger': selection['selected']['name'],
        'experimental_configurations': 18, 'exact_controls': 1, 'linear_heads_fitted': 6,
        'validation_brier': {k: selection[k]['validation_brier'] for k in ['control', 'selected']},
        'original_gate_checks': new['quality_checks'], 'promotion_checks': comparison['metric_checks'],
        'baseline_metrics': stats(old), 'challenger_metrics': stats(new),
        'minimum_label_recall': {'v4': minimum(old), 'v5': minimum(new)},
        'reserved_exact_matches': {'v4': exact(old), 'v5': exact(new), 'records': 26},
        'tests': tests, 'integrity': integrity, 'candidate_scalar_fields': selection['selected']['scalar_fields'],
        'new_raw_video_smoke_performed': False, 'reason_no_new_raw_smoke': 'Rejected at fixed-feature nonregression gate; no candidate was exported',
        'independent_blind_test': False, 'no_manual_collection_or_annotation': True,
        'main_network_training': False, 'web_agent_integration_changed': False, 'runtime_release_approved': False,
        'boundary': 'Public-keypoint plus frozen-feature metrics do not establish full raw-video performance. Existing v4 has one real-video smoke, not 26-video acceptance.'}
    e.o.write(e.REPORT/'optimization-summary.json', summary)
    text = ('本轮完成18组轻量配置探索，6个线性小分类头拟合；不采纳v5，当前仍为v4。\n'
        'v5整体/单错误/组合错误F1改善，但预留26样本F1及最低类别召回下降，触发事前冻结的不退步规则。\n'
        '17项校验通过；旧模型、原始划分、19个应用源码及当前候选指针未变。\n'
        '本轮未新增真实视频验证，未接入Web/Agent。当前测试为反复使用的工程回归集，不是独立盲测。\n'
        '下一步建议：固定v4完成26段RGB视频验收，区分视频感知误差和分类头误差，再决定是否优化。\n')
    (e.REPORT/'结论.txt').write_text(text, encoding='utf-8')
    def percent(v):
        return f'{v*100:.2f}%'
    metrics = [('全部片段 Macro-F1', 'all'), ('单错误片段 Macro-F1', 'single'),
               ('组合错误片段 Macro-F1', 'composite'), ('26个预留样本 Macro-F1', 'rgb_reserved_public_keypoints')]
    rows = []
    for label, key in metrics:
        a, b = old['evaluations'][key]['macro_f1'], new['evaluations'][key]['macro_f1']
        tone = 'bad' if b < a else 'good'
        rows.append(f'<tr><td>{label}</td><td>{percent(a)}</td><td>{percent(b)}</td><td class="{tone}">{(b-a)*100:+.2f} 个百分点</td></tr>')
    rows.extend([f'<tr><td>最低类别召回率</td><td>{percent(minimum(old))}</td><td>{percent(minimum(new))}</td><td class="bad">下降，不采纳</td></tr>',
                 '<tr><td>正常片段误报</td><td>1 / 20</td><td>1 / 20</td><td>不变</td></tr>',
                 f'<tr><td>预留样本错误向量完全匹配</td><td>{exact(old)} / 26</td><td>{exact(new)} / 26</td><td>不变</td></tr>'])
    candidate_rows = []
    selected_name = selection['selected']['name']
    for row in selection['all_candidates']:
        name = html.escape(row['name'])
        if name == selected_name:
            name += ' · 验证集选中'
        candidate_rows.append(f'<tr><td>{name}</td><td>{row["validation_cohorts"]["single"]["macro_f1"]:.4f}</td><td>{row["validation_cohorts"]["composite"]["macro_f1"]:.4f}</td><td>{row["validation_brier"]:.6f}</td></tr>')
    document = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CPR · 后续优化验证</title><style>
:root{font-family:Inter,"Segoe UI","Microsoft YaHei",sans-serif;color:#18364a;background:#f2f6f8;font-size:15px}*{box-sizing:border-box}
body{margin:0}main{max-width:1080px;margin:0 auto;padding:40px 26px 60px}h1{font-size:32px;line-height:1.3;margin:14px 0}h2{font-size:20px;margin:0 0 16px}h3{font-size:16px;margin:0 0 10px}
p{line-height:1.8;margin:10px 0}header{margin-bottom:26px}.tag{display:inline-block;background:#fff0d5;color:#80530c;font-weight:600;padding:7px 12px;border-radius:8px}
.sub{color:#617987}.card{background:white;border:1px solid #dce6eb;border-radius:15px;padding:24px;margin:18px 0;box-shadow:0 4px 18px #14304204}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.grid .card{margin:0}.notice{border-left:4px solid #d9a34b;background:#fff9ee}
table{border-collapse:collapse;width:100%;font-size:14px;line-height:1.6}th,td{text-align:left;padding:12px 10px;border-bottom:1px solid #e6edf1}th{color:#526e7f;background:#f4f8fa}td:first-child{width:42%}.tablewrap{overflow:auto}
.good{color:#13735b}.bad{color:#b44b3f;font-weight:600}.small{font-size:13px;color:#617987}.flow{display:flex;gap:10px;flex-wrap:wrap;margin-top:16px}.flow span{padding:8px 12px;border-radius:7px;background:#eef5f8}
a{color:#176e99;text-decoration:none;overflow-wrap:anywhere}a:hover{text-decoration:underline}ul{padding-left:20px;line-height:1.9}details summary{cursor:pointer;font-weight:600;padding:4px 0 18px}details table{min-width:640px}footer{font-size:13px;line-height:1.8;color:#617987;margin-top:24px}
@media(max-width:650px){main{padding:25px 14px 40px}h1{font-size:26px}.grid{grid-template-columns:1fr}.card{padding:18px 14px}th,td{padding:10px 7px}table{font-size:12px}.small{font-size:12px}}
@media print{body{background:white}main{max-width:none;padding:0}.card{box-shadow:none;break-inside:avoid}details{display:none}}
</style><main><header><span class="tag">验证结论 · 保留当前 v4</span><h1>有优化空间，尚不适合替换</h1>
<p class="sub">18组轻量配置已验证：候选提高总体识别，但削弱薄弱场景。按事前规则停止替换，演示版本保持稳定。</p></header>
<section class="card"><h2>本轮改了什么</h2><p>给线性分类头加入标准化特征截断，减少极端输入的影响；比较截断范围、正则强度和树／线性融合权重。仅拟合6个小分类头，复用原有几何树与冻结视觉特征。</p>
<div class="flow"><span>738条训练</span><span>196条验证筛选</span><span>只选1组候选</span><span>386条固定回归</span><span>触发退步规则 → 保留 v4</span></div>
<p class="small">无需采集、无需新标注、无需重训视觉主干。全部26个预留样本继续只用于测试。</p></section>
<section class="card"><h2>相同划分下的结果</h2><div class="tablewrap"><table><thead><tr><th>指标</th><th>当前 v4</th><th>试验 v5</th><th>变化</th></tr></thead><tbody>__METRICS__</tbody></table></div>
<p class="small">此表使用公开关键点及冻结特征，不能当作完整原始视频准确率；Macro-F1是各错误类别F1的平均值。</p></section>
<section class="card notice"><h2>为什么不采用总体更高的候选</h2><p>原始验收门槛均通过，但“手位错误”类别召回从67.31%降至63.46%，26个预留样本F1从53.26%降至51.44%。这不符合本轮预先冻结的“不牺牲薄弱项”规则。</p>
<p>验证集选中的配置为 <strong>±6截断、C=0.1、树权重0.75</strong>；其验证集Brier误差降低约1.66%。候选、阈值在固定测试复核前已确定，未在测试后换另一组模型。</p></section>
<div class="grid"><section class="card"><h2>已确认保留</h2><ul><li>当前候选仍是 v4，未改指针。</li><li>原始划分、标签和旧模型完整保留。</li><li>19个应用源码文件哈希未变。</li><li>17项实验／回归校验通过。</li></ul></section>
<section class="card"><h2>接下来最有价值的工作</h2><p>先固定 v4，完成已备好的26段RGB视频验收。对照公开关键点结果，区分视频检测／姿态提取误差和分类头误差，再针对实际瓶颈优化。</p><p class="small">本轮在分类头回归门槛即停止；未新增真实视频测试，未接入Web／Agent，也未改变发布状态。</p></section></div>
<section class="card"><details><summary>展开：全部验证集候选（19组，含固定对照）</summary><div class="tablewrap"><table><thead><tr><th>配置</th><th>单错误F1</th><th>组合错误F1</th><th>Brier ↓</th></tr></thead><tbody>__CANDIDATES__</tbody></table></div></details></section>
<section class="card"><h2>可核查记录</h2><p><a href="optimization-summary.json">结果摘要</a> · <a href="contract-tests.txt">17项校验记录</a> · <a href="source-integrity.json">完整性核对</a> · <a href="../../data/public_cpr/rgb-evaluation-v5/protocol.json">事前冻结协议</a> · <a href="../../data/public_cpr/rgb-evaluation-v5/promotion-comparison.json">不采纳判定</a> · <a href="../../data/public_cpr/current-offline-candidate.json">当前候选指针</a></p></section>
<footer>证据边界：这是历次迭代已查看过的工程回归集，不是独立盲测。v4已有1段真实视频冒烟验证，26段完整视频验收仍待完成。正常片段仅20条，不能据此保证真实使用中的误报率。</footer></main></html>'''
    document = document.replace('__METRICS__', ''.join(rows)).replace('__CANDIDATES__', ''.join(candidate_rows))
    (e.REPORT/'后续优化验证.html').write_text(document, encoding='utf-8')
    print(text, flush=True)


if __name__ == '__main__':
    main()
