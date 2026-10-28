"""Post-run public-label scoring. No fitting or test-time adaptation."""
from datetime import datetime, timezone
import html
from pathlib import Path
import sys

import numpy as np

from common import CODE, DATA, MODEL, OUTPUT, REPORT, read, digest, write_new, verify_lock, validate_result

sys.path.insert(0, str(CODE/'experiments/public_rgb_dense_head'))
from compact_head import load
from improve import predict
from baseline import metrics, LABEL_IDS, LABEL_NAMES
from geometry import describe


def scored(probability, targets, predictions):
    result = metrics(predictions.astype(float), targets)
    ranked = metrics(probability[:, :11], targets)
    result['mAP'] = ranked['mAP']
    for label, rank in zip(result['per_label'], ranked['per_label']):
        label['average_precision'] = rank['average_precision']
    return result


def quality_checks(protocol, measured, completed, total):
    gates = protocol['quality_gates']
    return {'full_classification_coverage': completed/total >= gates['classification_coverage'],
        'single_macro_f1': measured is not None and measured['macro_f1'] >= gates['single_macro_f1'],
        'correct_false_alarm': measured is not None and measured['correct_false_alarm_rate'] is not None
                               and measured['correct_false_alarm_rate'] <= gates['correct_false_alarm_rate'],
        'minimum_label_recall': measured is not None and min(r['recall'] for r in measured['per_label']) >= gates['minimum_label_recall']}


def main():
    if (OUTPUT/'evaluation.json').exists():
        raise FileExistsError('Preserve the first raw-video evaluation')
    protocol, lock = verify_lock(include_videos=True)
    scoring_lock = read(OUTPUT/'scoring-lock.json')
    for source, expected in scoring_lock['implementation_sources'].items():
        if digest(CODE/source) != expected:
            raise ValueError('Frozen scoring source changed: '+source)
    batch = read(OUTPUT/'batch-complete.json')
    jobs = read(OUTPUT/'jobs.json')
    targets = {r['record_id']: r for r in read(OUTPUT/'targets.json')}
    if batch['attempts'] != len(jobs):
        raise ValueError('Batch incomplete; do not score a convenient subset')
    model, state = load(CODE/lock['head'], read(MODEL/'protocol.json')['feature_profile_hash'], digest(MODEL/'protocol.json'))
    if state['operating_point'] != lock['operating_point']:
        raise ValueError('Operating point changed')
    with np.load(MODEL/'test-predictions.npz', allow_pickle=False) as public:
        public_map = {str(k): i for i, k in enumerate(public['record_ids'])}
        public_scores = np.stack([public['scores'][public_map[j['record_id']]] for j in jobs])
        public_predictions = np.stack([public['predictions'][public_map[j['record_id']]] for j in jobs])
        public_targets = np.stack([public['targets'][public_map[j['record_id']]] for j in jobs])
    truth = np.asarray([[i in targets[j['record_id']]['original_labels'] for i in LABEL_IDS] for j in jobs], dtype=bool)
    np.testing.assert_array_equal(truth, public_targets.astype(bool))
    rows, raw_scores, raw_predictions, completed_truth = [], [], [], []
    all_features = []
    for index, job in enumerate(jobs):
        folder = OUTPUT/'runs'/job['key']
        attempt = read(folder/'attempt.json')
        for artifact in attempt['artifacts']:
            if digest(folder/artifact['name']) != artifact['sha256']:
                raise ValueError('Attempt artifact modified')
        row = {'record_id': job['record_id'], 'original_labels': targets[job['record_id']]['original_labels'],
            'label_names': targets[job['record_id']]['label_names'], 'status': attempt['status'],
            'wall_seconds': attempt['wall_seconds'], 'expected_frames': job['expected_frames'],
            'public_keypoint_predicted_ids': [i for i, value in zip(LABEL_IDS, public_predictions[index]) if value],
            'public_keypoint_exact_match': bool(np.array_equal(public_predictions[index], truth[index])),
            'raw_predicted_ids': None, 'raw_exact_match': False, 'analysis': None}
        if attempt['status'] != 'completed':
            row['error'] = attempt.get('error')
            rows.append(row)
            continue
        analysis = read(folder/'inference.json')
        validate_result(job, analysis, lock['manifest_sha256'])
        with np.load(folder/'inference.npz', allow_pickle=False) as values:
            points, confidence = values['keypoint'], values['keypoint_score']
            features, probability, predictions = [values[k].copy() for k in ['features', 'scores', 'predictions']]
        if points.shape != (1, job['expected_frames'], 17, 2) or confidence.shape != (1, job['expected_frames'], 17):
            raise ValueError('Wrong raw skeleton shape')
        if features.shape != (1, 783) or probability.shape != (1, 12) or predictions.shape != (1, 11):
            raise ValueError('Wrong frozen feature/output shape')
        if not all(np.isfinite(v).all() for v in [points, confidence, features, probability]):
            raise ValueError('Nonfinite raw inference result')
        np.testing.assert_allclose(model.probability(features), probability, rtol=0, atol=1e-12)
        geometry, _ = describe(points[0], confidence[0])
        np.testing.assert_array_equal(geometry, features[0, 512:])
        op = state['operating_point']
        np.testing.assert_array_equal(predictions, predict(probability, op['thresholds'], op['normal_gate']))
        np.testing.assert_allclose([s['uncalibrated_score'] for s in analysis['scores']], probability[0, :11], rtol=0, atol=1e-12)
        np.testing.assert_array_equal([s['threshold'] for s in analysis['scores']], op['thresholds'])
        predicted_ids = [i for i, yes in zip(LABEL_IDS, predictions[0]) if yes]
        if predicted_ids != analysis['predicted_original_label_ids']:
            raise ValueError('JSON/NPZ label mismatch')
        row.update({'raw_predicted_ids': predicted_ids, 'raw_exact_match': bool(np.array_equal(predictions[0], truth[index])),
                    'analysis': {k: analysis[k] for k in ['decoded_frames', 'reported_fps', 'frames_with_person',
                         'frames_with_multiple_people', 'person_selection', 'geometry_quality', 'analysis_seconds',
                         'repeat_feature_max_difference', 'cuda_peak_allocated_bytes']}})
        rows.append(row)
        raw_scores.append(probability[0]); raw_predictions.append(predictions[0]); completed_truth.append(truth[index]); all_features.append(features[0])
    completed = len(raw_scores)
    if completed != batch['completed']:
        raise ValueError('Completion count mismatch')
    measured = scored(np.stack(raw_scores), np.stack(completed_truth), np.stack(raw_predictions)) if completed == len(jobs) else None
    reference = scored(public_scores, truth, public_predictions)
    technical = {'all_26_attempted': len(rows) == 26, 'all_26_classified': completed == 26,
        'all_expected_frames_decoded': sum(r['analysis']['decoded_frames'] for r in rows if r['analysis']) == protocol['frame_count'],
        'frozen_pipeline_and_videos_intact': True, 'head_and_geometry_replay_passed': True,
        'no_ground_truth_inference_input': all(not read(OUTPUT/'runs'/j['key']/'attempt.json')['ground_truth_supplied_to_inference'] for j in jobs)}
    checks = quality_checks(protocol, measured, completed, len(jobs))
    latencies = np.asarray([r['wall_seconds'] for r in rows])
    report = {'schema': 'cpr-raw-video-evaluation-v1', 'completed_utc': datetime.now(timezone.utc).isoformat(),
        'acceptance_protocol_sha256': digest(OUTPUT/'acceptance-protocol.json'), 'model_manifest_sha256': lock['manifest_sha256'],
        'scoring_lock_sha256': digest(OUTPUT/'scoring-lock.json'),
        'records': len(rows), 'completed_records': completed, 'failed_records': len(rows)-completed,
        'expected_frames': protocol['frame_count'], 'execution_checks': technical, 'quality_checks': checks,
        'raw_execution_passed': all(technical.values()), 'raw_quality_passed': all(technical.values()) and all(checks.values()),
        'raw_metrics': measured, 'same_26_public_keypoint_metrics': reference,
        'raw_exact_matches': sum(r['raw_exact_match'] for r in rows),
        'public_keypoint_exact_matches': sum(r['public_keypoint_exact_match'] for r in rows),
        'timing': {'sum_job_wall_seconds': float(latencies.sum()), 'median_job_wall_seconds': float(np.median(latencies)),
                   'p95_job_wall_seconds': float(np.percentile(latencies, 95)),
                   'sum_analysis_seconds': float(sum(r['analysis']['analysis_seconds'] for r in rows if r['analysis'])),
                   'max_pytorch_cuda_allocated_bytes': max((r['analysis']['cuda_peak_allocated_bytes'] for r in rows if r['analysis']), default=0),
                   'does_not_include': 'human preparation or reporting; GPU allocated bytes are not total device memory'},
        'rows': rows, 'no_fitting_or_threshold_changes': True, 'independent_blind_test': False,
        'composite_raw_video_quality': 'not evaluated: no composite RGB recordings in frozen collection',
        'cross_person_generalization': 'not evaluated: no verified subject identities',
        'runtime_release_approved': False, 'empty_prediction_semantics': 'No supported error exceeded threshold; not proof of correct CPR',
        'source_and_limits': protocol['disclosure']}
    if completed == 26:
        with (OUTPUT/'raw-video-predictions.npz').open('xb') as handle:
            np.savez_compressed(handle, record_ids=np.asarray([j['record_id'] for j in jobs]), scores=np.stack(raw_scores),
                                predictions=np.stack(raw_predictions), targets=truth, features=np.stack(all_features), label_ids=np.asarray(LABEL_IDS))
    write_new(OUTPUT/'evaluation.json', report)
    REPORT.mkdir(parents=True, exist_ok=True)
    write_new(REPORT/'evaluation-summary.json', {k: v for k, v in report.items() if k != 'rows'})
    render_report(report)
    print({k: report[k] for k in ['completed_records', 'failed_records', 'raw_execution_passed', 'raw_quality_passed', 'raw_exact_matches']}, flush=True)
    print('Raw quality checks', checks, flush=True)


def render_report(result):
    p = read(OUTPUT/'acceptance-protocol.json')
    raw = result['raw_metrics']
    reference = result['same_26_public_keypoint_metrics']
    pc = lambda v: '—' if v is None else f'{v*100:.2f}%'
    title = '真实视频验收通过' if result['raw_quality_passed'] else '验收已完成，识别质量未达标'
    if not result['raw_execution_passed']:
        title = '验收已完成，链路存在失败'
    if raw:
        metric_rows = f'<tr><td>单错误 Macro-F1（含正常样本误报）</td><td>{pc(raw["macro_f1"])}</td><td>≥{pc(p["quality_gates"]["single_macro_f1"])}</td></tr>'
        metric_rows += f'<tr><td>最低类别召回率</td><td>{pc(min(r["recall"] for r in raw["per_label"]))}</td><td>≥{pc(p["quality_gates"]["minimum_label_recall"])}</td></tr>'
        fp = round(raw['correct_false_alarm_rate']*raw['original_correct_records'])
        metric_rows += f'<tr><td>正常片段误报</td><td>{fp} / {raw["original_correct_records"]}</td><td>≤10%（本批须0/2）</td></tr>'
    else:
        metric_rows = '<tr><td>完整集合识别指标</td><td>不报告</td><td>存在失败，不能只选成功样本</td></tr>'
    detail = []
    for row in result['rows']:
        verdict = '匹配' if row['raw_exact_match'] else ('失败' if row['status'] != 'completed' else '不匹配')
        pred = '未输出' if row['raw_predicted_ids'] is None else (', '.join(map(str, row['raw_predicted_ids'])) or '无支持错误超阈值')
        label = ', '.join(map(str, row['original_labels']))
        relative = '../../data/public_cpr/rgb-video-acceptance-v4/runs/'+row['record_id'].replace('/', '--')+'/'
        detail.append(f'<tr><td>{html.escape(row["record_id"])}</td><td>{label}</td><td>{pred}</td><td class="{ "good" if row["raw_exact_match"] else "bad" }">{verdict}</td><td>{row["wall_seconds"]:.1f}s</td><td><a href="{relative}attempt.json">记录</a> · <a href="{relative}process.txt">日志</a></td></tr>')
    per_label = []
    if raw:
        for a, b in zip(raw['per_label'], reference['per_label']):
            per_label.append(f'<tr><td>{a["id"]} · {html.escape(a["name"])}</td><td>{a["support"]}</td><td>{pc(a["recall"])}</td><td>{pc(a["f1"])}</td><td>{pc(b["f1"])}</td></tr>')
    checks = {**result['execution_checks'], **result['quality_checks']}
    check_rows = ''.join(f'<li><span class="{ "good" if v else "bad" }">{ "通过" if v else "未通过" }</span> · {html.escape(k)}</li>' for k, v in checks.items())
    conclusion = ('本批所有片段的链路和预设识别门槛均通过。可进入受限场景下的应用接入验证，仍需评估组合错误和真实用户输入。'
                  if result['raw_quality_passed'] else '本轮保留v4及全部失败证据，不调整模型或阈值。当前不能将此版本作为已达标的真实视频错误识别模块接入演示主链路；需先依据视频与公开关键点对照，定位并修复识别薄弱环节。')
    document = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CPR v4 · 26段真实视频验收</title>
<style>:root{font-family:Inter,"Segoe UI","Microsoft YaHei",sans-serif;color:#17374b;background:#f2f6f8;font-size:15px}*{box-sizing:border-box}body{margin:0}main{max-width:1180px;margin:auto;padding:38px 24px 60px}h1{font-size:32px;line-height:1.35;margin:14px 0}h2{font-size:20px;margin:0 0 15px}p{line-height:1.85}header{margin-bottom:24px}.tag{display:inline-block;padding:7px 12px;border-radius:8px;background:#edf3fa;color:#245676;font-weight:600}.card{padding:24px;border:1px solid #dce6eb;border-radius:15px;background:white;margin:18px 0}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.grid .card{margin:0}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:25px 0}.stat{padding:17px;background:#fff;border:1px solid #dce6eb;border-radius:12px}.stat strong{display:block;font-size:27px;margin-bottom:7px}.stat span,.small{font-size:13px;color:#5e7988}.notice{background:#fff9ee;border-left:4px solid #d4a341}.tablewrap{overflow:auto}table{border-collapse:collapse;width:100%;line-height:1.7;font-size:14px}th,td{text-align:left;padding:11px 9px;border-bottom:1px solid #e6edf1}th{background:#f3f7fa;color:#526f80}.detail{min-width:850px}.classify{min-width:620px}.good{color:#13745c}.bad{color:#b5453b;font-weight:600}ul{padding-left:19px;line-height:1.9}a{color:#166b96;text-decoration:none;overflow-wrap:anywhere}a:hover{text-decoration:underline}details summary{cursor:pointer;font-weight:600;margin-bottom:15px}.flow{display:flex;flex-wrap:wrap;gap:9px}.flow span{padding:8px 10px;border-radius:7px;background:#edf5f8}footer{font-size:13px;color:#5e7988;line-height:1.85;margin-top:25px}@media(max-width:680px){main{padding:25px 14px 40px}h1{font-size:25px}.card{padding:18px 14px}.grid{grid-template-columns:1fr}.stats{grid-template-columns:1fr 1fr}.stat strong{font-size:24px}table{font-size:12px}th,td{padding:10px 7px}}@media print{body{background:white}main{max-width:none;padding:0}.card{break-inside:avoid}}</style>
<main><header><span class="tag">v4 固化 · 公开RGB · 单视角 ch0</span><h1>__TITLE__</h1><p class="small">26段 / 11,215帧；公开原标签；无新增采集或标注；模型、阈值和输入处理全程冻结。</p></header>
<div class="stats"><div class="stat"><strong>__COMPLETED__ / 26</strong><span>完整分类输出</span></div><div class="stat"><strong>__EXACT__ / 26</strong><span>错误向量与公开标签完全匹配</span></div><div class="stat"><strong>__RAWF1__</strong><span>真实视频 Macro-F1</span></div><div class="stat"><strong>__WALL__ 分钟</strong><span>串行处理总耗时，含模型启动</span></div></div>
<section class="card notice"><h2>验收结论</h2><p>__CONCLUSION__</p><p class="small">执行链路：__EXECUTION__；识别质量：__QUALITY__；Web／Agent发布状态：未批准。本次未训练模型、未调阈值，也未挑选成功片段替代完整集合。</p></section>
<section class="card"><h2>固化内容与实际处理流程</h2><div class="flow"><span>逐段MP4哈希校验</span><span>YOLO检测与施救者跟踪</span><span>AlphaPose身体关键点</span><span>冻结PoseC3D＋几何特征</span><span>v4小分类头及固定阈值</span><span>预测落盘后核对公开标签</span></div><p>26段均为v4的测试保留录像。每段独立进程，无公开参考标签参数；全流程输出原始视频提取的关键点与特征。辅助重放校验确认保存的特征、分数、阈值和输出一致。</p></section>
<div class="grid"><section class="card"><h2>事前冻结的识别门槛</h2><div class="tablewrap"><table><thead><tr><th>指标</th><th>本批结果</th><th>要求</th></tr></thead><tbody>__METRICS__<tr><td>分类覆盖率</td><td>__COVERAGE__</td><td>100%</td></tr></tbody></table></div></section>
<section class="card"><h2>与同一批公开关键点对照</h2><p>公开关键点特征 Macro-F1：<strong>__PUBLICF1__</strong>；真实视频：<strong>__RAWF1__</strong>。</p><p>公开关键点错误向量匹配：__PUBLICEXACT__ / 26；视频提取关键点后匹配：__EXACT__ / 26。此对照用于发现感知与特征分布差异，不能单凭差异断言具体原因。</p><p class="small">386条关键点回归集的结果属于另一种输入条件，不等于26段视频的识别效果。</p></section></div>
<section class="card"><h2>每类识别结果</h2><div class="tablewrap"><table class="classify"><thead><tr><th>公开错误类别</th><th>正例数</th><th>视频召回率</th><th>视频F1</th><th>公开关键点F1</th></tr></thead><tbody>__CLASSES__</tbody></table></div></section>
<section class="card"><h2>全部26段，逐段保留</h2><div class="tablewrap"><table class="detail"><thead><tr><th>录像ID</th><th>公开标签</th><th>视频预测错误ID</th><th>完整向量</th><th>耗时</th><th>证据</th></tr></thead><tbody>__ROWS__</tbody></table></div><p class="small">公开标签0表示原数据正常动作；空预测仅表示“没有支持的错误超过阈值”，不证明CPR整体正确。ID对应上方公开错误类别。</p></section>
<section class="card"><details><summary>展开：技术与质量检查</summary><ul>__CHECKS__</ul></details><p>单段墙钟耗时中位数 __MEDIAN__ 秒，P95 __P95__ 秒。PyTorch最高已分配显存 __MEMORY__ MiB，不能解释为设备总显存占用；本链路是离线分析。</p></section>
<section class="card"><h2>可复核文件</h2><p><a href="../../data/public_cpr/rgb-video-acceptance-v4/model-lock.json">模型与阈值锁定</a> · <a href="../../data/public_cpr/rgb-video-acceptance-v4/acceptance-protocol.json">事前验收协议</a> · <a href="../../data/public_cpr/rgb-video-acceptance-v4/evaluation.json">完整指标与逐段结果</a> · <a href="../../data/public_cpr/rgb-video-acceptance-v4/environment.json">实际运行环境</a> · <a href="batch-run.txt">批处理日志</a></p></section>
<footer>证据边界：本批含2段正常动作和11类单错误，部分录像曾在早期工程模型中使用或查看过，不属于独立盲测；各类样本很少。没有组合错误RGB、可靠受试者ID、精确错误起止、实测按压深度或BPM标注，因此不声称对应能力或临床有效性。</footer></main></html>'''
    timing = result['timing']
    replacements = {'TITLE': title, 'COMPLETED': str(result['completed_records']), 'EXACT': str(result['raw_exact_matches']),
        'RAWF1': pc(raw['macro_f1']) if raw else '—', 'WALL': f'{timing["sum_job_wall_seconds"]/60:.1f}',
        'CONCLUSION': conclusion, 'EXECUTION': '通过' if result['raw_execution_passed'] else '未通过',
        'QUALITY': '通过' if result['raw_quality_passed'] else '未通过', 'METRICS': metric_rows,
        'COVERAGE': pc(result['completed_records']/26), 'PUBLICF1': pc(reference['macro_f1']),
        'PUBLICEXACT': str(result['public_keypoint_exact_matches']), 'CLASSES': ''.join(per_label),
        'ROWS': ''.join(detail), 'CHECKS': check_rows, 'MEDIAN': f'{timing["median_job_wall_seconds"]:.1f}',
        'P95': f'{timing["p95_job_wall_seconds"]:.1f}', 'MEMORY': f'{timing["max_pytorch_cuda_allocated_bytes"]/2**20:.0f}'}
    for key, value in replacements.items():
        document = document.replace('__'+key+'__', value)
    (REPORT/'26段真实视频验收报告.html').write_text(document, encoding='utf-8')


if __name__ == '__main__':
    main()
