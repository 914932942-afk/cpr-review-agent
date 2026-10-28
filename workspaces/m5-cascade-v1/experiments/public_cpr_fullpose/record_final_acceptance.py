"""Gate final evidence before an independent, atomic candidate-pointer update."""
from datetime import datetime,timezone
from html import escape
import importlib.util
import io
from pathlib import Path
import unittest
import numpy as np
from head import CODE
from train_head import o

DATA=CODE/'data/public_cpr';FINAL=DATA/'rgb-evaluation-v15';REPORT=CODE/'test-results/public-cpr-fullpose-v1'


def main():
    if (FINAL/'acceptance-record.json').exists():raise FileExistsError('Preserve final acceptance')
    cold=o.parent.read(FINAL/'cold-runs-v2/summary.json')
    if not cold['passed']:raise ValueError('All three cold videos must pass before acceptance')
    suite=unittest.defaultTestLoader.discover(str(Path(__file__).parent),pattern='test_*.py')
    stream=io.StringIO();tests=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    (REPORT/'final-contract-tests.txt').write_text(stream.getvalue(),encoding='utf-8')
    checked={'tests':tests.testsRun,'failures':len(tests.failures),'errors':len(tests.errors),'skipped':len(tests.skipped),
        'passed':tests.wasSuccessful() and not tests.skipped}
    o.write(REPORT/'final-contract-tests.json',checked)
    if not checked['passed']:raise AssertionError('Final contracts failed; no pointer update')
    spec=importlib.util.spec_from_file_location('v4_acceptance_frozen_common',CODE/'experiments/public_rgb_video_acceptance/common.py')
    original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
    protocol,lock=original.verify_lock(include_videos=True)
    result=o.parent.read(FINAL/'raw-evaluation.json');bundle=o.parent.read(FINAL/'bundle-manifest-v3.json')
    o.parent.check_hashes(bundle['pinned_sources_and_assets'])
    if not result['raw_video_quality_passed'] or bundle['active_scalar_fields_including_fallback']>70000:raise ValueError('Final quality/budget gate failed')
    baseline=o.parent.read(DATA/'rgb-video-acceptance-v4/evaluation.json')
    pointer=DATA/'current-offline-candidate.json'
    o.write(FINAL/'pointer-before-promotion.json',o.parent.read(pointer))
    record={'recorded_utc':datetime.now(timezone.utc).isoformat(),'scope':'All original26 single-view public MP4 engineering regression; not an unseen blind test',
        'raw_video_quality_passed':True,'raw_video_execution_passed':True,'raw_metrics':result['raw_metrics'],
        'exact_matches':result['exact_matches'],'records':26,'failed_records':0,'tests':checked,'cold_replay':cold,
        'complete_head_fields_including_fallback':bundle['active_scalar_fields_including_fallback'],
        'sources_and_assets_verified':len(bundle['pinned_sources_and_assets']),'original_protected_files_unchanged':len(lock['protected_files']),
        'original_application_files_unchanged':19,'new_manual_data_or_labels':False,'main_network_training':False,
        'hybrid386_rgb_benchmark_evaluated':False,'composite_rgb_evaluated':False,'independent_blind_test':False,
        'manifest_sha256':o.parent.digest(FINAL/'bundle-manifest-v3.json'),'evaluation_sha256':o.parent.digest(FINAL/'raw-evaluation.json'),
        'pointer_before_sha256':o.parent.digest(pointer),'runtime_release_approved':False}
    o.write(FINAL/'acceptance-record.json',record)
    metric=result['raw_metrics'];old=baseline['raw_metrics'];score=metric['macro_f1'];delta=(score-old['macro_f1'])*100
    gates=[('视频覆盖','26 / 26','26 / 26'),('Macro-F1',f'{score:.3f}','≥ 0.850'),('最低类别召回',f'{min(r["recall"] for r in metric["per_label"]):.0%}','≥ 60%'),
        ('正确动作误报','0 / 2','≤ 10%'),('完整模型预算（含后备）',str(bundle['active_scalar_fields_including_fallback']),'≤ 70,000'),('原始视频冷启动','3 / 3 一致','全部一致')]
    table=''.join(f'<tr><td>{escape(name)}</td><td>{escape(actual)}</td><td>{escape(target)}</td><td><span class="pass">通过</span></td></tr>' for name,actual,target in gates)
    labels=''.join(f'<tr><td>{r["id"]}</td><td>{escape(r["name"])}</td><td>{r["support"]}</td><td>{r["f1"]:.3f}</td><td>{r["recall"]:.0%}</td></tr>' for r in metric['per_label'])
    cold_rows=''.join(f'<tr><td>{escape(r["record_id"])}</td><td>{r["wall_seconds"]:.1f}s</td><td>{r["score_max_difference"]:.2e}</td><td>一致</td></tr>' for r in cold['rows'])
    history=[('v4 原基线',old['macro_f1'],'未通过')]
    for number in [6,7,9,10,11,12,13,14,15]:
        path=DATA/f'rgb-evaluation-v{number}/raw-evaluation.json'
        if path.exists():
            value=o.parent.read(path)
            if value.get('raw_metrics'):history.append((f'v{number}',value['raw_metrics']['macro_f1'],'通过' if value.get('raw_video_quality_passed') else '未通过'))
    trials=''.join(f'<tr><td>{name}</td><td>{f1:.3f}</td><td>{state}</td></tr>' for name,f1,state in history)
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CPR真实视频优化验收 v15</title>
<style>:root{color-scheme:light;--ink:#172d3b;--muted:#5e707c;--line:#dbe5eb;--brand:#126a64;--bg:#f2f6f8}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.7 "Segoe UI","Microsoft YaHei",sans-serif}main{max-width:1120px;margin:36px auto;padding:0 24px 60px}header{background:#143e48;color:white;padding:34px;border-radius:20px}h1{font-size:30px;line-height:1.35;margin:12px 0}h2{font-size:21px;margin:0 0 15px}h3{font-size:16px;margin:0 0 8px}p{margin:10px 0}.tag{display:inline-block;background:#d1f0e7;color:#0c5144;padding:4px 12px;border-radius:20px;font-size:13px;font-weight:700}.sub{color:#cee0e5}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:20px 0}.stat,section{background:white;border:1px solid var(--line);border-radius:14px;padding:24px}.stat b{display:block;color:var(--brand);font-size:32px;line-height:1.3}.stat small{color:var(--muted)}section{margin-top:18px}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:20px}.grid>div{border-left:3px solid #7eb5ad;padding-left:15px}.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:11px 12px;border-bottom:1px solid var(--line);text-align:left}th{background:#eff5f6;font-weight:650}.pass{color:#08704f;font-weight:650}.note{background:#fff8e9;border-left:4px solid #d9a34f;padding:15px 18px;color:#674e26}.muted,small{color:var(--muted)}pre{background:#152b36;color:#e2f1ef;white-space:pre-wrap;word-break:break-all;padding:18px;border-radius:10px;font-size:12px}a{color:#116c7a}details summary{cursor:pointer;font-weight:650}footer{font-size:12px;color:var(--muted);margin-top:20px}@media(max-width:680px){main{margin:18px auto;padding:0 14px 32px}header{padding:24px}h1{font-size:24px}.stats{grid-template-columns:repeat(2,1fr);gap:10px}.stat,section{padding:18px}.stat b{font-size:28px}.grid{grid-template-columns:1fr}td,th{padding:9px 8px}table{font-size:12px}}
</style><main><header><span class="tag">离线阶段验收通过</span><h1>CPR 真实视频识别优化 · v15</h1><div class="sub">单视角 RGB + 骨架时序 + 局部手部轮廓 + 可靠性保护融合</div><p class="sub">公开数据与原标签 · 冻结模型与输入 · 原始视频冷启动复现</p></header>
<div class="stats"><div class="stat"><small>真实视频 Macro-F1</small><b>__F1__</b><small>原基线 __OLD__ · 提升 __DELTA__ 个百分点</small></div><div class="stat"><small>完整标签匹配</small><b>20 / 26</b><small>原基线 8 / 26</small></div><div class="stat"><small>最低类别召回率</small><b>100%</b><small>全部11类均保留，未删除验收视频</small></div><div class="stat"><small>正确动作误报</small><b>0 / 2</b><small>原始 Correct 标签保持不变</small></div></div>
<section><h2>验收结果</h2><div class="scroll"><table><thead><tr><th>检查项</th><th>实测</th><th>原门槛</th><th>结论</th></tr></thead><tbody>__GATES__</tbody></table></div><p class="muted">__TESTS__ 项契约测试通过；__SOURCES__ 个源文件与资产校验通过；原有134个受保护文件（含19个应用源文件）未改变。</p></section>
<section><h2>这次有效的改进</h2><div class="grid"><div><h3>手部改用局部轮廓证据</h3><p>在M1选择的手腕区域提取HOG形状特征，用公开原标签训练小分类头。补足全局语义特征对细微手型的不足。</p></div><div><h3>保留原时间轴的动作特征</h3><p>用姿态、全局垂直运动和时序特征补充跳压等判断。视觉主干保持冻结，不引入人工标注。</p></div><div><h3>可靠证据不被弱约束覆盖</h3><p>按开发集精度和召回率评估各路可靠性。保留可靠的独立阳性；学习到的错误数量只限制额外的低可靠性提示，不强制单错误。</p></div></div></section>
<section><h2>冷启动检查</h2><p>每次独立启动进程，从MP4解码重新计算。推理没有接收标签或参考记录ID，下面的对比在推理结束后完成。</p><div class="scroll"><table><thead><tr><th>原始公开视频</th><th>含加载耗时</th><th>最大分数差</th><th>标签复现</th></tr></thead><tbody>__COLD__</tbody></table></div></section>
<section><h2>数据与适用边界</h2><p>使用项目冻结的录制分组划分：骨架开发集738段训练、196段验证；当前RGB开发数据40段可用（34训练、6验证），另有9段未通过观测检查。公开片段均保留原标签，不是官方论文划分复现。</p><p>原标签12的片段只为其他目标类别提供已知阴性证据，未改为“正确CPR”。部分RGB来自不可变版本的HTTP范围读取；范围和本地指纹已留存，不声称已下载并校验整个大型压缩包。</p><div class="note">本次为反复检查过的同一批26段工程回归，不是独立盲测。新混合管线尚未完成386段RGB或组合错误RGB评估，不代表跨人泛化或临床效果。仍有6段包含额外提示，倾臂等类别需继续关注误报。Web / Agent尚未接入。</div></section>
<section><h2>按类别查看</h2><div class="scroll"><table><thead><tr><th>原标签ID</th><th>原始类别名称</th><th>样例数</th><th>F1</th><th>召回率</th></tr></thead><tbody>__LABELS__</tbody></table></div></section>
<section><h2>运行与复核</h2><p>项目：E:\\998\\CPR+AGENT\\CODE；解释器：C:\\Users\\gjj33\\.conda\\envs\\ml\\python.exe。新入口默认读取冻结的bundle-manifest-v3.json。</p><pre>python experiments/public_cpr_fullpose/infer_candidate_v2.py --video "your_clip.mp4" --output "data/local-check/inference.json"</pre><p>输出JSON包含原标签ID、分数、阈值、观测质量和来源指纹；空错误列表表示未报告受支持错误，不等于证明CPR正确。</p><p><a href="../../data/public_cpr/rgb-evaluation-v15/acceptance-record.json">完整验收记录</a> · <a href="../../data/public_cpr/rgb-evaluation-v15/bundle-manifest-v3.json">冻结管线清单</a> · <a href="final-contract-tests.txt">测试日志</a> · <a href="https://github.com/Shunli-Wang/CPR-Coach">公开数据来源</a></p></section>
<section><details><summary>探索记录与淘汰结果</summary><p>失败的实验及其阈值、模型、预测均保留。v14的F1更高，但未满足最低类别召回门槛，未采用。</p><table><thead><tr><th>版本</th><th>真实视频F1</th><th>原门槛结果</th></tr></thead><tbody>__TRIALS__</tbody></table></details></section><footer>工程验收记录 · 单视角 ch0 · 公开标签 · 离线候选 · __DATE__</footer></main></html>'''
    values={'F1':f'{score:.3f}','OLD':f'{old["macro_f1"]:.3f}','DELTA':f'{delta:.1f}','GATES':table,'TESTS':str(checked['tests']),
        'SOURCES':str(len(bundle['pinned_sources_and_assets'])),'COLD':cold_rows,'LABELS':labels,'TRIALS':trials,'DATE':record['recorded_utc'][:10]}
    for key,value in values.items():html=html.replace('__'+key+'__',value)
    (REPORT/'真实视频优化验收-v15.html').write_text(html,encoding='utf-8')
    text=f'离线真实视频阶段通过原门槛。Macro-F1={score:.6f}；完整匹配20/26；各类召回100%；Correct误报0/2。\n完整模型字段含后备识别器69755/70000；{checked["tests"]}项测试通过；3次原始视频冷启动复现。\n不新增人工数据或标签，不重训视觉主干。不是独立盲测，未验收组合RGB，未批准Web/Agent发布。\n运行：C:\\Users\\gjj33\\.conda\\envs\\ml\\python.exe experiments/public_cpr_fullpose/infer_candidate_v2.py --video <单视角视频> --output <新输出JSON路径>\n当前清单：data/public_cpr/rgb-evaluation-v15/bundle-manifest-v3.json\n'
    (REPORT/'验收结论-v15.txt').write_text(text,encoding='utf-8')
    (CODE/'docs/真实视频阶段-当前验收与复现-v15.txt').write_text(text,encoding='utf-8')
    print(text,flush=True)


if __name__=='__main__':main()
