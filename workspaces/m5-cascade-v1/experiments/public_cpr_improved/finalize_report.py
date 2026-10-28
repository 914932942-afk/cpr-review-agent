"""Summarize actual validation artifacts; never promote the application's tools."""
import html
import json
from pathlib import Path
from paths import CODE, DATA


def read(path): return json.loads(path.read_text(encoding="utf-8"))


def main():
    report = CODE / "test-results/step1-improved"
    quality = read(DATA / "improvement-v2/evaluation.json")
    selected = read(DATA / "improvement-v2/selection.json")["selected"]
    tests = read(report / "ml-contract-tests.json")
    integrity = read(report / "source-integrity.json")
    repeat = read(report / "end-to-end-repeat.json")
    cases = []
    for filename in ("raw-final-SupC02-r00.json", "raw-final-SupC02-r01.json", "raw-final-C13-r11.json", "raw-final-C13-r12.json"):
        result = read(report / filename)
        check = result["public_label_verification"]
        cases.append({**check, "predictions": result["predicted_original_label_ids"],
            "frames": result["decoded_frames"], "seconds_after_model_load": result["analysis_seconds"],
            "torch_peak_allocated_bytes": result["cuda_peak_allocated_bytes"],
            "manifest_sha256": result["manifest_sha256"], "report": filename})
    if len({r["manifest_sha256"] for r in cases}) != 1:
        raise ValueError("Video results do not use the same frozen pipeline")
    new_raw_pass = any(r["role"] == "preselected_raw_test" and r["exact_label_match"] for r in cases)
    checks = {**quality["quality_checks"], "preselected_public_raw_clip": new_raw_pass,
              "contract_tests": tests["passed"], "frozen_application_source": integrity["application_source_unchanged"],
              "repeated_raw_inference": repeat["predictions_identical"] and repeat["max_score_difference"] <= 1e-10}
    technical_pass = all(checks.values())
    summary = {"offline_technical_gate_passed": technical_pass,
        "raw_all_class_release_approved": False, "next_agent_integration_started": False,
        "ready_for_full_recognition_release": False,
        "reason": "Known Clenching Hands miss remains; raw clips cover only two error classes and no Correct clips.",
        "checks": checks, "public_pose_metrics": quality["evaluations"],
        "public_application_split": {"train": 758, "validation": 198, "test": 364},
        "protocol_disclosure": "Uses existing public composite labels in development; not original single-training/composite-zero-shot protocol. Earlier aggregate tests were inspected; no subject-disjoint claim.",
        "training_scope": "Frozen perception backbones; small classifier only; no collection or new manual labels.",
        "head_scalar_fields": selected["scalar_fields"], "head_fit_seconds": selected["seconds"],
        "raw_cases": cases, "raw_exact_matches": sum(r["exact_label_match"] for r in cases),
        "raw_records": len(cases), "raw_distinct_error_classes": 2,
        "raw_case_policy": "Convenience engineering samples, not an estimate of all-class or population accuracy; r00 is development, r01/r11 exposed regressions, r12 preselected raw test.",
        "normal_public_pose_false_alarms": "0/18 observed; small sample, not a zero-risk guarantee",
        "ml_contract_tests": tests, "application_regression": {"tests_passed": 14,
            "observed_command": ".venv/Scripts/python.exe -m pytest -q", "seconds": 10.99},
        "source_integrity": integrity, "repeat": repeat,
        "scope": "Offline CV baseline only; M1-M5 algorithm claims, RGB/fusion/calibration and Agent tool integration remain separately unaccepted."}
    (report / "validation-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    single = quality["evaluations"]["single"]; composite = quality["evaluations"]["composite"]
    rows = "".join(f'<tr><td>{html.escape(r["record_id"])}</td><td>{r["original_labels"]}</td><td>{r["predictions"] or "未检出支持的错误"}</td><td class="{("ok" if r["exact_label_match"] else "warn")}">{("匹配" if r["exact_label_match"] else "漏检，保留失败")}</td><td><a href="{r["report"]}">JSON</a></td></tr>' for r in cases)
    status = "通过" if technical_pass else "未通过"
    document = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CPR第一步 · 离线链路验证</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#f3f6fa;color:#233247;font:15px/1.7 "Segoe UI","Microsoft YaHei",sans-serif}}main{{max-width:1120px;margin:36px auto;padding:0 24px 50px}}h1{{font-size:29px;line-height:1.35;margin:8px 0}}h2{{font-size:19px;margin-top:0}}p{{margin:9px 0}}.muted{{color:#64748b}}.eyebrow{{color:#2864b4;font-weight:700;letter-spacing:.08em}}section,.card{{background:white;border:1px solid #dce4ef;border-radius:14px;padding:22px;margin-top:20px}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}}.card{{margin:0}}.number{{font-size:29px;font-weight:750;color:#215ba4}}.callout{{border-left:4px solid #dd982c;background:#fff9eb;padding:15px 19px;margin:20px 0}}.ok{{color:#147d59;font-weight:650}}.warn{{color:#ac6110;font-weight:650}}.flow{{background:#edf4fd;padding:14px;border-radius:8px;color:#31567e}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{text-align:left;padding:11px 10px;border-bottom:1px solid #e5eaf1;vertical-align:top}}th{{background:#f1f5fa}}.scroll{{overflow:auto}}a{{color:#2365b0}}code{{font-family:Consolas,monospace;font-size:13px;word-break:break-all}}li{{margin:7px 0}}@media(max-width:750px){{.grid{{grid-template-columns:1fr}}main{{padding:0 14px}}h1{{font-size:24px}}section{{padding:16px}}}}
</style><main><div class="eyebrow">CPR + AGENT / STEP 01 / 2026-10-21</div><h1>离线视频链路已打通<br>完整识别能力仍需验收</h1>
<p class="muted">公开原标签 · 冻结视觉主干 · 小型分类头 · 单视角 ch0 · 主程序保持原状</p>
<div class="callout"><b>本轮结论：</b>预先约定的离线技术门槛{status}。仍有一段握拳片段漏检，且实际视频只覆盖两类错误；<b>没有启用正式错误判断，也没有进入下一步Agent集成。</b></div>
<div class="grid"><div class="card"><span class="muted">公开骨架 · 单错误测试</span><div class="number">{single['macro_f1']:.1%}</div><span>Macro-F1 / {single['records']}段</span></div><div class="card"><span class="muted">公开骨架 · 组合错误测试</span><div class="number">{composite['macro_f1']:.1%}</div><span>Macro-F1 / {composite['records']}段</span></div><div class="card"><span class="muted">公开正常骨架 · 实测误报</span><div class="number">0 / 18</div><span>小样本观察值，不代表零误报保证</span></div></div>
<section><h2>实现了什么</h2><div class="flow">真实视频 → 检测与人物关联 → AlphaPose骨架 → 冻结PoseC3D + 几何特征 → 小型分类头 → 类别 / 分数 / 追踪记录</div><p>修复了低置信度关键点被显示层置零的问题，以及训练者与仿人模型之间的选人切换。改用更接近公开骨架来源的AlphaPose，权重完整校验；本轮视频验证没有修改分类头或阈值。</p><p class="muted">现成骨架测试成功，不能直接代替视频测试。下表保留全部最终视频结果。</p></section>
<section><h2>真实 RGB 视频验证</h2><div class="scroll"><table><thead><tr><th>公开录像</th><th>原标签</th><th>预测</th><th>结果</th><th>记录</th></tr></thead><tbody>{rows}</tbody></table></div><p>标签2：Clenching Hands（握拳）；标签13：Random Position Pressing（随机按压位置）。共{len(cases)}段，{sum(r['exact_label_match'] for r in cases)}段精确匹配；这是工程样例验证，<b>不能当成总体准确率</b>。</p><p class="muted">r00用于开发；r01与r11是已查看结果的回归样例。r12在视频推理前冻结代码、模型及阈值后验证，但其骨架所属测试集的汇总指标此前已查看。</p></section>
<section><h2>数据、训练与工程检查</h2><div class="scroll"><table><tbody><tr><th>数据与划分</th><td>公开原数据和标签；训练758 / 验证198 / 测试364。录像不跨划分，没有自行采集或人工新增标签。</td></tr><tr><th>监督范围</th><td>公开组合错误标签参与训练；与原论文“单错误训练、未见组合测试”不同。指标变化不能全部归为算法提升。</td></tr><tr><th>训练成本</th><td>只拟合小型分类头，导出{selected['scalar_fields']:,}个标量字段；选中头拟合约{selected['seconds']:.1f}秒，特征提取另计。视觉主干全部冻结。</td></tr><tr><th>已有门槛</th><td>单错误F1≥85%、组合F1≥80%、正常误报≤10%、各类召回≥60%、全测试覆盖与分类头预算均通过。没有靠拒判或删去困难样本提高骨架指标。</td></tr><tr><th>工程验证</th><td>ML契约测试{tests['tests_run']}项通过；原项目回归14项通过；{integrity['application_source_files_checked']}个既有应用源文件指纹一致。完整视频重新解码、推理的重复分数差={repeat['max_score_difference']}。</td></tr><tr><th>运行开销</th><td>模型加载后，14秒左右的视频分析约32–40秒；首次加载另计。PyTorch峰值张量分配约0.56 GiB，不等同于进程总显存；当前8GB GPU运行成功。</td></tr></tbody></table></div></section>
<section><h2>当前边界与后续阻断项</h2><ul><li>握拳r00仍漏检；17个人体点不直接表示手指状态，不能据此承诺手型类别可靠。</li><li>没有完成11类错误、真实正常片段、新机位和非CPR输入的视频验收。当前接受已截取的CPR动作片段，不提供完整课程视频的自动动作切分。</li><li>空错误列表不是“动作正确”；分数尚未校准，不输出真实深度、实测BPM、精确错误时刻或临床结论。</li><li>五个算法模块的独立改进、消融，以及RGB、融合、校准、Agent正式工具，仍须各自验收。本轮只保留独立离线组件，正式网页仍使用占位模块。</li></ul><p><b>本步后续工作：</b>先按原设计验证手部/RGB补充证据，并补齐公开正常片段及其余类别的视频测试；达到声明的能力边界后，再接正式工具。无需新增人工标签。</p></section>
<section><h2>查看与复现</h2><p><a href="validation-summary.json">完整验收JSON</a> · <a href="ml-contract-tests.txt">ML测试记录</a> · <a href="source-integrity.json">应用代码一致性</a> · <a href="end-to-end-repeat.json">端到端重复验证</a></p><p><code>CODE/experiments/public_cpr_improved/README.txt</code> 提供运行命令。模型和公开视频位于 <code>CODE/data/public_cpr/</code>；历史失败输出位于本目录的 <code>history/</code>。</p><p class="muted">来源：<a href="https://huggingface.co/datasets/ShunliWang/CPR-Coach">CPR-Coach公开数据</a> · <a href="https://github.com/Shunli-Wang/CPR-Coach">原项目</a> · <a href="https://github.com/MVIG-SJTU/AlphaPose">AlphaPose</a></p></section></main></html>'''
    (report / "验证结果.html").write_text(document, encoding="utf-8")
    text = f'''CPR第一步改进与验证（2026-10-21）

离线技术门槛：{status}。完整识别能力发布：不批准。未进入下一步Agent集成。
公开骨架：单错误Macro-F1 {single['macro_f1']:.4%}，组合Macro-F1 {composite['macro_f1']:.4%}，正常误报0/18。
当前使用公开组合标签训练的新应用划分；不可与原零样本组合任务直接比较算法提升。
训练758 / 验证198 / 测试364；仅训练小型分类头，冻结视觉主干；无采集与新增人工标注。

真实公开视频：{sum(r['exact_label_match'] for r in cases)}/{len(cases)}段精确匹配，仅2类错误；所有失败保留，不能当总体准确率。
''' + "\n".join(f"{r['record_id']}: 原标签{r['original_labels']}，预测{r['predictions']}，匹配={r['exact_label_match']}" for r in cases) + f'''

模型、阈值及代码已冻结；新公开视频C13 r12验证匹配，完整端到端重复分数差{repeat['max_score_difference']}。
ML契约测试{tests['tests_run']}项通过；原项目回归14项通过；既有应用源文件{integrity['application_source_files_checked']}项指纹一致。
目前保留独立离线实验，网页/五模块/Agent没有切换到真实预测。
已知握拳漏检、真实正常视频及其余类别覆盖缺口未解决，因此不进入正式工具集成。
详细说明与数据边界见同目录的验证结果.html；机器可读结论见validation-summary.json。
'''
    (report / "验证结论.txt").write_text(text, encoding="utf-8")
    print(json.dumps({"offline_technical_gate_passed": technical_pass, "raw_matches": summary["raw_exact_matches"],
                      "raw_records": len(cases), "runtime_release_approved": False,
                      "next_agent_integration_started": False}, indent=2))


if __name__ == "__main__": main()
