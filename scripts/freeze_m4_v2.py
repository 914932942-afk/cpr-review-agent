"""Keep v2 only after finite-policy, numerical-equivalence, real UI and recovery gates."""

import hashlib
import json
import shutil
import sqlite3
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WS = ROOT / "workspaces/m4-refined-v2"
PARENT = ROOT / "workspaces/m4-routing-v1"
DATA = ROOT / "data/m4-exploration-v2"
RESULTS = ROOT / "test-results/m4-exploration-v2"
RELEASE = ROOT / "releases/m4-refined-v2"
DOC = ROOT / "docs/M4优化验证-v2.html"


def read(p):
    return json.loads(p.read_text(encoding="utf-8"))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    if (DATA / "acceptance.json").exists() or RELEASE.exists():
        raise FileExistsError("Preserve accepted release")
    before = read(DATA / "before.json")
    offline = read(DATA / "offline.json")
    assert offline["passed"] and offline["strict_coverage_improvement_cases"] > 0
    assert sha(ROOT / ".env") == before["env_sha256"]
    # The old implementation, its entrypoint, artifacts and release are all preserved.
    # Read-only oracle requests append access logs on the retained v1 live server.
    assert all(
        sha(ROOT / k) == h
        for k, h in before["accepted_source_hashes"].items()
        if not k.startswith("test-results/m4-scheduling-v1/service/")
    )
    previous_release = read(ROOT / "releases/m4-routing-v1/manifest.json")
    assert (
        sha(ROOT / "releases/m4-routing-v1/source-parameters-and-evidence.zip")
        == previous_release["archive_sha256"]
    )
    assert all(
        sha(ROOT / "releases/m4-routing-v1" / n) == h
        for n, h in previous_release["database_snapshots"].items()
    )
    for name in ["http/acceptance.json", "ui/acceptance.json", "restart.json", "algorithm/acceptance.json"]:
        assert read(RESULTS / name)["passed"]
    unique = set()
    for name in ["core-final.xml", "new-contracts-final.xml"]:
        xml = ET.parse(RESULTS / name)
        assert not list(xml.iter("failure")) and not list(xml.iter("error")) and not list(xml.iter("skipped"))
        unique.update((c.get("classname"), c.get("name")) for c in xml.iter("testcase"))
    assert len(unique) == 258
    profile_path = WS / "data/m4-exploration-v2/pipeline-manifest.json"
    profile = read(profile_path)
    parent = read(PARENT / "data/m4-scheduling-v1/pipeline-manifest.json")
    assert all(sha(WS / k) == h for k, h in profile["sources"].items())
    protected = [
        k
        for k in parent["sources"]
        if k.startswith(
            (
                "experiments/m1_fast_v1/",
                "experiments/m2_competence_v3/",
                "experiments/m3_calibration_v1/",
                "experiments/branch_pipeline_v1/",
                "data/m1-m2-exploration-v1/",
                "data/m3-calibration-v1/",
            )
        )
    ]
    assert all(sha(WS / k) == parent["sources"][k] for k in protected)
    assert sha(WS / "backend/cpr_agent/m3_quality.py") == parent["sources"]["backend/cpr_agent/m3_quality.py"]
    assert sha(PARENT / "data/m4-scheduling-v1/pipeline-manifest.json") == before["parent_profile_sha256"]
    params = read(WS / "data/m4-exploration-v2/parameters.json")
    oldparams = read(ROOT / "data/m4-scheduling-v1/parameters.json")
    assert (
        params["cost_model"] == oldparams["cost_model"]
        and params["closing_reserve_seconds"] == oldparams["closing_reserve_seconds"]
    )
    bundle = read(WS / "data/public_cpr/web-integration-v1/bundle-manifest.json")
    assert all(sha(WS / k) == h for k, h in bundle["pinned_sources_and_assets"].items())
    runtime = WS / "data/m4-exploration-v2/runtime"
    db = sqlite3.connect(runtime / "business.sqlite3")
    runs = [json.loads(r[0]) for r in db.execute("select body from runs")]
    assert all(
        r["status"] in {"completed", "partially_completed", "failed", "rejected", "cancelled"} for r in runs
    )
    counts = dict(db.execute("select status,count(*) from tools group by status").fetchall())
    db.close()
    assert counts == {"failed": 1, "succeeded": 17}, counts
    ui = read(RESULTS / "ui/acceptance.json")
    acceptance = {
        "accepted": True,
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "algorithm": "consent-coverage-dag-v2",
        "workspace": str(WS),
        "url": "http://127.0.0.1:8012",
        "entrypoint": "scripts/start_m4_v2.ps1",
        "profile_sha256": sha(profile_path),
        "parent_profile_sha256": before["parent_profile_sha256"],
        "graph_version": "cpr-review-12-v0.8-adaptive",
        "nodes": 12,
        "default_equal_observations": offline["default_equal_cases"],
        "optin_observations": offline["optin_cases"],
        "strict_coverage_improvement_observations": offline["strict_coverage_improvement_cases"],
        "unique_engineering_checks": len(unique),
        "policy_boundary_checks": 20,
        "ml_stage_checks": 5,
        "actual_current_profile_video_analyses": 5,
        "actual_current_profile_successful_tool_calls": 14,
        "total_runtime_tool_counts_including_initial_profile_diagnostics": counts,
        "api_exact_old_raw_scores_and_m3_states": True,
        "unanalysed_never_negative": True,
        "real_user_consent_and_budget_clicks": True,
        "full_expansion_only_two_missing_stages": True,
        "partial_source_and_history_and_export_supported": True,
        "terminal_restart_and_global_gpu_lock": True,
        "m1_m2_m3_numerical_sources_unchanged": True,
        "protected_sources_checked": len(protected),
        "m5": "baseline",
        "new_training": False,
        "manual_labels": False,
        "new_data_collection": False,
        "provider_calls": 0,
        "old_m4_source_entrypoint_release_preserved": True,
        "env_unchanged": True,
        "classification_accuracy_gain_claimed": False,
        "latency_speedup_claimed": False,
        "improvement_scope": "Availability and bounded explicit scope under insufficient complete-route budget; per-stage replanning. Not new CPR accuracy or a general routing optimum outside two certified routes.",
        "timing_limits": "Same six historical cost traces and observed-max x1.25 guard; no new timing fit or deadline completion guarantee. Full4, focused3, focused-then-expand5 remain unchanged.",
        "initial_issue": "Initial partial video succeeded, but follow-up source selection rejected partial status. Fixed source/history/export handling, six source-status regression cases added; initial profile/artifacts retained. Separate startup containment issue fixed by isolating copied inputs rather than weakening path validation.",
        "ui_partial_run_id": ui["partial_run_id"],
        "ui_expand_run_id": ui["expand_run_id"],
    }
    # A compact, self-contained report for teammates; no external libraries or secrets.
    DOC.write_text(
        """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>M4 优化与验证 · v2</title>
<style>*{box-sizing:border-box}body{margin:0;background:#f2f6f6;color:#253c46;font:15px/1.8 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1040px;margin:auto;padding:30px 20px 48px}header{background:#193f4e;color:white;border-radius:18px;padding:30px}header h1{font-size:29px;line-height:1.4;margin:8px 0}header p{color:#c8dde0;margin:8px 0}.eyebrow{letter-spacing:2px;font-size:12px;color:#8fd3ca}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:18px 0}.metric{background:white;border:1px solid #dbe5e8;border-radius:12px;padding:16px}.metric b{font-size:25px;display:block;color:#197c70}.metric span{font-size:12px;color:#607983}section{background:white;border:1px solid #dbe5e8;border-radius:14px;padding:22px 25px;margin-top:17px}h2{font-size:19px;margin:0 0 13px}p{margin:8px 0}.note{background:#e8f5f1;border-left:4px solid #208b79;padding:12px 16px;margin:15px 0}.muted{font-size:12px;color:#6c8188}.table{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;padding:13px;border-bottom:1px solid #e2e9ec;vertical-align:top}th{background:#edf4f6}code{font-size:12px;background:#eff4f5;padding:2px 5px;overflow-wrap:anywhere}.steps{display:flex;gap:8px;flex-wrap:wrap}.steps span{padding:8px 12px;border-radius:7px;background:#edf5f3;color:#286b61}a{color:#167d70}li{margin:7px 0}@media(max-width:650px){.cards{grid-template-columns:repeat(2,1fr)}header{padding:23px}header h1{font-size:23px}section{padding:18px}th,td{padding:10px;min-width:125px}}@media print{body{background:white}section,.metric{break-inside:avoid}main{padding:0}header{color:#193f4e;background:#eef5f4}header p{color:#345763}}</style>
<main><header><div class="eyebrow">M4 · VALIDATED V2 · 2026-10-27</div><h1>授权约束下的最大覆盖<br>与逐步重规划 Agent 调度</h1><p>预算不足时，经用户允许交付可验证的部分结果；完整检查仍是默认目标。</p></header>
<div class="cards"><div class="metric"><b>81,880</b><span>相同观察条件下，默认选择与 v1 一致</span></div><div class="metric"><b>7,620</b><span>有限测试网格中，授权后覆盖严格提升</span></div><div class="metric"><b>258 + 25</b><span>唯一工程检查 + 策略 / 执行检查通过</span></div><div class="metric"><b>14</b><span>当前版本真实成功工具调用</span></div></div>
<div class="note"><strong>保留理由：</strong>改进的是“有用结果的可交付性”和执行中的决策能力。没有降低分类门槛，没有改动 M1–M3 参数，也不宣称 CPR 识别精度或整体时延提升。</div>
<section><h2>从什么问题开始</h2><p>v1 要求一次覆盖全部请求目标。保守预算不足时，即使能完成已验证的4类检查，也会直接停止。工具开始执行后，原路径也不会在每个阶段重新评估。</p><div class="table"><table><tr><th>情况</th><th>v1</th><th>v2 保留的改进</th></tr><tr><td>完整路径超预算</td><td>停止，无新结果</td><td>默认仍停止；明确授权后选择有用的部分覆盖</td></tr><tr><td>工具阶段完成</td><td>继续原计划</td><td>扣除实际耗时，重算剩余依赖、覆盖与预算</td></tr><tr><td>已采用部分路径</td><td>无此交付方式</td><td>本轮范围固定，生成结果后停止；扩展另起请求</td></tr><tr><td>部分报告后续使用</td><td>上下文要求全部完成状态</td><td>验证来源后可解释、恢复、导出并扩展</td></tr></table></div></section>
<section><h2>算法边界清晰，可以独立验证</h2><p><strong>选择次序：</strong>合法工具与依赖 → 来源有效 → 预算可行 → 最大化新增请求类别覆盖 → 同等覆盖下选择较低边际成本。</p><div class="steps"><span>N05 评估已有证据</span><span>→ N06 约束规划</span><span>→ N07–N09 执行并验证一步</span><span>→ N05 重新评估</span><span>→ 满足目标或停止</span></div><p>仍只有两条认证路径：完整11类；骨架支持的单手、屈臂、站立、低频4类。覆盖表示“该类别已检查”，不等于识别一定正确。LLM 不能替用户授予部分检查权限。</p><p class="muted">新增实现：<code>consent-coverage-dag-v2</code>。保留12个状态节点。成本仍使用原6条历史执行记录和原保守保护值，没有为实验结果重新拟合成本。</p></section>
<section><h2>用户能实际感受到什么</h2><p>展开 <strong>Review options</strong>，选择时间上限；“允许部分检查”默认关闭。部分结果明确显示“4 / 11 已检查”和未检查类别，保留一次扩大全量入口。预算停止会解释原因并提供延长时间的重试入口。</p><p>扩展仅新增 RGB 分析与汇总，原报告保留。重开旧部分报告不会被新完整报告静默替换。不确定项仍是不确定，重复调用同一模型不能制造新的确定性。</p><div class="note"><strong>调用数没有改变：</strong>完整4步；局部3步；局部后再扩展共5步。已知需要全部类别时，直接完整检查更合适。</div></section>
<section><h2>验证结果与证据范围</h2><ul><li>穷举全部2,047种目标组合 × 4种依赖前缀 × 10档预算：默认与授权各81,880个观察条件。授权组目标覆盖不下降，7,620个条件严格提升；不把该比例当成真实用户收益。</li><li>258项唯一工程检查通过；20项策略边界和5项执行适配检查通过。RGB 失败后的部分回退在策略层测试，不冒充真实故障推理实验。</li><li>当前版本真实 HTTP 与浏览器共完成5次分析、14次成功工具调用。实际点击时间限制、授权、开始检查和扩展；分数、判断与 M3 校准结果和旧版对应结果一致。</li><li>桌面与手机无页面横向溢出；部分报告可追问、导出和重开；重启后来源与结果恢复，全量已有结果0次新工具调用；全局 GPU 锁能阻止冲突推理。</li></ul><p class="muted">初次联调暴露“部分状态不能作为上下文来源”的缺口，已修复并新增6种来源状态回归检查。初始记录单独保留，不将失败尝试计为通过。没有采集或标注新数据；没有重训视觉模型；DeepSeek 本轮关闭。</p></section>
<section><h2>运行与复核</h2><p>优化版：<a href="http://127.0.0.1:8012">http://127.0.0.1:8012</a>，启动 <code>.\\scripts\\start_m4_v2.ps1</code>。旧 M4 仍在8011，旧代码、入口与归档保留；M5仍为基线。</p><div class="table"><table><tr><td>运行代码 / 参数</td><td><code>workspaces/m4-refined-v2</code></td></tr><tr><td>离线策略对比</td><td><code>data/m4-exploration-v2/offline.json</code></td></tr><tr><td>工程、HTTP、界面与恢复证据</td><td><code>test-results/m4-exploration-v2</code></td></tr><tr><td>验收与源码归档</td><td><code>releases/m4-refined-v2</code></td></tr></table></div><p class="muted">这是两条已知路径上的项目算法改进，不承诺通用任务规划最优性、截止时间必完成或临床效果。收益随目标、预算和已有缓存而变化。</p></section></main></html>""",
        encoding="utf-8",
    )
    (DATA / "acceptance.json").write_text(json.dumps(acceptance, indent=2), encoding="utf-8")
    record = read(WS / "workspace.json")
    record["candidate_not_accepted"] = False
    record["accepted"] = True
    (WS / "workspace.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    RELEASE.mkdir()
    for name in ["business", "checkpoints"]:
        source = sqlite3.connect(runtime / (name + ".sqlite3"))
        target = sqlite3.connect(RELEASE / (name + ".sqlite3"))
        source.backup(target)
        target.close()
        source.close()
    files = {
        DOC,
        WS / "workspace.json",
        profile_path,
        WS / "tests/test_m4_v2.py",
        WS / "tests/test_m4_profile.py",
    }
    files.update(WS / k for k, h in profile["sources"].items() if parent["sources"].get(k) != h)
    files.update((ROOT / "experiments/m4_policy_v2").glob("*.py"))
    files.update(p for p in DATA.glob("*.json"))
    files.update(
        p
        for p in RESULTS.rglob("*")
        if p.is_file() and p.suffix in {".json", ".xml", ".log"} and p.parent != RESULTS / "service"
    )
    static_logs = RESULTS / "archived-service"
    static_logs.mkdir(exist_ok=False)
    for p in (RESULTS / "service").glob("*.log"):
        target = static_logs / p.name
        shutil.copy2(p, target)
        files.add(target)
    for pattern in ["*m4*v2*.py", "m4_v2*.cjs", "start_m4_v2.ps1"]:
        files.update((ROOT / "scripts").glob(pattern))
    files.update(
        WS / "data/m4-exploration-v2" / n for n in ["protocol.json", "decision.json", "parameters.json"]
    )
    manifest = {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(files)}
    archive = RELEASE / "source-and-evidence.zip"
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(files):
            assert p.is_relative_to(ROOT) and p.name != ".env"
            z.write(p, p.relative_to(ROOT).as_posix())
    (RELEASE / "manifest.json").write_text(
        json.dumps(
            {
                "files": manifest,
                "archive_sha256": sha(archive),
                "database_snapshots": {p.name: sha(p) for p in RELEASE.glob("*.sqlite3")},
                "contains_keys_or_vision_weights_or_video": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "accepted": True,
                "unique_checks": len(unique),
                "current_profile_tools": 14,
                "profile_sha256": sha(profile_path),
                "archive_files": len(files),
            }
        )
    )


if __name__ == "__main__":
    main()
