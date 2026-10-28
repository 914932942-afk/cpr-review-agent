"""Promote M5 only after numerical, paired-resource, live and recovery gates."""

import hashlib
import json
import shutil
import sqlite3
import statistics
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WS = ROOT / "workspaces/m5-cascade-v1"
PARENT = ROOT / "workspaces/m4-refined-v2"
DATA = ROOT / "data/m5-cascade-v1"
RESULTS = ROOT / "test-results/m5-cascade-v1"
RELEASE = ROOT / "releases/m5-cascade-v1"
DOC = ROOT / "docs/M5视觉级联算法与验收-v1.html"


def read(p):
    return json.loads(p.read_text(encoding="utf-8"))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    assert not RELEASE.exists() and not (DATA / "acceptance.json").exists(), "Preserve accepted release"
    before = read(DATA / "before.json")
    protocol = read(DATA / "protocol.json")
    assert sha(ROOT / ".env") == before["env_sha256"]
    assert sha(PARENT / "data/m4-exploration-v2/pipeline-manifest.json") == before["parent_profile_sha256"]
    assert sha(ROOT / "releases/m4-refined-v2/manifest.json") == before["parent_release_manifest_sha256"]
    assert all(sha(PARENT / p) == h for p, h in before["parent_sources"].items())
    for name in ["offline.json", "decision.json"]:
        assert read(DATA / name).get("passed", read(DATA / name).get("accepted"))
    for name in [
        "benchmark/acceptance.json",
        "raw26/acceptance.json",
        "ablation.json",
        "algorithm/acceptance.json",
        "http/acceptance.json",
        "ui/acceptance.json",
        "restart.json",
    ]:
        assert read(RESULTS / name)["passed"]
    assert sha(ROOT / "experiments/m5_cascade_v1/algorithm.py") == protocol["algorithm_sha256"]
    assert sha(ROOT / "experiments/m5_cascade_v1/runtime.py") == protocol["runtime_sha256"]
    assert all(
        sha(ROOT / p) == h for p, h in {**protocol["models"], **protocol["additional_sources"]}.items()
    )
    unique = set()
    for name in ["core-final.xml", "new-contracts.xml"]:
        xml = ET.parse(RESULTS / name)
        assert not list(xml.iter("failure")) and not list(xml.iter("error")) and not list(xml.iter("skipped"))
        unique.update((c.get("classname"), c.get("name")) for c in xml.iter("testcase"))
    assert len(unique) == 278, len(unique)
    profile_path = WS / "data/m5-cascade-v1/pipeline-manifest.json"
    profile = read(profile_path)
    assert all(sha(WS / p) == h for p, h in profile["sources"].items())
    protected = [
        p
        for p in before["parent_sources"]
        if p.startswith(
            (
                "experiments/m1_fast_v1/",
                "experiments/m2_competence_v3/",
                "experiments/m3_calibration_v1/",
                "experiments/m4_policy_v2/",
                "data/m1-m2-exploration-v1/",
                "data/m3-calibration-v1/",
                "data/m4-exploration-v2/",
            )
        )
    ]
    assert all(sha(WS / p) == before["parent_sources"][p] for p in protected)
    bundle = read(WS / "data/public_cpr/web-integration-v1/bundle-manifest.json")
    assert all(sha(WS / p) == h for p, h in bundle["pinned_sources_and_assets"].items())
    runtime = WS / "data/m5-cascade-v1/runtime"
    db = sqlite3.connect(runtime / "business.sqlite3")
    assert all(
        json.loads(r[0])["status"] in {"completed", "partially_completed", "failed", "rejected", "cancelled"}
        for r in db.execute("select body from runs")
    )
    counts = dict(db.execute("select status,count(*) from tools group by status").fetchall())
    db.close()
    assert counts == {"failed": 1, "succeeded": 11}, counts
    bench = read(RESULTS / "benchmark/acceptance.json")
    raw = read(RESULTS / "raw26/acceptance.json")
    ablation = read(RESULTS / "ablation.json")
    assert (
        bench["median_relative_process_time_reduction"] >= 0.1
        and bench["median_relative_cuda_peak_reduction"] >= 0.1
        and raw["max_score_difference"] == 0
    )
    acceptance = {
        "accepted": True,
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "workspace": str(WS),
        "entrypoint": "scripts/start_m5.ps1",
        "url": "http://127.0.0.1:8013",
        "graph_version": "cpr-review-12-v0.9-cascade",
        "nodes": 12,
        "algorithm": "quality-dependency-cascade-v1",
        "profile_sha256": sha(profile_path),
        "parent_profile_sha256": before["parent_profile_sha256"],
        "model_weights_and_thresholds_unchanged": True,
        "m1_m2_m3_m4_numerical_algorithms_and_parameters_unchanged": True,
        "protected_source_files": len(protected),
        "all_five_modules_implemented": True,
        "unique_engineering_checks": len(unique),
        "algorithm_checks": 15,
        "cached_public_records": 66,
        "quality_readout_comparisons": 132,
        "counterfactual_feature_invariance_cases": 200,
        "actual_public_videos": 26,
        "actual_video_max_score_difference": 0,
        "paired_resource_tests": 9,
        "unique_benchmark_videos": 3,
        "benchmark_processes_including_controlled_fallback": 20,
        "median_paired_relative_process_time_reduction": bench["median_relative_process_time_reduction"],
        "median_paired_relative_cuda_peak_reduction": bench["median_relative_cuda_peak_reduction"],
        "controlled_quality_fallback_matches_original": True,
        "quality_gate_removed_ablation_decision_changes": ablation["quality_gate_removed"][
            "different_class_decisions"
        ],
        "current_live_successful_tools": 11,
        "current_live_successful_m5_calls": 3,
        "gpu_lock_intentional_failures": 1,
        "real_browser_source_and_uncertainty_preserved": True,
        "restart_verified": True,
        "new_training": False,
        "new_manual_labels": False,
        "new_data_collection": False,
        "provider_calls": 0,
        "env_unchanged": True,
        "accuracy_improvement_claimed": False,
        "whole_system_latency_improvement_claimed": False,
        "resource_scope": "M5-only cold processes; nine alternating paired runs on three previously inspected videos. Paired relative-change median, not ratio of overall medians. Same16 sample budget and original minimum8 observability.",
        "contribution": "Compile actual selected expert input dependencies after the original hand-observability gate; execute only required feature domains. Full frozen RGB fallback when any active dependency needs dense visual features. Never persist uncomputed embeddings as observations.",
        "limitations": "Exact pruning is certified for the pinned heads and original gates. Existing26 is an exposed engineering regression, not independent clinical/cross-person evidence. M4 cost bounds remain the earlier conservative parameters.",
    }
    time_gain = 100 * bench["median_relative_process_time_reduction"]
    memory_gain = 100 * bench["median_relative_cuda_peak_reduction"]
    base_mem = statistics.median(r["baseline_memory"] for r in bench["records"]) / (1024**2)
    new_mem = statistics.median(r["candidate_memory"] for r in bench["records"]) / (1024**2)
    DOC.write_text(
        f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>M5 视觉级联算法与验收</title>
<style>*{{box-sizing:border-box}}body{{margin:0;background:#f0f5f4;color:#263e45;font:15px/1.8 system-ui,"Microsoft YaHei",sans-serif}}main{{max-width:1050px;margin:auto;padding:30px 20px 48px}}header{{background:#193e4b;border-radius:17px;padding:30px;color:#fff}}header h1{{font-size:29px;line-height:1.4;margin:9px 0}}header p{{color:#c9dfe0;margin:8px 0}}.eyebrow{{font-size:12px;letter-spacing:2px;color:#97d6ca}}.metrics{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:18px 0}}.metric,section{{background:#fff;border:1px solid #dbe6e5;border-radius:13px}}.metric{{padding:17px}}.metric b{{display:block;color:#167b69;font-size:28px}}.metric span{{font-size:12px;color:#698087}}section{{padding:23px 25px;margin-top:17px}}h2{{font-size:19px;margin:0 0 12px}}p{{margin:8px 0}}.note{{background:#e6f4ed;border-left:4px solid #248874;padding:13px 16px;margin:15px 0}}.muted{{color:#6e8289;font-size:12px}}.flow{{display:flex;gap:8px;flex-wrap:wrap}}.flow span{{padding:8px 12px;border-radius:7px;background:#edf5f2}}.table{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:12px;vertical-align:top;text-align:left;border-bottom:1px solid #e1e9e8}}th{{background:#edf4f5}}code{{font-size:12px;background:#edf3f4;padding:3px 5px;overflow-wrap:anywhere}}a{{color:#157967}}li{{margin:7px 0}}.bar{{height:12px;background:#e5eeee;border-radius:20px;margin:6px 0 14px;overflow:hidden}}.bar i{{display:block;height:100%;background:#377b72}}@media(max-width:650px){{.metrics{{grid-template-columns:repeat(2,1fr)}}header{{padding:23px}}header h1{{font-size:23px}}section{{padding:18px}}th,td{{padding:9px;min-width:120px}}}}@media print{{body{{background:white}}main{{padding:0}}section,.metric{{break-inside:avoid}}header{{background:#eaf3ef;color:#193e4b}}header p{{color:#3d6561}}}}</style>
<main><header><div class="eyebrow">M5 · XIAOYU LI · VALIDATED V1</div><h1>质量门控与特征依赖驱动的<br>轻量化视觉级联识别</h1><p>根据实际生效的分类专家，证明哪些特征必须计算，再执行相应视觉分支。识别依据、标签与阈值保持原样。</p></header>
<div class="metrics"><div class="metric"><b>{time_gain:.1f}%</b><span>M5 配对进程耗时相对降幅中位数</span></div><div class="metric"><b>{memory_gain:.1f}%</b><span>M5 配对 CUDA 峰值显存相对降幅中位数</span></div><div class="metric"><b>26 / 26</b><span>实际视频输出分数与旧版完全一致</span></div><div class="metric"><b>278 + 15</b><span>唯一工程检查 + 算法检查通过</span></div></div>
<div class="note"><strong>M5 的改进价值：</strong>将视觉推理改为可验证的条件计算，减少未进入最终判定的特征计算。保持同样16帧采样、原有最低8帧支持和全部模型参数；无需新增训练、采集或标注。</div>
<section><h2>为什么可以省下这些计算</h2><p>当前手型专家在局部图像可用时只读取 HOG 手部形状；另外三类专家读取姿态与接触区域特征。原流程仍会生成 DINO 特征，随后其中部分预测又被手型专家覆盖。</p><div class="table"><table><tr><th>功能</th><th>实际使用的特征</th><th>M5 执行方式</th></tr><tr><td>手型相关1 / 2类</td><td>原可用性规则通过时，采用 HOG 专家</td><td>保留原专家和阈值，跳过已被替代的分支</td></tr><tr><td>视觉支持的5 / 7 / 9类</td><td>共享人物、姿态和接触区域上下文</td><td>仍对原16帧做人物与 ROI 检查</td></tr><tr><td>手部可用帧不足 / 存在视觉特征依赖</td><td>完整原 RGB 特征</td><td>启用完整视觉回退</td></tr></table></div><p class="muted">可用性门控沿用原规则：有效手部 ROI 至少8帧。它不额外宣称能判断图像模糊程度或动作正确性。</p></section>
<section><h2>算法设计与可证明的边界</h2><div class="flow"><span>检查原手部可用性</span><span>→ 解析当前生效专家</span><span>→ 合并输入特征依赖</span><span>→ 按需计算 / 完整回退</span><span>→ 原分类头输出</span></div><p><strong>依赖不相交，输出不变：</strong>若所有生效专家都不读取未计算的特征位置，这些位置就不会影响其分数与判断。运行时会核验输入索引、模型指纹和依赖计划，模型结构改变后不能沿用旧证书。</p><p>未计算的 DINO 嵌入不会作为真实观测写进产物。轻量路径只保存实际计算的上下文与手型特征；完整回退保存完整 RGB 特征。人物绑定、ROI 有效性和样本支持判据均保留。</p><p class="muted">实现：<code>quality-dependency-cascade-v1</code>。贡献属于推理算法与条件计算优化，未修改识别网络权重或分类阈值。</p></section>
<section><h2>对照与消融：收益来自哪里</h2><div class="table"><table><tr><th>对照</th><th>结果</th><th>说明</th></tr><tr><td>原始全量计算</td><td>作为同输入、同分类头基线</td><td>3段既有视频 × 3次配对，顺序交替；共18个独立进程</td></tr><tr><td>完整 M5</td><td>配对耗时降幅中位数 {time_gain:.1f}%；峰值显存降幅中位数 {memory_gain:.1f}%</td><td>3段视频各自中位耗时均改善；保持所有判断</td></tr><tr><td>移除质量门控</td><td>受控低手部置信度条件下，与合法基线出现1项判断变化</td><td>正确回退版本与原流程一致；不是新增真实遮挡标注</td></tr><tr><td>改变生效专家的特征依赖</td><td>需要语义特征时，不能继续使用上下文轻量路径</td><td>由依赖编译器和回归测试保护</td></tr></table></div><p>基线 CUDA 峰值约 <strong>{base_mem:.1f} MiB</strong>，轻量路径约 <strong>{new_mem:.1f} MiB</strong>。手型路径满足条件时，DINO 骨干不加载；YOLO 人物与 ROI 检查继续执行。</p><p class="muted">时间指标是逐对相对降幅的中位数，不是两个耗时中位数的比值。测量对象仅为 M5 阶段进程，不能直接推导全系统加速比例。</p></section>
<section><h2>验证已完成</h2><ul><li>40条开发记录与26条既有缓存：原可用状态及强制不可用状态共132次读出对照，结果完全一致；另有200组任意特征不干扰检查。</li><li>重新处理全部26段原始视频：上下文一致，分数最大差异0，类别判断一致，没有把未计算的嵌入保存为观测。</li><li>278项唯一工程检查与15项算法检查通过，覆盖模型变更、伪造依赖、错误支持数量、缺失回退特征和来源绑定。</li><li>真实主链路与浏览器点击：局部检查跳过 M5；扩展仅补 RGB 与汇总；M5 元数据可追溯，不确定项保留；桌面与手机布局通过。</li><li>重启后恢复报告与来源、已有结果0次新增分析、全局 GPU 锁冲突控制均通过。</li></ul><p class="muted">既有26段是反复使用过的工程回归样例，不是新的独立盲测。这里不宣称识别准确率提高；M4 继续使用原有保守成本参数。DeepSeek 本轮关闭，.env 未修改。</p></section>
<section><h2>与五模块系统的关系</h2><div class="table"><table><tr><th>模块 / 负责人</th><th>职责</th><th>本轮状态</th></tr><tr><td>M1 · Jinlai GE</td><td>骨架质量处理与稳定识别</td><td>数值算法与参数保持</td></tr><tr><td>M2 · Wenbo Zhang</td><td>类别 / 质量感知融合</td><td>保持</td></tr><tr><td>M3 · Wenxin Yan</td><td>质量条件校准与不确定项复核</td><td>保持</td></tr><tr><td>M4 · Mingmin Guo</td><td>目标 / 预算感知的工具调度</td><td>保持；决定是否调用视觉工具</td></tr><tr><td>M5 · Xiaoyu Li</td><td>视觉工具内部的质量门控与依赖感知级联</td><td>已实现、验证并接入</td></tr></table></div><p>12节点 LangGraph 主链路保留。M4 决定调用什么工具，M5 决定工具内部计算哪些视觉特征；两者各自有独立的改进算法与验证指标。</p></section>
<section><h2>运行与查看</h2><p>当前入口：<a href="http://127.0.0.1:8013">http://127.0.0.1:8013</a>。在 CODE 根目录运行 <code>.\\scripts\\start_m5.ps1</code>。运行代码：<code>workspaces/m5-cascade-v1</code>。</p><p>协议与证书：<code>data/m5-cascade-v1</code>；测试证据：<code>test-results/m5-cascade-v1</code>；源码、测试输出与数据库归档：<code>releases/m5-cascade-v1</code>。原 M4 v2 入口8012及其代码保留。</p></section></main></html>""",
        encoding="utf-8",
    )
    (DATA / "acceptance.json").write_text(json.dumps(acceptance, indent=2), encoding="utf-8")
    old_readme = (ROOT / "README.md").read_text(encoding="utf-8")
    (DATA / "README-before-M5.md").write_text(old_readme, encoding="utf-8")
    intro = f"""M1–M5 已接入并完成当前版本验收，推荐运行 `.\\scripts\\start_m5.ps1`，打开 http://127.0.0.1:8013 。M5 使用质量门控与特征依赖驱动的视觉级联；9组配对测试的 M5 进程耗时相对降幅中位数为 {time_gain:.1f}%，CUDA 峰值显存相对降幅中位数为 {memory_gain:.1f}%，26段实际视频分数和判断保持一致。M1–M4 数值算法与参数不变，DeepSeek 默认关闭。代码在 `workspaces/m5-cascade-v1`；278项唯一工程检查通过。见 [M5算法与验收](docs/M5视觉级联算法与验收-v1.html)。

### 历史验收入口（按版本保留）

"""
    (ROOT / "README.md").write_text(
        old_readme.replace("## 当前系统主线\n\n", "## 当前系统主线\n\n" + intro, 1), encoding="utf-8"
    )
    (WS / "README.md").write_text(
        "# CPR Review · M1–M5\n\nValidated M5 quality/dependency visual cascade. Start from the CODE root with `scripts/start_m5.ps1`, then open http://127.0.0.1:8013 .\n\nM1–M4 numerical algorithms and parameters are unchanged; M5 retains the same sixteen-frame sampling and frozen classifier thresholds. DeepSeek is disabled by this launcher.\n\n[Algorithm and acceptance](../../docs/M5视觉级联算法与验收-v1.html)\n",
        encoding="utf-8",
    )
    record = read(WS / "workspace.json")
    record.update(candidate_not_accepted=False, accepted=True)
    (WS / "workspace.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    RELEASE.mkdir()
    for name in ["business", "checkpoints"]:
        source = sqlite3.connect(runtime / (name + ".sqlite3"))
        target = sqlite3.connect(RELEASE / (name + ".sqlite3"))
        source.backup(target)
        target.close()
        source.close()
    static = RESULTS / "archived-service"
    static.mkdir()
    for p in (RESULTS / "service").glob("*.log"):
        shutil.copy2(p, static / p.name)
    files = {
        DOC,
        ROOT / "README.md",
        WS / "README.md",
        WS / "workspace.json",
        profile_path,
        WS / "tests/test_m4_profile.py",
        WS / "tests/test_m5_profile.py",
    }
    files.update(WS / p for p, h in profile["sources"].items() if before["parent_sources"].get(p) != h)
    files.update((ROOT / "experiments/m5_cascade_v1").glob("*.py"))
    files.update(p for p in DATA.glob("*") if p.is_file())
    files.update(
        p
        for p in RESULTS.rglob("*")
        if p.is_file() and p.suffix in {".json", ".xml", ".log", ".npz"} and p.parent != RESULTS / "service"
    )
    for pattern in ["*m5*.py", "m5*.cjs", "start_m5.ps1"]:
        files.update((ROOT / "scripts").glob(pattern))
    files.update(
        WS / "data/m5-cascade-v1" / n for n in ["protocol.json", "certificate.json", "decision.json"]
    )
    hashes = {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(files)}
    archive = RELEASE / "source-and-evidence.zip"
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(files):
            assert (
                p.is_relative_to(ROOT)
                and p.name != ".env"
                and p.suffix not in {".pt", ".pth", ".mp4", ".safetensors"}
            )
            z.write(p, p.relative_to(ROOT).as_posix())
    (RELEASE / "manifest.json").write_text(
        json.dumps(
            {
                "files": hashes,
                "archive_sha256": sha(archive),
                "database_snapshots": {p.name: sha(p) for p in RELEASE.glob("*.sqlite3")},
                "contains_keys_weights_or_video": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "accepted": True,
                "profile_sha256": sha(profile_path),
                "unique_checks": len(unique),
                "time_gain": time_gain,
                "memory_gain": memory_gain,
                "archive_files": len(files),
            }
        )
    )


if __name__ == "__main__":
    main()
