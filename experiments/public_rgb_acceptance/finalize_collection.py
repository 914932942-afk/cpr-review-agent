"""Audit material artifacts and freeze a collection manifest, without inference."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import html
import json
from common import CODE, DATA, ROOT, REPORT, SIZE, SHA, LABELS, read, sha, write
from build_media import folder_for, verify_frame_numbers


def main():
    if (ROOT/"collection-manifest.json").exists():
        raise FileExistsError("Collection is already frozen; use a new version for changes")
    plan=read(ROOT/"frozen-record-plan.json");proof=read(ROOT/"source-archive-integrity.json")
    fingerprint=read(REPORT/"record-plan-fingerprint.json");tests=read(REPORT/"curation-tests.json")
    if sha(ROOT/"frozen-record-plan.json")!=fingerprint["sha256"] or not tests["passed"]:
        raise ValueError("Frozen plan or curation tests did not pass")
    if not proof["verified"] or proof["sha256"]!=SHA or proof["bytes"]!=SIZE:
        raise ValueError("Whole supplemental archive integrity failed")
    current_checks={"head_checkpoint_unchanged":sha(DATA/"improvement-v2/combined-64.pt")==plan["current_head_checkpoint_sha256"],
        "training_split_unchanged":sha(DATA/"improvement-v2/record-splits.json")==plan["current_head_split_sha256"],
        "source_record_labels_unchanged":sha(DATA/"run/records.json")==plan["source_records_sha256"]}
    protocol=read(DATA/"improvement-v2/protocol.json")
    current_checks["split_matches_trained_protocol"]=sha(DATA/"improvement-v2/record-splits.json")==protocol["split_sha256"]
    current_checks["labels_match_trained_protocol"]=sha(DATA/"run/records.json")==protocol["source_records_sha256"]
    pipeline=read(CODE/"experiments/public_cpr_improved/inference-manifest.json")
    current_checks["existing_inference_pipeline_unchanged"]=all(sha(CODE/r["path"])==r["sha256"] for r in pipeline["pinned_files"])
    snapshot=read(CODE/"test-results/public-head-validation/application-source-before.json")
    changed=[name for name,digest in snapshot.items() if not (CODE/name).is_file() or sha(CODE/name)!=digest]
    current_checks["application_source_unchanged"]=not changed
    if not all(current_checks.values()):raise ValueError("An existing model, split, labels or application source changed")
    records=[];source_bytes=video_bytes=frames=0
    for expected in plan["records"]:
        folder=folder_for(expected);manifest_path=folder/"manifest.json";manifest=read(manifest_path)
        for name in ("record_id","original_labels","expected_frames","current_head_split"):
            if manifest[name]!=expected[name]:raise ValueError("Media record differs from frozen plan")
        if manifest["state"]!="video_verified":raise ValueError("Video not fully verified")
        verify_frame_numbers(manifest["images"],expected["expected_frames"])
        if len(list((folder/"images").glob("*.jpg")))!=expected["expected_frames"]:
            raise ValueError("On-disk JPEG count differs from manifest")
        for image in manifest["images"]:
            path=folder/"images"/image["filename"]
            if path.stat().st_size!=image["bytes"] or sha(path)!=image["sha256"]:
                raise ValueError("On-disk public JPEG hash differs")
            source_bytes+=image["bytes"]
        video=folder/"video.mp4"
        if sha(video)!=manifest["video_sha256"] or manifest["video_validation"]["decoded_frames"]!=expected["expected_frames"]:
            raise ValueError("Video integrity or complete-decode count differs")
        frames+=expected["expected_frames"];video_bytes+=video.stat().st_size
        records.append({**expected,"image_shape":manifest["image_shape"],
            "source_manifest":str(manifest_path.relative_to(ROOT)).replace("\\","/"),
            "source_manifest_sha256":sha(manifest_path),"video":str(video.relative_to(ROOT)).replace("\\","/"),
            "video_sha256":manifest["video_sha256"],"video_bytes":video.stat().st_size,
            "duration_seconds":manifest["video_validation"]["duration_seconds"],"media_verified":True})
        print("Final on-disk hashes verified",expected["record_id"],flush=True)
    if frames!=plan["total_frames"] or {i for r in records for i in r["original_labels"]}!=set(LABELS):
        raise ValueError("Final material coverage differs from frozen plan")
    collection={"schema":"public-rgb-acceptance-collection-v1","step_1_complete":True,
        "completed_utc":datetime.now(timezone.utc).isoformat(),"dataset_revision":plan["dataset_revision"],
        "frozen_record_plan_sha256":fingerprint["sha256"],"supplemental_archive_sha256":SHA,
        "supplemental_archive_bytes":SIZE,"supplemental_archive_fully_verified":True,
        "records":records,"record_count":len(records),"newly_added_records":22,"reused_previous_records":4,
        "source_frames":frames,"normal_records":sum(r["original_labels"]==[0] for r in records),
        "error_class_count":11,"label_record_counts":dict(Counter(str(i) for r in records for i in r["original_labels"])),
        "jpeg_bytes":source_bytes,"mp4_bytes":video_bytes,"total_seconds":sum(r["duration_seconds"] for r in records),
        "current_head_split_counts":plan["current_head_split_counts"],"current_model_checks":current_checks,
        "head_reference":{"path":"data/public_cpr/improvement-v2/combined-64.pt",
            "sha256":plan["current_head_checkpoint_sha256"],"protocol_sha256":sha(DATA/"improvement-v2/protocol.json")},
        "original_application_files_checked":len(snapshot),"curation_tests":tests,
        "model_training_performed":False,"model_inference_performed":False,"new_manual_labels":False,
        "independent_unseen_test_claim":False,"recognition_accuracy_evaluated":False,
        "step_2_split_or_refit_started":False,"existing_inference_manifest_modified":False,
        "limitations":["Small single-error/Correct engineering collection; no composite RGB coverage",
            "20 recordings used for current-head fitting and 2 for tuning; cannot claim independent current-head test",
            "4 existing test-video results were previously inspected",
            "No error-onset, measured depth, participant identity or cross-person labels created",
            "Derived MP4 is lossy; exact public JPEG bytes and provenance retained"],
        "next_step":"Exclude future-test recording groups from fitting and tuning before refitting a small head in a new protocol."}
    write(ROOT/"collection-manifest.json",collection)
    write(REPORT/"collection-fingerprint.json",{"path":"data/public_cpr/rgb-acceptance-v1/collection-manifest.json",
        "sha256":sha(ROOT/"collection-manifest.json")})
    write(REPORT/"collection-summary.json",{k:v for k,v in collection.items() if k!="records"})
    split_names={"train":"训练相关","validation":"调参相关","test":"原测试，已查看"}
    rows=""
    for record in records:
        prefix="../../data/public_cpr/rgb-acceptance-v1/";labels=" / ".join(record["label_names"])
        shape=record["image_shape"]
        rows+=f'<tr><td><code>{html.escape(record["record_id"])}</code></td><td>{html.escape(labels)}</td><td>{record["expected_frames"]}</td><td>{shape[1]} × {shape[0]}</td><td>{split_names[record["current_head_split"]]}</td><td><a href="{prefix+record["video"]}">视频</a> · <a href="{prefix+record["source_manifest"]}">出处</a></td></tr>'
    document=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>公开RGB素材验收 · 第一项</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#f3f6fa;color:#233247;font:15px/1.7 "Segoe UI","Microsoft YaHei",sans-serif}}main{{max-width:1160px;margin:32px auto;padding:0 24px 48px}}h1{{font-size:30px;line-height:1.35;margin:6px 0}}h2{{font-size:19px;margin:0 0 12px}}p{{margin:9px 0}}.muted{{color:#63738a}}.eyebrow{{color:#2864b4;font-weight:700}}section,.card{{background:white;border:1px solid #dce4ef;border-radius:14px;padding:21px;margin-top:20px}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}}.card{{margin:0}}.number{{font-size:30px;font-weight:750;color:#215ba4}}.notice{{background:#edf7f0;border-left:4px solid #27865b;padding:15px 18px;margin:20px 0}}.boundary{{background:#fff8e9;border-left:4px solid #dd982c;padding:15px 18px;margin-top:16px}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{text-align:left;padding:9px 8px;border-bottom:1px solid #e5eaf1;vertical-align:top}}th{{background:#edf3fa}}code{{font:12px/1.5 Consolas,monospace;word-break:break-all}}a{{color:#2365b0}}.scroll{{overflow:auto}}li{{margin:7px 0}}@media(max-width:760px){{.grid{{grid-template-columns:1fr}}main{{padding:0 14px}}h1{{font-size:25px}}section{{padding:15px}}}}
</style><main><div class="eyebrow">CPR + AGENT / PUBLIC RGB / STEP 01</div><h1>公开 RGB 验收素材已补齐</h1><p class="muted">正常动作 + 11 类单错误 · 原始公开标签 · 只取 ch0 · 无采集、无新增人工标注</p>
<div class="notice"><b>第一项完成：</b>26段素材的原标签、图片完整性、连续帧号和视频完整解码均通过核对。本轮没有训练、推理或调整阈值；第二项划分与重拟合尚未执行。</div>
<div class="grid"><div class="card"><span class="muted">可追踪的公开片段</span><div class="number">26 段</div><span>新增22段，复用原有4段</span></div><div class="card"><span class="muted">类别覆盖</span><div class="number">正常 + 11 类</div><span>正常2段；错误片段24段</span></div><div class="card"><span class="muted">完整验证的帧</span><div class="number">{frames:,}</div><span>视频总时长约{collection['total_seconds']/60:.1f}分钟</span></div></div>
<section><h2>获取与验证</h2><p>已获取固定版本的补充发布包，大小9.68 GB，完整SHA-256与公开值一致。只解出24段ch0 RGB；另外2段主集录像复用已有固定Parquet来源与逐帧哈希，没有下载其完整大包。</p><p>每段均核对原标签、预期帧数、连续帧号、JPEG结构、分辨率、逐帧哈希、MP4完整解码帧数和25fps时间基准。保留精确原始JPEG，新增MP4使用H.264 CRF18有损导出，并记录首、中、末帧的像素差异。</p><p class="muted">原始JPEG约{source_bytes/10**9:.2f} GB，MP4约{video_bytes/10**9:.2f} GB。素材工具{tests['tests_run']}项测试通过；既有模型、训练划分和{len(snapshot)}个应用源文件保持不变。</p></section>
<section><h2>当前训练关系必须保留</h2><div class="boundary"><b>20段训练相关，2段调参相关，4段原测试录像的结果已查看。</b>本批当前用于工程回归和下一步划分准备，不能直接据此宣称当前模型的独立测试效果。第二项应先排除未来测试录像组，再拟合小分类头。</div><p>这次不改变任何原标签或现有划分，不把素材完整性通过解释为模型识别通过。每类样例仍少，没有新增组合错误RGB、错误精确时刻、实测深度或受试者身份标签。</p></section>
<section><h2>固定素材清单</h2><p class="muted">标签名称沿用公开数据。训练关系描述当前分类头所用骨架的来源；“原测试”条目的视频结果此前已查看。</p><div class="scroll"><table><thead><tr><th>录像编号</th><th>原标签名称</th><th>帧数</th><th>分辨率</th><th>当前模型关系</th><th>查看</th></tr></thead><tbody>{rows}</tbody></table></div></section>
<section><h2>文件与后续边界</h2><p><a href="../../data/public_cpr/rgb-acceptance-v1/collection-manifest.json">完整素材清单JSON</a> · <a href="../../data/public_cpr/rgb-acceptance-v1/source-archive-integrity.json">发布包SHA校验</a> · <a href="curation-tests.txt">测试记录</a> · <a href="collection-summary.json">验收摘要</a></p><p>数据位于 <code>CODE/data/public_cpr/rgb-acceptance-v1/records/</code>，每段都有原图、视频和出处清单。获取脚本及复现说明位于 <code>CODE/experiments/public_rgb_acceptance/</code>。</p><p><b>下一项：</b>按录像组处理未来测试与训练/调参之间的隔离，在新协议下重拟合小分类头；保留旧协议和历史结果。</p><p class="muted">来源：<a href="https://huggingface.co/datasets/ShunliWang/CPR-Coach">CPR-Coach公开数据</a> · <a href="https://github.com/Shunli-Wang/CPR-Coach">原项目</a></p></section></main></html>'''
    (REPORT/"素材验收报告.html").write_text(document,encoding="utf-8")
    (REPORT/"验收结论.txt").write_text(f"第一项完成：26段公开RGB素材、{frames}帧，覆盖正常动作与11类单错误。\n完整补充包SHA-256一致，图片与MP4逐段核验通过。\n新增22段，复用4段；当前20段训练相关、2段调参相关、4段原测试录像结果已查看。\n本批不宣称独立测试，不计算识别准确率。\n未采集或新增人工标注，未训练/推理，未修改模型、阈值或原划分。\n下一项尚未执行。详情见素材验收报告.html。\n",encoding="utf-8")
    print(json.dumps({"step_1_complete":True,"records":len(records),"frames":frames,
        "new_records":22,"current_model_checks":current_checks,"training_or_inference_performed":False},indent=2),flush=True)


if __name__=="__main__":main()
