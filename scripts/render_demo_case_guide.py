"""Render a local, concise casebook from verified receipts. Does not call the application."""

from __future__ import annotations

import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def esc(value):
    return html.escape(str(value), quote=True)


def main():
    cases = json.loads((ROOT / "data/demo-cases-v1/cases.json").read_text(encoding="utf-8"))
    target = ROOT / "docs/演示案例-v1.html"
    if target.exists():
        raise FileExistsError("Preserve the existing guide")
    by_id = {c["id"]: c for c in cases["cases"]}
    base = cases["base_url"]

    def copy(label, value):
        return f'<button class="copy" data-copy="{esc(value)}">{esc(label)}</button>'

    def name(case):
        return f'<p class="workspace">左侧练习：<strong>{esc(case["workspace"]["name"])}</strong></p>'

    def source(case):
        item = case["source"]
        path = ROOT / item["video_rel"]
        return (
            "<details><summary>素材与来源</summary>"
            f"<p>{esc(item['record_id'])} · {item['frames']} 帧 · {item['frames'] / item['fps']:.2f} 秒</p>"
            f"{copy('复制原视频路径', str(path))}"
            f'<video controls preload="none" aria-label="{esc(case["id"])} public clip" src="../{esc(item["video_rel"])}"></video>'
            f'<p class="small mono">SHA-256 {esc(item["video_sha256"])}</p></details>'
        )

    def report(case):
        return f'<a class="button secondary" target="_blank" href="{esc(base)}/api/reports/{esc(case["verified"]["report_id"])}/export">打开已验证报告 ↗</a>'

    content = []
    labels = {
        "C01": ("Bending Arms · #4", "弯臂 · #4"),
        "C02": ("Tilting Arms · #5", "手臂倾斜 · #5"),
        "C03": ("Correct · #0", "11 类均未标记"),
    }
    steps = {
        "C01": "选择 video.mp4；在 Review history 中打开完整范围的 Video review。查看弯臂提示、回放原视频，再展示已有摘要或参考。",
        "C02": "选择 follow-up-practice.mp4，打开最新 Video review。此记录已展开全部 11 类，显示手臂倾斜；原来只看弯臂时不显示该提示，是范围不同。",
        "C03": "选择 case-03-normal-labelled.mp4。展示未标记提示及支持类别范围，说明系统不会把“没有提示”包装成动作合格。",
    }
    for identity in ["C01", "C02", "C03"]:
        case = by_id[identity]
        truth, flags = labels[identity]
        content.append(
            f'<article class="card{" wide" if identity == "C03" else ""}" id="{identity}"><div class="case-id">{identity} · 真实视频</div><h2>{esc(case["title"])}</h2>'
            f'<div class="comparison"><div><span>公开数据标签</span><strong>{esc(truth)}</strong></div><div><span>当前模型提示 · 全部11类</span><strong>{esc(flags)}</strong></div></div>'
            f'{name(case)}<p>{esc(steps[identity])}</p><div class="actions">{copy("复制练习名称", case["workspace"]["name"])}{report(case)}</div>'
            + (
                '<p class="note">数据集标为 Correct；本系统实际输出是“未标记支持错误”，不是临床合格认证。</p>'
                if identity == "C03"
                else ""
            )
            + source(case)
            + "</article>"
        )

    bad, short = by_id["C04"], by_id["C05"]
    content.append(
        '<section class="card wide" id="failures"><div class="case-id">C04 / C05 · 失败与恢复</div><h2>失败时不制造识别结论</h2>'
        '<div class="boundary-grid"><article><h3>C04 · 损坏的 MP4 容器</h3><p>在任意已有练习中点击 Change video，选择此测试文件。上传立即拒绝，原练习仍保持打开，不创建新分析。</p>'
        f'{copy("复制损坏文件路径", str(ROOT / bad["source"]["path"]))}<p class="result">已核验：HTTP 415 · INVALID_CONTAINER · 无视觉调用</p></article>'
        f"<article><h3>C05 · 只有三帧的可解码视频</h3>{name(short)}<p>选择保存的失败记录，即可展示失败提示、没有发现/预测的状态，再切回正常案例。若重新上传，它会在首个工具的前置检查失败。</p>"
        f"{copy('复制练习名称', short['workspace']['name'])} {copy('复制三帧文件路径', str(ROOT / short['source']['path']))}"
        '<p class="result">已核验：首阶段失败 · 后三个阶段不执行 · 无分类结论</p></article></div>'
        '<p class="small">这两个是明确标记的边界测试输入。三帧片段由已下载公开视频截取生成，不是新增 CPR 数据集样本，不带 CPR 标签，不参与训练或精度评估。</p></section>'
    )
    ref, continuity = by_id["C06"], by_id["C07"]
    content.append(
        '<section class="card" id="C06"><div class="case-id">C06 · AGENT / 有来源的工具结果</div><h2>参考片段对照</h2>'
        f"{name(ref)}<p>在 Review history 选择 Reference comparison。切换正常标签参考与弯臂示例，点击 Play both from start；来源可展开查看。</p>"
        f'{copy("复制练习名称", ref["workspace"]["name"])}<p class="result">已保存真实 Agent 请求与检索回执，现场回看不再次调用模型或 CV。</p>'
        '<p class="small">两段视频从各自起点播放，没有动作时间对齐；参考不是认证教学示范。</p></section>'
    )
    content.append(
        '<section class="card" id="C07"><div class="case-id">C07 · AGENT / 显式记忆</div><h2>跨次衔接，不继承旧结论</h2>'
        f"{name(continuity)}<p>查看已沿用的 Bent arms 起始范围，点击 View earlier recap。弹窗属于旧视频；当前视频没有分析结果。右侧已有根据起始目标找到的参考。</p>"
        f'{copy("复制练习名称", continuity["workspace"]["name"])}<p class="result">已核验：记住目标 #4 · 当前分析来源为空 · 无预测或错误结论</p>'
        '<p class="small">这展示“目标记忆 + 当前视频上下文 + 按需检索”。不把上次提示复制到本次，也不计算进步分数。</p></section>'
    )
    html_text = r"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CPR Review · 演示案例手册</title><style>
:root{font:14px/1.75 "Segoe UI","Microsoft YaHei",sans-serif;color:#25473f;background:#f5f7f3}*{box-sizing:border-box}body{margin:0}main{max-width:1120px;margin:auto;padding:45px 25px 50px}a{color:#26765d;text-underline-offset:4px}header{display:grid;grid-template-columns:1fr auto;gap:20px;align-items:center}.eyebrow,.case-id{font-size:11px;letter-spacing:1.8px;color:#6b8565;font-weight:600}h1{font-size:36px;line-height:1.35;letter-spacing:-1px;margin:12px 0}h2{font-size:21px;line-height:1.5;margin:8px 0 16px}h3{font-size:16px;margin:0 0 10px}p{margin:10px 0}.lead{color:#6c7c6c;max-width:800px}.button,button{display:inline-flex;align-items:center;justify-content:center;gap:7px;border:1px solid #dce5d6;border-radius:8px;padding:10px 14px;background:white;color:#2d6751;text-decoration:none;font:inherit;font-size:12px;cursor:pointer}.button.primary{background:#27745b;color:white;border-color:#27745b}.button:hover,button:hover{filter:brightness(.97)}button:focus-visible,a:focus-visible,summary:focus-visible{outline:3px solid #9bc7ac;outline-offset:3px}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:13px;margin:28px 0}.stat{border-top:2px solid #d6e3ce;padding:15px 0}.stat strong{font-size:28px;font-weight:500;display:block;line-height:1.4}.stat span{font-size:12px;color:#74806e}.route{background:#edf3e7;border:1px solid #dee8d5;border-radius:12px;padding:19px 22px;margin:15px 0 24px}.route strong{font-size:13px}.steps{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px}.steps a{padding:5px 9px;background:white;border-radius:6px;font-size:12px;text-decoration:none}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.card{background:white;border:1px solid #e0e7da;border-radius:13px;padding:24px;min-width:0}.wide{grid-column:1/-1}.card>p{color:#617365;font-size:13px}.comparison{display:grid;grid-template-columns:1fr 1fr;gap:12px;background:#f4f7ef;border-radius:9px;padding:13px;margin:14px 0 17px}.comparison span{display:block;font-size:10px;color:#7d886f}.comparison strong{display:block;font-size:13px;margin-top:5px;font-weight:600}.workspace{font-size:12px;overflow-wrap:anywhere}.workspace strong{color:#2f5546}.actions{display:flex;gap:8px;flex-wrap:wrap;margin:16px 0}.note{background:#fbf6e9;border-left:3px solid #c6b278;padding:10px 12px;font-size:12px!important;color:#7b6a3f!important}.small{font-size:11px!important;color:#7c8774!important}.result{color:#477b56!important;font-size:12px!important}.boundary-grid{display:grid;grid-template-columns:1fr 1fr;gap:26px}.boundary-grid p{font-size:13px;color:#617365}details{border-top:1px solid #e7ebdf;margin-top:20px;padding-top:12px;font-size:12px;color:#6c7b61}summary{cursor:pointer;font-weight:600;color:#50764c}video{width:100%;aspect-ratio:16/9;display:block;background:#20352a;border-radius:9px;margin-top:13px}.mono{font-family:Consolas,monospace;overflow-wrap:anywhere}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f6ef;border:1px solid #e3eadc;border-radius:8px;padding:14px;font-size:12px}footer{margin-top:24px;color:#7b8874;font-size:11px}.copy-status{min-height:20px;font-size:11px;color:#47825c}li{margin:7px 0}ol{padding-left:22px}#copy-status{position:fixed;bottom:20px;left:50%;transform:translateX(-50%);background:#285f48;color:white;border-radius:8px;padding:7px 14px;max-width:90vw;font-size:12px}#copy-status:empty{display:none}@media(max-width:700px){main{padding:25px 16px}header,.grid,.boundary-grid{grid-template-columns:1fr}h1{font-size:29px}.card{padding:19px}.stats{grid-template-columns:1fr 1fr;gap:5px 16px}.wide{grid-column:auto}.comparison{gap:8px;padding:11px}.route{padding:16px}}@media print{.copy,#copy-status,video{display:none}body{background:white}main{max-width:none;padding:0}.card{break-inside:avoid}.grid{display:block}.card{margin:12px 0}header{display:block}}
</style></head><body><main><header><div><div class="eyebrow">CPR REVIEW / PRESENTATION CASEBOOK / 2026.10.23</div><h1>七个案例，演示完整使用价值。</h1><p class="lead">不同错误、未标记结果、输入失败，以及 Agent 的参考检索与目标记忆。每个案例都对应真实记录；正常标签与模型输出分开展示。</p></div><a class="button primary" href="BASE" target="_blank">打开系统 ↗</a></header>
<div class="stats"><div class="stat"><strong>3</strong><span>完整范围的视频案例</span></div><div class="stat"><strong>2</strong><span>失败与恢复案例</span></div><div class="stat"><strong>2</strong><span>Agent 使用场景</span></div><div class="stat"><strong>0</strong><span>本轮新增大模型调用</span></div></div>
<section class="route"><strong>推荐顺序 · 约 5–7 分钟，使用已完成记录</strong><div class="steps"><a href="#C01">① 弯臂</a><a href="#C02">② 倾斜手臂</a><a href="#C03">③ 未标记的边界</a><a href="#failures">④ 输入失败与恢复</a><a href="#C06">⑤ 参考对照</a><a href="#C07">⑥ 跨次目标记忆</a></div><p class="small">默认回看真实已完成的分析，不把缓存回看冒充现场推理。若需要现场首次分析，另留约 2–3 分钟及冷启动余量；解释和复盘可以复用已保存结果。</p></section>
<div class="grid">CARDS</div>
<section class="card" style="margin-top:20px"><h2>演示前，确认这五件事</h2><ol><li>可先运行 <code>.\.venv\Scripts\python.exe scripts/check_demo_case_readiness.py</code>，只读检查7个案例，不创建任务或外呼。</li><li>服务位于 <a href="BASE" target="_blank">127.0.0.1:8007</a>，页面显示 Connected。尚未启动时，在 CODE 下运行 <code>.\scripts\start_product_demo.ps1</code>。</li><li>左侧练习按最近活动排序，不依赖固定位置。按上方给出的练习名定位；C01 可在 Review history 打开完整 Video review，C06 使用 Reference comparison。</li><li>已经保存的报告、参考和起始目标可直接回看，不必启用 DeepSeek。若要现场新发文字请求，再用 <code>-EnableDeepSeek</code> 启动；关闭时按确认表单完成。</li><li>失败演示后切回 C01 / C02 / C03，确认可恢复。不要在系统仍分析时连续启动多个 GPU 任务。</li></ol><details><summary>工程验收与素材说明</summary><p>三段素材沿用已下载的 CPR-Coach，使用其公开标签，无新增采集、标注或训练。这是已知样片的工程演示，不是盲测、泛化或临床验证。</p><p>当前目录没有核验过的多错误视频真值，因而没有将单标签样片上的多个模型提示称为“多错误识别成功”。完整视频仍留在原本地数据目录，未新打包作公开分发。</p><p><a href="../test-results/demo-cases-v1/http/acceptance.json">真实链路记录</a> · <a href="../data/demo-cases-v1/cases.json">案例清单与来源哈希</a> · <a href="https://huggingface.co/datasets/ShunliWang/CPR-Coach" target="_blank">公开数据来源 ↗</a></p></details></section>
<footer>本机演示索引 · M1–M5 算法、阈值及应用代码保持不变 · 视频标签不会作为识别输入</footer></main><div id="copy-status" role="status" aria-live="polite"></div><script>
document.querySelectorAll('[data-copy]').forEach(button=>button.addEventListener('click',async()=>{let copied=false;try{await navigator.clipboard.writeText(button.dataset.copy);copied=true}catch{const input=document.createElement('textarea');input.value=button.dataset.copy;input.style.position='fixed';input.style.opacity='0';document.body.append(input);input.select();copied=document.execCommand('copy');input.remove()}document.getElementById('copy-status').textContent=copied?'已复制，可粘贴使用':'请在素材清单中复制路径';setTimeout(()=>document.getElementById('copy-status').textContent='',3000)}));
</script></body></html>"""
    html_text = html_text.replace("BASE", esc(base)).replace("CARDS", "\n".join(content))
    target.write_text(html_text, encoding="utf-8")
    (ROOT / "docs/演示案例-v1-快速操作.txt").write_text(
        "CPR Review 演示案例 v1\n\n打开 docs/演示案例-v1.html，按卡片操作。案例清单在 data/demo-cases-v1/cases.json。\n"
        "启动：.\\scripts\\start_product_demo.ps1，访问 http://127.0.0.1:8007；回看无需大模型。\n\n"
        "C01 弯臂：video.mp4，Review history中的完整Video review，提示#4。\n"
        "C02 手臂倾斜：follow-up-practice.mp4，最新Video review，11类范围内提示#5。\n"
        "C03 正常标签：case-03-normal-labelled.mp4，11类均未标记；不是动作合格认证。\n"
        "C04 损坏容器：data/demo-cases-v1/controls/case-04-invalid-container.mp4，上传返回415，旧练习不切换。\n"
        "C05 三帧视频：case-05-too-short-3frames.mp4，已有失败记录可回看；无预测、无分类结论。\n"
        "C06 参考对照：video.mp4，历史中的Reference comparison，切换正常标签与弯臂示例，查看来源。\n"
        "C07 跨次目标：next-practice-focus.mp4，保存了弯臂起始范围；打开旧摘要，但新视频仍没有分析结果。\n\n"
        "默认演示已完成记录，约5–7分钟；真正现场重跑首次分析另留2–3分钟及余量。历史回看不等于新推理。\n"
        "本轮新增1次正常样片完整分析、1次短视频输入探测；倾斜案例只扩展已有结果的显示范围，0次新视觉计算。\n"
        "未新增模型请求、数据采集、人工标签或训练；应用代码、M1–M5及模型版本未改。\n"
        "这是已知样片的工程演示，不是未知场景准确率实验；三帧与损坏文件是边界控制，不带CPR标签。\n"
        "案例准备脚本会保存运行ID与幂等键，禁止覆盖已验收清单，避免重复计算。\n",
        encoding="utf-8",
    )
    print(json.dumps({"passed": True, "case_cards": 7, "guide": target.relative_to(ROOT).as_posix()}))


if __name__ == "__main__":
    main()
