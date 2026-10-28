"""Summarize observed results; deliberately no runtime release approval."""
import argparse
import json
from pathlib import Path

from baseline import digest, write_json


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--project",type=Path,required=True)
    parser.add_argument("--snapshot",type=Path,required=True)
    parser.add_argument("--report",type=Path,required=True)
    args=parser.parse_args()
    before=read(args.snapshot)
    after={name:digest(args.project/name) for name in before}
    if before != after:
        raise AssertionError("Application implementation changed during isolated verification")
    audit=read(args.run/"data-audit.json")
    extraction=read(args.run/"extraction-audit.json")
    training=read(args.run/"head-training-results.json")
    smoke=read(args.run/"raw-video-smoke.json")
    selected=training["models"][training["selected_by_validation"]]
    evaluations=selected["evaluations"]
    assert not audit["missing_keypoint_records"] and not audit["rejected_records"]
    assert extraction["all_encoder_parameters_frozen"] and extraction["encoder_file_unchanged"]
    assert all(m["reload_max_difference"]==0 for m in training["models"].values())
    assert len(smoke)==2 and all(s["feature_shape"]==[1,512] and len(s["uncalibrated_scores"])==11 for s in smoke)
    observed={name:values["validation_selected_threshold"] for name,values in evaluations.items()}
    weak=[r for r in observed["single_test"]["per_label"] if r["f1"]<.5]
    weak_composite=[r for r in observed["composite_test"]["per_label"] if r["f1"]<.5]
    result={"data_integrity_passed":True,"frozen_encoder_and_light_head_training_passed":True,
            "checkpoint_round_trip_passed":True,"raw_video_interface_smoke_passed":True,
            "application_sources_unchanged":True,"runtime_release_approved":False,
            "manual_collection_or_annotation":False,"visual_backbone_training":False,
            "selected_head":training["selected_by_validation"],"parameters":selected["parameters"],
            "observed_metrics":observed,"single_test_f1_below_05_diagnostic":weak,
            "composite_test_f1_below_05_diagnostic":weak_composite,
            "performance_limitation":"High original-Correct false-alarm rate and weaker unseen-composite generalization; not a deployable 11-error detector yet.",
            "supported_claim":"Public labels suffice to fit and test a small CPR-specific head on frozen single-view skeleton features.",
            "unsupported_claims":["All 11 errors are reliable on arbitrary raw video","AlphaPose-to-YOLO transfer accuracy",
                                  "Five improved algorithms finished","RGB/fusion/Agent recognition tools integrated"]}
    args.report.mkdir(parents=True,exist_ok=True)
    write_json(args.report/"verification-summary.json",result)
    pct=lambda value:f"{value*100:.1f}%"
    lines=["CPR公开数据与轻量分类头：下一步验证结果", "",
           "范围：保持单视角、五模块与12节点Agent路线。本轮只验证骨架识别基线，不接入Web正式判断。",
           "",
           "1. 数据与监督",
           f"CPR-Coach固定版本 {audit['source_revision']}，ann与Keypoints两个公开包SHA-256均已校验。",
           f"ch0视角；实际划分：{audit['prepared_counts']}。无缺失、无拒绝记录。",
           "原始标签直接使用；68点确定性取前17人体点，置信度裁剪至[0,1]。没有采集和新增人工标签。",
           "深度类10/12不纳入当前目标；深度单类样本未伪装成正常样本。保留原始测试划分；不宣称跨受试者泛化。",
           "",
           "2. 训练与工程验证",
           "PoseC3D主干全部冻结，全部主干张量已严格核对，只拟合线性头与小MLP。",
           f"按验证集选择：{training['selected_by_validation']}，参数{selected['parameters']}；训练耗时{selected['head_training_seconds']:.2f}秒（特征提取另计）。",
           "类别顺序/编码器profile校验、缓存校验、保存恢复一致性和两段本地视频输入链路通过。",
           f"特征提取：{extraction['elapsed_seconds']:.1f}秒；峰值GPU分配{extraction['cuda_peak_allocated_bytes']/2**30:.2f}GiB；原始权重文件未改变。",
           "本地视频均未用于训练，未添加真值标签，因此未测其错误识别准确率。",
           "",
           "3. 独立测试观察值（阈值和候选头只在验证集选择）"]
    for name in ("single_test","composite_test"):
        value=observed[name]
        negative=evaluations[name]["all_negative_baseline"]
        positive=evaluations[name]["all_positive_baseline"]
        lines.append(f"{name}：n={value['records']}，Macro-F1={pct(value['macro_f1'])}，Micro-F1={pct(value['micro_f1'])}，mAP={pct(value['mAP'])}；全阴性/全阳性Macro-F1={pct(negative['macro_f1'])}/{pct(positive['macro_f1'])}。")
        if value["correct_false_alarm_rate"] is not None:
            lines.append(f"原始Correct样本{value['original_correct_records']}段，至少一项误报的比例={pct(value['correct_false_alarm_rate'])}。")
    lines += ["", "单错误测试逐类F1 / AP："]
    for label in observed["single_test"]["per_label"]:
        lines.append(f"ID{label['id']:02d} {label['name']}：F1 {pct(label['f1'])}；AP {pct(label['average_precision'])}；正例{label['support']}。")
    lines += ["", "4. 验收结论与边界",
              "通过：只依赖公开已有数据与标签、冻结视觉主干、训练小分类头的技术链路可运行。",
              "识别质量以以上真实逐类结果为准；低F1类别不算已通过发布验收。",
              f"主要缺口：组合错误Macro-F1仅{pct(observed['composite_test']['macro_f1'])}，正常片段误报{pct(observed['single_test']['correct_false_alarm_rate'])}；目前不批准正式工具输出这些错误判断。",
              "本地输入使用YOLO骨架，公开训练使用AlphaPose骨架；提取器迁移、多人帧选择和本地机位效果仍未验证。",
              "当前仅有片段级标签，不输出精确错误起止时刻、真实按压深度或临床合格结论。分数尚未校准。",
              "这是后续改进的基线；M1—M5改进均未在本轮宣称完成。RGB分支、融合、校准与Agent正式工具仍需独立验证。",
              "应用源代码快照一致；正式系统继续保留未实现状态，不把本实验分数冒充正式功能。",
              "", "公开来源：https://huggingface.co/datasets/ShunliWang/CPR-Coach",
              "原始项目：https://github.com/Shunli-Wang/CPR-Coach"]
    (args.report/"验证结论.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("\n".join(lines),flush=True)


if __name__=="__main__":
    main()
