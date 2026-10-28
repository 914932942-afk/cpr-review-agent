"""Extend the accepted real M4 API scenarios with M5 artifact assertions."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    destination = ROOT / "scripts/m5_smoke.py"
    assert not destination.exists()
    s = (ROOT / "scripts/m4_v2_smoke.py").read_text(encoding="utf-8")
    s = s.replace("test-results/m4-exploration-v2/http", "test-results/m5-cascade-v1/http").replace(
        "http://127.0.0.1:8012", "http://127.0.0.1:8013"
    )
    s = (
        s.replace(
            "test-results/m4-scheduling-v1/http/state.json", "test-results/m4-exploration-v2/http/state.json"
        )
        .replace("http://127.0.0.1:8011", "http://127.0.0.1:8012")
        .replace('oldstate["body"]', 'oldstate["partial"]')
    )
    marker = '        caps = get("/capabilities")'
    assert s.count(marker) == 1
    s = s.replace(
        marker,
        marker
        + '\n        assert caps["module_slots"]["M5"]["implementation_id"]=="quality-dependency-cascade-v1" and caps["module_slots"]["M5"]["kind"]=="improved"',
    )
    marker = "        result = dict("
    assert s.count(marker) == 1
    s = s.replace(
        marker,
        """        assert r["response"].get("visual_processing") is None
        for current,current_tools in [(fr,ft),(ex,et)]:
            summary=current["response"]["visual_processing"]
            assert summary["mode"]=="lightweight" and summary["sampled_frames"]>=8 and summary["scope_unchanged"] is True
            rgb=next(t for t in current_tools if t["request"]["tool_id"]=="rgb.predict")
            artifact=ROOT/"workspaces/m5-cascade-v1/data/m5-cascade-v1/runtime"/rgb["output"]["artifact_rel"]
            raw=json.loads(artifact.read_text(encoding="utf-8"))
            assert raw["m5_cascade"]["dense_visual_crop_evaluations"]==0
            assert raw["module_executions"]["M5"]["implementation_id"]=="quality-dependency-cascade-v1"
        result = dict(""",
    )
    s = s.replace(
        "actual_successful_tool_calls=9,",
        "actual_successful_tool_calls=9, m5_actual_dependency_pruning=True, m1_m2_m3_states_unchanged=True,",
    )
    destination.write_text(s, encoding="utf-8")


if __name__ == "__main__":
    main()
