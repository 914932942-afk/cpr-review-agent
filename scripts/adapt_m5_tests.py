"""Retain M4 checks in the new workspace and update only version/kind expectations."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = ROOT / "workspaces/m5-cascade-v1/tests/test_m4_profile.py"
    s = p.read_text(encoding="utf-8")
    assert "m4-refined-v2" in s
    s = (
        s.replace("m4-refined-v2", "m5-cascade-v1")
        .replace("data/m4-exploration-v2/pipeline-manifest.json", "data/m5-cascade-v1/pipeline-manifest.json")
        .replace("cpr-review-12-v0.8-adaptive", "cpr-review-12-v0.9-cascade")
    )
    s = s.replace("test_m4_is_improved_and_m5_remains_a_baseline", "test_m4_and_m5_are_improved").replace(
        'assert caps["module_slots"]["M5"]["kind"] == "baseline"',
        'assert caps["module_slots"]["M5"]["kind"] == "improved"',
    )
    p.write_text(s, encoding="utf-8")


if __name__ == "__main__":
    main()
