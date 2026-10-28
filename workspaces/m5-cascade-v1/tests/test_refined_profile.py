"""Selection/version checks for the isolated refined application, without loading ML models."""

import json
import sys
from pathlib import Path

import pytest
from cpr_agent.api import create_app
from cpr_agent.branch_vision import validate_pipeline
from cpr_agent.config import ROOT, Settings
from cpr_agent.evidence import file_hash
from cpr_agent.vision import VisionError
from fastapi.testclient import TestClient

WORKSPACE = ROOT if ROOT.name == "m1-m2-refined-v1" else ROOT / "workspaces/m1-m2-refined-v1"
PROFILE = ROOT / "data/m1-m2-refined-v1/pipeline-manifest.json"
pytestmark = pytest.mark.skipif(ROOT != WORKSPACE, reason="Run with the isolated refined backend/root")


def settings(tmp_path, profile=PROFILE):
    return Settings(
        data_dir=tmp_path,
        vision_mode="real",
        ml_python=Path(sys.executable),
        vision_pipeline="branches",
        branch_manifest=profile,
        context_mode="bound-v1",
    )


def test_accepted_profile_validates_all_source_and_parameter_pins(tmp_path):
    assert validate_pipeline(settings(tmp_path)) == file_hash(PROFILE)


def test_capabilities_select_only_m2_and_preserve_other_slots(tmp_path):
    with TestClient(create_app(settings(tmp_path))) as c:
        caps = c.get("/api/capabilities").json()
        assert caps["module_slots"]["M2"]["implementation_id"] == "competence-scope-disagreement-v3"
        assert caps["module_slots"]["M2"]["kind"] == "improved"
        assert all(caps["module_slots"][k]["kind"] == "baseline" for k in ["M3", "M4", "M5"])
        assert caps["llm"]["mode"] == "disabled"
        assert len(caps["nodes"]) == 12


@pytest.mark.parametrize(
    "source",
    [
        "experiments/m1_fast_v1/algorithm.py",
        "experiments/m2_competence_v3/fusion_algorithm.py",
        "data/m1-m2-exploration-v1/m1-coherent/selection.json",
        "data/m1-m2-exploration-v1/m2-competence/selection.json",
    ],
)
def test_tampered_selected_code_or_parameters_block_startup(tmp_path, source):
    value = json.loads(PROFILE.read_text())
    value["sources"][source] = "0" * 64
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(value))
    with pytest.raises(VisionError, match="PIPELINE_SOURCE_CHANGED"):
        validate_pipeline(settings(tmp_path, path))


def test_selected_m1_algorithm_and_config_are_the_accepted_pair():
    folder = ROOT / "data/m1-m2-exploration-v1/m1-coherent"
    decision = json.loads((folder / "decision.json").read_text())
    selection = json.loads((folder / "selection.json").read_text())
    protocol = json.loads((folder / "protocol.json").read_text())
    assert decision["accepted"] and decision["selected"] == selection["name"]
    assert selection["selected_before_validation"]
    assert protocol["algorithm_sha256"] == file_hash(ROOT / "experiments/m1_fast_v1/algorithm.py")
