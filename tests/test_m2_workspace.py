"""M2 is isolated, version bound and lazy; these tests do not measure CV accuracy."""

import json
import os
import subprocess
import sys

import pytest
from cpr_agent.branch_vision import validate_pipeline
from cpr_agent.config import ROOT
from cpr_agent.evidence import file_hash
from cpr_agent.modules.registry import build_registry
from test_branches import configured

WORKSPACE = ROOT / "workspaces/m2-quality-v1"


def test_stable_module_profile_is_preserved(tmp_path):
    original = ROOT / "data/module-slots-v1/pipeline-manifest.json"
    assert validate_pipeline(configured(tmp_path, branch_manifest=original)) == file_hash(original)
    with pytest.raises(ValueError, match="MODULE_NOT_REGISTERED"):
        build_registry().spec("M2", "quality-gated-disagreement-v1")


@pytest.mark.skipif(not WORKSPACE.exists(), reason="Generate the explicit isolated M2 workspace first")
def test_m2_profile_changes_only_m2_and_pins_sources():
    profile = json.loads((WORKSPACE / "data/m2-quality-v1/pipeline-manifest.json").read_text())
    original = json.loads((ROOT / "data/module-slots-v1/pipeline-manifest.json").read_text())
    assert profile["module_slots"]["M2"] == "quality-gated-disagreement-v1"
    assert {k: v for k, v in profile["module_slots"].items() if k != "M2"} == {
        k: v for k, v in original["module_slots"].items() if k != "M2"
    }
    for name, expected in profile["sources"].items():
        assert file_hash(WORKSPACE / name) == expected, name
    assert not (WORKSPACE / ".env").exists()
    assert "ROOT.parents[1]" in (WORKSPACE / "backend/cpr_agent/branch_vision.py").read_text()


def test_public_pairing_and_selection_are_explicit_and_disjoint():
    base = ROOT / "data/m2-quality-v1"
    audit = json.loads((base / "prepared/audit.json").read_text())
    assert len(audit["train_ids"]) == 32 and len(audit["validation_ids"]) == 6
    assert not set(audit["train_ids"]) & set(audit["validation_ids"])
    assert audit["locked_test_opened"] is False and audit["new_labels"] is False
    selection = json.loads((base / "experiment-01/selection.json").read_text())
    protocol = json.loads((base / "experiment-01/protocol.json").read_text())
    assert selection["validation_used_for_selection"] is False
    assert len(protocol["candidate_configs"]) == 9
    assert selection["selected"] in protocol["candidate_configs"]
    assert selection["protocol_sha256"] == file_hash(base / "experiment-01/protocol.json")


@pytest.mark.skipif(not WORKSPACE.exists(), reason="Generate the explicit isolated M2 workspace first")
def test_web_registry_does_not_import_numpy_or_load_gpu_models():
    env = {k: v for k, v in os.environ.items() if not k.startswith(("CPR_", "DEEPSEEK_"))}
    env["PYTHONPATH"] = str(WORKSPACE / "backend")
    code = (
        "import sys;from cpr_agent.modules.registry import build_registry;"
        "s=build_registry().spec('M2','quality-gated-disagreement-v1');"
        "assert s.kind=='improved';assert 'numpy' not in sys.modules;"
        "assert 'torch' not in sys.modules;print(s.implementation_id)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=WORKSPACE, env=env, capture_output=True, text=True, timeout=15
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "quality-gated-disagreement-v1"
