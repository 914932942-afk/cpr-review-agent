"""M5 metadata/source gates are checked without loading numerical libraries on web."""

import copy
import json
from types import SimpleNamespace

import pytest
from cpr_agent.config import ROOT
from cpr_agent.m5_validation import validate_visual_metadata, visual_summary
from cpr_agent.vision import VisionError

PROFILE = ROOT / "data/m5-cascade-v1/pipeline-manifest.json"
CERT = json.loads((ROOT / "data/m5-cascade-v1/certificate.json").read_text(encoding="utf-8"))


def raw(available=True):
    plan = CERT["available" if available else "unavailable"]
    return {
        "decoded_frames": 362,
        "hand_shape_audit": {
            "available": available,
            "requested_frames": 16,
            "usable_frames": 16 if available else 0,
        },
        "rgb_audit": {"frames": 16, "requested_frames": 16},
        "m5_cascade": {
            **copy.deepcopy(plan),
            "dense_visual_crop_evaluations": 0 if available else 32,
            "weights_and_thresholds_unchanged": True,
            "observed_frames_unchanged": True,
            "uncomputed_dense_features_persisted": False,
        },
    }


@pytest.mark.parametrize("available", [True, False])
def test_certified_routes_are_accepted(available):
    validate_visual_metadata(raw(available), SimpleNamespace(branch_manifest=PROFILE))


@pytest.mark.parametrize(
    "change",
    [
        "quality",
        "indices",
        "scope",
        "crops",
        "support",
        "weights",
        "fabrication",
        "sample_count",
        "shape_count",
    ],
)
def test_untrusted_visual_certificates_fail_closed(change):
    r = raw()
    if change == "quality":
        r["hand_shape_audit"]["available"] = "true"
    if change == "indices":
        r["m5_cascade"]["required_rgb_indices"] = [0]
    if change == "scope":
        r["m5_cascade"]["active_experts"]["1"] = "rgb-context"
    if change == "crops":
        r["m5_cascade"]["dense_visual_crop_evaluations"] = 32
    if change == "support":
        r["rgb_audit"]["frames"] = 7
    if change == "weights":
        r["m5_cascade"]["weights_and_thresholds_unchanged"] = False
    if change == "fabrication":
        r["m5_cascade"]["uncomputed_dense_features_persisted"] = True
    if change == "sample_count":
        r["rgb_audit"]["requested_frames"] = 15
    if change == "shape_count":
        r["hand_shape_audit"]["usable_frames"] = 7
    with pytest.raises(VisionError):
        validate_visual_metadata(r, SimpleNamespace(branch_manifest=PROFILE))


def test_nonvisual_body_route_has_no_visual_summary():
    assert (
        visual_summary(None, SimpleNamespace(), SimpleNamespace(analysis_route="body-qualified-v1")) is None
    )


def test_monolithic_legacy_predictions_do_not_read_branch_bindings():
    assert (
        visual_summary(
            None, SimpleNamespace(vision_pipeline="monolithic"), SimpleNamespace(analysis_route="full-v15")
        )
        is None
    )


def test_web_guard_does_not_import_torch_or_numpy():
    import sys

    # This process runs only web contract tests; module guards remain standard-library-only.
    assert "torch" not in sys.modules and "numpy" not in sys.modules


def test_profile_preserves_m4_numerical_algorithm_and_cost_parameters():
    import hashlib

    p = json.loads(PROFILE.read_text(encoding="utf-8"))
    parent = ROOT.parent / "m4-refined-v2"
    for name in [
        "experiments/m4_policy_v2/algorithm.py",
        "data/m4-exploration-v2/parameters.json",
        "experiments/m1_fast_v1/algorithm.py",
        "experiments/m2_competence_v3/fusion_algorithm.py",
        "experiments/m3_calibration_v1/algorithm.py",
    ]:
        assert (
            hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            == hashlib.sha256((parent / name).read_bytes()).hexdigest()
        )
    assert p["module_slots"]["M5"] == "quality-dependency-cascade-v1"


@pytest.mark.parametrize("pruned", [True, False])
def test_only_materialised_feature_domains_are_stored(tmp_path, pruned):
    from zipfile import ZipFile

    from cpr_agent.m5_validation import validate_visual_archive

    artifact = tmp_path / "result.json"
    names = (
        ["context_features", "computed_rgb_feature_indices", "shape_features"]
        if pruned
        else ["rgb_features", "shape_features"]
    )
    with ZipFile(tmp_path / "values.npz", "w") as z:
        for n in names:
            z.writestr(n + ".npy", b"explicit-field-presence-fixture")
    validate_visual_archive(
        {
            "files": [{"name": "values.npz"}],
            "m5_cascade": {
                "algorithm": "quality-dependency-cascade-v1",
                "route": "dependency-pruned" if pruned else "full-rgb-fallback",
            },
        },
        artifact,
    )


@pytest.mark.parametrize("pruned", [True, False])
def test_invented_dense_features_or_missing_fallback_are_rejected(tmp_path, pruned):
    from zipfile import ZipFile

    from cpr_agent.m5_validation import validate_visual_archive

    with ZipFile(tmp_path / "values.npz", "w") as z:
        z.writestr(
            "rgb_features.npy" if pruned else "context_features.npy", b"explicit-invalid-domain-fixture"
        )
    with pytest.raises(VisionError):
        validate_visual_archive(
            {
                "files": [{"name": "values.npz"}],
                "m5_cascade": {
                    "algorithm": "quality-dependency-cascade-v1",
                    "route": "dependency-pruned" if pruned else "full-rgb-fallback",
                },
            },
            tmp_path / "result.json",
        )


def test_changed_model_is_rejected_before_loading_numerical_runtime(tmp_path):
    import importlib.util
    import shutil

    folder = tmp_path / "experiments/m5_cascade_v1"
    folder.mkdir(parents=True)
    for name in ["module.py", "algorithm.py", "runtime.py"]:
        shutil.copy2(ROOT / "experiments/m5_cascade_v1" / name, folder / name)
    params = tmp_path / "data/m5-cascade-v1"
    params.mkdir(parents=True)
    for name in ["protocol.json", "certificate.json", "decision.json"]:
        shutil.copy2(ROOT / "data/m5-cascade-v1" / name, params / name)
    protocol = json.loads((params / "protocol.json").read_text(encoding="utf-8"))
    for name in {**protocol["models"], **protocol["additional_sources"]}:
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, p)
    spec = importlib.util.spec_from_file_location("isolated_m5_constructor", folder / "module.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.QualityDependencyRGB()
    target = tmp_path / next(iter(protocol["models"]))
    target.write_bytes(target.read_bytes() + b"explicit-tampering-fixture")
    with pytest.raises(ValueError, match="M5_CERTIFICATE_SOURCE_CHANGED"):
        module.QualityDependencyRGB()
