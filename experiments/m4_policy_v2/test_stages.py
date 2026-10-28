"""ML-process import and finalisation contracts, using explicit synthetic readouts only."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
WS = ROOT / "workspaces/m4-refined-v2"
sys.path[:0] = [
    str(WS / "backend"),
    str(WS / "experiments/module_slots_v1"),
    str(WS / "experiments/m4_policy_v2"),
]
import stages  # noqa: E402
from pipeline import LABELS, SCHEMA, array_file, sha  # noqa: E402


class StageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        self.video = self.folder / "video.mp4"
        self.video.write_bytes(b"explicit-test-input-not-a-real-video")
        self.manifest = self.folder / "bundle.json"
        self.manifest.write_text("{}")
        self.profile = self.folder / "profile.json"
        self.profile.write_text(
            json.dumps(
                {
                    "module_slots": {"M4": "consent-coverage-dag-v2"},
                    "m4_routes": {"body-qualified-v1": {"supported_label_ids": [3, 4, 8, 11]}},
                }
            )
        )
        self.source = self.folder / "skeleton.json"
        self.output = self.folder / "final.json"
        scores = np.full((1, 12), 0.1)
        scores[0, LABELS.index(4)] = 0.9
        eligible = scores[:, :11] >= 0.5
        value = {
            "schema": SCHEMA,
            "tool_id": "skeleton.predict",
            "video_sha256": sha(self.video),
            "bundle_sha256": sha(self.manifest),
            "pipeline_sha256": sha(self.profile),
            "labels_used": False,
            "decoded_frames": 100,
            "reported_fps": 25,
            "fullpose_audit": {},
            "files": [
                array_file(
                    self.folder / "skeleton.npz",
                    scores=scores,
                    thresholds=np.full((1, 11), 0.5),
                    eligible=eligible,
                    predicted_count=np.array([1]),
                )
            ],
        }
        self.source.write_text(json.dumps(value))

    def tearDown(self):
        self.temp.cleanup()

    def call(self, **inputs):
        return stages.run_stage(
            "fusion.predict",
            self.video,
            self.output,
            self.manifest,
            self.profile,
            {"__route_profile__": "body-qualified-v1", "skeleton.predict": str(self.source), **inputs},
            "clip",
            "execution",
        )

    def test_body_finalisation_does_not_call_rgb_or_m2(self):
        previous = stages.full_stage
        try:

            def forbidden(*args):
                raise AssertionError("Body projection must not run full fusion")

            stages.full_stage = forbidden
            value = self.call()
        finally:
            stages.full_stage = previous
        self.assertEqual(value["schema"], "cpr-offline-analysis-v2")
        self.assertEqual(value["supported_label_ids"], [3, 4, 8, 11])
        self.assertEqual(value["predicted_original_label_ids"], [4])
        self.assertEqual(value["module_executions"], {})
        self.assertFalse(value["m4_finalization"]["rgb_executed"])

    def test_skipped_classes_have_no_scores(self):
        value = self.call()
        self.assertEqual([v["original_label_id"] for v in value["scores"]], [3, 4, 8, 11])

    def test_foreign_video_or_pipeline_source_is_rejected(self):
        value = json.loads(self.source.read_text())
        value["video_sha256"] = "a" * 64
        self.source.write_text(json.dumps(value))
        with self.assertRaises(ValueError):
            self.call()

    def test_unexpected_rgb_dependency_is_rejected(self):
        with self.assertRaises(ValueError):
            self.call(**{"rgb.predict": "invented"})

    def test_existing_outputs_are_never_overwritten(self):
        self.call()
        before = self.output.read_bytes()
        with self.assertRaises(FileExistsError):
            self.call()
        self.assertEqual(self.output.read_bytes(), before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
