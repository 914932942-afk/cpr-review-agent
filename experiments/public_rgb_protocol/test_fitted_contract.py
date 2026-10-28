"""Integration checks for real fitted artifacts, after the frozen evaluation."""
import unittest

import numpy as np

import rgb_protocol as p


class FittedArtifactContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output = p.OUTPUT
        cls.protocol = p.checked_protocol(cls.output)
        cls.selected = p.read(cls.output / "selection.json")["selected"]
        cls.head, cls.state = p.load(cls.output / cls.selected["checkpoint"],
                                    cls.protocol["feature_profile_hash"], p.digest(cls.output / "protocol.json"))

    def test_head_is_bound_to_the_new_protocol_and_small_budget(self):
        self.assertLessEqual(self.selected["scalar_fields"], 70000)
        self.assertEqual(self.state["input_dimension"], 783)
        self.assertEqual(self.protocol["candidate_configs"], [{"features": "combined", "trees": 64}])
        self.assertEqual(p.digest(self.output / self.selected["checkpoint"]), self.selected["checkpoint_sha256"])
        self.assertNotEqual(self.selected["checkpoint_sha256"], p.digest(p.BASE / "combined-64.pt"))
        self.assertLessEqual(self.selected["portable_vs_training_max_difference"], 1e-10)

    def test_validation_result_is_reproducible_without_refitting(self):
        with np.load(self.output / "development.npz", allow_pickle=False) as bundle:
            mask = bundle["splits"] == "validation"
            x, y = bundle["features"][mask], bundle["targets"][mask]
        probability = self.head.probability(x[:, self.state["feature_indices"].numpy()])
        op = self.state["operating_point"]
        predicted = p.predict(probability, op["thresholds"], op["normal_gate"])
        measured = p.metrics(predicted.astype(float), y)
        self.assertEqual(measured, self.selected["operating_point"]["validation_metrics"])
        self.assertEqual(len(y), 196)

    def test_saved_test_predictions_include_every_record_and_replay_exactly(self):
        with np.load(self.output / "locked-test.npz", allow_pickle=False) as bundle:
            x, y, ids = (bundle[k].copy() for k in ("features", "targets", "record_ids"))
        with np.load(self.output / "test-predictions.npz", allow_pickle=False) as saved:
            np.testing.assert_array_equal(ids, saved["record_ids"])
            np.testing.assert_array_equal(y, saved["targets"])
            np.testing.assert_array_equal(self.head.probability(x[:, self.state["feature_indices"].numpy()]), saved["scores"])
            self.assertEqual(saved["predictions"].shape, (386, 11))
        self.assertTrue(set(self.protocol["raw_test_reservations"]).issubset(set(map(str, ids))))
        self.assertEqual(len(set(ids)), 386)

    def test_inference_registry_uses_new_head_and_all_26_frozen_videos(self):
        manifest = p.load_manifest(self.output / "inference-manifest.json")
        self.assertEqual(manifest["head"], (self.output / self.selected["checkpoint"]).relative_to(p.CODE).as_posix())
        self.assertEqual(manifest["protocol"], (self.output / "protocol.json").relative_to(p.CODE).as_posix())
        self.assertEqual({r["record_id"] for r in manifest["public_samples"].values()}, set(self.protocol["raw_test_reservations"]))
        self.assertEqual(len(manifest["public_samples"]), 26)
        self.assertFalse(manifest["runtime_release_approved"])
        self.assertTrue(manifest["raw_video_acceptance_pending"])
        self.assertFalse(p.read(self.output / "evaluation.json")["raw_video_acceptance_performed"])

    def test_fitting_access_and_actual_development_are_recording_disjoint(self):
        audit = p.read(self.output / "fit-access-audit.json")
        self.assertFalse(audit["test_npz_decoded_by_fitter"])
        self.assertEqual(audit["reserved_rgb_in_fit_or_tuning"], 0)
        self.assertEqual(audit["npz_files_decoded_by_fitter"], [(self.output / "development.npz").relative_to(p.CODE).as_posix()])
        with np.load(self.output / "development.npz", allow_pickle=False) as bundle:
            development = {p.group_id(str(k)) for k in bundle["record_ids"]}
            self.assertEqual(len(development), 934)
            self.assertEqual(int((bundle["splits"] == "train").sum()), 738)
        self.assertFalse(development & {p.group_id(k) for k in self.protocol["raw_test_reservations"]})

    def test_old_head_and_overwriting_completed_phases_are_rejected(self):
        with self.assertRaises(ValueError):
            p.load(p.BASE / "combined-64.pt", self.protocol["feature_profile_hash"], p.digest(self.output / "protocol.json"))
        for stage in (p.prepare, p.train, p.evaluate):
            with self.assertRaises(FileExistsError):
                stage(self.output)


if __name__ == "__main__":
    unittest.main(verbosity=2)
