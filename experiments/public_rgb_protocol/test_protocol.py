from collections import Counter
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import rgb_protocol as protocol


class RGBProtocolContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parent = protocol.read(protocol.BASE / "record-splits.json")
        cls.reservations = protocol.read(protocol.COLLECTION / "collection-manifest.json")["records"]

    def test_all_26_reserved_groups_are_test_without_losing_existing_test(self):
        result = protocol.reserve_groups(self.parent, self.reservations)
        self.assertEqual(Counter(r["application_split"] for r in result), {"train": 738, "validation": 196, "test": 386})
        groups = {protocol.group_id(r["record_id"]) for r in self.reservations}
        self.assertEqual(sum(r["recording_group"] in groups for r in result), 26)
        self.assertTrue(all(r["application_split"] == "test" for r in result if r["recording_group"] in groups))
        previous = {r["record_id"] for r in self.parent if r["application_split"] == "test"}
        current = {r["record_id"] for r in result if r["application_split"] == "test"}
        self.assertTrue(previous.issubset(current))
        self.assertEqual(len(previous), 364)

    def test_reservation_covers_other_views_of_the_same_recording(self):
        candidate = dict(next(r for r in self.parent if r["record_id"] == "SupC01/r00/ch0"))
        candidate.update(record_id="SupC01/r00/ch1", view="ch1")
        result = protocol.reserve_groups(self.parent + [candidate], self.reservations)
        self.assertEqual({r["application_split"] for r in result if r["recording_group"] == "SupC01/r00"}, {"test"})

    def test_nonreserved_cross_view_overlap_is_rejected(self):
        reserved = {protocol.group_id(r["record_id"]) for r in self.reservations}
        row = next(r for r in self.parent if r["application_split"] == "train" and protocol.group_id(r["record_id"]) not in reserved)
        camera = {**row, "record_id": row["record_id"][:-1] + "1", "application_split": "validation"}
        with self.assertRaisesRegex(ValueError, "crosses partitions"):
            protocol.reserve_groups(self.parent + [camera], self.reservations)

    def test_duplicate_and_missing_reservations_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate or missing"):
            protocol.reserve_groups(self.parent, self.reservations + [self.reservations[0]])
        with self.assertRaisesRegex(ValueError, "Duplicate or missing"):
            protocol.reserve_groups(self.parent, [{"record_id": "absent/r00/ch0"}])

    def test_original_single_test_cannot_be_used_for_fitting(self):
        row = dict(next(r for r in self.parent if r["original_partition"] == "official_single_test"))
        row["application_split"] = "train"
        with self.assertRaisesRegex(ValueError, "Original single-error test"):
            protocol.reserve_groups([row], [])

    def test_public_labels_and_source_records_are_not_mutated(self):
        before = [(r["record_id"], tuple(r["original_labels"]), r["application_split"]) for r in self.parent]
        result = protocol.reserve_groups(self.parent, self.reservations)
        self.assertEqual(before, [(r["record_id"], tuple(r["original_labels"]), r["application_split"]) for r in self.parent])
        self.assertEqual([r["original_labels"] for r in result], [r["original_labels"] for r in self.parent])

    def test_invalid_ids_and_duplicate_records_fail_closed(self):
        for value in ("../r00/ch0", "SupC01/../ch0", "SupC01/r00/unknown", "SupC01//ch0"):
            with self.assertRaises(ValueError):
                protocol.group_id(value)
        with self.assertRaisesRegex(ValueError, "Duplicate record IDs"):
            protocol.reserve_groups(self.parent + [self.parent[0]], self.reservations)

    def test_no_overwrite_of_a_frozen_protocol_or_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileExistsError):
                protocol.prepare(root)
            protocol.write_new(root / "result.json", {"first": True})
            with self.assertRaises(FileExistsError):
                protocol.write_new(root / "result.json", {"second": True})

    def test_fitter_cannot_decode_the_locked_test_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.savez(root / "development.npz", record_ids=np.array(["train/r00/ch0"]))
            def forbidden(output):
                np.load(output / "locked-test.npz", allow_pickle=False)
            with patch.object(protocol, "checked_protocol", return_value={"raw_test_reservations": ["test/r00/ch0"]}), \
                 patch.object(protocol.fit_quality_head, "train", side_effect=forbidden):
                with self.assertRaisesRegex(PermissionError, "development feature bundle"):
                    protocol.train(root)

    def test_fitter_rejects_another_camera_from_reserved_recording(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.savez(root / "development.npz", record_ids=np.array(["test/r00/ch1"]))
            with patch.object(protocol, "checked_protocol", return_value={"raw_test_reservations": ["test/r00/ch0"]}):
                with self.assertRaisesRegex(ValueError, "leaked"):
                    protocol.train(root)

    def test_incompatible_feature_targets_are_not_silently_relabelled(self):
        parent = protocol.read(protocol.BASE / "protocol.json")
        invalid = [dict(r) for r in self.parent]
        invalid[0] = {**invalid[0], "original_labels": [0]}
        with self.assertRaisesRegex(ValueError, "public targets differ"):
            protocol.load_parent_features(invalid, parent)


if __name__ == "__main__":
    unittest.main(verbosity=2)
