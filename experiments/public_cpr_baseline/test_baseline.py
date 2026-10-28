"""Contract checks for the isolated public-label experiment (no GPU needed)."""
import io
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from baseline import LABEL_IDS, REVISION, Head, RestrictedUnpickler, canonical_path, choose_splits, encode_labels, load_checked_head, metrics


class BaselineContracts(unittest.TestCase):
    def test_original_labels_preserved(self):
        self.assertEqual(encode_labels([0]).tolist(), [0.] * 11)
        self.assertEqual(encode_labels([1, 4, 11]).sum(), 3)
        self.assertEqual(encode_labels([10, 12]).sum(), 0)
        for invalid in ([], [0, 1], [-1], [14], [1.5], [True]):
            with self.assertRaises(ValueError):
                encode_labels(invalid)

    def test_path_canonicalization(self):
        self.assertEqual(canonical_path(r"prefix\CPR_Dataset_S0_BIG\C00_S0\r00\ch0"),
                         "CPR_Dataset_S0/C00_S0/r00/ch0")
        with self.assertRaises(ValueError):
            canonical_path("../CPR_Dataset_S0/x")
        self.assertEqual(canonical_path("SupC14_Dataset/SupC00/r00/ch0"), "SupC00/r00/ch0")
        self.assertEqual(canonical_path("SupDC59_Dataset/SupDC00/r01/ch0"), "SupDC00/r01/ch0")
        self.assertEqual(canonical_path("SupMC15_Dataset/SupMC12/r02/ch0"), "SupMC12/r02/ch0")

    def test_original_partitions_not_reassigned(self):
        records=[]
        for label in (0, *LABEL_IDS):
            for i in range(24):
                records.append({"record_id":f"label{label}/r{i}/ch0", "original_partition":"official_train",
                                "original_labels":[label]})
        for partition in ("official_single_test", "official_composite_test"):
            records.append({"record_id":partition, "original_partition":partition, "original_labels":[1]})
        result=choose_splits(records)
        self.assertEqual(result, choose_splits(records))
        self.assertEqual(sum(r["split"]=="train" for r in result), 240)
        self.assertEqual(sum(r["split"]=="validation" for r in result), 48)
        self.assertEqual(next(r for r in result if r["record_id"]=="official_single_test")["split"], "single_test")
        self.assertEqual(next(r for r in result if r["record_id"]=="official_composite_test")["split"], "composite_test")
        with self.assertRaises(ValueError):
            choose_splits(records+[records[0]])

    def test_pickle_restricts_globals(self):
        value={"keypoint":np.ones((1, 2, 68, 2), dtype=np.float64), "label":[1, 4]}
        restored=RestrictedUnpickler(io.BytesIO(pickle.dumps(value, protocol=4))).load()
        np.testing.assert_array_equal(restored["keypoint"], value["keypoint"])
        with self.assertRaises(pickle.UnpicklingError):
            RestrictedUnpickler(io.BytesIO(pickle.dumps(Path("x")))).load()

    def test_average_precision_ties_and_constants(self):
        rng=np.random.default_rng(42)
        target=rng.integers(0, 2, size=(32, 11)).astype(np.float32)
        for probability in (np.zeros_like(target), np.ones_like(target), rng.integers(0,4,target.shape)/3):
            result=metrics(probability,target)
            expected=average_precision_score(target,probability,average=None)
            np.testing.assert_allclose([r["average_precision"] for r in result["per_label"]],expected)

    def test_head_contract_and_reload(self):
        torch.manual_seed(42)
        for kind in ("linear", "mlp"):
            model=Head(kind).eval()
            feature=torch.randn(3,512)
            with tempfile.TemporaryDirectory() as temporary:
                path=Path(temporary)/"head.pt"
                torch.save(model.state_dict(),path)
                restored=Head(kind).eval()
                restored.load_state_dict(torch.load(path,weights_only=True))
                torch.testing.assert_close(model(feature),restored(feature),atol=0,rtol=0)
            with self.assertRaises(ValueError):
                model(torch.randn(1,1024))

    def test_head_rejects_incompatible_profile(self):
        checkpoint={"schema_version":1,"kind":"linear","label_ids":list(LABEL_IDS),
                    "source_revision":REVISION,"encoder_profile_hash":"expected","head_state":Head("linear").state_dict()}
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/"head.pt"
            torch.save(checkpoint,path)
            load_checked_head(path,"expected")
            with self.assertRaises(ValueError):
                load_checked_head(path,"another-encoder")


if __name__ == "__main__":
    unittest.main(verbosity=2)
