import hashlib
import json
import sys
import unittest
from pathlib import Path
import numpy as np

from geometry import DIMENSION,describe,repair
from improve import grouped_split,predict,tune_operating_point,read,digest
from quality_head import load

DATA=(Path(__file__).resolve().parents[2] / "data/public_cpr")


class ImprovementContracts(unittest.TestCase):
    def test_geometry_translation_scale_invariance(self):
        rng=np.random.default_rng(2)
        p=rng.normal(size=(48,17,2))*30+500;q=np.ones((48,17))
        a,_=describe(p,q);b,_=describe(p*2.5+[123,-45],q)
        np.testing.assert_allclose(a,b,atol=1e-6,rtol=1e-5)

    def test_only_short_internal_gaps_repaired(self):
        p=np.ones((30,17,2));q=np.ones((30,17));q[5:8,9]=0;q[12:18,10]=0;q[:2,5]=0
        fixed,_,count=repair(p,q)
        self.assertEqual(count,3)
        self.assertTrue(np.isfinite(fixed[5:8,9]).all())
        self.assertTrue(np.isnan(fixed[12:18,10]).all())
        self.assertTrue(np.isnan(fixed[:2,5]).all())

    def test_unavailable_geometry_not_fabricated(self):
        a,meta=describe(np.ones((20,17,2)),np.zeros((20,17)))
        self.assertFalse(meta["geometry_valid"])
        np.testing.assert_array_equal(a,np.zeros(DIMENSION))
        with self.assertRaises(ValueError):describe(np.ones((20,16,2)),np.ones((20,16)))

    def test_split_disjoint_and_raw_sample_held_out(self):
        source=read(DATA/"run/records.json");a=grouped_split(source);b=grouped_split(source)
        self.assertEqual(a,b)
        self.assertEqual(len({r["record_id"].rsplit('/',1)[0] for r in a}),len(a))
        self.assertTrue(all(r["application_split"]=="test" for r in a if r["record_id"].startswith("SupC02/")))
        self.assertTrue(all(r["application_split"]=="test" for r in a if r["original_partition"]=="official_single_test"))

    def test_exported_predictor_and_incompatible_profile(self):
        root=DATA/"improvement-v2";p=read(root/"protocol.json");s=read(root/"selection.json")
        selected=s["selected"];path=root/selected["checkpoint"]
        self.assertEqual(digest(path),selected["checkpoint_sha256"])
        model,state=load(path,p["feature_profile_hash"],digest(root/"protocol.json"))
        with np.load(root/"development.npz",allow_pickle=False) as f:x=f["features"][:3,state["feature_indices"].numpy()]
        np.testing.assert_array_equal(model.probability(x),model.probability(x))
        self.assertLessEqual(selected["portable_vs_training_max_difference"],1e-10)
        with self.assertRaises(ValueError):load(path,"wrong-profile",digest(root/"protocol.json"))
        with self.assertRaises(ValueError):model.probability(np.full_like(x,np.nan))

    def test_threshold_search_keeps_all_records(self):
        y=np.zeros((10,11),dtype=np.float32);y[2:,0]=1
        p=np.zeros((10,12));p[2:,0]=.9;p[:2,11]=.9
        op=tune_operating_point(p,y);pred=predict(p,op["thresholds"],op["normal_gate"])
        self.assertEqual(pred.shape,y.shape)
        self.assertFalse(pred[:2].any())
        self.assertTrue(pred[2:,0].all())

    def test_raw_source_is_existing_public_label(self):
        manifest=read(DATA/"video-verification/manifest.json")
        self.assertEqual(manifest["record_id"],"SupC02/r00/ch0")
        self.assertEqual(manifest["original_label_ids"],[2])
        self.assertEqual(len(manifest["images"]),415)
        folder=DATA/"video-verification/SupC02-r00-ch0"
        for record in manifest["images"]:
            self.assertEqual(digest(folder/record["filename"]),record["sha256"])


if __name__=="__main__":unittest.main(verbosity=2)
