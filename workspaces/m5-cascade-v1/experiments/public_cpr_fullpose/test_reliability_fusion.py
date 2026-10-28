import unittest
import numpy as np
from reliability_fusion import guarded,reliability
from hand_shape import extract


class ReliabilityFusionTests(unittest.TestCase):
    def test_qualified_evidence_survives_weaker_count_and_higher_score(self):
        eligible=np.array([[True,True,True]])
        result=guarded(eligible,np.array([[.99,.8,.75]]),np.ones((1,3))*.5,np.array([1]),np.array([False,True,True]))
        np.testing.assert_array_equal(result,[[False,True,True]])

    def test_only_eligible_weak_errors_fit_remaining_budget(self):
        result=guarded(np.array([[True,True,True,False]]),np.array([[.7,.9,.8,.99]]),np.ones((1,4))*.5,np.array([2]),np.array([True,False,False,False]))
        np.testing.assert_array_equal(result,[[True,True,False,False]])

    def test_empty_evidence_stays_empty_even_when_count_positive(self):
        result=guarded(np.zeros((1,3),bool),np.ones((1,3))*.9,np.ones((1,3))*.5,np.array([2]),np.ones(3,bool))
        self.assertFalse(result.any())

    def test_same_development_qualification_rule_for_every_label(self):
        self.assertTrue(reliability({'tp':8,'fp':2,'recall':.8})['qualified'])
        self.assertFalse(reliability({'tp':7,'fp':2,'recall':.9})['qualified'])
        self.assertFalse(reliability({'tp':9,'fp':0,'recall':.7})['qualified'])

    def test_invalid_fusion_does_not_emit_predictions(self):
        with self.assertRaises(ValueError):guarded(np.ones((1,2),bool),np.full((1,2),np.nan),np.ones((1,2))*.5,np.array([1]),np.ones(2,bool))

    def test_missing_wrist_shape_is_unavailable_not_a_correct_gesture(self):
        frame=np.zeros((200,200,3),np.uint8)
        feature,audit=extract([frame]*16,np.ones((16,17,2))*100,np.zeros((16,17)))
        self.assertEqual(feature.shape,(576,));self.assertFalse(audit['available'])
        self.assertEqual(audit['usable_frames'],0)


if __name__=='__main__':unittest.main()
