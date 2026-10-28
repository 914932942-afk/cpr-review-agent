import unittest
import numpy as np
import torch
from raw_pose_adapter import make_result, select_active_track


class RawPoseAdapterTests(unittest.TestCase):
    @staticmethod
    def detection(x, shift):
        points = np.ones((17, 3), dtype=np.float32)
        points[:, :2] = np.array([x + shift, 50.])
        return {"box": np.array([x, 0., x + 100., 100., .9, 0.]), "points": points}

    def test_track_survives_detector_order_changes(self):
        frames = []
        for t in range(30):
            pair = [self.detection(0, 5. * np.sin(t)), self.detection(200, .01 * np.sin(t))]
            frames.append(pair if t % 2 else pair[::-1])
        points, audit = select_active_track(frames)
        self.assertTrue((points[:, :, 0] < 100).all())
        self.assertEqual(audit["track_coverage"], 1.)
        self.assertEqual(len(audit["eligible_tracks"]), 2)

    def test_ambiguous_motion_and_no_person_fail_closed(self):
        frames = [[self.detection(0, 5. * np.sin(t)), self.detection(200, 5. * np.sin(t))] for t in range(30)]
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            select_active_track(frames)
        with self.assertRaisesRegex(ValueError, "No persistent"):
            select_active_track([[] for _ in range(30)])

    def test_display_mask_does_not_mutate_classifier_input(self):
        points = torch.ones((1, 17, 3), dtype=torch.float32)
        points[:, :, :2] = torch.tensor([80., 90.])
        points[0, 0, 2] = .3
        before = points.clone()
        result = make_result(np.zeros((100, 100, 3), np.uint8), "test", {0: "person"},
                             torch.tensor([[0, 0, 99, 99, .9, 0]]), points)
        self.assertTrue(torch.equal(points, before))
        self.assertTrue(torch.equal(result.cpr_raw_keypoints, before))
        self.assertEqual(result.keypoints.data[0, 0, :2].tolist(), [0., 0.])
        self.assertTrue(torch.equal(result.keypoints.data[0, 1:], before[0, 1:]))
        self.assertTrue(torch.equal(result.keypoints.conf, before[:, :, 2]))

    def test_empty_and_invalid_shape(self):
        result = make_result(np.zeros((100, 100, 3), np.uint8), "test", {0: "person"},
                             torch.empty((0, 6)), torch.empty((0, 17, 3)))
        self.assertEqual(tuple(result.cpr_raw_keypoints.shape), (0, 17, 3))
        with self.assertRaises(ValueError):
            make_result(None, "test", {}, None, torch.ones((1, 16, 3)))
        invalid = torch.ones((1, 17, 3)); invalid[0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            make_result(None, "test", {}, None, invalid)


if __name__ == "__main__":
    unittest.main()
