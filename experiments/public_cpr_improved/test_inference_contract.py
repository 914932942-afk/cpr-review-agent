import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch
import cv2
import numpy as np
import torch
from torchvision.ops import deform_conv2d
from infer_video import load_manifest
from raw_pose_adapter import decode_pose, make_result


class InferenceContracts(unittest.TestCase):
    def test_frozen_manifest_and_corruption(self):
        path = Path(__file__).with_name("inference-manifest.json")
        original = load_manifest(path)
        invalid = copy.deepcopy(original)
        invalid["pinned_files"][0]["sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as directory:
            fake = Path(directory) / "manifest.json"
            fake.write_text(json.dumps(invalid), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "mismatch"):
                load_manifest(fake)

    def test_over_budget_and_no_person_do_not_emit_scores(self):
        model = SimpleNamespace(predictor=None, overrides={}, callbacks={})
        capture = MagicMock()
        capture.isOpened.return_value = True
        capture.get.side_effect = lambda prop: 25. if prop == cv2.CAP_PROP_FPS else 3001.
        with patch("cv2.VideoCapture", return_value=capture):
            with self.assertRaisesRegex(ValueError, "frame budget"):
                decode_pose("synthetic", model)
        capture.release.assert_called()
        capture = MagicMock()
        capture.isOpened.return_value = True
        capture.get.side_effect = lambda prop: 25. if prop == cv2.CAP_PROP_FPS else 8.
        frame = np.zeros((64, 64, 3), np.uint8)
        capture.read.side_effect = [(True, frame)] * 8 + [(False, None)]
        empty = make_result(frame, "synthetic", {0: "person"}, torch.empty((0, 6)), torch.empty((0, 17, 3)))
        model.predict = MagicMock(return_value=[empty])
        with patch("cv2.VideoCapture", return_value=capture):
            with self.assertRaisesRegex(ValueError, "Insufficient observable"):
                decode_pose("synthetic", model)
        capture.release.assert_called()

    def test_deformable_operator_nonzero_yx_offsets(self):
        # Fixed displacements have a known ordinary-convolution equivalent in
        # the interior, checking offset order beyond the zero-offset smoke test.
        generator = torch.Generator().manual_seed(2)
        image = torch.randn((1, 2, 9, 9), generator=generator)
        weight = torch.randn((3, 2, 3, 3), generator=generator)
        regular = torch.nn.functional.conv2d(image, weight, padding=1)
        for channel, expected in [(0, regular[:, :, 2:-1, 1:-2]),
                                  (1, regular[:, :, 1:-2, 2:-1])]:
            offset = torch.zeros((1, 18, 9, 9)); offset[:, channel::2] = 1.
            actual = deform_conv2d(image, offset, weight, padding=(1, 1))
            torch.testing.assert_close(actual[:, :, 1:-2, 1:-2], expected, atol=1e-5, rtol=1e-5)


if __name__ == "__main__": unittest.main()
