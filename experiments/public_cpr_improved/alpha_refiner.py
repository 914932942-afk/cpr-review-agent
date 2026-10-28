"""Offline AlphaPose compatibility probe, frozen public Halpe68 model.

Uses upstream model/transforms in an isolated vendor namespace. The original
CUDA DCN operator is replaced by torchvision's available deformable operator;
strict checkpoint loading and a zero-offset numerical test are mandatory.
This is an extractor compatibility experiment, not a released application tool.
"""
from pathlib import Path
import hashlib
import importlib
import sys
import numpy as np
import cv2
import torch

ROOT = Path(__file__).resolve().parent / "alpha-pose-source"
HASH = "679cb94eb30b0d587425037e36b1b9921c2648ed4ad079f2c4dd93163a7c27c9"


def prepare_vendor():
    import json
    audit = json.loads((ROOT / "vendor-manifest.json").read_text(encoding="utf-8"))
    for item in audit:
        if hashlib.sha256((ROOT / item["path"]).read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("Frozen AlphaPose compatibility source changed")
    sys.path.insert(0, str(ROOT))
    return importlib.import_module("cpr_alpha_vendor.fastpose").FastPose


class AlphaRefiner:
    def __init__(self):
        path = Path(__file__).resolve().parents[2] / "data/public_cpr/weights/noface_fast50_dcn_combined_256x192.pth"
        if hashlib.sha256(path.read_bytes()).hexdigest() != HASH:
            raise ValueError("Unverified AlphaPose public weights")
        FastPose = prepare_vendor()
        self.model = FastPose(PRESET={"NUM_JOINTS": 68}, CONV_DIM=256, NUM_LAYERS=50,
            DCN={"MODULATED": False, "DEFORM_GROUP": 1, "FALLBACK_ON_STRIDE": False},
            STAGE_WITH_DCN=(False, True, True, True)).eval().requires_grad_(False)
        saved = torch.load(path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(saved, strict=True)
        if any(not torch.equal(v, saved[k]) for k, v in self.model.state_dict().items()):
            raise ValueError("AlphaPose checkpoint tensor mismatch")
        self.model.cuda()
        self.transforms = importlib.import_module("cpr_alpha_vendor.transforms")
        self.bbox = importlib.import_module("cpr_alpha_vendor.bbox")
        from torchvision.ops import deform_conv2d
        generator = torch.Generator().manual_seed(12)
        x = torch.randn((1, 2, 8, 8), generator=generator).cuda()
        weight = torch.randn((3, 2, 3, 3), generator=generator).cuda()
        deform = deform_conv2d(x, torch.zeros((1, 18, 8, 8), device="cuda"), weight, padding=(1, 1))
        regular = torch.nn.functional.conv2d(x, weight, padding=1)
        self.zero_offset_difference = float((deform - regular).abs().max())
        if self.zero_offset_difference > 1e-4:
            raise ValueError("Available DCN operator failed zero-offset equivalence")

    def __call__(self, frame, result):
        if len(result.boxes) == 0: return []
        crops, crop_boxes = [], []
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        for box in result.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = box
            center, scale = self.bbox._box_to_center_scale(x1, y1, x2-x1, y2-y1, .75)
            matrix = self.transforms.get_affine_transform(center, scale, 0, [192, 256])
            crop = cv2.warpAffine(rgb, matrix, (192, 256), flags=cv2.INTER_LINEAR)
            tensor = self.transforms.im_to_torch(crop)
            tensor -= torch.tensor([.406, .457, .480])[:, None, None]
            crops.append(tensor); crop_boxes.append(self.bbox._center_scale_to_box(center, scale))
        with torch.inference_mode():
            heatmaps = self.model(torch.stack(crops).cuda()).cpu()
        points = []
        for heatmap, box in zip(heatmaps, crop_boxes):
            xy, score = self.transforms.heatmap_to_coord_simple(heatmap[:17], box)
            points.append(np.concatenate([xy, score], 1).astype(np.float32))
        return points
