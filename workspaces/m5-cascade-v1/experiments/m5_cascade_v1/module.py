"""M5 adapter: accepted dependency certificate, immutable models, bound RGB input."""

import hashlib
import importlib.util
import json
from pathlib import Path

from cpr_agent.modules.baselines import prediction_from_arrays
from cpr_agent.modules.contracts import MLResult

ROOT = Path(__file__).resolve().parents[2]


class QualityDependencyRGB:
    def __init__(self):
        folder = ROOT / "data/m5-cascade-v1"

        def sha(p):
            return hashlib.sha256(p.read_bytes()).hexdigest()

        def read(p):
            return json.loads(p.read_text(encoding="utf-8"))

        protocol, decision = read(folder / "protocol.json"), read(folder / "decision.json")
        if (
            not decision["accepted"]
            or decision["protocol_sha256"] != sha(folder / "protocol.json")
            or decision["certificate_sha256"] != sha(folder / "certificate.json")
        ):
            raise ValueError("M5_CASCADE_NOT_ACCEPTED")
        for filename, key in [("algorithm.py", "algorithm_sha256"), ("runtime.py", "runtime_sha256")]:
            if sha(Path(__file__).with_name(filename)) != protocol[key]:
                raise ValueError("M5_SOURCE_CHANGED")
        for name, expected in {**protocol["models"], **protocol["additional_sources"]}.items():
            if sha(ROOT / name) != expected:
                raise ValueError("M5_CERTIFICATE_SOURCE_CHANGED")

    def predict(self, request):
        if request.sampling_profile != "uniform16-v15":
            raise ValueError("M5_SAMPLE_PROFILE_CHANGED")
        spec = importlib.util.spec_from_file_location(
            "m5_cascade_inference", Path(__file__).with_name("inference.py")
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        arrays, audit = module.infer(request.video_path, request.points, request.confidence)
        return MLResult(prediction_from_arrays(request, arrays, arrays["label_ids"].tolist()), arrays, audit)
