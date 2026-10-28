"""M4 pure policy adapter; no Store, ML runtime, network or model API."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

from cpr_agent.modules.contracts import Decision

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("cpr_m4_algorithm", Path(__file__).with_name("algorithm.py"))
algorithm = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = algorithm
spec.loader.exec_module(algorithm)


class GoalCostPolicy:
    def __init__(self):
        folder = ROOT / "data/m4-exploration-v2"

        def read(p):
            return json.loads(p.read_text(encoding="utf-8"))

        def sha(p):
            return hashlib.sha256(p.read_bytes()).hexdigest()

        protocol, decision = read(folder / "protocol.json"), read(folder / "decision.json")
        if (
            not decision["accepted"]
            or decision["protocol_sha256"] != sha(folder / "protocol.json")
            or protocol["algorithm_sha256"] != sha(Path(__file__).with_name("algorithm.py"))
            or protocol["parameters_sha256"] != sha(folder / "parameters.json")
        ):
            raise ValueError("M4_POLICY_NOT_ACCEPTED")
        self.parameters = read(folder / "parameters.json")

    def choose(self, observation):
        return Decision(**algorithm.choose(observation.model_dump(mode="json"), self.parameters))
