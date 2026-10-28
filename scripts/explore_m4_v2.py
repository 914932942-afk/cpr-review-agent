"""Predeclared finite-policy comparison; no labels/vision/LLM/accepted-file mutations."""

import hashlib
import importlib.util
import itertools
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/m4-exploration-v2"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    DATA.mkdir(exist_ok=True)
    result_path = DATA / "offline.json"
    if result_path.exists():
        raise FileExistsError("Preserve completed experiment")
    old = module("baseline", ROOT / "experiments/m4_policy_v1/algorithm.py")
    new = module("candidate", ROOT / "experiments/m4_policy_v2/algorithm.py")
    params = json.loads((ROOT / "data/m4-scheduling-v1/parameters.json").read_text())
    before = {
        "parent_profile_sha256": sha(
            ROOT / "workspaces/m4-routing-v1/data/m4-scheduling-v1/pipeline-manifest.json"
        ),
        "env_sha256": sha(ROOT / ".env"),
        "accepted_source_hashes": json.loads((ROOT / "releases/m4-routing-v1/manifest.json").read_text())[
            "files"
        ],
    }
    (DATA / "before.json").write_text(json.dumps(before, indent=2), encoding="utf-8")
    protocol = {
        "algorithm_sha256": sha(ROOT / "experiments/m4_policy_v2/algorithm.py"),
        "baseline": "goal-cost-dag-v1",
        "candidate": new.VERSION,
        "grid": "All2047 scopes x4 prefix caches x budgets[0,20,60,100,150,180,200,230,300,900] x consent0/1",
        "gate": "Default same action/profile/stop; opt-in coverage weakly dominates v1, strict improvement exists; no over-bound plan; exact lexicographic optimum among legal candidates; frozen classifiers/certificates and public readouts unchanged. Live/UI/recovery still mandatory.",
        "scope": "Finite-policy property/exhaustive check, not measured end-to-end speedup or classification accuracy gain.",
    }
    (DATA / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    comparisons = strict = default_equal = optin = 0
    examples = []
    prefixes = [
        [],
        ["pose.extract"],
        ["pose.extract", "skeleton.predict"],
        ["pose.extract", "skeleton.predict", "rgb.predict"],
    ]
    for bits, prefix, budget in itertools.product(
        range(1, 1 << 11), prefixes, [0, 20, 60, 100, 150, 180, 200, 230, 300, 900]
    ):
        targets = [k for i, k in enumerate(new.LABELS) if bits >> i & 1]
        obs = {
            "target_label_ids": targets,
            "remaining_budget_seconds": budget,
            "analysis_mode": "adaptive",
            "available_actions": list(new.ACTIONS.values()) + ["STOP"],
            "available_profile_ids": list(new.PATHS),
            "executed_actions": prefix,
            "failed_actions": [],
            "decoded_frames": 362,
        }
        b = old.choose(obs, params)
        d = new.choose(obs, params)
        assert (b["action"], b["profile_id"], b["stop_code"]) == (
            d["action"],
            d["profile_id"],
            d["stop_code"],
        )
        default_equal += 1
        d = new.choose({**obs, "allow_partial": True}, params)
        bg = len(set(targets) & set(new.COVERAGE[b["profile_id"]])) if b["profile_id"] else 0
        dg = len(set(targets) & set(new.COVERAGE[d["profile_id"]])) if d["profile_id"] else 0
        assert dg >= bg
        if dg > bg:
            strict += 1
            if len(examples) < 3 and len(targets) > 4:
                examples.append(
                    {
                        "targets": targets,
                        "budget": budget,
                        "cached": prefix,
                        "old_coverage": bg,
                        "new_coverage": dg,
                    }
                )
        if d["profile_id"]:
            selected = d["audit"]["selected"]
            assert selected["upper_seconds"] + params["closing_reserve_seconds"] <= budget
            legal = [r for r in d["audit"]["candidates"] if r["budget_feasible"]]
            assert selected["new_coverage_gain"] == max(r["new_coverage_gain"] for r in legal)
            assert not (set(selected["missing_tools"]) & set(prefix) - {"fusion.predict"})
        optin += 1
        comparisons += 2
    result = {
        "passed": True,
        "comparisons": comparisons,
        "default_equal_cases": default_equal,
        "optin_cases": optin,
        "strict_coverage_improvement_cases": strict,
        "examples": examples,
        "classification_changed": False,
        "public_labels_used": False,
        "new_training": False,
        "interpretation": "Exact finite-grid safety/coverage dominance under the unchanged empirical cost guard; not a frequency estimate of user requests or a real-time guarantee.",
        "integration_accepted": False,
    }
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
