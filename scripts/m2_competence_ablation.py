"""Post-selection factorial ablation, without refitting any model or operating point."""

import json
from dataclasses import replace

import numpy as np
from evaluate_m2_quality import load, metric
from explore_m2_competence import OUT, ROOT, evaluate, new, prediction


def main():
    cfg = new.Config(**json.loads((OUT / "selection.json").read_text())["config"])
    variants = {
        "accepted_v1": None,
        "competence_only": replace(cfg, lock_unshared=False),
        "scope_only": replace(cfg, boundary_band=0),
        "without_competence_protection": replace(cfg, protect_specialists=False),
        "selected_v3": cfg,
    }
    raw = load(ROOT / "data/m2-quality-v1/experiment-01/raw-readouts.npz")
    args = (
        {k[3:]: v for k, v in raw.items() if k.startswith("sk_")},
        {k[4:]: v for k, v in raw.items() if k.startswith("rgb_")},
        raw["usable"],
        raw["requested"],
        raw["hand_available"],
    )
    bundle = load(ROOT / "data/m2-quality-v1/prepared/development.npz")
    val = np.flatnonzero(bundle["splits"] == "validation")
    report = {"no_refit": True, "previously_inspected_engineering_regression": True, "variants": {}}
    for name, c in variants.items():
        p, _ = prediction(args, c)
        report["variants"][name] = {
            "raw26": metric(p, raw["targets"]),
            "validation6": evaluate(bundle, val, c),
        }
    base, _ = prediction(args, None)
    selected, audits = prediction(args, cfg)
    report["changed_records"] = [
        {
            "record_id": str(raw["record_ids"][i]),
            "before": np.flatnonzero(base[i]).tolist(),
            "after": np.flatnonzero(selected[i]).tolist(),
            "audit": audits[i],
        }
        for i in range(len(base))
        if (base[i] != selected[i]).any()
    ]
    with (OUT / "ablation.json").open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: v["raw26"]["macro_f1"] for k, v in report["variants"].items()}))
    print(json.dumps(report["changed_records"]))


if __name__ == "__main__":
    main()
