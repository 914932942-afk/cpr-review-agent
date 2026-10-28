"""Fixed, post-selection ablation; never refits or replaces the frozen M2 choice."""

import json

import numpy as np
from evaluate_m2_quality import OUT, Config, condition, load, metric, predict


def fixed_average(sk, rgb, used, requested, hands):
    # Keep the same class support, conflict rules and hand protection; remove only quality adaptation.
    return predict(sk, rgb, np.full(len(used), 8), np.full(len(used), 16), hands, Config(0.75, 0.5))


def main():
    target = OUT / "experiment-01/ablation.json"
    if target.exists():
        raise FileExistsError("Preserve the first post-selection ablation")
    protocol = target.with_name("ablation-protocol.json")
    protocol.write_text(
        json.dumps(
            {
                "variants": ["frozen_baseline", "fixed_half_margin", "selected_quality_gate"],
                "selection_changes": False,
                "purpose": "Compare fixed .5 margin mixing against frozen two-parameter quality gating; same class support, hand protection and final guard. No new tuning.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    config = Config(**json.loads((OUT / "experiment-01/selection.json").read_text())["selected"])
    bundle = load(OUT / "prepared/development.npz")
    ids = np.flatnonzero(bundle["splits"] == "validation")
    result = {}
    for name in ["clean", "rgb_noise_half_support", "rgb_noise_reduced_support"]:
        args = condition(bundle, ids, name)
        result[name] = {
            "frozen_baseline": metric(predict(*args, None)[0], bundle["targets"][ids]),
            "fixed_half_margin": metric(fixed_average(*args)[0], bundle["targets"][ids]),
            "selected_quality_gate": metric(predict(*args, config)[0], bundle["targets"][ids]),
        }
    raw = load(OUT / "experiment-01/raw-readouts.npz")
    n = len(raw["record_ids"])
    args = (
        {k[3:]: v for k, v in raw.items() if k.startswith("sk_")},
        {k[4:]: v for k, v in raw.items() if k.startswith("rgb_")},
        raw["usable"],
        raw["requested"],
        raw["hand_available"],
    )
    result["raw26"] = {
        "frozen_baseline": metric(predict(*args, None)[0], raw["targets"]),
        "fixed_half_margin": metric(fixed_average(*args)[0], raw["targets"]),
        "selected_quality_gate": metric(predict(*args, config)[0], raw["targets"]),
    }
    target.write_text(
        json.dumps(
            {
                "passed": True,
                "records": {"validation": len(ids), "raw_regression": n},
                "no_selection_changes": True,
                "results": result,
                "scope": "Post-selection engineering ablation on previously inspected data; no new independent accuracy evidence.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                phase: {
                    key: {
                        "macro_f1": value["macro_f1"],
                        "shared_hamming_error": value["shared_hamming_error"],
                        "normal_false_alarms": value["normal_false_alarms"],
                    }
                    for key, value in variants.items()
                }
                for phase, variants in result.items()
            }
        )
    )


if __name__ == "__main__":
    main()
