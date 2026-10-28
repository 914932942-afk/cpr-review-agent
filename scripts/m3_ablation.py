"""Post-selection ablation. Every alternative is chosen/refit on calibration training groups only."""

import numpy as np
from fit_m3_calibration import OUT, POOLS, fit, keep, load, metrics, predict, read, take, write


def main():
    data = load()
    selection = read(OUT / "selection.json")
    training = read(OUT / "training.json")
    with np.load(OUT / "raw26.npz", allow_pickle=False) as f:
        raw = {k: f[k].copy() for k in data}
    variants = {}
    parameters = {}
    for name, labels in POOLS.items():
        train = take(data, (data["splits"] == "train") & np.isin(data["label_ids"], labels))
        score_only = min(
            (v for v in training[name]["candidates"] if v["q"] == 0 and v["d"] == 0),
            key=lambda v: (v["brier"], -v["ridge"]),
        )
        plain = fit(train, 0, 0, score_only["ridge"])
        selected = selection["parameters"][name]
        parameters[name] = {
            "ordinary_temperature": training[name]["temperature_parameters"],
            "score_only_two_parameter": plain,
            "selected_quality": selected,
        }
    for phase, source in [("validation", take(data, data["splits"] == "validation")), ("raw26", raw)]:
        variants[phase] = {}
        for variant in ["ordinary_temperature", "score_only_two_parameter", "selected_quality"]:
            p = np.zeros(len(source["targets"]))
            accepted = np.zeros(len(p), bool)
            for name, labels in POOLS.items():
                mask = np.isin(source["label_ids"], labels)
                subset = take(source, mask)
                p[mask] = predict(subset, parameters[name][variant])
                accepted[mask] = keep(subset, p[mask], selection["parameters"][name])
            variants[phase][variant] = metrics(source, p, accepted)
    write(
        "ablation.json",
        {
            "training_only_alternative_fits": True,
            "selection_unchanged": True,
            "parameters": parameters,
            "results": variants,
            "scope": "Previously inspected engineering splits. Same selected review thresholds for fixed-policy ablation; not an oracle policy for each alternative.",
        },
    )
    print({phase: {k: round(v["brier"], 6) for k, v in values.items()} for phase, values in variants.items()})


if __name__ == "__main__":
    main()
