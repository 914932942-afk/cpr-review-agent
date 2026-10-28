"""Prepare labelled public calibration records with frozen accepted M1/M2; no locked test read."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/branch_pipeline_v1"))
from pipeline import LABELS, rgb_readout, sha, skeleton_readout  # noqa: E402

OUT = ROOT / "data/m3-calibration-v1"
DATA = ROOT / "data/public_cpr"


def mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


m1 = mod("m3_frozen_m1", ROOT / "experiments/m1_coherent_v3/algorithm.py")
m2 = mod("m3_frozen_m2", ROOT / "experiments/m2_competence_v3/fusion_algorithm.py")


def read(p):
    return json.loads(p.read_text(encoding="utf-8"))


def load(p):
    with np.load(p, allow_pickle=False) as f:
        return {k: f[k].copy() for k in f.files}


def source_group(key):
    return str(key).rsplit("/", 1)[0]


def pose_loss(p, q, key):
    p, q = p.copy(), q.copy()
    rng = np.random.default_rng(int(hashlib.sha256(str(key).encode()).hexdigest()[:8], 16))
    length = max(6, int(len(p) * 0.25))
    for joint in rng.choice(17, 6, replace=False):
        start = int(rng.integers(2, max(3, len(p) - length - 2)))
        p[start : start + length, joint], q[start : start + length, joint] = 0, 0
    return p, q


def prepare():
    target = OUT / "prepared.npz"
    if target.exists():
        raise FileExistsError("Preserve M3 prepared data")
    cfg1 = m1.Config(**read(ROOT / "data/m1-m2-exploration-v1/m1-coherent/selection.json")["config"])
    cfg2 = m2.Config(**read(ROOT / "data/m1-m2-exploration-v1/m2-competence/selection.json")["config"])
    full = load(DATA / "fullpose-v2/development.npz")
    paired = load(ROOT / "data/m2-quality-v1/prepared/development.npz")
    rgb_lookup = {str(k): i for i, k in enumerate(paired["record_ids"])}
    metadata = {r["record_id"]: r for r in read(DATA / "run/records.json")}
    skeleton_indices = np.flatnonzero(full["splits"] == "validation")
    needed = sorted(
        set(skeleton_indices.tolist() + [i for i, k in enumerate(full["record_ids"]) if str(k) in rgb_lookup])
    )
    inputs, motions, points, confidence, audits, index = [], [], [], [], [], {}
    body_noisy_full, body_noisy_motion, body_noisy_quality = [], [], []
    hashes = {}
    for ordinal, i in enumerate(needed):
        key = str(full["record_ids"][i])
        row = metadata[key]
        path = DATA / "run" / row["record_file"]
        assert sha(path) == row["record_file_sha256"]
        hashes[path.relative_to(ROOT).as_posix()] = sha(path)
        a = load(path)
        p, q = a["keypoint"][0], a["keypoint_score"][0]
        g, m, audit = m1.features(p, q, cfg1)
        value = full["features"][i].copy()
        value[512:783] = g
        index[key] = len(inputs)
        inputs.append(value)
        motions.append(m)
        points.append(p)
        confidence.append(q)
        audits.append(audit)
        pp, qq = pose_loss(p, q, key)
        g, m, _ = m1.features(pp, qq, cfg1)
        value = full["features"][i].copy()
        value[512:783] = g
        body_noisy_full.append(value)
        body_noisy_motion.append(m)
        body_noisy_quality.append(float((qq >= 0.2).mean()))
        if (ordinal + 1) % 40 == 0:
            print("Prepared original public poses", ordinal + 1, "/", len(needed), flush=True)
    sk = skeleton_readout(np.stack(inputs), np.stack(motions))
    noisy = skeleton_readout(np.stack(body_noisy_full), np.stack(body_noisy_motion))
    rows = []
    for i in skeleton_indices:
        key = str(full["record_ids"][i])
        j = index[key]
        group = source_group(key)
        split = (
            "validation" if int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 5 in (0, 1) else "train"
        )
        for name, readout, quality in [
            ("clean", sk, float((confidence[j] >= 0.2).mean())),
            ("pose_loss", noisy, body_noisy_quality[j]),
        ]:
            for col, label in enumerate(LABELS):
                if label in (1, 2, 5, 7, 9):
                    continue
                rows.append(
                    (
                        key,
                        group,
                        split,
                        name,
                        label,
                        float(readout["scores"][j, col]),
                        float(readout["thresholds"][j, col]),
                        bool(readout["eligible"][j, col]),
                        float(quality),
                        0,
                        bool(full["targets"][i, col]),
                    )
                )
    for key, j in rgb_lookup.items():
        k = index[key]
        skrow = {name: value[k : k + 1].copy() for name, value in sk.items()}
        for condition in ["clean", "rgb_noise_half_support", "rgb_noise_reduced_support"]:
            x = paired["rgb_features"][j].copy()
            hand = bool(paired["hand_available"][j])
            used = int(paired["usable"][j])
            requested = int(paired["requested"][j])
            if condition != "clean":
                rng = np.random.default_rng(
                    int(hashlib.sha256((key + condition).encode()).hexdigest()[:8], 16)
                )
                x += (
                    rng.normal(0, 0.75 if condition == "rgb_noise_half_support" else 1.5, size=x.shape)
                    * paired["noise_scale"]
                )
                hand = False
                used = int(np.ceil(requested * (0.5 if condition == "rgb_noise_half_support" else 0.625)))
            rg = rgb_readout(x[None], paired["hand_features"][j : j + 1], np.array([hand]))
            fused, _ = m2.fuse(skrow, rg, used, requested, hand, cfg2)
            for rc, label in enumerate(rg["label_ids"]):
                col = LABELS.index(int(label))
                disagree = int(bool(skrow["eligible"][0, col]) != bool(rg["eligible"][0, rc]))
                rows.append(
                    (
                        key,
                        source_group(key),
                        str(paired["splits"][j]),
                        condition,
                        int(label),
                        float(fused["scores"][0, col]),
                        float(fused["thresholds"][0, col]),
                        bool(fused["predictions"][0, col]),
                        used / requested,
                        disagree,
                        bool(paired["targets"][j, col]),
                    )
                )
    names = [
        "record_ids",
        "groups",
        "splits",
        "conditions",
        "label_ids",
        "scores",
        "thresholds",
        "flagged",
        "quality",
        "disagreement",
        "targets",
    ]
    arrays = {name: np.asarray([row[i] for row in rows]) for i, name in enumerate(names)}
    np.savez_compressed(target, **arrays)
    report = {
        "prepared_sha256": sha(target),
        "source_hashes": hashes,
        "locked_test_opened": False,
        "new_manual_labels": False,
        "body_scope": "Non-RGB six classes, frozen head predictions on original validation196; deterministic record-group 60/40 split for this post-hoc engineering calibration. Earlier projects already inspected these records.",
        "rgb_scope": "Five RGB-covered classes, paired32 training +6 validation. Frozen upstream had previously used these training records; grouped calibration OOF does not make upstream predictions out-of-sample.",
        "conditions": "Programmed pose/feature stress retains public action labels; not real occlusion ground truth. All variants of one recording remain in its original group.",
        "upstream_sources": {
            p: sha(ROOT / p)
            for p in [
                "experiments/m1_coherent_v3/algorithm.py",
                "experiments/m2_competence_v3/fusion_algorithm.py",
                "data/m1-m2-exploration-v1/m1-coherent/selection.json",
                "data/m1-m2-exploration-v1/m2-competence/selection.json",
            ]
        },
        "summary": {
            pool: {
                split: {
                    "records": len(
                        set(
                            arrays["record_ids"][
                                (arrays["splits"] == split) & np.isin(arrays["label_ids"], labels)
                            ]
                        )
                    ),
                    "positive_labels": int(
                        arrays["targets"][
                            (arrays["splits"] == split)
                            & np.isin(arrays["label_ids"], labels)
                            & (arrays["conditions"] == "clean")
                        ].sum()
                    ),
                }
                for split in ["train", "validation"]
            }
            for pool, labels in {"body": [3, 4, 6, 8, 11, 13], "rgb": [1, 2, 5, 7, 9]}.items()
        },
    }
    (OUT / "prepared-audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"]), flush=True)


if __name__ == "__main__":
    prepare()
