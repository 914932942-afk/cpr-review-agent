"""Offline public-label feasibility baseline, separate from the Agent runtime.

No manual labels and no encoder optimization. Original official test partitions
remain test-only. This is a baseline, not an implementation of M1's improvement.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import io
import json
import pickle
import re
import sys
import tarfile
import time
import types
from collections import Counter
from pathlib import Path, PurePosixPath

import numpy as np
import torch
from torch import nn

REVISION = "b8667b8fd6a70ac63cc09fc6b763a9459b7c160a"
LABEL_IDS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13)
LABEL_NAMES = ("Overlap Hands", "Clenching Hands", "Single Hand", "Bending Arms", "Tilting Arms",
               "Jump Pressing", "Squatting", "Standing", "Wrong Position", "Slow Frequency", "Random Position Pressing")
EXPECTED_HASHES = {"ann.tar.gz": "72d99337152cf5873ddc5cf6608d360119f85fa57378cf94699688f6540951e4",
                   "Keypoints.tar.gz": "daf93b63af7663ef07988e2dcdc04400029fc1a82ed1afe101ff93a6c0a62441"}
ENCODER_HASH = "041f49c67281a1630eaa2931a000d7dc3e7a5a3dd87ee8a1b7c2ca0da2e71ab8"
SEED = 20261020


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def canonical_path(value):
    text = str(value).replace("\\", "/").strip("/")
    parts = PurePosixPath(text).parts
    if ".." in parts:
        raise ValueError("Parent traversal in dataset key")
    # Public keypoints use *_BIG for the corresponding primary RGB directory.
    parts = [p[:-4] if p.endswith("_BIG") else p for p in parts]
    known = [i for i,p in enumerate(parts) if p.startswith(("CPR_Dataset_", "CPR_Double_Dataset_", "CPR_Triple_Dataset_", "CPR_Quadruple_Dataset_", "CPR_TriQuad_Dataset_"))]
    if known:
        parts = parts[known[-1]:]
    # Public TXT includes a container directory omitted in supplemental PKLs.
    supplemental={"SupC14_Dataset":r"SupC\d{2}","SupDC59_Dataset":r"SupDC\d{2}",
                  "SupMC15_Dataset":r"SupMC\d{2}"}
    for index,part in enumerate(parts[:-1]):
        if part in supplemental and re.fullmatch(supplemental[part],parts[index+1]):
            parts=parts[index+1:]
            break
    return "/".join(parts)


def encode_labels(labels):
    if any(isinstance(x, bool) or int(x) != x for x in labels):
        raise ValueError("Original labels must be integer IDs")
    labels = sorted(set(int(x) for x in labels))
    if not labels or any(x < 0 or x > 13 for x in labels) or (0 in labels and labels != [0]):
        raise ValueError(f"Invalid original labels: {labels}")
    return np.asarray([int(label in labels) for label in LABEL_IDS], dtype=np.float32)


class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {
            ("numpy", "ndarray"): np.ndarray, ("numpy", "dtype"): np.dtype,
            ("numpy.core.multiarray", "_reconstruct"): np._core.multiarray._reconstruct,
            ("numpy._core.multiarray", "_reconstruct"): np._core.multiarray._reconstruct,
            ("numpy.core.numeric", "_frombuffer"): np._core.numeric._frombuffer,
            ("numpy._core.numeric", "_frombuffer"): np._core.numeric._frombuffer,
        }
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(f"Disallowed pickle global: {module}.{name}")
        return allowed[module, name]


def read_annotations(archive, view):
    selected = {}
    counts = {}
    names = {"trainlist.txt": "official_train", "testlist.txt": "official_single_test",
             "all_errors_testlist.txt": "official_composite_test"}
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle:
            filename = PurePosixPath(member.name).name
            if filename not in names or not member.isfile():
                continue
            if member.size > 2**20:
                raise ValueError("Unexpected annotation size")
            lines = bundle.extractfile(member).read().decode("utf-8-sig").splitlines()
            counts[filename] = 0
            for line in lines:
                parts = line.split()
                if not parts:
                    continue
                key = canonical_path(parts[0])
                if not key.endswith("/" + view):
                    continue
                counts[filename] += 1
                labels = [int(x) for x in parts[2:]]
                target = encode_labels(labels)
                # Only retain true original Correct or a supported target error.
                # Depth-only samples are not re-labelled as Correct.
                if not target.any() and labels != [0]:
                    continue
                if key in selected:
                    raise ValueError(f"Official partition overlap: {key}")
                selected[key] = {"record_id": key, "original_partition": names[filename],
                                 "original_labels": labels, "target": target.tolist(),
                                 "annotation_frame_dir":parts[0],
                                 "expected_frames": int(parts[1]), "view": view}
    if set(counts) != set(names):
        raise ValueError(f"Missing original annotation files: {counts}")
    return selected, counts


def choose_splits(records):
    result = [dict(r) for r in records]
    rng = np.random.default_rng(SEED)
    training = [r for r in result if r["original_partition"] == "official_train"]
    by_label = {}
    for record in training:
        if len(record["original_labels"]) != 1:
            raise ValueError("Original training partition is not single-label")
        by_label.setdefault(record["original_labels"][0], []).append(record)
    expected = {0, *LABEL_IDS}
    if set(by_label) != expected:
        raise ValueError("Missing supported training class")
    for label in sorted(by_label):
        group = sorted(by_label[label], key=lambda r: r["record_id"])
        if len(group) < 10:
            raise ValueError(f"Too few records for label {label}: {len(group)}")
        order = rng.permutation(len(group))
        validation = set(int(i) for i in order[:4])
        for i, record in enumerate(group):
            record["split"] = "validation" if i in validation else "train"
    for record in result:
        if record["original_partition"] == "official_single_test":
            record["split"] = "single_test"
        elif record["original_partition"] == "official_composite_test":
            record["split"] = "composite_test"
    keys = [r["record_id"] for r in result]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate recordings")
    return sorted(result, key=lambda r: (r["split"], r["record_id"]))


def prepare(assets, output, view):
    for name, expected in EXPECTED_HASHES.items():
        if digest(assets / name) != expected:
            raise ValueError(f"Public source hash mismatch: {name}")
    wanted, annotation_counts = read_annotations(assets / "ann.tar.gz", view)
    prepared = []
    rejected = []
    seen = set()
    output.mkdir(parents=True, exist_ok=True)
    record_dir = output / "records"
    record_dir.mkdir(exist_ok=True)
    layouts, max_confidence, out_of_declared_bounds = Counter(), 0.0, 0
    source_partitions={"train_keypoints.pkl":"official_train","test_keypoints.pkl":"official_single_test",
                       "all_errors_keypoints.pkl":"official_composite_test"}
    with tarfile.open(assets / "Keypoints.tar.gz", mode="r|gz") as bundle:
        for member in bundle:
            basename=PurePosixPath(member.name).name
            # double_errors_keypoints.pkl duplicates the double-error subset
            # already present in all_errors_keypoints.pkl; do not sample twice.
            if not member.isfile() or basename not in source_partitions:
                continue
            print(f"Reading restricted public pickle: {member.name} ({member.size/2**20:.1f} MiB)", flush=True)
            payload = RestrictedUnpickler(bundle.extractfile(member)).load()
            records = payload.get("annotations") if isinstance(payload, dict) else payload
            if not isinstance(records, list):
                raise ValueError("Unexpected keypoint archive structure")
            for record in records:
                if not isinstance(record, dict) or "frame_dir" not in record:
                    raise ValueError("Invalid keypoint record")
                key = canonical_path(record["frame_dir"])
                if key not in wanted:
                    continue
                if key in seen:
                    raise ValueError(f"Duplicate keypoint record: {key}")
                seen.add(key)
                meta = wanted[key]
                if meta["original_partition"] != source_partitions[basename]:
                    raise ValueError(f"Keypoint/annotation partition mismatch: {key}")
                raw_labels = np.asarray(record["label"]).reshape(-1)
                if raw_labels.size == 14 and np.isin(raw_labels, [0,1]).all():
                    raw_labels = np.flatnonzero(raw_labels)
                if sorted(int(x) for x in raw_labels) != sorted(meta["original_labels"]):
                    raise ValueError(f"Public pickle/TXT label mismatch: {key}")
                kp = np.asarray(record["keypoint"])
                confidence = np.asarray(record["keypoint_score"])
                if kp.ndim != 4 or kp.shape[0] != 1 or kp.shape[-1] != 2 or kp.shape[:3] != confidence.shape:
                    raise ValueError(f"Unsupported pose shape: {key}: {kp.shape}, {confidence.shape}")
                if kp.shape[2] not in {17,26,68}:
                    raise ValueError(f"Unverified joint layout: {kp.shape[2]}")
                if kp.shape[1] != record["total_frames"] or kp.shape[1] != meta["expected_frames"]:
                    rejected.append({"record_id": key, "reason": "frame_count_mismatch"})
                    continue
                if not np.isfinite(kp).all() or not np.isfinite(confidence).all():
                    rejected.append({"record_id": key, "reason": "non_finite_input"})
                    continue
                layouts[int(kp.shape[2])] += 1
                max_confidence = max(max_confidence, float(confidence.max()))
                image_shape = tuple(int(x) for x in record["img_shape"])
                out_of_declared_bounds += int((kp[...,0] >= image_shape[1]).any() or (kp[...,1] >= image_shape[0]).any())
                # Halpe-68 retains the first 26 Halpe body joints; first 17 match COCO.
                body = kp[...,:17,:].astype(np.float32)
                scores = np.clip(confidence[...,:17], 0, 1).astype(np.float32)
                filename = hashlib.sha256(key.encode()).hexdigest() + ".npz"
                np.savez_compressed(record_dir / filename, keypoint=body, keypoint_score=scores,
                                    img_shape=np.asarray(image_shape), record_id=np.asarray(key))
                prepared.append({**meta, "source_pickle_member": member.name,
                                 "source_joint_count": int(kp.shape[2]), "record_file": "records/"+filename,
                                 "record_file_sha256": digest(record_dir / filename)})
            del records, payload
            gc.collect()
            print(f"Prepared selected view records: {len(prepared)}", flush=True)
    split_records = choose_splits(prepared)
    missing = sorted(set(wanted)-seen)
    summary = {"source_revision": REVISION, "source": "https://huggingface.co/datasets/ShunliWang/CPR-Coach",
               "source_sha256": EXPECTED_HASHES, "view": view, "label_ids": list(LABEL_IDS),
               "label_names": list(LABEL_NAMES), "original_annotation_counts": annotation_counts,
               "prepared_counts": dict(Counter(r["split"] for r in split_records)),
               "missing_keypoint_records": missing, "rejected_records": rejected,
               "source_joint_layout_counts": dict(layouts), "source_confidence_max": max_confidence,
               "source_member_selection":source_partitions,
               "duplicate_member_policy":"Skip double_errors_keypoints.pkl because the all_errors member already includes this test subset.",
               "records_with_coordinates_outside_declared_image": out_of_declared_bounds,
               "normalization_policy": "First17 COCO-compatible body joints; confidence clipped to [0,1]. PoseCompact allow_imgpad=True derives its crop from joint coordinates; no guessed image rescaling and no physical-distance measurement.",
               "split_policy": "4 validation recordings per original training class; seed 20261020. Official single/composite test lists never used for training or threshold selection. No subject IDs inferred.",
               "manual_labels_added": False, "local_videos_used_for_training": False}
    write_json(output / "records.json", split_records)
    write_json(output / "data-audit.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if missing or rejected:
        raise ValueError("Incomplete public input coverage: inspect data-audit.json before continuing")


def install_compatibility():
    if "mmaction.models.localizers.drn" not in sys.modules:
        package = types.ModuleType("mmaction.models.localizers.drn")
        package.__path__ = []
        module = types.ModuleType("mmaction.models.localizers.drn.drn")
        module.DRN = type("DRN", (), {})
        sys.modules[package.__name__], sys.modules[module.__name__] = package, module
    if "mmaction.models.multimodal" not in sys.modules:
        package = types.ModuleType("mmaction.models.multimodal")
        package.__path__ = []
        sys.modules[package.__name__] = package
    np.Inf = np.inf


class FrozenEncoder:
    def __init__(self, checkpoint, device="cuda"):
        if digest(checkpoint) != ENCODER_HASH:
            raise ValueError("Unexpected frozen encoder checkpoint")
        install_compatibility()
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True
        import mmaction
        from mmaction.apis import init_recognizer
        from mmengine import Config
        from mmengine.dataset import Compose
        config_path = Path(mmaction.__file__).parent / ".mim/configs/skeleton/posec3d/slowonly_r50_8xb32-u48-240e_k400-keypoint.py"
        config = Config.fromfile(str(config_path))
        self.pipeline_config = [copy.deepcopy(x) for x in config.val_pipeline if x["type"] != "DecompressPose"]
        self.model = init_recognizer(config, str(checkpoint), device=device).eval().requires_grad_(False)
        saved=torch.load(checkpoint,map_location="cpu",weights_only=True)["state_dict"]
        current=self.model.state_dict()
        backbone_keys={key for key in current if key.startswith("backbone.")}
        if backbone_keys != set(saved) or any(not torch.equal(current[key].cpu(),saved[key]) for key in backbone_keys):
            raise ValueError("Frozen backbone not fully and exactly loaded")
        self.backbone_tensor_count=len(backbone_keys)
        self.pipeline = Compose(self.pipeline_config)
        self.device = device
        if any(p.requires_grad for p in self.model.parameters()):
            raise AssertionError("Encoder must be frozen")

    def encode(self, records):
        from mmengine.dataset import pseudo_collate
        items = []
        for record in records:
            kp = np.asarray(record["keypoint"], dtype=np.float32)
            scores = np.asarray(record["keypoint_score"], dtype=np.float32)
            if (kp.ndim != 4 or kp.shape[0] != 1 or kp.shape[1] < 1 or
                kp.shape[:3] != scores.shape or kp.shape[2:] != (17,2) or
                not np.isfinite(kp).all() or not np.isfinite(scores).all() or
                (scores < 0).any() or (scores > 1).any()):
                raise ValueError("Expected finite [M,T,17,2] keypoints and matching scores")
            item = {"frame_dir": "", "label": -1, "img_shape": tuple(record["img_shape"]),
                    "original_shape": tuple(record["img_shape"]), "total_frames": kp.shape[1],
                    "keypoint": kp.copy(), "keypoint_score": scores.copy(), "start_index": 0,
                    "modality": "Pose", "test_mode": True}
            items.append(self.pipeline(item))
        batch = pseudo_collate(items)
        with torch.inference_mode():
            processed = self.model.data_preprocessor(batch, training=False)
            feature = self.model.extract_feat(processed["inputs"], stage="backbone", test_mode=True)
            if isinstance(feature, (tuple,list)):
                feature = feature[0]
            feature = torch.nn.functional.normalize(feature.float().mean((2,3,4)), dim=1)
        value = feature.cpu().numpy()
        if value.shape != (len(records),512) or not np.isfinite(value).all():
            raise ValueError("Invalid frozen features")
        return value


def encoder_profile(encoder):
    profile={"backbone_sha256": ENCODER_HASH, "pipeline": encoder.pipeline_config,
             "source_joint_layout": "Halpe68/26 first17 (COCO17)", "confidence": "clip[0,1]",
             "feature": "backbone spatial-temporal mean + L2; 512D", "main_network_trained": False,
             "source_revision": REVISION,"precision":"float32; TF32 disabled; cuDNN deterministic"}
    return profile,hashlib.sha256(json.dumps(profile,sort_keys=True).encode()).hexdigest()


def extract(output, checkpoint, batch_size):
    torch.set_num_threads(4)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    records = json.loads((output / "records.json").read_text(encoding="utf-8"))
    audit = json.loads((output / "data-audit.json").read_text(encoding="utf-8"))
    if audit["missing_keypoint_records"] or audit["rejected_records"]:
        raise ValueError("Public data audit has unresolved inputs")
    for record in records:
        if digest(output / record["record_file"]) != record["record_file_sha256"]:
            raise ValueError("Prepared input hash mismatch")
    encoder = FrozenEncoder(checkpoint)
    profile,profile_hash=encoder_profile(encoder)
    folder = output / "features" / profile_hash[:16]
    folder.mkdir(parents=True,exist_ok=True)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for offset in range(0,len(records),batch_size):
        group = records[offset:offset+batch_size]
        pending = [r for r in group if not (folder/(Path(r["record_file"]).stem+".npz")).exists()]
        if pending:
            loaded = []
            for record in pending:
                with np.load(output / record["record_file"],allow_pickle=False) as source:
                    loaded.append({k:source[k].copy() for k in ("keypoint","keypoint_score","img_shape")})
            features = encoder.encode(loaded)
            for record, feature in zip(pending,features):
                np.savez_compressed(folder/(Path(record["record_file"]).stem+".npz"), feature=feature,
                                    record_id=np.asarray(record["record_id"]), profile_hash=np.asarray(profile_hash),
                                    record_file_sha256=np.asarray(record["record_file_sha256"]))
        if offset % (batch_size*10) == 0:
            print(f"Frozen features {min(offset+batch_size,len(records))}/{len(records)} elapsed={time.perf_counter()-started:.1f}s", flush=True)
    all_features = []
    for record in records:
        with np.load(folder/(Path(record["record_file"]).stem+".npz"),allow_pickle=False) as f:
            if (str(f["profile_hash"]) != profile_hash or str(f["record_id"]) != record["record_id"] or
                str(f["record_file_sha256"]) != record["record_file_sha256"]):
                raise ValueError("Stale/incompatible feature cache")
            all_features.append(f["feature"].copy())
    with np.load(output / records[0]["record_file"], allow_pickle=False) as f:
        repeat = encoder.encode([{k:f[k].copy() for k in ("keypoint","keypoint_score","img_shape")}])
    repeat_error = float(np.abs(repeat[0]-all_features[0]).max())
    if repeat_error > 1e-5:
        raise AssertionError(f"Feature repeat mismatch: {repeat_error}")
    np.savez_compressed(output / "feature-bundle.npz", features=np.stack(all_features),
                        targets=np.asarray([r["target"] for r in records],dtype=np.float32),
                        splits=np.asarray([r["split"] for r in records]),
                        record_ids=np.asarray([r["record_id"] for r in records]),
                        label_ids=np.asarray(LABEL_IDS), profile_hash=np.asarray(profile_hash))
    if digest(checkpoint) != ENCODER_HASH:
        raise AssertionError("Encoder file changed")
    write_json(output / "encoder-profile.json", {**profile,"profile_hash":profile_hash})
    write_json(output / "extraction-audit.json", {"records":len(records),"elapsed_seconds":time.perf_counter()-started,
               "batch_size":batch_size,"gpu":torch.cuda.get_device_name(),"torch":str(torch.__version__),
               "cuda_peak_allocated_bytes":torch.cuda.max_memory_allocated(),"repeat_max_difference":repeat_error,
               "all_encoder_parameters_frozen":all(not p.requires_grad for p in encoder.model.parameters()),
               "encoder_file_unchanged":True,"exactly_loaded_backbone_tensors":encoder.backbone_tensor_count,
               "unused_pretraining_classifier":"Checkpoint contains backbone only; randomly initialized K400 classification head is never executed.",
               "profile_hash":profile_hash})


class Head(nn.Module):
    def __init__(self, kind="mlp"):
        super().__init__()
        self.kind = kind
        self.register_buffer("feature_mean", torch.zeros(512))
        self.register_buffer("feature_scale", torch.ones(512))
        self.classifier = nn.Linear(512,11) if kind == "linear" else nn.Sequential(nn.Linear(512,64),nn.ReLU(),nn.Dropout(.2),nn.Linear(64,11))

    def forward(self, feature):
        if feature.ndim != 2 or feature.shape[1] != 512:
            raise ValueError("Head requires [N,512] frozen PoseC3D features")
        return self.classifier((feature-self.feature_mean)/self.feature_scale)


def load_checked_head(path, profile_hash):
    checkpoint=torch.load(path,map_location="cpu",weights_only=True)
    if (checkpoint.get("schema_version") != 1 or checkpoint.get("label_ids") != list(LABEL_IDS) or
        checkpoint.get("encoder_profile_hash") != profile_hash or
        checkpoint.get("source_revision") != REVISION or checkpoint.get("kind") not in {"linear","mlp"}):
        raise ValueError("Incompatible head labels, backbone profile, or source revision")
    head=Head(checkpoint["kind"]).eval()
    head.load_state_dict(checkpoint["head_state"],strict=True)
    return head, checkpoint


def metrics(probability, target, threshold=.5):
    prediction = probability >= threshold
    target = target.astype(bool)
    tp = (prediction & target).sum(0)
    fp = (prediction & ~target).sum(0)
    fn = (~prediction & target).sum(0)
    f1 = 2*tp/np.maximum(2*tp+fp+fn,1)
    recall = tp/np.maximum(target.sum(0),1)
    average_precision=[]
    for k in range(11):
        order=np.argsort(-probability[:,k],kind="stable")
        truth=target[order,k]
        scores=probability[order,k]
        # Integrate at distinct thresholds so tied/constant scores cannot gain
        # artificial AP from the arbitrary order of labelled records.
        ends=np.r_[np.flatnonzero(np.diff(scores)),len(scores)-1]
        cumulative=np.cumsum(truth)[ends]
        increments=np.diff(np.r_[0,cumulative])
        average_precision.append(float((increments*cumulative/(ends+1)).sum()/max(truth.sum(),1)))
    normal = ~target.any(1)
    return {"records":len(target),"micro_f1":float(2*tp.sum()/max(2*tp.sum()+fp.sum()+fn.sum(),1)),
            "macro_f1":float(f1.mean()),"mAP":float(np.mean(average_precision)),
            "exact_match":float((prediction==target).all(1).mean()),
            "original_correct_records":int(normal.sum()),
            "correct_false_alarm_rate":float(prediction[normal].any(1).mean()) if normal.any() else None,
            "threshold":float(threshold),
            "per_label":[{"id":label,"name":name,"support":int(target[:,k].sum()),
                          "predicted_positive":int(prediction[:,k].sum()),"f1":float(f1[k]),
                          "recall":float(recall[k]),"average_precision":average_precision[k]}
                         for k,(label,name) in enumerate(zip(LABEL_IDS,LABEL_NAMES))]}


def train(output):
    torch.set_num_threads(4)
    np.random.seed(SEED)
    with np.load(output / "feature-bundle.npz",allow_pickle=False) as f:
        if not np.array_equal(f["label_ids"],LABEL_IDS):
            raise ValueError("Incompatible original label order")
        x=torch.from_numpy(f["features"].copy()).float()
        y=torch.from_numpy(f["targets"].copy()).float()
        split=f["splits"].copy()
        profile=str(f["profile_hash"])
    masks={name:torch.from_numpy(split==name) for name in ("train","validation","single_test","composite_test")}
    if any(not mask.any() for mask in masks.values()):
        raise ValueError("All required partitions must be present")
    x_train,y_train=x[masks["train"]],y[masks["train"]]
    x_val,y_val=x[masks["validation"]],y[masks["validation"]]
    results={}
    for kind in ("linear","mlp"):
        torch.manual_seed(SEED)
        head=Head(kind)
        head.feature_mean.copy_(x_train.mean(0))
        head.feature_scale.copy_(x_train.std(0,unbiased=False).clamp_min(1e-4))
        weights=(len(y_train)-y_train.sum(0))/y_train.sum(0).clamp_min(1)
        loss_fn=nn.BCEWithLogitsLoss(pos_weight=weights)
        optimizer=torch.optim.AdamW(head.parameters(),lr=1e-3,weight_decay=1e-3)
        best_loss=float("inf")
        best_state=None
        best_epoch=0
        stale=0
        history=[]
        started=time.perf_counter()
        for epoch in range(1,201):
            head.train()
            optimizer.zero_grad(set_to_none=True)
            loss=loss_fn(head(x_train),y_train)
            loss.backward()
            optimizer.step()
            head.eval()
            with torch.inference_mode():
                validation=float(loss_fn(head(x_val),y_val))
            if not np.isfinite(validation):
                raise FloatingPointError("Non-finite head loss")
            history.append({"epoch":epoch,"train_loss":float(loss.detach()),"validation_loss":validation})
            if validation < best_loss-1e-5:
                best_loss=validation
                best_epoch=epoch
                best_state={k:v.detach().clone() for k,v in head.state_dict().items()}
                stale=0
            else:
                stale+=1
            if stale>=25:
                break
        head.load_state_dict(best_state)
        head.eval()
        with torch.inference_mode():
            validation_probability=head(x_val).sigmoid().numpy()
        candidates=[(metrics(validation_probability,y_val.numpy(),t)["macro_f1"],-abs(t-.5),t)
                    for t in np.linspace(.1,.9,17)]
        threshold=float(max(candidates)[2])
        checkpoint={"schema_version":1,"kind":kind,"head_state":head.state_dict(),"label_ids":list(LABEL_IDS),
                    "label_names":list(LABEL_NAMES),"encoder_profile_hash":profile,"threshold":threshold,
                    "source_revision":REVISION,"seed":SEED,"best_epoch":best_epoch,
                    "training_record_count":len(x_train),"validation_record_count":len(x_val),
                    "score_semantics":"uncalibrated sigmoid scores; no clinical or all-class release claim"}
        destination=output/f"{kind}-head.pt"
        torch.save(checkpoint,destination)
        restored=Head(kind)
        restored.load_state_dict(torch.load(destination,map_location="cpu",weights_only=True)["head_state"])
        restored.eval()
        with torch.inference_mode():
            before=head(x).sigmoid()
            after=restored(x).sigmoid()
        reload_error=float((before-after).abs().max())
        if reload_error != 0:
            raise AssertionError("Saved head reload differs")
        evaluations={}
        for name,mask in masks.items():
            probability=after[mask].numpy()
            truth=y[mask].numpy()
            evaluations[name]={"default_threshold":metrics(probability,truth,.5),
                               "validation_selected_threshold":metrics(probability,truth,threshold),
                               "all_negative_baseline":metrics(np.zeros_like(truth),truth),
                               "all_positive_baseline":metrics(np.ones_like(truth),truth)}
            np.savez_compressed(output/f"{kind}-{name}-predictions.npz",probabilities=probability,targets=truth)
        results[kind]={"parameters":sum(p.numel() for p in head.parameters()),"best_epoch":best_epoch,
                       "epochs_executed":len(history),"head_training_seconds":time.perf_counter()-started,
                       "reload_max_difference":reload_error,"checkpoint_sha256":digest(destination),
                       "threshold_selected_on_validation_only":threshold,"evaluations":evaluations,
                       "history":history}
        print(f"{kind}: head parameters={results[kind]['parameters']}; best epoch={best_epoch}; single-test macroF1={evaluations['single_test']['validation_selected_threshold']['macro_f1']:.4f}",flush=True)
    selected=max(results,key=lambda name:results[name]["evaluations"]["validation"]["validation_selected_threshold"]["macro_f1"])
    write_json(output/"head-training-results.json",{"selected_by_validation":selected,"models":results,
               "main_network_trained":False,"manual_data_or_labels_added":False,
               "task":"original single-error training; unseen composite-error test retained", "final_test_used_for_selection":False})


def main():
    paths_file=Path(__file__).resolve().parent/"paths.json"
    defaults=json.loads(paths_file.read_text(encoding="utf-8")) if paths_file.exists() else {}
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=["prepare","extract","train"])
    parser.add_argument("--assets",type=Path,default=Path(defaults.get("dataset_assets",Path(__file__).resolve().parent/"assets")))
    parser.add_argument("--output",type=Path,default=Path(defaults.get("output",Path(__file__).resolve().parent/"run")))
    parser.add_argument("--view",choices=["ch0","ch1","ch2","ch3"],default="ch0")
    parser.add_argument("--backbone",type=Path,default=Path(defaults["backbone"]) if "backbone" in defaults else None)
    parser.add_argument("--batch-size",type=int,default=4)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if args.stage=="prepare":
        prepare(args.assets,args.output,args.view)
    elif args.stage=="extract":
        if args.backbone is None:
            parser.error("extract requires --backbone or a configured paths.json")
        extract(args.output,args.backbone,args.batch_size)
    else:
        train(args.output)


if __name__=="__main__":
    main()
