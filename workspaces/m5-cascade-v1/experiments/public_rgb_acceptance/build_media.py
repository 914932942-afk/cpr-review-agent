"""Extract only frozen ch0 records, verify JPEGs, create/decode demo MP4s.

Class labels are copied from verified public annotations. This program does
not import the recognition model, add labels, or run predictions.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
import tarfile
import time
import cv2
import numpy as np
from PIL import Image
from common import CODE, DATA, ROOT, REPORT, SHA, SIZE, CHUNK, chunk_path, read, write, sha, ChunksReader


def safe_member(name, byte_count, selected):
    parts = PurePosixPath(name).parts
    if name.startswith(("/", "\\")) or ".." in parts or "\\" in name:
        raise ValueError("Unsafe archive member path")
    directory = "/".join(parts[:-1])
    if directory not in selected: return None
    if not re.fullmatch(r"\d{8}\.jpg", parts[-1]) or not 4 <= byte_count <= 4*2**20:
        raise ValueError("Unexpected selected RGB member")
    return selected[directory], parts[-1]


def verify_frame_numbers(images, expected):
    if expected < 1 or not images: raise ValueError("Empty public frame sequence")
    numbers = sorted(int(Path(r["filename"]).stem) for r in images)
    if len(numbers) != expected or len(set(numbers)) != expected or numbers != list(range(numbers[0], numbers[0]+expected)):
        raise ValueError("Public frame sequence missing, duplicated or non-contiguous")
    return numbers[0]


def jpeg_information(payload):
    if payload[:2] != b"\xff\xd8" or payload[-2:] != b"\xff\xd9":
        raise ValueError("Incomplete JPEG boundary markers")
    with Image.open(BytesIO(payload)) as image:
        if image.format != "JPEG": raise ValueError("Unexpected image format")
        shape = (int(image.height), int(image.width))
        if min(shape) <= 0 or max(shape) > 8192: raise ValueError("Unsupported JPEG dimensions")
        image.verify()
    return shape


def folder_for(record): return ROOT / "records" / record["record_id"].replace("/", "--")


def extract_supplemental(plan):
    proof = read(ROOT / "source-archive-integrity.json")
    if not proof["verified"] or proof["sha256"] != SHA or proof["bytes"] != SIZE:
        raise ValueError("Whole public archive is not verified")
    selected = {r["annotation_frame_dir"]:r for r in plan["records"] if r["source_kind"] == "supplemental_archive"}
    frames = {r["record_id"]:{} for r in selected.values()}; shapes = {}
    for r in selected.values(): (folder_for(r)/"images").mkdir(parents=True,exist_ok=True)
    reader = ChunksReader([chunk_path(i) for i in range(math.ceil(SIZE/CHUNK))])
    count = 0; members = 0; started = time.monotonic(); last_report = started
    try:
        with tarfile.open(fileobj=reader, mode="r|gz") as archive:
            for member in archive:
                members += 1
                if not member.isfile(): continue
                match = safe_member(member.name, member.size, selected)
                if match is not None:
                    record, filename = match; rid = record["record_id"]
                    if filename in frames[rid]: raise ValueError("Repeated source JPEG: " + rid + "/" + filename)
                    payload = archive.extractfile(member).read()
                    if len(payload) != member.size: raise EOFError("Incomplete public JPEG payload")
                    shape = jpeg_information(payload)
                    if rid in shapes and shapes[rid] != shape: raise ValueError("Dimensions vary within a public recording")
                    shapes[rid] = shape
                    path = folder_for(record)/"images"/filename
                    path.write_bytes(payload)
                    frames[rid][filename] = {"filename":filename,"bytes":len(payload),
                        "sha256":hashlib.sha256(payload).hexdigest(),"archive_member":member.name}
                    count += 1
                now=time.monotonic()
                if now-last_report>10:
                    print(f"Extract ch0 RGB: {count}/{plan['supplemental_frames']} frames; archive members scanned={members}",flush=True)
                    write(ROOT/"media-build-status.json",{"phase":"extracting_verified_archive","selected_frames":count,
                        "expected_selected_frames":plan["supplemental_frames"],"archive_members_scanned":members})
                    last_report=now
    finally: reader.close()
    for record in selected.values():
        images=sorted(frames[record["record_id"]].values(),key=lambda r:r["filename"])
        start=verify_frame_numbers(images,record["expected_frames"])
        manifest={**record,"image_shape":list(shapes[record["record_id"]]),"source_start_frame_number":start,
            "source_archive_sha256":SHA,"whole_archive_hash_verified":True,"dataset_revision":plan["dataset_revision"],
            "frozen_plan_sha256":sha(ROOT/"frozen-record-plan.json"),"images":images,
            "labels_from":"Existing public ann.tar.gz + validated record inventory; no new labels",
            "new_manual_labels":False,"state":"source_images_verified"}
        write(folder_for(record)/"manifest.json",manifest)
    print("Supplemental extraction verified",count,"frames,",len(selected),"records",flush=True)


def import_primary(plan):
    source_root=DATA/"video-verification"
    old_manifest={"CPR_Dataset_S0/C13_S0/r11/ch0":"manifest-public-C13-r11.json",
                  "CPR_Dataset_S0/C13_S0/r12/ch0":"manifest-CPR_Dataset_S0-C13_S0-r12-ch0.json"}
    for record in plan["records"]:
        if record["source_kind"]!="existing_pinned_parquet_subset":continue
        path=source_root/old_manifest[record["record_id"]]; prior=read(path)
        if prior["record_id"]!=record["record_id"] or prior["original_label_ids"]!=record["original_labels"]:
            raise ValueError("Existing primary provenance differs from public annotation")
        source_folder=source_root/record["record_id"].replace("/","-")
        folder=folder_for(record);(folder/"images").mkdir(parents=True,exist_ok=True)
        shapes=set()
        for image in prior["images"]:
            source=source_folder/image["filename"]
            if sha(source)!=image["sha256"]:raise ValueError("Existing primary JPEG hash differs")
            shapes.add(jpeg_information(source.read_bytes()))
            shutil.copyfile(source,folder/"images"/image["filename"])
        if len(shapes)!=1:raise ValueError("Existing primary image dimensions vary")
        start=verify_frame_numbers(prior["images"],record["expected_frames"])
        write(folder/"manifest.json",{**record,"image_shape":list(next(iter(shapes))),"source_start_frame_number":start,
            "dataset_revision":plan["dataset_revision"],"conversion_revision":prior["conversion_revision"],
            "source_manifest":str(path.relative_to(CODE)),"source_manifest_sha256":sha(path),
            "frozen_plan_sha256":sha(ROOT/"frozen-record-plan.json"),"images":prior["images"],
            "whole_archive_hash_verified":False,"transport":"Pinned public Parquet subset + stored per-image hashes",
            "labels_from":"Existing public annotations; no new labels","new_manual_labels":False,"state":"source_images_verified"})
        print("Imported verified existing primary RGB",record["record_id"],flush=True)


def inspect_video(path, manifest):
    capture=cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise ValueError("Derived MP4 cannot be opened")
    count=0;fps=float(capture.get(cv2.CAP_PROP_FPS));shape=tuple(manifest["image_shape"])
    samples={0,manifest["expected_frames"]//2,manifest["expected_frames"]-1};errors=[]
    try:
        while True:
            ok,frame=capture.read()
            if not ok:break
            if frame.shape[:2]!=shape:raise ValueError("MP4 dimensions differ from source JPEGs")
            if count in samples:
                original=cv2.imread(str(path.parent/"images"/manifest["images"][count]["filename"]))
                if original is None:raise ValueError("Source JPEG cannot be decoded")
                errors.append({"frame_ordinal":count,"mean_absolute_pixel_difference":float(np.abs(frame.astype(np.int16)-original.astype(np.int16)).mean())})
            count+=1
    finally:capture.release()
    if count!=manifest["expected_frames"] or abs(fps-25)>1e-6:
        raise ValueError("MP4 frame count/timebase differs from its derivation contract")
    return {"decoded_frames":count,"fps":fps,"image_shape":list(shape),"duration_seconds":count/fps,
            "sample_pixel_comparison":errors,"pixel_comparison_is_not_recognition_accuracy":True}


def encode_record(record):
    folder=folder_for(record);manifest=read(folder/"manifest.json");output=folder/"video.mp4"
    # Resume only a fully verified prior export with the same image inventory.
    if manifest.get("state")=="video_verified" and output.exists() and sha(output)==manifest["video_sha256"]:
        return manifest
    previous_videos={"SupC02/r00/ch0":"SupC02-r00-ch0.mp4", "SupC02/r01/ch0":"SupC02-r01-ch0.mp4",
        "CPR_Dataset_S0/C13_S0/r11/ch0":"public-C13-r11-ch0.mp4",
        "CPR_Dataset_S0/C13_S0/r12/ch0":"public-C13-r12-ch0.mp4"}
    previous=DATA/"video-verification"/previous_videos[record["record_id"]] if record["record_id"] in previous_videos else None
    temporary=folder/"video.partial.mp4"
    if previous is not None:
        shutil.copyfile(previous,temporary)
    else:
        command=["ffmpeg","-hide_banner","-loglevel","error","-xerror","-framerate","25",
            "-start_number",str(manifest["source_start_frame_number"]),"-i",str(folder/"images/%08d.jpg"),
            "-c:v","libx264","-threads","2","-preset","fast","-crf","18","-pix_fmt","yuv420p","-y",str(temporary)]
        completed=subprocess.run(command,capture_output=True,text=True,encoding="utf-8",errors="replace",timeout=900)
        if completed.returncode:raise RuntimeError("Public JPEG export failed: "+completed.stderr[-2000:])
    temporary.replace(output)
    validation=inspect_video(output,manifest)
    manifest.update({"state":"video_verified","video":"video.mp4","video_sha256":sha(output),
        "video_bytes":output.stat().st_size,"video_validation":validation,
        "derivation":{"codec":"libx264","crf":18,"pixel_format":"yuv420p","fps":25,
            "source_order":"numeric filename order, no frame drops or resampling",
            "fps_source":"CPR-Coach published dataset FPS=25; not a physical compression-rate estimate",
            "lossy_video_export":True,"original_jpeg_bytes_retained":True}})
    if previous is not None:
        manifest["derivation"]["reused_previous_video"]=str(previous.relative_to(CODE))
        if manifest["video_sha256"]!=sha(previous):raise ValueError("Reused public video checksum differs")
    write(folder/"manifest.json",manifest)
    print("Video fully decoded and verified",record["record_id"],validation["decoded_frames"],flush=True)
    return manifest


def main():
    parser=argparse.ArgumentParser();parser.add_argument("stage",choices=["extract","videos"])
    parser.add_argument("--workers",type=int,default=2);args=parser.parse_args()
    if (ROOT/"collection-manifest.json").exists():
        raise FileExistsError("Verified collection is frozen; do not overwrite its media")
    plan=read(ROOT/"frozen-record-plan.json")
    expected=read(REPORT/"record-plan-fingerprint.json")
    if sha(ROOT/"frozen-record-plan.json")!=expected["sha256"]:raise ValueError("Frozen inventory changed")
    if args.stage=="extract":extract_supplemental(plan);import_primary(plan)
    else:
        if not 1<=args.workers<=4:raise ValueError("Encoding worker budget must be 1..4")
        results=[]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(encode_record,r) for r in plan["records"]]
            for future in as_completed(futures):results.append(future.result())
        write(ROOT/"media-build-status.json",{"phase":"complete","records_verified":len(results),
            "total_decoded_frames":sum(r["video_validation"]["decoded_frames"] for r in results),
            "training_or_inference_performed":False})


if __name__=="__main__":main()
