"""Bounded, hash-verified range download of the pinned public CPR-Coach asset."""
from __future__ import annotations

import concurrent.futures
import argparse
import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "assets"
REVISION = "b8667b8fd6a70ac63cc09fc6b763a9459b7c160a"
ASSETS = {
    "ann.tar.gz": (35485, "72d99337152cf5873ddc5cf6608d360119f85fa57378cf94699688f6540951e4"),
    "Keypoints.tar.gz": (2598001567, "daf93b63af7663ef07988e2dcdc04400029fc1a82ed1afe101ff93a6c0a62441"),
}
CHUNK = 32 * 1024 * 1024


def file_sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def download(name, size, expected):
    target = ROOT / name
    if target.exists() and target.stat().st_size == size and file_sha(target) == expected:
        print(f"Already verified: {name}", flush=True)
        return
    partial = ROOT / (name + ".part")
    chunks = ROOT / (name + ".ranges")
    chunks.mkdir(parents=True, exist_ok=True)
    if not partial.exists() or partial.stat().st_size != size:
        with partial.open("wb") as stream:
            stream.truncate(size)
        # Completion markers are bound to their bytes below, so stale markers are harmless.

    def transfer(index):
        start = index * CHUNK
        end = min(size, start + CHUNK) - 1
        marker = chunks / f"{index:04d}.json"
        if marker.exists():
            old = json.loads(marker.read_text())
            with partial.open("rb") as stream:
                stream.seek(start)
                h = hashlib.sha256(stream.read(end-start+1)).hexdigest()
            if old.get("sha256") == h and old.get("end") == end:
                return end-start+1
        progress_path = chunks / f"{index:04d}.progress.json"
        received = 0
        if progress_path.exists():
            progress = json.loads(progress_path.read_text())
            if progress.get("start") == start and progress.get("end") == end:
                received = int(progress["received"])
        for attempt in range(15):
            try:
                h = hashlib.sha256()
                if received:
                    with partial.open("rb") as prefix:
                        prefix.seek(start)
                        h.update(prefix.read(received))
                request_start = start + received
                if request_start > end:
                    marker.write_text(json.dumps({"start": start, "end": end, "sha256": h.hexdigest()}))
                    return received
                url = f"https://huggingface.co/datasets/ShunliWang/CPR-Coach/resolve/{REVISION}/{name}?download=true&range_start={request_start}&attempt={attempt}"
                request = urllib.request.Request(url, headers={
                    "Range": f"bytes={request_start}-{end}", "Accept-Encoding": "identity",
                    "User-Agent": "CPR-Student-Project-Verified-Downloader/1.0"})
                with urllib.request.urlopen(request, timeout=60) as response, partial.open("r+b") as output:
                    wanted = f"bytes {request_start}-{end}/{size}"
                    if response.status != 206 or response.headers.get("Content-Range") != wanted:
                        raise RuntimeError(f"Unexpected range response: {response.status}, {response.headers.get('Content-Range')}")
                    output.seek(request_start)
                    while received < end-start+1:
                        block = response.read(min(2**20, end-start+1-received))
                        if not block:
                            break
                        output.write(block)
                        h.update(block)
                        received += len(block)
                        progress_temp = progress_path.with_suffix(".tmp")
                        progress_temp.write_text(json.dumps({"start": start, "end": end, "received": received}))
                        progress_temp.replace(progress_path)
                    output.flush()
                if received != end-start+1:
                    raise RuntimeError(f"Incomplete range: {received}")
                marker.write_text(json.dumps({"start": start, "end": end, "sha256": h.hexdigest()}))
                return received
            except Exception as error:
                if attempt == 14:
                    raise
                print(f"Retry {name} range {index}: {type(error).__name__}: {error}", flush=True)
                time.sleep(min(8, 2**attempt))

    total_chunks = (size + CHUNK - 1) // CHUNK
    started = time.perf_counter()
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
        jobs = [executor.submit(transfer, index) for index in range(total_chunks)]
        for job in concurrent.futures.as_completed(jobs):
            done += job.result()
            elapsed = max(0.001, time.perf_counter()-started)
            print(f"{name}: {done/size:.1%}, {done/2**20:.1f}/{size/2**20:.1f} MiB, {done/2**20/elapsed:.2f} MiB/s", flush=True)
    actual = file_sha(partial)
    if actual != expected:
        raise RuntimeError(f"SHA256 mismatch for {name}: {actual}")
    os.replace(partial, target)
    print(f"SHA256 verified: {name} {actual}", flush=True)


if __name__ == "__main__":
    paths_file=Path(__file__).resolve().parent/"paths.json"
    if paths_file.exists():
        ROOT=Path(json.loads(paths_file.read_text(encoding="utf-8"))["dataset_assets"])
    parser=argparse.ArgumentParser()
    parser.add_argument("--assets",type=Path,default=ROOT)
    ROOT=parser.parse_args().assets.resolve()
    ROOT.mkdir(parents=True, exist_ok=True)
    manifest = {"source": "https://huggingface.co/datasets/ShunliWang/CPR-Coach", "revision": REVISION,
                "files": {name: {"bytes": size, "sha256": sha} for name, (size, sha) in ASSETS.items()}}
    (ROOT / "public-source-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for name, (size, digest) in ASSETS.items():
        download(name, size, digest)
