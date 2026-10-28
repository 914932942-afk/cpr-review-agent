"""Resume immutable HTTP ranges, reuse verified prefixes, verify whole SHA-256.

Large HTTP requests amortize redirect latency; storage remains 8-MiB chunks.
No second full .tar.gz copy and no dataset extraction is performed here.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import math
import os
import shutil
import threading
import time
import urllib.request
from common import ROOT, REPORT, REVISION, ARCHIVE, SIZE, SHA, CHUNK, chunk_size, chunk_path, read, write, sha

LOCK = threading.Lock()
VALID = set()
STARTED = time.monotonic()
LAST_REPORT = 0.
INITIAL = 0


def validate_cached(index):
    path = chunk_path(index); marker = path.with_suffix(".json")
    if not path.exists() or not marker.exists() or path.stat().st_size != chunk_size(index): return False
    meta = read(marker)
    return meta.get("start") == index*CHUNK and meta.get("end") == index*CHUNK+chunk_size(index)-1 and meta.get("sha256") == sha(path)


def progress(force=False, phase="downloading"):
    global LAST_REPORT
    with LOCK:
        now = time.monotonic()
        if not force and now-LAST_REPORT < 10: return
        done = sum(chunk_size(i) for i in VALID)
        result = {"phase":phase,"verified_range_bytes":done,"archive_bytes":SIZE,
            "percent":100*done/SIZE,"new_verified_bytes":done-INITIAL,"seconds":now-STARTED,
            "chunks":len(VALID),"total_chunks":math.ceil(SIZE/CHUNK),
            "updated_utc":datetime.now(timezone.utc).isoformat(),"whole_archive_sha256_verified":phase=="complete"}
        write(ROOT/"download-status.json",result); LAST_REPORT=now
        print(f"{phase}: {done/2**30:.2f}/{SIZE/2**30:.2f} GiB ({result['percent']:.1f}%), chunks={len(VALID)}",flush=True)


def fetch_group(group):
    first=group*8; last=min(first+7,math.ceil(SIZE/CHUNK)-1)
    for attempt in range(12):
        missing=[i for i in range(first,last+1) if i not in VALID]
        if not missing: return
        index=missing[0]; partial=chunk_path(index).with_suffix(".part")
        retained=partial.stat().st_size if partial.exists() else 0
        if retained>chunk_size(index): raise ValueError("Oversized partial range")
        start=index*CHUNK+retained; end=last*CHUNK+chunk_size(last)-1
        try:
            if retained == chunk_size(index):
                partial.replace(chunk_path(index)); path=chunk_path(index)
                write(path.with_suffix(".json"),{"start":index*CHUNK,"end":index*CHUNK+chunk_size(index)-1,"sha256":sha(path)})
                with LOCK: VALID.add(index)
                continue
            url=f"https://huggingface.co/datasets/ShunliWang/CPR-Coach/resolve/{REVISION}/{ARCHIVE}?acceptance_start={start}&end={end}&attempt={attempt}"
            request=urllib.request.Request(url,headers={"Range":f"bytes={start}-{end}","Accept-Encoding":"identity"})
            with urllib.request.urlopen(request,timeout=45) as response:
                if response.status!=206 or response.headers.get("Content-Range")!=f"bytes {start}-{end}/{SIZE}":
                    raise ValueError("Source returned incorrect HTTP range")
                for current in range(index,last+1):
                    expected=chunk_size(current); path=chunk_path(current); part=path.with_suffix(".part")
                    existing=retained if current==index else 0
                    with part.open("ab" if existing else "wb") as output:
                        while existing<expected:
                            block=response.read(min(2**20,expected-existing))
                            if not block: raise EOFError("Truncated public archive range")
                            output.write(block); existing+=len(block)
                    part.replace(path)
                    write(path.with_suffix(".json"),{"start":current*CHUNK,"end":current*CHUNK+expected-1,"sha256":sha(path)})
                    with LOCK: VALID.add(current)
                    progress()
            return
        except Exception as error:
            if attempt==11: raise
            print(f"Group {group} retry {attempt+1}: {type(error).__name__}",flush=True)
            time.sleep(min(10,1+attempt))


def main():
    global INITIAL
    parser=argparse.ArgumentParser();parser.add_argument("--workers",type=int,default=12)
    parser.add_argument("--reuse-cache",type=Path);args=parser.parse_args()
    if not 1<=args.workers<=16: raise ValueError("Worker budget must be 1..16")
    plan=read(ROOT/"frozen-record-plan.json")
    if plan["archive"]["sha256"]!=SHA or plan["archive"]["bytes"]!=SIZE: raise ValueError("Frozen source mismatch")
    (ROOT/"archive-ranges").mkdir(exist_ok=True)
    if args.reuse_cache:
        for old in args.reuse_cache.glob("*.bin"):
            if not old.stem.isdigit(): continue
            index=int(old.stem); marker=old.with_suffix(".json")
            if index>=math.ceil(SIZE/CHUNK) or not marker.exists():continue
            meta=read(marker)
            if old.stat().st_size!=chunk_size(index) or meta.get("sha256")!=sha(old): continue
            path=chunk_path(index)
            if path.exists():continue
            # Same-volume hard link reuses trusted immutable bytes; no moving or deletion.
            try: os.link(old,path)
            except OSError: shutil.copyfile(old,path)
            shutil.copyfile(marker,path.with_suffix(".json"))
    for index in range(math.ceil(SIZE/CHUNK)):
        if validate_cached(index):VALID.add(index)
    INITIAL=sum(chunk_size(i) for i in VALID)
    if shutil.disk_usage(ROOT).free < SIZE-INITIAL+3*2**30:
        raise ValueError("Insufficient disk budget for missing archive + selected RGB outputs")
    progress(force=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(fetch_group,g) for g in range(math.ceil(SIZE/(8*CHUNK)))]
        for future in as_completed(futures): future.result()
    progress(force=True,phase="verifying-whole-archive")
    result=hashlib.sha256();total=0
    for index in range(math.ceil(SIZE/CHUNK)):
        with chunk_path(index).open("rb") as stream:
            for block in iter(lambda:stream.read(2**20),b""):
                result.update(block);total+=len(block)
    if total!=SIZE or result.hexdigest()!=SHA: raise ValueError("Whole public archive SHA-256 mismatch")
    write(ROOT/"source-archive-integrity.json",{"verified":True,"bytes":total,"sha256":result.hexdigest(),
        "expected_sha256":SHA,"dataset_revision":REVISION,"archive":ARCHIVE,
        "frozen_record_plan_sha256":sha(ROOT/"frozen-record-plan.json"),"verified_utc":datetime.now(timezone.utc).isoformat()})
    progress(force=True,phase="complete")


if __name__=="__main__":main()
