"""Bounded, public, immutable RGB source acquisition for input-aligned heads."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import requests

from head import CODE
from prepare_features import write_new

ROOT = CODE/'data/public_cpr/rgb-domain-training-v1'
REVISION = 'b8667b8fd6a70ac63cc09fc6b763a9459b7c160a'
NAME = 'SupDC59_Dataset.tar.gz.00'
CHUNK = 16*2**20


def main():
    plan_path = ROOT/'source-plan.json'
    if plan_path.exists():
        plan = json.loads(plan_path.read_text(encoding='utf-8'))
    else:
        response = requests.get('https://huggingface.co/api/datasets/ShunliWang/CPR-Coach/tree/'+REVISION, timeout=30)
        response.raise_for_status()
        item = next(r for r in response.json() if r['path'] == NAME)
        plan = {'frozen_utc': datetime.now(timezone.utc).isoformat(), 'dataset_revision': REVISION,
            'name': NAME, 'bytes': item['size'], 'sha256': item['lfs']['oid'], 'chunk_bytes': CHUNK,
            'maximum_download_bytes': 10*2**30,
            'purpose': 'Public non-held-out composite RGB frames for frozen visual features; original labels only; never use the 26 held-out videos for fitting'}
        if item['size'] > plan['maximum_download_bytes']:
            raise ValueError('Source exceeds declared budget')
        write_new(plan_path, plan)
    folder = ROOT/'transport/part00'; folder.mkdir(parents=True, exist_ok=True)
    size = plan['bytes']
    def fetch(i):
        start = i*CHUNK; count = min(CHUNK, size-start); end = start+count-1
        path = folder/f'{i:05d}.bin'
        if path.exists() and path.stat().st_size == count:
            return count
        for attempt in range(12):
            try:
                url = f'https://huggingface.co/datasets/ShunliWang/CPR-Coach/resolve/{REVISION}/{NAME}?start={start}&attempt={attempt}&fresh={int(time.time())}'
                response = requests.get(url, headers={'Range': f'bytes={start}-{end}', 'Accept-Encoding': 'identity'}, timeout=(20, 90))
                if response.status_code != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{size}':
                    raise ValueError('Invalid immutable source range')
                data = response.content
                if len(data) != count:
                    raise ValueError('Short range')
                tmp = path.with_suffix('.part'); tmp.write_bytes(data); tmp.replace(path)
                return count
            except Exception:
                if attempt == 11:
                    raise
                time.sleep(min(2**attempt, 15))
    done = 0; started = time.perf_counter(); last = started
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending = [pool.submit(fetch, i) for i in range((size+CHUNK-1)//CHUNK)]
        for future in as_completed(pending):
            done += future.result()
            if time.perf_counter()-last >= 10:
                print(f'Public RGB part00 {done/2**30:.2f}/{size/2**30:.2f} GiB elapsed={time.perf_counter()-started:.1f}s', flush=True)
                last = time.perf_counter()
    digest = hashlib.sha256()
    for path in sorted(folder.glob('*.bin')):
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(2**20), b''):
                digest.update(block)
    actual = digest.hexdigest()
    if actual != plan['sha256']:
        raise ValueError('Public LFS part SHA256 mismatch')
    write_new(ROOT/'source-integrity.json', {'verified': True, 'bytes': size, 'sha256': actual,
        'source_kind': 'First complete official 10GiB multipart segment; not a claim that the full 40GiB archive was downloaded',
        'elapsed_seconds': time.perf_counter()-started})
    print('Public RGB part00 fully SHA256 verified; original labels, no collection', flush=True)


if __name__ == '__main__':
    main()
