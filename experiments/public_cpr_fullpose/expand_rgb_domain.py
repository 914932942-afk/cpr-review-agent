"""Acquire two additional immutable public source parts with a disk budget."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json
import hashlib
import shutil
import time
import requests
from download_rgb_domain import REVISION, CHUNK, CODE
from prepare_features import write_new

ROOT = CODE/'data/public_cpr/rgb-domain-training-v2'


def main():
    plan_path = ROOT/'source-plan.json'
    if plan_path.exists():
        plan = json.loads(plan_path.read_text(encoding='utf-8'))
    else:
        reply = requests.get('https://huggingface.co/api/datasets/ShunliWang/CPR-Coach/tree/'+REVISION, timeout=40)
        reply.raise_for_status()
        parts = []
        for number in [1, 2]:
            name = f'SupDC59_Dataset.tar.gz.{number:02d}'
            row = next(v for v in reply.json() if v['path'] == name)
            parts.append({'name': name, 'bytes': row['size'], 'sha256': row['lfs']['oid'], 'number': number})
        total = sum(v['bytes'] for v in parts)
        if total > 20*2**30 or shutil.disk_usage(CODE).free-total < 22*2**30:
            raise ValueError('Public RGB acquisition exceeds 20GiB or free-space reserve')
        plan = {'revision': REVISION, 'parts': parts, 'purpose': 'Expand public train/validation RGB coverage; exclude all386 test and26 reserved recordings',
            'maximum_download_bytes': 20*2**30, 'minimum_free_disk_reserve_bytes': 22*2**30,
            'existing_part00': 'data/public_cpr/rgb-domain-training-v1/transport/part00', 'labels': 'Original public labels only'}
        write_new(plan_path, plan)
    completed = []
    for part in plan['parts']:
        folder = ROOT/'transport'/f'part{part["number"]:02d}'; folder.mkdir(parents=True, exist_ok=True)
        size = part['bytes']
        def fetch(i):
            start = i*CHUNK; end = min(start+CHUNK, size)-1
            path = folder/f'{i:05d}.bin'
            if path.exists() and path.stat().st_size == end-start+1:
                return end-start+1
            for attempt in range(12):
                try:
                    if shutil.disk_usage(CODE).free < plan['minimum_free_disk_reserve_bytes']:
                        raise ValueError('Disk reserve reached')
                    url = f'https://huggingface.co/datasets/ShunliWang/CPR-Coach/resolve/{REVISION}/{part["name"]}?start={start}&attempt={attempt}&fresh={int(time.time())}'
                    r = requests.get(url, headers={'Range': f'bytes={start}-{end}', 'Accept-Encoding': 'identity'}, timeout=(20, 90))
                    if r.status_code != 206 or r.headers.get('Content-Range') != f'bytes {start}-{end}/{size}' or len(r.content) != end-start+1:
                        raise ValueError('Immutable range response mismatch')
                    tmp = path.with_suffix('.part'); tmp.write_bytes(r.content); tmp.replace(path)
                    return end-start+1
                except Exception:
                    if attempt == 11: raise
                    time.sleep(min(2**attempt, 15))
        done = 0; started = last = time.perf_counter()
        with ThreadPoolExecutor(max_workers=16) as pool:
            futures = [pool.submit(fetch, i) for i in range((size+CHUNK-1)//CHUNK)]
            for future in as_completed(futures):
                done += future.result()
                if time.perf_counter()-last >= 10:
                    print(part['name'], f'{done/2**30:.2f}/{size/2**30:.2f} GiB', flush=True); last = time.perf_counter()
        h = hashlib.sha256()
        for path in sorted(folder.glob('*.bin')):
            with path.open('rb') as f:
                for block in iter(lambda: f.read(2**20), b''): h.update(block)
        if h.hexdigest() != part['sha256']:
            raise ValueError('Full public part LFS hash mismatch')
        completed.append({**part, 'verified': True, 'seconds': time.perf_counter()-started})
        write_new(ROOT/f'part{part["number"]:02d}-integrity.json', completed[-1])
        print('Full source part verified', part['name'], flush=True)
    write_new(ROOT/'source-integrity.json', {'verified': True, 'parts': completed,
        'total_prefix_bytes_including_part00': 30*2**30, 'not_a_full_archive_claim': True})


if __name__ == '__main__': main()
