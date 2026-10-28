"""Pin Meta's public frozen DINOv2-small safetensors locally on E:."""
from datetime import datetime, timezone
import hashlib
import json
import time
import requests
from head import CODE
from prepare_features import write_new

ROOT = CODE/'data/public_cpr/weights/dinov2-small'


def main():
    if (ROOT/'source.json').exists():
        print('Pinned DINO weights already present', flush=True)
        return
    info = requests.get('https://huggingface.co/api/models/facebook/dinov2-small', timeout=30).json()
    revision = info['sha']
    files = requests.get(f'https://huggingface.co/api/models/facebook/dinov2-small/tree/{revision}', timeout=30).json()
    selected = [f for f in files if f['path'] in ['config.json', 'preprocessor_config.json', 'model.safetensors']]
    if len(selected) != 3:
        raise ValueError('Required safe model/config files unavailable')
    ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in selected:
        path = ROOT/item['path']
        for attempt in range(8):
            try:
                response = requests.get(f'https://huggingface.co/facebook/dinov2-small/resolve/{revision}/{item["path"]}?fresh={time.time()}', timeout=(20, 90))
                response.raise_for_status(); content = response.content
                expected = item.get('lfs', {}).get('oid')
                actual = hashlib.sha256(content).hexdigest()
                if len(content) != item['size'] or (expected and expected != actual):
                    raise ValueError('Official model size/hash mismatch')
                path.write_bytes(content)
                rows.append({'file': path.name, 'bytes': len(content), 'sha256': actual})
                print('Pinned', path.name, len(content), flush=True)
                break
            except Exception:
                if attempt == 7:
                    raise
                time.sleep(min(attempt+1, 8))
    write_new(ROOT/'source.json', {'source': 'https://huggingface.co/facebook/dinov2-small', 'revision': revision,
        'downloaded_utc': datetime.now(timezone.utc).isoformat(), 'files': rows,
        'purpose': 'Frozen visual feature extraction; no CPR backbone training', 'trust_remote_code': False})


if __name__ == '__main__':
    main()
