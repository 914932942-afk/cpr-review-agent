"""Recover 16 predefined ch0 RGB frames from non-test public recording groups."""
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import tarfile
import time

from PIL import Image
from head import CODE
from descriptor import sample_indices
from prepare_features import write_new
from baseline import canonical_path, digest
from download_rgb_domain import ROOT, CHUNK


class StreamingParts(io.RawIOBase):
    def __init__(self, size):
        self.size, self.position = size, 0
    def read(self, size=-1):
        if size < 0:
            raise ValueError('Unbounded transport read')
        output = []
        remaining = min(size, self.size-self.position)
        while remaining:
            index, offset = divmod(self.position, CHUNK)
            path = ROOT/'transport/part00'/f'{index:05d}.bin'
            deadline = time.monotonic()+240
            while not path.exists():
                if time.monotonic() > deadline:
                    raise TimeoutError('Waiting for immutable transport chunk')
                time.sleep(.5)
            count = min(remaining, path.stat().st_size-offset)
            with path.open('rb') as f:
                f.seek(offset); data = f.read(count)
            if len(data) != count or count <= 0:
                raise EOFError('Transport range is incomplete')
            output.append(data); self.position += count; remaining -= count
        return b''.join(output)


def main():
    output = ROOT/'rgb-subset-v2'
    if output.exists():
        raise FileExistsError('Preserve RGB subset extraction')
    records = json.loads((CODE/'data/public_cpr/rgb-evaluation-v4/record-splits.json').read_text(encoding='utf-8'))
    wanted = {r['record_id']: r for r in records if r['record_id'].startswith('SupDC') and r['application_split'] in ['train', 'validation']}
    indices = {key: set((sample_indices(r['expected_frames'])+1).tolist()) for key, r in wanted.items()}
    source = json.loads((ROOT/'source-plan.json').read_text(encoding='utf-8'))
    write_new(output/'plan.json', {'frozen_utc': datetime.now(timezone.utc).isoformat(),
        'source_plan_sha256': digest(ROOT/'source-plan.json'), 'sampling': '16 uniformly spaced original frame ordinals, ch0 only',
        'selected_groups': {k: {'split': r['application_split'], 'original_labels': r['original_labels'], 'frame_ordinals': sorted(indices[k])} for k, r in wanted.items()},
        'no_manual_labels': True, 'no_test_groups': True})
    seen, count, directories, error = {}, 0, set(), None
    stream = StreamingParts(source['bytes'])
    try:
        with tarfile.open(fileobj=stream, mode='r|gz') as archive:
            for member in archive:
                parts = member.name.split('/')
                if member.isdir() and len(parts) == 3:
                    print('Public source directory', member.name, flush=True)
                    directories.add(member.name)
                if not member.isfile() or not member.name.endswith('.jpg'):
                    continue
                key = canonical_path('/'.join(parts[:-1]))
                if key not in wanted:
                    continue
                number = int(parts[-1].removesuffix('.jpg'))
                if number not in indices[key]:
                    continue
                data = archive.extractfile(member).read()
                with Image.open(io.BytesIO(data)) as image:
                    image.verify()
                folder = output/'records'/key.replace('/', '--'); folder.mkdir(parents=True, exist_ok=True)
                path = folder/f'{number:08d}.jpg'
                if path.exists():
                    raise ValueError('Repeated public frame')
                path.write_bytes(data)
                seen.setdefault(key, []).append({'frame_ordinal': number-1, 'filename': path.relative_to(output).as_posix(),
                    'sha256': hashlib.sha256(data).hexdigest(), 'archive_member': member.name, 'bytes': len(data)})
                count += 1
                if count % 100 == 0:
                    print('Selected non-test RGB frames', count, 'groups', len(seen), flush=True)
    except (tarfile.ReadError, EOFError) as exception:
        # The downloaded object is one complete, hashed transport part, not the
        # full multipart gzip archive. Only complete predefined samples qualify.
        error = type(exception).__name__+': '+str(exception)
    completed = []
    for key, images in seen.items():
        if {r['frame_ordinal']+1 for r in images} != indices[key]:
            continue
        completed.append({'record_id': key, 'split': wanted[key]['application_split'],
            'original_labels': wanted[key]['original_labels'], 'expected_video_frames': wanted[key]['expected_frames'],
            'images': sorted(images, key=lambda r: r['frame_ordinal'])})
    write_new(output/'manifest.json', {'records': completed, 'complete_record_count': len(completed),
        'partial_groups_excluded': len(seen)-len(completed), 'extracted_frames': count,
        'partial_archive_termination': error, 'source_bytes_scanned': stream.position,
        'requires_source_integrity_json_before_training': True, 'labels': 'unchanged public CPR-Coach annotations',
        'no_test_groups_used': True})
    print('Predefined RGB sample groups complete', len(completed), 'partial groups not used', len(seen)-len(completed), flush=True)


if __name__ == '__main__':
    main()
