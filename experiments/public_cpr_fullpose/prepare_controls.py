"""Public original-class12 partial-label negative controls, never Correct labels."""
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import tarfile

import numpy as np
from PIL import Image

from head import CODE
from baseline import canonical_path, RestrictedUnpickler, digest
from descriptor import sample_indices
from prepare_features import write_new

DATA = CODE/'data/public_cpr'
ROOT = DATA/'rgb-partial-controls-v1'


class PrefixReader:
    def __init__(self):
        self.paths = iter(sorted(ROOT.glob('*.bin'))); self.stream = None
    def read(self, size=-1):
        if size < 0:
            raise ValueError('Unbounded source read')
        output = []
        while size:
            if self.stream is None:
                path = next(self.paths, None)
                if path is None:
                    break
                self.stream = path.open('rb')
            block = self.stream.read(size)
            if not block:
                self.stream.close(); self.stream = None; continue
            output.append(block); size -= len(block)
        return b''.join(output)
    def close(self):
        if self.stream is not None:
            self.stream.close()


def main():
    if (ROOT/'manifest.json').exists():
        raise FileExistsError('Preserve partial-label curation')
    proof = json.loads((ROOT/'transport-proof.json').read_text(encoding='utf-8'))
    metadata = {}
    with tarfile.open(DATA/'assets/ann.tar.gz', 'r:gz') as archive:
        for member in archive:
            if Path(member.name).name != 'trainlist.txt':
                continue
            for line in archive.extractfile(member).read().decode('utf-8-sig').splitlines():
                values = line.split()
                if not values:
                    continue
                key = canonical_path(values[0])
                if key.startswith('CPR_Dataset_S1/C12_S1/') and key.endswith('/ch0'):
                    labels = list(map(int, values[2:]))
                    if labels != [12]:
                        raise ValueError('Unexpected public control annotation')
                    metadata[key] = {'record_id': key, 'expected_video_frames': int(values[1]),
                        'split': 'train', 'original_labels': labels, 'can_train_correct_auxiliary': False,
                        'supervised_targets': 'known negatives of supported error classes only; not relabelled Correct'}
    selected = dict(sorted(metadata.items())[:3])
    write_new(ROOT/'curation-plan.json', {'frozen_utc': datetime.now(timezone.utc).isoformat(), 'records': selected,
        'source_proof_sha256': digest(ROOT/'transport-proof.json'), 'annotation_sha256': digest(DATA/'assets/ann.tar.gz')})
    samples = {key: set((sample_indices(r['expected_video_frames'])+1).tolist()) for key, r in selected.items()}
    images = {key: [] for key in selected}
    stream = PrefixReader()
    try:
        with tarfile.open(fileobj=stream, mode='r|gz') as archive:
            for member in archive:
                if not member.isfile() or not member.name.endswith('.jpg'):
                    continue
                key = canonical_path(str(Path(member.name).parent).replace('\\', '/'))
                if key not in selected:
                    continue
                number = int(Path(member.name).stem)
                if number not in samples[key]:
                    continue
                payload = archive.extractfile(member).read()
                with Image.open(io.BytesIO(payload)) as image:
                    image.verify()
                path = ROOT/'records'/key.replace('/', '--')/f'{number:08d}.jpg'
                path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(payload)
                images[key].append({'frame_ordinal': number-1, 'filename': path.relative_to(ROOT).as_posix(),
                    'sha256': hashlib.sha256(payload).hexdigest(), 'archive_member': member.name})
    except (EOFError, tarfile.ReadError):
        pass
    finally:
        stream.close()
    complete = {key for key, values in images.items() if len(values) == len(samples[key])}
    if not complete:
        raise ValueError('Public prefix contains no complete predefined control samples')
    (ROOT/'poses').mkdir(exist_ok=True)
    found = set()
    with tarfile.open(DATA/'assets/Keypoints.tar.gz', mode='r|gz') as archive:
        for member in archive:
            if Path(member.name).name != 'train_keypoints.pkl':
                continue
            payload = RestrictedUnpickler(archive.extractfile(member)).load()
            records = payload.get('annotations') if isinstance(payload, dict) else payload
            for row in records:
                key = canonical_path(row['frame_dir'])
                if key not in complete:
                    continue
                np.savez_compressed(ROOT/'poses'/(hashlib.sha256(key.encode()).hexdigest()+'.npz'),
                    keypoint=np.asarray(row['keypoint'], dtype=np.float32),
                    keypoint_score=np.clip(np.asarray(row['keypoint_score'], dtype=np.float32), 0, 1))
                found.add(key)
    if found != complete:
        raise ValueError('Public negative controls lack matching public body references')
    write_new(ROOT/'manifest.json', {'records': [{**selected[key], 'images': sorted(images[key], key=lambda r: r['frame_ordinal']),
        'pose_file': 'poses/'+hashlib.sha256(key.encode()).hexdigest()+'.npz'} for key in sorted(complete)],
        'partial_label_training_only': True, 'original_class12_preserved': True, 'new_correct_labels_created': False,
        'full_source_archive_verification_claim': False, 'pinned_source_range_transport_verified': proof['range_content_headers_verified']})
    print('Public partial-label controls', len(complete), 'original label12, no Correct relabelling', flush=True)


if __name__ == '__main__':
    main()
