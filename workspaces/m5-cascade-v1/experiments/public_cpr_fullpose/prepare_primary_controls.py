"""Bounded public Parquet range read for existing non-test class13 recordings."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import importlib.util
import hashlib
import json
import io
import pyarrow.parquet as pq
from PIL import Image
from head import CODE
from descriptor import sample_indices
from baseline import digest
from prepare_features import write_new

ARCHIVE=CODE.parent/'workspace-archive/work/cpr-step1-improve'
ROOT=CODE/'data/public_cpr/rgb-primary-controls-v1'
REVISION='883b8e79c731936deb776dbb072ef69678bc5e4f'


def main():
    if (ROOT/'manifest.json').exists(): raise FileExistsError('Preserve primary controls')
    metadata=json.loads((ARCHIVE/'parquet-index-metadata.json').read_text(encoding='utf-8'))
    rows=json.loads((CODE/'data/public_cpr/rgb-evaluation-v4/record-splits.json').read_text(encoding='utf-8'))
    wanted={r['record_id']:r for r in rows if r['record_id'].startswith('CPR_Dataset_S0/C13_S0/') and r['application_split'] in ['train','validation']}
    plan=[]
    for entry in metadata:
        groups=[]
        for group in entry['groups']:
            if any(group['min'].startswith('./'+key+'/') or group['max'].startswith('./'+key+'/') for key in wanted): groups.append(group['index'])
        if groups: plan.append({**entry,'selected_groups':groups})
    if not (ROOT/'plan.json').exists():
        write_new(ROOT/'plan.json',{'frozen_utc':datetime.now(timezone.utc).isoformat(),'conversion_revision':REVISION,
            'source_index_sha256':digest(ARCHIVE/'parquet-index-metadata.json'),'groups':plan,
            'records':{k:{'split':r['application_split'],'original_labels':r['original_labels'],'frame_ordinals':sample_indices(r['expected_frames']).tolist()} for k,r in wanted.items()},
            'labels':'Original13 only. Known negatives for six other errors, never Correct labels.', 'all_test386_excluded':True})
    spec=importlib.util.spec_from_file_location('pinned_parquet_reader',ARCHIVE/'extract_parquet_clip.py')
    transport=importlib.util.module_from_spec(spec); spec.loader.exec_module(transport)
    transport.MAX_BYTES=192*2**20
    indices={k:set((sample_indices(r['expected_frames'])+1).tolist()) for k,r in wanted.items()}
    def fetch(entry):
        url=f'https://huggingface.co/datasets/ShunliWang/CPR-Coach/resolve/{REVISION}/default/partial-train/'+entry['filename']
        source=transport.RangeFile(url,entry['size'],ROOT/('ranges-'+Path(entry['filename']).stem))
        file=pq.ParquetFile(source); selected=[]
        for group in entry['selected_groups']:
            part=file.read_row_group(group,columns=['__key__','jpg']).to_pylist()
            for row in part:
                key,number=str(row['__key__']).removeprefix('./').rsplit('/',1)
                if key not in wanted or int(number) not in indices[key]: continue
                data=row['jpg']['bytes']
                with Image.open(io.BytesIO(data)) as im: im.verify()
                target=ROOT/'records'/key.replace('/','--')/f'{int(number):08d}.jpg'; target.parent.mkdir(parents=True,exist_ok=True)
                if target.exists() and target.read_bytes()!=data: raise ValueError('Duplicate source image mismatch')
                target.write_bytes(data)
                selected.append((key,{'frame_ordinal':int(number)-1,'filename':target.relative_to(ROOT).as_posix(),
                    'sha256':hashlib.sha256(data).hexdigest(),'public_key':row['__key__'],'parquet':entry['filename'],'row_group':group}))
            print('Primary public RGB',entry['filename'],group,'selected',len(selected),flush=True)
        return selected,source.audit,source.received
    images={k:[] for k in wanted}; audit=[]; total=0
    with ThreadPoolExecutor(max_workers=8) as pool:
        for future in as_completed([pool.submit(fetch,entry) for entry in plan]):
            selected,ranges,received=future.result(); audit.extend(ranges); total+=received
            for key,row in selected: images[key].append(row)
    completed=[]; incomplete=[]
    for key,parts in images.items():
        by_number={r['frame_ordinal']+1:r for r in parts}
        if set(by_number)!=indices[key]: incomplete.append(key); continue
        r=wanted[key]
        completed.append({'record_id':key,'split':r['application_split'],'original_labels':r['original_labels'],
            'expected_video_frames':r['expected_frames'],'images':sorted(by_number.values(),key=lambda v:v['frame_ordinal']),
            'source':'Pinned public Parquet ranges','can_train_correct_auxiliary':False})
    write_new(ROOT/'manifest.json',{'records':completed,'incomplete_records_not_used':incomplete,'downloaded_bytes':total,
        'transport_ranges':audit,'full_shard_lfs_verified':False,'conversion_revision':REVISION,'labels_unchanged':True,'new_manual_labels':False})
    print('Public primary controls complete',len(completed),'incomplete',incomplete,flush=True)


if __name__=='__main__': main()
