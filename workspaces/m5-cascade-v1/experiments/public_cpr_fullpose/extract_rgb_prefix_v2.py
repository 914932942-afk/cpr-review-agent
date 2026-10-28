"""Freeze a bounded already-downloaded public range prefix and sample complete clips.

This is explicitly a partial-transport proof, not an LFS whole-file verification.
The acquisition code validates HTTPS Content-Range for every original source block.
"""
from pathlib import Path
import json
from datetime import datetime, timezone
from head import CODE
from baseline import digest
from prepare_features import write_new
import extract_rgb_domain_v2 as extractor

ROOT = CODE/'data/public_cpr/rgb-domain-prefix-v2'
CHUNK = 16*2**20


class PrefixReader:
    def __init__(self, size):
        self.size=size; self.position=0
        self.rows=json.loads((ROOT/'source-integrity.json').read_text(encoding='utf-8'))['blocks']
        self.handle=None; self.block_index=None
    def read(self, size=-1):
        if size < 0: raise ValueError('Unbounded prefix read')
        remaining=min(size,self.size-self.position); output=[]
        while remaining:
            index,offset=divmod(self.position,CHUNK)
            path=CODE/self.rows[index]['path']; count=min(remaining,CHUNK-offset)
            if self.block_index!=index:
                if self.handle is not None: self.handle.close()
                self.handle=path.open('rb',buffering=2**20); self.block_index=index
            self.handle.seek(offset); block=self.handle.read(count)
            if len(block)!=count: raise EOFError('Frozen prefix changed')
            output.append(block); remaining-=count; self.position+=count
        return b''.join(output)


def main():
    if ROOT.exists(): raise FileExistsError('Preserve prefix snapshot')
    plan=CODE/'data/public_cpr/rgb-domain-training-v2/source-plan.json'
    previous=CODE/'data/public_cpr/rgb-domain-training-v1/source-integrity.json'
    if not json.loads(previous.read_text(encoding='utf-8'))['verified']: raise ValueError('Part00 not verified')
    paths=sorted((CODE/'data/public_cpr/rgb-domain-training-v1/transport/part00').glob('*.bin'))
    for index in range(640):
        path=CODE/'data/public_cpr/rgb-domain-training-v2/transport/part01'/f'{index:05d}.bin'
        if not path.exists() or path.stat().st_size!=CHUNK: break
        paths.append(path)
    if len(paths)<=640: raise ValueError('No additional complete public source blocks')
    write_new(ROOT/'source-plan.json',{'frozen_utc':datetime.now(timezone.utc).isoformat(),
        'upstream_source_plan_sha256':digest(plan),'revision':'b8667b8fd6a70ac63cc09fc6b763a9459b7c160a',
        'verification_scope':'Part00 full LFS SHA verified. Part01 prefix uses strict HTTPS Content-Range plus local block fingerprints; full LFS01 verification pending.',
        'implementation_sources':{Path(__file__).resolve().relative_to(CODE).as_posix():digest(Path(__file__).resolve()),
            'experiments/public_cpr_fullpose/extract_rgb_domain_v2.py':digest(Path(extractor.__file__))}})
    rows=[{'path':p.relative_to(CODE).as_posix(),'bytes':p.stat().st_size,'sha256':digest(p)} for p in paths]
    write_new(ROOT/'source-integrity.json',{'verified':True,'verification_scope':'Acquired immutable HTTPS range prefix and local block fingerprints only',
        'full_part01_lfs_verified':False,'full_gzip_archive_verified':False,'blocks':rows,'total_prefix_bytes_including_part00':len(rows)*CHUNK})
    extractor.ROOT=ROOT; extractor.StreamingParts=PrefixReader
    extractor.main()


if __name__=='__main__': main()
