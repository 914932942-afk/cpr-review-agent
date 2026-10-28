"""Offline loader adapter; old video/perception code stays byte-identical."""
import argparse
import json
from pathlib import Path
import sys
from compact_head import CODE, load

sys.path.insert(0,str(CODE/'experiments/public_cpr_improved'))
import infer_video


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--video',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference-record');parser.add_argument('--manifest',type=Path,default=CODE/'data/public_cpr/rgb-evaluation-v4/inference-manifest.json');args=parser.parse_args()
    manifest=infer_video.load_manifest(args.manifest)
    if manifest.get('head_kind')!='compact':raise ValueError('Expected compact offline manifest')
    original=infer_video.load
    try:
        infer_video.load=load
        result=infer_video.analyze(args.video,args.output,args.manifest,args.reference_record)
    finally:infer_video.load=original
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
