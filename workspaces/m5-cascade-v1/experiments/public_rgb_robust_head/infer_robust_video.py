"""Isolated offline entrypoint; no web or Agent runtime changes."""
import argparse
import json
from pathlib import Path
import sys
from robust_head import CODE, load

sys.path.insert(0, str(CODE/'experiments/public_cpr_improved'))
import infer_video


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference-record')
    parser.add_argument('--manifest', type=Path, default=CODE/'data/public_cpr/rgb-evaluation-v5/inference-manifest.json')
    args = parser.parse_args()
    if infer_video.load_manifest(args.manifest).get('head_kind') != 'robust_compact':
        raise ValueError('Expected isolated robust-head manifest')
    original = infer_video.load
    try:
        infer_video.load = load
        result = infer_video.analyze(args.video, args.output, args.manifest, args.reference_record)
    finally:
        infer_video.load = original
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
