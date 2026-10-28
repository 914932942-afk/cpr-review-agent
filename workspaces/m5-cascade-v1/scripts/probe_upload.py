"""CPU-only upload check using the same input policy as the frozen vision worker."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from vision_worker import InputRejected, probe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    args = parser.parse_args()
    policy = Path(__file__).with_name("vision_worker.py")
    checksum = hashlib.sha256(policy.read_bytes()).hexdigest()
    try:
        result = {"status": "passed", "metadata": probe(args.video), "policy_sha256": checksum}
    except InputRejected as exc:
        result = {"status": "rejected", "code": str(exc), "policy_sha256": checksum}
    print(json.dumps(result))


if __name__ == "__main__":
    main()
