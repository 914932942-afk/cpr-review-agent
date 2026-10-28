"""Public RGB curation only: no classifier import, fit or prediction."""
from pathlib import Path
import hashlib
import json
import os

CODE = Path(__file__).resolve().parents[2]
DATA = CODE / "data/public_cpr"
ROOT = DATA / "rgb-acceptance-v1"
REPORT = CODE / "test-results/public-rgb-acceptance-v1"
REVISION = "b8667b8fd6a70ac63cc09fc6b763a9459b7c160a"
ARCHIVE = "SupC14_Dataset.tar.gz"
SIZE = 9683930812
SHA = "6ee601c3634089a2794a4d61bb8f786cf6744b9cc391d3ce57e992e25a54ef4e"
ANN_SHA = "72d99337152cf5873ddc5cf6608d360119f85fa57378cf94699688f6540951e4"
CHUNK = 8 * 2**20
LABELS = {0:"Correct",1:"Overlap Hands",2:"Clenching Hands",3:"Single Hand",4:"Bending Arms",
          5:"Tilting Arms",6:"Jump Pressing",7:"Squatting",8:"Standing",9:"Wrong Position",
          11:"Slow Frequency",13:"Random Position Pressing"}


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""): result.update(block)
    return result.hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def chunk_size(index): return min(CHUNK, SIZE - index * CHUNK)


def chunk_path(index): return ROOT / "archive-ranges" / f"{index:04d}.bin"


class ChunksReader:
    """Sequential, bounded concatenation; does not create a second full archive."""
    def __init__(self, paths):
        self.paths = iter(paths); self.stream = None; self.position = 0
    def read(self, size=-1):
        if size < 0: raise ValueError("Unbounded archive read prohibited")
        parts = []
        while size:
            if self.stream is None:
                path = next(self.paths, None)
                if path is None: break
                self.stream = Path(path).open("rb")
            block = self.stream.read(size)
            if not block:
                self.stream.close(); self.stream = None; continue
            parts.append(block); size -= len(block); self.position += len(block)
        return b"".join(parts)
    def close(self):
        if self.stream is not None: self.stream.close(); self.stream = None
