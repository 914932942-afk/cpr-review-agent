"""Refuse a graph-mode switch while an incompatible request is still active."""

import argparse
import json
import sqlite3
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--graph-version", required=True)
    args = parser.parse_args()
    db = (args.data_dir / "business.sqlite3").resolve()
    incompatible = []
    if db.is_file():
        with sqlite3.connect(db.as_uri() + "?mode=ro", uri=True) as conn:
            for row in conn.execute("SELECT body FROM runs"):
                run = json.loads(row[0])
                if run["status"] not in {
                    "completed",
                    "partially_completed",
                    "rejected",
                    "failed",
                    "cancelled",
                }:
                    if run["graph_version"] != args.graph_version:
                        incompatible.append(run["id"])
    if incompatible:
        print(
            json.dumps(
                {"passed": False, "code": "ACTIVE_GRAPH_VERSION_CONFLICT", "active_run_ids": incompatible}
            )
        )
        raise SystemExit(2)
    print(json.dumps({"passed": True, "graph_version": args.graph_version}))


if __name__ == "__main__":
    main()
