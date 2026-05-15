"""Download the public CAP benchmark split from Hugging Face.

The dataset (Warrior0302/CAP) ships task descriptions for the 192 public tasks.
This script materializes them as one JSON file per task at:

    <output_dir>/task-<id>.json

Each JSON has the shape:
    {"task_id": "task-...", "instruction": "..."}

Usage:
    python scripts/download_dataset.py
    python scripts/download_dataset.py --output_dir src/evaluate/tasks
    python scripts/download_dataset.py --hf_repo Warrior0302/CAP --split test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset


DEFAULT_REPO = "Warrior0302/CAP"
DEFAULT_SPLIT = "test"
DEFAULT_OUTPUT = Path("src/evaluate/tasks")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf_repo", default=DEFAULT_REPO)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument(
        "--output_dir", type=Path, default=DEFAULT_OUTPUT,
        help="Where to write per-task JSON files (default: src/evaluate/tasks)",
    )
    return parser.parse_args()


def _normalize_columns(row: dict) -> dict:
    """Strip the BOM that may prefix 'task id' when loaded from CSV."""
    cleaned = {}
    for k, v in row.items():
        cleaned[k.lstrip("﻿").strip()] = v
    return cleaned


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.hf_repo} (split={args.split})...")
    ds = load_dataset(args.hf_repo, split=args.split)

    written = 0
    for row in ds:
        row = _normalize_columns(row)
        task_id = row.get("task id") or row.get("task_id")
        instruction = row.get("instruction")
        if not task_id:
            continue
        payload = {"task_id": task_id, "instruction": instruction}
        out_path = args.output_dir / f"{task_id}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        written += 1

    print(f"Wrote {written} task files to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
