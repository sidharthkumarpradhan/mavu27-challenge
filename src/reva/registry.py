"""Append-only JSONL logs kept on the `state` branch: runs, submissions, board snapshots.

The repo is public, so these hold metrics only. Never test predictions or probabilities.
"""

from __future__ import annotations

import json
from pathlib import Path


def append(path: str | Path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def read(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def best(rows: list[dict], metric: str) -> dict | None:
    scored = [r for r in rows if isinstance(r.get("metrics", {}).get(metric), (int, float))]
    return max(scored, key=lambda r: r["metrics"][metric]) if scored else None
