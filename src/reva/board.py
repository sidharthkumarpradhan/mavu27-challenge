"""Live Codabench leaderboard: fetch, snapshot, and the per-task gap to the leader.

GET /api/phases/<phase>/get_leaderboard/ needs no login (checked 6 Oct 2026). Each submission row
holds `owner`, `id`, `created_when` and `scores` (one per column_key, the score as a string).
"""

from __future__ import annotations

import csv
from pathlib import Path

import requests

from reva.score import OVERALL, TASK_COLUMNS

BASE = "https://www.codabench.org"


def fetch(phase: int, base: str = BASE, timeout: int = 60) -> list[dict]:
    r = requests.get(f"{base}/api/phases/{phase}/get_leaderboard/", timeout=timeout)
    r.raise_for_status()
    return parse(r.json())


def parse(payload: dict) -> list[dict]:
    """Rows sorted by overall accuracy: {"owner", "submission", "created", <column_key>: float}."""
    rows = []
    for s in payload.get("submissions", []):
        row = {"owner": s["owner"], "submission": s["id"], "created": s.get("created_when", "")}
        row.update({c["column_key"]: float(c["score"]) for c in s.get("scores", [])})
        rows.append(row)
    return sorted(rows, key=lambda r: -r.get(OVERALL, 0.0))


def snapshot(rows: list[dict], path: str | Path, when: str) -> int:
    """Append rows not seen before (keyed by submission id) to a CSV. Returns how many were new."""
    path = Path(path)
    cols = ["seen", "owner", "submission", "created", OVERALL, *TASK_COLUMNS.values()]
    seen = set()
    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            seen = {int(r["submission"]) for r in csv.DictReader(f)}
    new = [r for r in rows if r["submission"] not in seen]
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        if write_header:
            w.writeheader()
        for r in new:
            w.writerow({**r, "seen": when})
    return len(new)


def gap(rows: list[dict], ours: dict[str, float], test_counts: dict[str, int]) -> list[dict]:
    """Per task: how many points of overall accuracy we lose to the current leader.

    lost = (leader_acc - our_acc) * n_test(task) / n_test. Sorted largest first, so the top line
    is where the next experiment should aim. `ours` uses Codabench column keys.
    """
    if not rows:
        return []
    leader = rows[0]
    total = sum(test_counts.values())
    out = []
    for task, col in TASK_COLUMNS.items():
        mine = ours.get(col, 0.0)
        out.append({"task": task, "column": col, "leader": leader.get(col, 0.0), "ours": mine,
                    "lost": (leader.get(col, 0.0) - mine) * test_counts.get(task, 0) / total})
    return sorted(out, key=lambda g: -g["lost"])


def rank_of(rows: list[dict], owner: str) -> int | None:
    for i, r in enumerate(rows, 1):
        if r["owner"].lower() == owner.lower():
            return i
    return None
