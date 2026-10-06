"""Local scorer that mirrors the Codabench leaderboard columns.

Codabench reports overall accuracy plus one accuracy per task. The column keys below are copied
from GET /api/competitions/18274/ (leaderboard 20415, read 6 Oct 2026).
"""

from __future__ import annotations

from collections import defaultdict

TASK_COLUMNS = {
    "General Understanding": "generalunderstand_accuracy",
    "Object and Land Cover Recognition": "objectlandcover_accuracy",
    "Change Detection": "changedetection_accuracy",
    "Temporal Grounding": "temporalgrounding_accuracy",
    "Trend and Pattern": "trendpattern_accuracy",
    "Geometric Relation": "geometricrelation_accuracy",
    "Structural Layout": "structurallayout_accuracy",
    "Perspective and Viewpoint": "perspectiveviewpoint_accuracy",
    "Causation Reasoning": "causation_accuracy",
    "Consequence Reasoning": "consequence_accuracy",
    "Hypothetical Reasoning": "hypothetical_accuracy",
}
OVERALL = "overall_accuracy"


def columns(rows: list[dict], preds: dict[str, str]) -> dict[str, float]:
    """Codabench-style table: overall plus per-task accuracy over labeled rows."""
    hit: dict[str, list[int]] = defaultdict(list)
    for r in rows:
        ok = int(preds.get(r["qa_id"]) == r["correct_answer"])
        hit[OVERALL].append(ok)
        hit[TASK_COLUMNS[r["task"]]].append(ok)
    return {k: sum(v) / len(v) for k, v in hit.items()}


def weighted(rows: list[dict], preds: dict[str, str], test: list[dict], min_cell: int = 20) -> float:
    """Dev accuracy re-weighted to the test mix of (source, task).

    Each test question counts with the dev accuracy of its (source, task) cell. A cell with fewer
    than min_cell dev questions falls back to the task's accuracy, so a tiny cell cannot swing it.
    """
    cell: dict[tuple, list[int]] = defaultdict(list)
    task: dict[str, list[int]] = defaultdict(list)
    for r in rows:
        ok = int(preds.get(r["qa_id"]) == r["correct_answer"])
        cell[(r["dataset_name"], r["task"])].append(ok)
        task[r["task"]].append(ok)
    total = 0.0
    for t in test:
        c = cell.get((t["dataset_name"], t["task"]), [])
        src = c if len(c) >= min_cell else task.get(t["task"], [])
        if not src:
            raise ValueError(f"dev set has no questions for task {t['task']!r}")
        total += sum(src) / len(src)
    return total / len(test)


def summary(rows: list[dict], preds: dict[str, str], test: list[dict], min_cell: int = 20) -> dict[str, float]:
    out = columns(rows, preds)
    out["weighted_accuracy"] = weighted(rows, preds, test, min_cell)
    return out
