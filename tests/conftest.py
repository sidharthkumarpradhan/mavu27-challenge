import json
from pathlib import Path

import pytest

from reva import preflight
from reva.score import TASK_COLUMNS

BOARD_COLUMNS = ("overall_accuracy", {"overall_accuracy", *TASK_COLUMNS.values()})


@pytest.fixture(autouse=True)
def offline_columns(monkeypatch):
    """Every upload path reads the live leaderboard columns first. Tests answer for Codabench, so
    no test reaches the network (a fixture in one test module covers only that module)."""
    monkeypatch.setattr(preflight, "live_columns", lambda *a, **k: BOARD_COLUMNS)


SOURCES = [("VisDrone", "VisDrone"), ("Hawk_UAV", "NJ"), ("ERA_Tra", "Baseball")]


def make_rows(split: str, per_cell: int, seed: int = 0) -> list[dict]:
    """Rows in the ReVA schema: every (source, task) cell gets per_cell questions."""
    rows = []
    for name, sub in SOURCES:
        for t in TASK_COLUMNS:
            for k in range(per_cell):
                n = len(rows) + 1
                rows.append({"qa_id": f"{split}_{n:06d}", "video_path": f"{name}/{sub}/v{k % 3}.mp4", "subdir": sub,
                             "dataset_name": name, "category": "c", "task": t, "question": f"q{n}",
                             "options": {L: f"{L}{n}" for L in "ABCD"},
                             "correct_answer": "" if split == "test" else "ABCD"[(n + seed) % 4]})
    return rows


@pytest.fixture
def ann(tmp_path: Path) -> Path:
    for split, n in (("train", 10), ("val", 3), ("test", 2)):
        (tmp_path / f"{split}.json").write_text(json.dumps({"metadata": {}, "QA": make_rows(split, n)}))
    return tmp_path
