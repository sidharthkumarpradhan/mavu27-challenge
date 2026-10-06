"""ReVA annotations and the local dev split.

The three JSON files on Hugging Face hold {"metadata": ..., "QA": [...]}. Each QA row has
qa_id, video_path, dataset_name, subdir, category, task, question, options {A..D}, correct_answer.

Test shares its videos with train (1,012 of 1,014), so the hidden test asks new questions about
seen videos. A faithful local check therefore holds out questions, not videos. The official val
split does this already, but it has almost no ERA_Tra questions while test has 1,285. So the dev
set is val plus a small train holdout stratified by (source, task), which also covers ERA.
"""

from __future__ import annotations

import json
import random
import urllib.request
from collections import defaultdict
from pathlib import Path

SPLITS = ("train", "val", "test")
LETTERS = ("A", "B", "C", "D")


def url(repo: str, name: str) -> str:
    return f"https://huggingface.co/datasets/{repo}/resolve/main/{name}"


def fetch_annotations(root: str | Path, repo: str) -> list[Path]:
    """Download train/val/test JSON (about 14 MB together). Skips files already present."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    out = []
    for split in SPLITS:
        dest = root / f"{split}.json"
        if not dest.exists():
            tmp = dest.with_suffix(".part")
            urllib.request.urlretrieve(url(repo, dest.name), tmp)
            tmp.replace(dest)
        out.append(dest)
    return out


def load_split(root: str | Path, split: str) -> list[dict]:
    rows = json.loads((Path(root) / f"{split}.json").read_text(encoding="utf-8"))["QA"]
    check_rows(rows, labeled=split != "test")
    return rows


def check_rows(rows: list[dict], labeled: bool) -> None:
    """Fail loudly on a schema change upstream instead of scoring garbage later."""
    ids = [r["qa_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate qa_id in split")
    for r in rows:
        if tuple(sorted(r["options"])) != LETTERS:
            raise ValueError(f"{r['qa_id']}: options are {sorted(r['options'])}, expected A-D")
        if labeled and r["correct_answer"] not in LETTERS:
            raise ValueError(f"{r['qa_id']}: bad label {r['correct_answer']!r}")


def stratified_holdout(rows: list[dict], frac: float, seed: int) -> tuple[list[dict], list[dict]]:
    """Split rows into (keep, holdout), taking round(frac * n) from every (source, task) cell."""
    if frac <= 0:
        return list(rows), []
    cells: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        cells[(r["dataset_name"], r["task"])].append(r)
    rng = random.Random(seed)
    held: set[str] = set()
    for key in sorted(cells):
        group = sorted(cells[key], key=lambda r: r["qa_id"])
        rng.shuffle(group)
        held.update(r["qa_id"] for r in group[: round(frac * len(group))])
    return [r for r in rows if r["qa_id"] not in held], [r for r in rows if r["qa_id"] in held]


def make_splits(root: str | Path, holdout_frac: float, seed: int, refit: bool = False,
                refit_with_val: bool = False) -> dict[str, list[dict]]:
    """{"fit", "dev", "test"} for one experiment.

    fit is what the model trains on. dev is never trained on during selection. For a final refit
    the holdout returns to fit, and val joins only when the owner allows it (see CLAUDE.md).
    """
    train, val, test = (load_split(root, s) for s in SPLITS)
    fit, holdout = stratified_holdout(train, holdout_frac, seed)
    dev = val + holdout
    if refit:
        fit = train + (val if refit_with_val else [])
    return {"fit": fit, "dev": dev, "test": test}
