"""ReVA annotations and the local dev split.

The three JSON files on Hugging Face hold {"metadata": ..., "QA": [...]}. Each QA row has
qa_id, video_path, dataset_name, subdir, category, task, question, options {A..D}, correct_answer.

Test shares its videos with train (1,012 of 1,014), so the hidden test asks new questions about
seen videos. A faithful local check therefore holds out questions, not videos. The official val
split does this already, but it has almost no ERA_Tra questions while test has 1,285. So the dev
set is val plus a small train holdout stratified by (source, task), which also covers ERA.

1,456 of the 2,000 val questions are exact copies of train questions: same video, question, option
set and answer (measured 8 Oct 2026). Test has 2. A fine-tune that saw a copy is being asked to
recall it, not to answer it, so dev leaves out every question whose copy is in fit (and any repeat
within dev). That leaves about 1,800 questions that measure what test measures.
"""

from __future__ import annotations

import json
import random
import urllib.request
from collections import defaultdict
from pathlib import Path

SPLITS = ("train", "val", "test")
DEV_SET = "unseen-v1"  # recorded with each run; runs scored on an older dev set are rescored (reva.autopilot)
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


def load_metadata(root: str | Path, split: str) -> dict:
    """The split file's "metadata" block. A fill_test submission keeps test.json's own, so the
    file we upload has the same layout as the hidden answer key (see reva.package)."""
    return json.loads((Path(root) / f"{split}.json").read_text(encoding="utf-8")).get("metadata", {})


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


def qa_key(r: dict) -> tuple:
    """A question's identity regardless of letter order: video, question, option set, answer text."""
    opts = r["options"]
    return r["video_path"], r["question"].strip(), tuple(sorted(opts.values())), opts.get(r.get("correct_answer"))


def unseen(dev: list[dict], fit: list[dict]) -> list[dict]:
    """dev without the questions fit already holds, and without repeats."""
    seen = {qa_key(r) for r in fit}
    out = []
    for r in dev:
        if (k := qa_key(r)) not in seen:
            seen.add(k)
            out.append(r)
    return out


def make_splits(root: str | Path, holdout_frac: float, seed: int, refit: bool = False,
                refit_with_val: bool = False) -> dict[str, list[dict]]:
    """{"fit", "dev", "test"} for one experiment.

    fit is what the model trains on. dev is never trained on during selection. For a final refit
    the holdout returns to fit, and val joins only when the owner allows it (see CLAUDE.md).
    """
    train, val, test = (load_split(root, s) for s in SPLITS)
    fit, holdout = stratified_holdout(train, holdout_frac, seed)
    dev = unseen(val + holdout, fit)
    if refit:
        fit = train + (val if refit_with_val else [])
    return {"fit": fit, "dev": dev, "test": test}
