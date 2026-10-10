"""Error analysis on dev probabilities: where a run loses points, and what a blend would buy.

This is the local half of the board. It reads dev probabilities only (`dev_probs.json` from a
run's private dataset). Test rows give the (source, task) mix, never labels, so nothing here can
look at the test answers.

    python -m reva.cli analyze --probs ft8b=path/dev_probs.json --probs zs32b=path/dev_probs.json
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from reva import score
from reva.arena import argmax

CONFIDENCE_EDGES = (0.6, 0.7, 0.8, 0.9)
BLEND_WEIGHTS = (0.1, 0.2, 0.3, 0.4, 0.5)


def load(path: str | Path, dev: list[dict]) -> dict[str, list[float]]:
    """A run's dev probabilities, cut to this dev set. Fails if any dev question is missing."""
    probs = json.loads(Path(path).read_text())
    missing = [r["qa_id"] for r in dev if r["qa_id"] not in probs]
    if missing:
        raise ValueError(f"{path} has no probabilities for {len(missing)} dev questions, e.g. {missing[0]}")
    return {r["qa_id"]: probs[r["qa_id"]] for r in dev}


def losses(dev: list[dict], preds: dict[str, str], test: list[dict], min_cell: int = 20) -> list[dict]:
    """Test-weighted points lost per (source, task) cell, largest first.

    Uses the same cell accuracy as score.weighted (a cell under min_cell dev questions falls back to
    its task), so the points add up to 100 * (1 - weighted accuracy).
    """
    cell: dict[tuple, list[int]] = defaultdict(list)
    task: dict[str, list[int]] = defaultdict(list)
    for r in dev:
        ok = int(preds[r["qa_id"]] == r["correct_answer"])
        cell[(r["dataset_name"], r["task"])].append(ok)
        task[r["task"]].append(ok)
    out = []
    for (src, t), n in Counter((r["dataset_name"], r["task"]) for r in test).items():
        c = cell.get((src, t), [])
        hits = c if len(c) >= min_cell else task.get(t, [])
        if not hits:
            raise ValueError(f"dev set has no questions for task {t!r}")
        acc = sum(hits) / len(hits)
        out.append({"source": src, "task": t, "dev_acc": acc, "dev_n": len(c), "test_n": n,
                    "points": 100 * n * (1 - acc) / len(test)})
    return sorted(out, key=lambda x: -x["points"])


def confidence(dev: list[dict], probs: dict[str, list[float]], edges=CONFIDENCE_EDGES) -> list[dict]:
    """Accuracy by top probability (normalized over the 4 options), one bin per edge interval."""
    bounds = [0.0, *edges, 1.01]
    bins = [[0, 0] for _ in range(len(bounds) - 1)]
    preds = argmax(probs)
    for r in dev:
        p = probs[r["qa_id"]]
        top = max(p) / sum(p)
        k = next(i for i in range(len(bins)) if top < bounds[i + 1])
        bins[k][0] += preds[r["qa_id"]] == r["correct_answer"]
        bins[k][1] += 1
    return [{"lo": bounds[i], "hi": min(1.0, bounds[i + 1]), "n": n, "accuracy": ok / n if n else None}
            for i, (ok, n) in enumerate(bins)]


def oracle(dev: list[dict], runs: dict[str, dict[str, list[float]]]) -> float:
    """Share of dev questions at least one run answers right: a ceiling for any mix of these runs."""
    preds = [argmax(p) for p in runs.values()]
    return sum(any(p[r["qa_id"]] == r["correct_answer"] for p in preds) for r in dev) / len(dev)


def blend(a: dict[str, list[float]], b: dict[str, list[float]], w: float) -> dict[str, list[float]]:
    """(1 - w) * a + w * b, question by question."""
    return {q: [(1 - w) * x + w * y for x, y in zip(a[q], b[q])] for q in a}


def blend_search(dev: list[dict], test: list[dict], a: dict, b: dict, weights=BLEND_WEIGHTS,
                 min_cell: int = 20) -> list[tuple[float, float]]:
    """Weighted dev accuracy of each blend weight on b. Pick on dev only, never on the board."""
    return [(w, score.weighted(dev, argmax(blend(a, b, w)), test, min_cell)) for w in weights]


def report(dev: list[dict], test: list[dict], runs: dict[str, dict[str, list[float]]], min_cell: int = 20,
           top: int = 12) -> str:
    """Plain-text report: per-run scores by task, the best run's losses, confidence, oracle, blends."""
    preds = {k: argmax(p) for k, p in runs.items()}
    summ = {k: score.summary(dev, p, test, min_cell) for k, p in preds.items()}
    names = sorted(runs, key=lambda k: -summ[k]["weighted_accuracy"])
    best = names[0]
    lines = [f"dev {len(dev)} questions, test mix {len(test)}", "", "weighted dev accuracy"]
    lines += [f"  {k:32} {summ[k]['weighted_accuracy']:.4f}" for k in names]
    lines += ["", "per task  " + "  ".join(f"{k[:10]:>10}" for k in names)]
    for t, col in score.TASK_COLUMNS.items():
        lines.append(f"  {t[:30]:30} " + "  ".join(f"{summ[k].get(col, float('nan')):10.3f}" for k in names))
    lost = losses(dev, preds[best], test, min_cell)
    lines += ["", f"test-weighted points lost by {best} (total {sum(x['points'] for x in lost):.2f})"]
    lines += [f"  {x['points']:5.2f}  {x['source']:9} {x['task'][:32]:32} dev {x['dev_acc']:.3f} "
              f"(n {x['dev_n']}), test n {x['test_n']}" for x in lost[:top]]
    lines += ["", f"confidence of {best} (top probability: accuracy, n)"]
    lines += [f"  {b['lo']:.1f}-{b['hi']:.1f}: " + (f"{b['accuracy']:.3f}" if b["n"] else "-") + f"  n {b['n']}"
              for b in confidence(dev, runs[best])]
    if len(runs) > 1:
        lines += ["", f"oracle (any run right): {oracle(dev, runs):.4f}", "", f"blends of {best} with"]
        for k in names[1:]:
            found = blend_search(dev, test, runs[best], runs[k], min_cell=min_cell)
            lines.append(f"  {k:32} " + "  ".join(f"w{w:.1f} {acc:.4f}" for w, acc in found))
    return "\n".join(lines)
