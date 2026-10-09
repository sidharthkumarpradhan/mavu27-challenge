"""Where a run loses points, measured on dev and weighted by the test mix.

Run this before choosing the next experiment (CLAUDE.md: measure before building). It reads a run's
dev_probs.json and the labeled dev split, and counts how the hidden test set splits over tasks and
sources (test.json carries both; it has no labels). "Test points lost" in a cell is
(test share of the cell) x (1 - dev accuracy in the cell) x 100: the overall test accuracy that
cell costs if test behaves like dev. The largest cells are where an experiment can win the most.

Found with it on 9 Oct 2026: Temporal Grounding on Hawk_UAV cost the most (2.8 points). Its
options sit 0.5 s apart in 8 s clips, and Qwen3-VL prints one timestamp per pair of frames, so 16
frames give marks about 1 s apart.
"""

from __future__ import annotations

import collections

LETTERS = "ABCD"


def pred(p: list[float]) -> str:
    return LETTERS[max(range(4), key=p.__getitem__)]


def breakdown(dev: list[dict], probs: dict[str, list[float]], test: list[dict],
              keys: tuple[str, ...] = ("task",)) -> list[dict]:
    """One row per cell of `keys` that the test set has, sorted by test points lost (most first).
    A test cell without dev questions has dev accuracy None: it is not measured."""
    hits: dict[tuple, list[bool]] = collections.defaultdict(list)
    for r in dev:
        if r["qa_id"] in probs:
            hits[tuple(r[k] for k in keys)].append(pred(probs[r["qa_id"]]) == r["correct_answer"])
    counts = collections.Counter(tuple(r[k] for k in keys) for r in test)
    out = []
    for cell, n_test in counts.items():
        v = hits.get(cell, [])
        acc = sum(v) / len(v) if v else None
        share = n_test / len(test)
        out.append({"cell": cell, "dev_acc": acc, "dev_n": len(v), "test_n": n_test, "test_share": share,
                    "test_points_lost": None if acc is None else share * (1 - acc) * 100})
    # unmeasured cells first: a blind spot is worth knowing about before any loss
    return sorted(out, key=lambda c: (c["test_points_lost"] is not None, -(c["test_points_lost"] or 0)))


def confidence_bins(dev: list[dict], probs: dict[str, list[float]], bins: int = 5) -> list[float]:
    """Dev accuracy by quintile of the top probability, least confident first."""
    pairs = sorted((max(probs[r["qa_id"]]) / sum(probs[r["qa_id"]]), pred(probs[r["qa_id"]]) == r["correct_answer"])
                   for r in dev if r["qa_id"] in probs)
    k = len(pairs) // bins
    return [sum(ok for _, ok in pairs[i * k:(i + 1) * k]) / k for i in range(bins)] if k else []


def agreement(dev: list[dict], a: dict[str, list[float]], b: dict[str, list[float]]) -> dict:
    """How two runs split the dev questions, and the oracle accuracy (either one right)."""
    both = only_a = only_b = neither = 0
    for r in dev:
        if r["qa_id"] in a and r["qa_id"] in b:
            x, y = pred(a[r["qa_id"]]) == r["correct_answer"], pred(b[r["qa_id"]]) == r["correct_answer"]
            both += x and y
            only_a += x and not y
            only_b += y and not x
            neither += not x and not y
    n = both + only_a + only_b + neither
    return {"both": both, "only_a": only_a, "only_b": only_b, "neither": neither,
            "oracle": (both + only_a + only_b) / n if n else 0.0}


def report(dev: list[dict], runs: dict[str, dict[str, list[float]]], test: list[dict], top: int = 10) -> str:
    """Markdown: per task and per (task, source) losses, letter balance and confidence, per run."""
    lines = ["# Error analysis (dev, weighted by the test mix)", ""]
    for name, probs in runs.items():
        rows = [r for r in dev if r["qa_id"] in probs]
        acc = sum(pred(probs[r["qa_id"]]) == r["correct_answer"] for r in rows) / max(1, len(rows))
        lines += [f"## {name}", "", f"{len(rows)} dev questions, accuracy {acc:.4f}.", "",
                  "| task | dev acc | dev n | test share | test points lost |", "|---|---|---|---|---|"]
        for c in breakdown(rows, probs, test):
            lines.append(_row(c))
        lines += ["", f"Top {top} task x source cells:", "",
                  "| task, source | dev acc | dev n | test share | test points lost |", "|---|---|---|---|---|"]
        for c in breakdown(rows, probs, test, ("task", "dataset_name"))[:top]:
            lines.append(_row(c))
        letters = collections.Counter(pred(probs[r["qa_id"]]) for r in rows)
        lines += ["", "Predicted letters: " + ", ".join(f"{k} {letters[k]}" for k in LETTERS) + ".",
                  "Accuracy by confidence quintile (least confident first): "
                  + ", ".join(f"{x:.3f}" for x in confidence_bins(rows, probs)) + ".", ""]
    names = list(runs)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            g = agreement(dev, runs[a], runs[b])
            lines.append(f"{a} vs {b}: both right {g['both']}, only the first {g['only_a']}, only the second "
                         f"{g['only_b']}, neither {g['neither']}; oracle {g['oracle']:.4f}.")
    return "\n".join(lines) + "\n"


def _row(c: dict) -> str:
    acc = "-" if c["dev_acc"] is None else f"{c['dev_acc']:.3f}"
    lost = "not measured" if c["test_points_lost"] is None else f"{c['test_points_lost']:.2f}"
    return f"| {', '.join(c['cell'])} | {acc} | {c['dev_n']} | {c['test_share']:.3f} | {lost} |"
