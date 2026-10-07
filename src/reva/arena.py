"""Our own arena: every finished run competes on dev, and so do averages of the best of them.

Each lane already writes dev_probs.json and test_probs.json to its private Kaggle output. Here the
loop averages the probabilities of the top k runs (by weighted dev accuracy), keeps the k that
scores best on dev, and adds that ensemble to runs.jsonl as one more candidate. The usual gate and
the pre-upload checks then decide whether it goes to Codabench, exactly as for a single run.

Selection uses dev only, never the board. Only plain single runs take part: a refit run trained on
the dev holdout (its dev score is inflated), and a text-only probe never touches the test set.
Picking k on dev flatters the ensemble a little; submit.min_gain and the board calibration absorb it.
The pre-upload checks read every member's own run.json, so one non-compliant member blocks it.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from reva import score

MAX_K = 5


def eligible(runs: list[dict]) -> list[dict]:
    """Single runs with a full test prediction whose dev score is fair to compare."""
    out = []
    for r in runs:
        cfg = r.get("config") or {}
        if (r.get("status") == "ok" and r.get("zip") and r.get("n", {}).get("test") == 4000 and not r.get("members")
                and cfg.get("model.use_video") is not False and not cfg.get("train.refit")
                and not cfg.get("train.refit_with_val")):
            out.append(r)
    return sorted(out, key=lambda r: -r["metrics"]["weighted_accuracy"])


def ensemble_id(members: list[str]) -> str:
    return "ens-" + hashlib.sha1(",".join(sorted(members)).encode()).hexdigest()[:8]


def load_probs(kaggle, run: dict, work: Path, name: str) -> dict[str, list[float]]:
    """dev_probs.json or test_probs.json of a run, fetched from its private kernel output when this
    runner has not got it. For an ensemble, the mean over its members."""
    if run.get("members"):
        return mean([load_probs(kaggle, m, work, name) for m in run["members"]])
    dest = work / run["kernel"].split("/")[-1]
    path = dest / run["run_id"] / name
    if not path.exists():
        kaggle.output(run["kernel"], dest)
    return json.loads(path.read_text())


def mean(probs: list[dict[str, list[float]]]) -> dict[str, list[float]]:
    keys = probs[0].keys()
    if any(p.keys() != keys for p in probs):
        raise ValueError("ensemble members predicted different questions")
    return {q: [sum(p[q][j] for p in probs) / len(probs) for j in range(4)] for q in keys}


def argmax(probs: dict[str, list[float]]) -> dict[str, str]:
    return {q: "ABCD"[max(range(4), key=p.__getitem__)] for q, p in probs.items()}


def p_better(dev: list[dict], a: dict[str, str], b: dict[str, str]) -> float:
    """Chance that a beats b on questions like these: a one-sided paired test on dev correctness."""
    d = [int(a[r["qa_id"]] == r["correct_answer"]) - int(b[r["qa_id"]] == r["correct_answer"]) for r in dev]
    m = sum(d) / len(d)
    var = sum((x - m) ** 2 for x in d) / max(1, len(d) - 1)
    if var == 0:
        return 1.0 if m > 0 else 0.0 if m < 0 else 0.5
    return 0.5 * (1 + math.erf(m / math.sqrt(var / len(d)) / math.sqrt(2)))


def best_ensemble(singles: list[dict], dev_probs: dict[str, dict], dev: list[dict], test: list[dict],
                  min_cell: int = 20) -> dict | None:
    """The top-k average (k from 2) with the best weighted dev accuracy, if it beats the best single."""
    if len(singles) < 2:
        return None
    best_single = argmax(dev_probs[singles[0]["run_id"]])
    found = None
    for k in range(2, min(MAX_K, len(singles)) + 1):
        members = singles[:k]
        preds = argmax(mean([dev_probs[m["run_id"]] for m in members]))
        metrics = score.summary(dev, preds, test, min_cell)
        if found is None or metrics["weighted_accuracy"] > found["metrics"]["weighted_accuracy"]:
            found = {"members": members, "metrics": metrics, "p": p_better(dev, preds, best_single)}
    if found["metrics"]["weighted_accuracy"] <= singles[0]["metrics"]["weighted_accuracy"]:
        return None
    return found


def step(kaggle, runs: list[dict], dev: list[dict], test: list[dict], work: Path, seen: list[str], now: str,
         min_cell: int = 20) -> tuple[dict | None, list[str], str]:
    """One arena round. Returns (new ensemble row or None, the run ids considered, a note).

    `seen` is the set considered last time; with no new single run there is nothing to redo, so
    the loop does not re-download every kernel output each cycle.
    """
    singles = eligible(runs)
    ids = sorted(r["run_id"] for r in singles)
    if ids == sorted(seen) or len(singles) < 2:
        return None, ids, ""
    dev_ids = {r["qa_id"] for r in dev}
    dev_probs = {}
    for r in singles:
        p = load_probs(kaggle, r, work, "dev_probs.json")
        if set(p) == dev_ids:  # a run scored on another dev split cannot be averaged with the rest
            dev_probs[r["run_id"]] = p
    singles = [r for r in singles if r["run_id"] in dev_probs]
    found = best_ensemble(singles, dev_probs, dev, test, min_cell)
    if not found:
        return None, ids, f"arena: no average of the top runs beats {singles[0]['run_id'] if singles else 'any run'} on dev"
    members = [m["run_id"] for m in found["members"]]
    run_id = ensemble_id(members)
    if any(r["run_id"] == run_id for r in runs):
        return None, ids, f"arena: best is still {run_id}"
    best = found["members"][0]
    row = {"run_id": run_id, "status": "ok", "members": [{"run_id": m["run_id"], "kernel": m["kernel"]} for m in found["members"]],
           "metrics": found["metrics"], "n": {"dev": len(dev), "test": 4000}, "zip": "rebuilt from members",
           "hours": 0, "finished": now, "config": {"model.use_video": True, "ensemble": members},
           "why": f"mean of top {len(members)} on dev: {', '.join(members)}; "
                  f"P(beats {best['run_id']}) {found['p']:.2f}"}
    return row, ids, (f"arena: {run_id} = mean of {len(members)} runs, dev weighted "
                      f"{found['metrics']['weighted_accuracy']:.4f} vs best single {best['metrics']['weighted_accuracy']:.4f}")
