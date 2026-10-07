"""Last checks before a zip goes to Codabench, and what we expect it to score there.

The test labels are hidden, so nothing local can prove a board score. What can be checked:
- the file is what Codabench asks for, and it holds exactly this run's answers (validate, round trip);
- the run is allowed on the test set: its own run.json says video input on and no training on
  val (a missing flag counts as a failure);
- the run is not broken in a way dev accuracy can miss (non-finite probabilities, one letter for
  almost every answer);
- the live leaderboard still ranks by overall accuracy over the 11 task columns we score locally.

The projection is local weighted dev accuracy plus the mean (board - dev) gap of our earlier
scored submissions. It is a forecast for STATUS.md, not a gate: the gate stays "beats our own best
on dev" (reva.autopilot.gate), so the first submission can confirm the format and calibrate.

Problem messages land in the public STATUS.md, so they carry counts and shares, never answers.
"""

from __future__ import annotations

import json
import math
import zipfile

import requests

from reva import config as C
from reva import package
from reva.codabench import DONE
from reva.data import LETTERS
from reva.score import OVERALL, TASK_COLUMNS

# A letter outside these shares means a broken run, not a weak one: train and val labels are
# balanced at about 25% per letter (measured 6 Oct 2026), and even a strongly letter-biased
# zero-shot model stays well inside them.
MIN_SHARE, MAX_SHARE = 0.05, 0.60


class Blocked(Exception):
    """A run that must not be uploaded. The problems carry counts and shares, never answers."""

    def __init__(self, problems: list[str]):
        super().__init__("; ".join(problems))
        self.problems = problems


def probs_problems(probs: dict[str, list[float]], test: list[dict]) -> list[str]:
    """Why these test probabilities cannot become a submission, before any answer is derived."""
    want = {r["qa_id"] for r in test}
    out = []
    if missing := want - set(probs):
        out.append(f"probabilities: {len(missing)} test questions have none")
    if extra := set(probs) - want:
        out.append(f"probabilities: {len(extra)} are for questions not in test.json")
    bad = [q for q, p in probs.items()
           if not isinstance(p, list) or len(p) != 4
           or not all(isinstance(x, (int, float)) and math.isfinite(x) and x >= 0 for x in p)]
    if bad:
        out.append(f"probabilities: {len(bad)} rows are not 4 finite non-negative numbers")
    return out


def compliance_problems(configs: list[dict]) -> list[str]:
    """From each run's full config (its run.json). A missing flag fails: no evidence, no upload."""
    out = []
    for cfg in configs:
        video, with_val = C.get(cfg, "model.use_video"), C.get(cfg, "train.refit_with_val")
        if video is False:
            out.append("compliance: a text-only run never goes to the test set")
        elif video is not True:
            out.append("compliance: the run does not record model.use_video")
        if with_val is True:
            out.append("compliance: train.refit_with_val is the owner's call and stays false")
        elif with_val is not False:
            out.append("compliance: the run does not record train.refit_with_val")
    return out


def answers(zip_path, fmt: str) -> dict[str, str]:
    with zipfile.ZipFile(zip_path) as z:
        return dict(package.parse(json.loads(z.read(package.MEMBER)), fmt))


def letter_shares(preds: dict[str, str]) -> dict[str, float]:
    n = max(1, len(preds))
    return {L: sum(a == L for a in preds.values()) / n for L in LETTERS}


def live_columns(competition: int, base: str, timeout: int = 60) -> tuple[str, set[str]]:
    """(primary column key, all column keys) of the competition's leaderboard, from the public API."""
    r = requests.get(f"{base}/api/competitions/{competition}/", timeout=timeout)
    r.raise_for_status()
    lb = r.json()["leaderboards"][0]
    cols = sorted(lb["columns"], key=lambda c: c["index"])
    return cols[lb.get("primary_index", 0)]["key"], {c["key"] for c in cols}


def metric_problem(columns: tuple[str, set[str]]) -> str | None:
    """Why our local scorer no longer mirrors the board, or None. Not a fault of any one run."""
    primary, keys = columns
    if primary != OVERALL or keys != {OVERALL, *TASK_COLUMNS.values()}:
        return f"the board now ranks by {primary!r} over {sorted(keys)}; update reva.score before submitting"
    return None


def check(zip_path, test: list[dict], probs: dict[str, list[float]], fmt: str, configs: list[dict]) -> list[str]:
    """Every reason not to upload this zip. `configs` are the full configs of the run (or of every
    member of an ensemble). An empty list means ready."""
    problems = compliance_problems(configs)
    try:
        package.validate(zip_path, test, fmt)
    except ValueError as e:
        problems.append(f"format: {e}")
    if bad := probs_problems(probs, test):
        problems += bad
    else:
        want = {q: LETTERS[max(range(4), key=p.__getitem__)] for q, p in probs.items()}
        try:
            got = answers(zip_path, fmt)
        except (KeyError, TypeError, ValueError, zipfile.BadZipFile) as e:
            problems.append(f"round trip: cannot read the zip back ({type(e).__name__})")
        else:
            if diff := sum(got.get(q) != a for q, a in want.items()) + len(set(got) - set(want)):
                problems.append(f"round trip: {diff} answers differ from the run's own probabilities")
            off = {L: s for L, s in letter_shares(got).items() if not MIN_SHARE <= s <= MAX_SHARE}
            if off:
                problems.append("letter balance: " + ", ".join(f"{L} {s:.1%}" for L, s in off.items())
                                + f" of answers (expected {MIN_SHARE:.0%} to {MAX_SHARE:.0%} each)")
    return problems


def projected(dev_weighted: float, subs: list[dict]) -> tuple[float, int]:
    """Expected board overall accuracy, and how many scored submissions the correction rests on."""
    gaps = [s["scores"][OVERALL] - s["dev_weighted"] for s in subs
            if s["status"] in DONE and OVERALL in (s.get("scores") or {})]
    return dev_weighted + (sum(gaps) / len(gaps) if gaps else 0.0), len(gaps)


def margin(acc: float, n: int) -> float:
    """Half-width of a 95% interval for an accuracy measured on n questions."""
    return 1.96 * math.sqrt(max(acc * (1 - acc), 1e-9) / max(n, 1))


def forecast(run: dict, subs: list[dict], leader: float | None) -> str:
    """One line for STATUS.md: dev, the projection, and where it would land against the leader."""
    w = run["metrics"]["weighted_accuracy"]
    p, k = projected(w, subs)
    line = (f"next candidate {run['run_id']}: dev weighted {w:.4f} +/- {margin(w, run.get('n', {}).get('dev', 0)):.4f}, "
            f"projected board {p:.4f} ({k} calibration pair{'s' if k != 1 else ''})")
    if leader is not None:
        line += f"; leader {leader:.4f}, {'above' if p > leader else 'below'} by {abs(p - leader):.4f}"
    return line
