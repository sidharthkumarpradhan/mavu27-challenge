"""One unattended cycle. GitHub Actions runs it every hour (.github/workflows/autopilot.yml).

    board snapshot -> poll open submissions -> collect a finished Kaggle job -> gated submit
    -> push the next queued lanes -> write STATUS.md

State lives in a directory that the workflow keeps on the `state` branch:
- active.json        the job running now: {kernel, pushed, sha, runs}
- jobs.jsonl         one row per finished job, with wall hours (the weekly GPU tally)
- runs.jsonl         one row per lane: run id, status, dev metrics, hours, config summary
- submissions.jsonl  one row per Codabench submission, updated by appending newer rows
- board.csv          every leaderboard row ever seen
- STATUS.md          the human summary
The repo is public. Nothing here holds test predictions or probabilities.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from reva import board, data, package, registry, remote
from reva import config as C
from reva.codabench import DONE, FAILED, CodabenchError, scores
from reva.kaggle import DONE as K_DONE
from reva.kaggle import FAILED as K_FAILED
from reva.kaggle import tail

TEST_COUNTS = {"General Understanding": 180, "Object and Land Cover Recognition": 660, "Change Detection": 500,
               "Temporal Grounding": 640, "Trend and Pattern": 360, "Geometric Relation": 400,
               "Structural Layout": 340, "Perspective and Viewpoint": 480, "Causation Reasoning": 180,
               "Consequence Reasoning": 100, "Hypothetical Reasoning": 160}  # Codabench Data page


def iso(t: dt.datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_iso(s: str) -> dt.datetime:
    return dt.datetime.strptime(s[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=dt.timezone.utc)


def latest_submissions(rows: list[dict]) -> list[dict]:
    """submissions.jsonl appends a new row per status change; keep the last row per submission."""
    last = {}
    for r in rows:
        last[r["submission_id"]] = r
    return sorted(last.values(), key=lambda r: r["submitted"])


def gpu_hours(jobs: list[dict], now: dt.datetime, days: int = 7) -> float:
    since = now - dt.timedelta(days=days)
    return sum(j.get("hours", 0) for j in jobs if parse_iso(j["collected"]) >= since)


def pick_format(subs: list[dict], cfg: dict) -> str | None:
    """The predictions.json layout to use. A layout Codabench scored once is kept. Until then the
    configured one goes first, then the others, skipping any that came back Failed. Failed
    submissions do not count against the budget, so this probe costs nothing."""
    done = [s.get("format") for s in subs if s["status"] in DONE and s.get("format")]
    if done:
        return done[-1]
    failed = {s.get("format") for s in subs if s["status"] in FAILED}
    first = C.get(cfg, "submit.format")
    return next((f for f in [first, *[f for f in package.FORMATS if f != first]] if f not in failed), None)


def gate(run: dict, subs: list[dict], cfg: dict, now: dt.datetime) -> tuple[bool, str]:
    """Should this run be submitted now? Pure function of the state, so it is unit-tested."""
    if run.get("status") != "ok" or not run.get("zip"):
        return False, "no validated zip"
    if run.get("n", {}).get("test") != 4000:
        return False, "not a full test run"
    if any(s["run_id"] == run["run_id"] and s["status"] not in FAILED for s in subs):
        return False, "already submitted"
    if any(s["status"] not in DONE | FAILED for s in subs):
        return False, "a submission is still being scored"
    if pick_format(subs, cfg) is None:
        return False, "Codabench failed every predictions.json layout; the scorer needs a code fix"
    used = [s for s in subs if s["status"] not in FAILED]
    today = [s for s in used if s["submitted"][:10] == iso(now)[:10]]
    if len(today) >= C.get(cfg, "submit.max_per_day"):
        return False, "daily pacing cap reached"
    deadline = dt.datetime.fromisoformat(C.get(cfg, "competition.deadline")).replace(tzinfo=dt.timezone.utc)
    left = C.get(cfg, "submit.total_budget") - len(used)
    if left <= 0 or ((deadline - now).days > 7 and left <= C.get(cfg, "submit.reserve")):
        return False, f"submission budget: {left} left, reserve {C.get(cfg, 'submit.reserve')}"
    mine = run["metrics"]["weighted_accuracy"]
    best = max((s["dev_weighted"] for s in used), default=None)
    if best is not None and mine < best + C.get(cfg, "submit.min_gain"):
        return False, f"dev {mine:.4f} does not beat best submitted {best:.4f} by {C.get(cfg, 'submit.min_gain')}"
    return True, f"dev {mine:.4f}" + (f" vs best submitted {best:.4f}" if best is not None else " (first submission)")


def candidate(runs: list[dict], subs: list[dict]) -> dict | None:
    """Best finished full-test run by weighted dev accuracy that has no live submission yet."""
    taken = {s["run_id"] for s in subs if s["status"] not in FAILED}
    ok = [r for r in runs if r["status"] == "ok" and r.get("zip") and r["run_id"] not in taken]
    return max(ok, key=lambda r: r["metrics"]["weighted_accuracy"], default=None)


def submission_zip(kaggle, run: dict, fmt: str, cfg: dict, work: Path) -> Path:
    """Rebuild the run's zip in the chosen layout from its private test_probs.json (re-downloaded
    from the run's own kernel when this runner has not got it), validated against test.json."""
    dest = work / run["kernel"].split("/")[-1]
    probs_path = dest / run["run_id"] / "test_probs.json"
    if not probs_path.exists():
        kaggle.output(run["kernel"], dest)
    probs = json.loads(probs_path.read_text())
    ann = work / "annotations"
    data.fetch_annotations(ann, C.get(cfg, "data.hf_repo"))
    preds = {q: data.LETTERS[max(range(4), key=p.__getitem__)] for q, p in probs.items()}
    return package.write(work / "submit" / f"{run['run_id']}.zip", data.load_split(ann, "test"), preds, fmt,
                         {"run_id": run["run_id"]})


def summarize_config(cfg: dict) -> dict:
    keys = ["model.id", "model.load_4bit", "model.use_video", "frames.n", "frames.max_side", "train.enabled",
            "train.epochs", "train.max_samples", "train.refit", "infer.perms"]
    return {k: C.get(cfg, k) for k in keys}


def collect(kaggle, active: dict, work: Path, now: dt.datetime) -> tuple[list[dict], dict, str]:
    """Download a finished job. Returns (run rows, job row, log tail)."""
    dest = work / active["kernel"].split("/")[-1]
    files, log = kaggle.output(active["kernel"], dest)
    rows = []
    for run_id in active["runs"]:
        rj = dest / run_id / "run.json"
        if rj.exists():
            r = json.loads(rj.read_text())
            rows.append({"run_id": run_id, "status": "ok", "kernel": active["kernel"], "metrics": r["metrics"],
                         "hours": r["hours"], "zip": r["zip"], "n": r["n"], "train": r["train"],
                         "timings": r["timings"], "versions": r["versions"], "why": r["config"].get("why", ""),
                         "config": summarize_config(r["config"]), "finished": r["finished"], "sha": active["sha"]})
        else:
            lane_log = dest / run_id / "log.txt"
            text = lane_log.read_text(errors="replace") if lane_log.exists() else log
            rows.append({"run_id": run_id, "status": "failed", "kernel": active["kernel"], "finished": iso(now),
                         "sha": active["sha"], "error": tail(text, 30)})
    hours = (now - parse_iso(active["pushed"])).total_seconds() / 3600
    job = {"kernel": active["kernel"], "pushed": active["pushed"], "collected": iso(now), "hours": round(hours, 2),
           "runs": active["runs"], "sha": active["sha"]}
    return rows, job, tail(log, 60)


def status_md(cfg: dict, rows: list[dict], runs: list[dict], subs: list[dict], active: dict | None,
              jobs: list[dict], now: dt.datetime, notes: list[str]) -> str:
    owner = C.get(cfg, "competition.owner") or ""
    lines = [f"# ReVA autopilot status ({iso(now)})", ""]
    lines += [f"- {n}" for n in notes] + [""]
    lines += ["## Leaderboard (top 8)", "", "| # | owner | overall |", "|---|---|---|"]
    lines += [f"| {i} | {r['owner']} | {r.get('overall_accuracy', 0):.4f} |" for i, r in enumerate(rows[:8], 1)]
    if owner:
        lines += ["", f"Our rank: {board.rank_of(rows, owner) or 'not on the board'}"]
    fin = [s for s in subs if s["status"] in DONE and s.get("scores")]
    if fin and rows:
        best = max(fin, key=lambda s: s["scores"].get("overall_accuracy", 0))
        lines += ["", f"## Gap to the leader (best submission {best['run_id']})", "",
                  "| task | leader | ours | overall points lost |", "|---|---|---|---|"]
        for g in board.gap(rows, best["scores"], TEST_COUNTS):
            lines.append(f"| {g['task']} | {g['leader']:.3f} | {g['ours']:.3f} | {g['lost'] * 100:.2f} |")
        lines += ["", "## Calibration (board minus local weighted dev)", "", "| run | dev weighted | board | diff |",
                  "|---|---|---|---|"]
        for s in fin:
            b = s["scores"].get("overall_accuracy", 0)
            lines.append(f"| {s['run_id']} | {s['dev_weighted']:.4f} | {b:.4f} | {b - s['dev_weighted']:+.4f} |")
    lines += ["", "## Runs (newest first)", "", "| run | status | dev weighted | dev overall | hours | why |",
              "|---|---|---|---|---|---|"]
    for r in sorted(runs, key=lambda r: r["finished"], reverse=True)[:20]:
        m = r.get("metrics") or {}
        lines.append(f"| {r['run_id']} | {r['status']} | {m.get('weighted_accuracy', float('nan')):.4f} | "
                     f"{m.get('overall_accuracy', float('nan')):.4f} | {r.get('hours', '')} | {r.get('why', '')} |")
    used = len([s for s in subs if s["status"] not in FAILED])
    lines += ["", f"Submissions used: {used} of {C.get(cfg, 'submit.total_budget')}. "
              f"GPU hours, last 7 days: {gpu_hours(jobs, now):.1f} of {C.get(cfg, 'remote.weekly_gpu_hours')}.",
              f"Active job: {active['kernel'] + ' ' + str(active['runs']) if active else 'none'}"]
    return "\n".join(lines) + "\n"


def submit_as(client, cfg: dict) -> int | None:
    """Codabench organization id for `competition.organization`, or None to submit as the account."""
    name = C.get(cfg, "competition.organization")
    return client.organization_id(name) if name else None


def cycle(cfg: dict, queue: list[dict], state: Path, work: Path, kaggle, sha: str, kaggle_user: str,
          client=None, auto_submit: bool = False, now: dt.datetime | None = None,
          fetch_board=None, push: bool = True, notes: list[str] | None = None) -> dict:
    """One step. `client` is a logged-in reva.codabench.Client or None (then nothing is submitted)."""
    now = now or dt.datetime.now(dt.timezone.utc)
    state.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    notes = list(notes or [])  # e.g. the Codabench login result from the caller
    out = {"collected": [], "submitted": [], "pushed": None, "needs_fix": False}

    # 1. leaderboard
    fetch_board = fetch_board or (lambda: board.fetch(C.get(cfg, "competition.phase"), C.get(cfg, "competition.base")))
    try:
        rows = fetch_board()
        new = board.snapshot(rows, state / "board.csv", iso(now))
        notes.append(f"leaderboard: {len(rows)} rows, {new} new; leader {rows[0]['owner']} "
                     f"{rows[0].get('overall_accuracy', 0):.4f}" if rows else "leaderboard empty")
    except Exception as e:  # the board is information, never a reason to stop the loop
        rows = []
        notes.append(f"leaderboard fetch failed: {e}")

    # 2. open submissions
    subs = latest_submissions(registry.read(state / "submissions.jsonl"))
    if client:
        for s in subs:
            if s["status"] not in DONE | FAILED:
                try:
                    rec = client.submission(s["submission_id"])
                except CodabenchError as e:
                    notes.append(f"could not poll submission {s['submission_id']}: {e}")
                    continue
                if rec.get("status") != s["status"]:
                    registry.append(state / "submissions.jsonl", {**s, "status": rec.get("status"), "scores": scores(rec)})
        subs = latest_submissions(registry.read(state / "submissions.jsonl"))

    # 3. collect a finished job
    active_path = state / "active.json"
    active = json.loads(active_path.read_text()) if active_path.exists() else None
    fresh: list[dict] = []
    if active:
        kstate, message = kaggle.status(active["kernel"])
        if kstate in K_DONE | K_FAILED:
            fresh, job, log = collect(kaggle, active, work, now)
            job["state"] = kstate
            for r in fresh:
                registry.append(state / "runs.jsonl", r)
            registry.append(state / "jobs.jsonl", job)
            active_path.unlink()
            active = None
            out["collected"] = [r["run_id"] for r in fresh]
            bad = [r["run_id"] for r in fresh if r["status"] != "ok"]
            if bad or kstate in K_FAILED:
                out["needs_fix"] = True
                out["log_tail"] = log
                notes.append(f"job {job['kernel']} {kstate}; failed lanes: {bad} {message}")
            notes.append(f"collected {out['collected']}")
        else:
            notes.append(f"job {active['kernel']} is {kstate}")

    # 4. gated submission: the best run not yet submitted, from any cycle
    r = candidate(registry.read(state / "runs.jsonl"), subs)
    if r:
        allowed, why = gate(r, subs, cfg, now)
        if not allowed or not auto_submit or not client:
            notes.append(f"not submitting {r['run_id']}: "
                         f"{why if not allowed else 'automatic submission off' if not auto_submit else 'no Codabench login'}")
        else:
            try:
                can, reason = client.can_submit(C.get(cfg, "competition.phase"))
                if not can:
                    raise CodabenchError(f"Codabench refuses submissions for this account: {reason}")
                fmt = pick_format(subs, cfg)
                zip_path = submission_zip(kaggle, r, fmt, cfg, work)
                org = submit_as(client, cfg)
                sid = client.submit(zip_path, C.get(cfg, "competition.id"), C.get(cfg, "competition.phase"),
                                    [C.get(cfg, "competition.task")], organization=org)
                rec = client.wait(sid, timeout_s=60 * C.get(cfg, "submit.wait_minutes"))
                row = {"run_id": r["run_id"], "kernel": r["kernel"], "submission_id": sid, "submitted": iso(now),
                       "status": rec.get("status"), "scores": scores(rec), "format": fmt,
                       "organization": C.get(cfg, "competition.organization"),
                       "dev_weighted": r["metrics"]["weighted_accuracy"], "dev_overall": r["metrics"]["overall_accuracy"]}
                registry.append(state / "submissions.jsonl", row)
                out["submitted"].append(row)
                notes.append(f"submitted {r['run_id']} as {sid} ({fmt}): {why}; status {row['status']}")
            except CodabenchError as e:
                out["submit_error"] = str(e)
                notes.append(f"submission of {r['run_id']} not made: {e}")

    # 5. push the next lanes
    subs = latest_submissions(registry.read(state / "submissions.jsonl"))  # include this cycle's submission
    runs = registry.read(state / "runs.jsonl")
    jobs = registry.read(state / "jobs.jsonl")
    if push and not active:
        done = {r["run_id"] for r in runs if r["status"] == "ok"}
        failed: dict[str, int] = {}
        for r in runs:
            if r["status"] != "ok":
                failed[r["run_id"]] = failed.get(r["run_id"], 0) + 1
        lanes = remote.pending(cfg, queue, done, failed)[: C.get(cfg, "remote.lanes")]
        used = gpu_hours(jobs, now)
        if not lanes:
            notes.append("queue empty: add experiments to configs/queue.yaml")
        elif used + C.get(cfg, "remote.max_hours") > C.get(cfg, "remote.weekly_gpu_hours"):
            notes.append(f"GPU quota pacing: {used:.1f} h used in 7 days; waiting")
        else:
            kdir = work / "kernel"
            slug = remote.build(cfg, lanes, sha, kdir, kaggle_user)
            pushed = kaggle.push(kdir, timeout_s=int(3600 * C.get(cfg, "remote.max_hours")),
                                 accelerator=C.get(cfg, "remote.accelerator"))
            active = {"kernel": slug, "pushed": iso(now), "sha": sha, "runs": [c["run_id"] for c in lanes],
                      "url": pushed.url}
            active_path.write_text(json.dumps(active, indent=1))
            out["pushed"] = active
            notes.append(f"pushed {slug} with {active['runs']}")

    (state / "STATUS.md").write_text(status_md(cfg, rows, runs, subs, active, jobs, now, notes))
    out["notes"] = notes
    return out
