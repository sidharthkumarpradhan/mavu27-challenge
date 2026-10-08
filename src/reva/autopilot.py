"""One unattended cycle. GitHub Actions runs it twice an hour (.github/workflows/autopilot.yml).

    board snapshot -> poll open submissions -> collect a finished Kaggle job -> gated submit
    -> push the next queued lanes -> write STATUS.md

State lives in a directory that the workflow keeps on the `state` branch:
- active.json        the job running now: {kernel, pushed, sha, runs}
- jobs.jsonl         one row per finished job, with wall hours (the weekly GPU tally)
- runs.jsonl         one row per lane: run id, status, dev metrics, hours, config summary
- submissions.jsonl  one row per Codabench submission, updated by appending newer rows
- blocked.jsonl      runs the pre-upload checks stopped (reva.preflight), never retried
- arena.json         the run ids the arena last averaged (reva.arena), so it reruns only on new runs
- dev_clean.json     dev scores of runs recorded before dev left out train copies (data.unseen)
- board.csv          every leaderboard row ever seen
- STATUS.md          the human summary
The repo is public. Nothing here holds test predictions or probabilities.

Every lane's output is archived in a private Kaggle dataset under the account that ran it, named
reva-run-<run id>, one dataset version per session: config, logs (the lane's and the kernel's),
the training checkpoint, the finished adapter, dev predictions and train stats. Test predictions
stay out of it; they live only in the kernel's own output (CLAUDE.md compliance rules).
An unfinished run loses nothing: its next lane mounts that dataset and carries on
(reva.remote.restore). When the next lane runs on the other account, the dataset is copied there
first, since a private dataset mounts only in its owner's kernels. The Kaggle keys stay in the
workflow; no secret enters a kernel.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import shutil
import tarfile
import traceback
from pathlib import Path

from reva import arena, board, data, package, preflight, registry, remote, score
from reva import config as C
from reva.codabench import DONE, FAILED, KNOWN_REASONS, CodabenchError, scores
from reva.kaggle import DONE as K_DONE
from reva.kaggle import FAILED as K_FAILED
from reva.kaggle import KaggleError
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


def gpu_hours(jobs: list[dict], now: dt.datetime, days: int = 7, user: str | None = None) -> float:
    """GPU hours of jobs collected in the last `days`, for one account's kernels when `user` is set."""
    since = now - dt.timedelta(days=days)
    return sum(j.get("hours", 0) for j in jobs if parse_iso(j["collected"]) >= since
               and (user is None or j.get("kernel", "").split("/")[0].lower() == user.lower()))


MIN_JOB_HOURS = 4.0  # below this a job cannot train and still score dev and test
QUOTA_MARGIN_H = 0.5  # setup and our own rounding against Kaggle's count

NO_INTERNET = ("Could not resolve host", "Temporary failure in name resolution")


def no_internet(run: dict) -> bool:
    """A lane that died because its kernel had no internet. That is the account's fault, not the
    experiment's: Kaggle turns internet off for an account without a verified phone (7 Oct 2026)."""
    return run.get("status") != "ok" and any(s in run.get("error", "") for s in NO_INTERNET)


# A broken environment, not the experiment: the Kaggle image shipped torchao 0.10.0 on 8 Oct 2026,
# and peft 0.21.2 refuses to add LoRA next to an older torchao.
ENV_ERRORS = NO_INTERNET + ("Found an incompatible version of",)
ENV_FREE_TRIES = 3  # environment failures start to count after this many, so a bad fix cannot loop


def env_failure(run: dict) -> bool:
    return run.get("status") != "ok" and any(s in run.get("error", "") for s in ENV_ERRORS)


FREE_SESSIONS = 4  # a run that spans sessions (train.span_sessions) stops on its own; this caps a runaway


def failures(runs: list[dict]) -> dict[str, int]:
    """Failures that count against each run's retries. Environment failures are the account's or
    the image's fault, so the first ENV_FREE_TRIES of them are free. A session that a spanning run
    ended on purpose ("partial") is not a failure until there are more than FREE_SESSIONS of them."""
    own: dict[str, int] = {}
    env: dict[str, int] = {}
    part: dict[str, int] = {}
    for r in runs:
        if r["status"] != "ok":
            bucket = part if r["status"] == "partial" else env if env_failure(r) else own
            bucket[r["run_id"]] = bucket.get(r["run_id"], 0) + 1
    return {k: own.get(k, 0) + max(0, env.get(k, 0) - ENV_FREE_TRIES) + max(0, part.get(k, 0) - FREE_SESSIONS)
            for k in own.keys() | env.keys() | part.keys()}


def read_runs(state: Path) -> list[dict]:
    """Every run row: Kaggle lanes (runs.jsonl, with the arena's rows) and Colab sessions
    (colab_runs.jsonl, written by reva.colab). Two files, so the two workflows never edit the same one."""
    return registry.read(state / "runs.jsonl") + registry.read(state / "colab_runs.jsonl")


def kaggle_queue(queue: list[dict]) -> list[dict]:
    return [q for q in queue if q.get("backend", "kaggle") == "kaggle"]


# what reva.job leaves behind that a later session can carry on from
RESUMABLE = ("ckpt", "ckpt.old", "train.json", "dev_probs.part.json", "test_probs.part.json")


def stash_ref(user: str, run_id: str) -> str:
    """The private dataset that archives a run's output on `user`'s account."""
    return f"{user.lower()}/" + f"reva-run-{run_id}"[:50].rstrip("-").lower()


def test_predictions(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
    """tarfile filter: drop test probabilities and submission zips (they stay in kernel output only)."""
    name = Path(info.name).name
    return None if name.startswith("test_probs") or name.endswith(".zip") else info


def stash(kaggle, run_dir: Path, ref: str, work: Path, kernel_log: Path | None = None, message: str = "",
          keep_test: bool = False) -> None:
    """Upload a run's output folder, as one tar that Kaggle unpacks, as a new version of the private
    dataset `ref`. Earlier versions stay: each holds one session's state and logs. Test predictions
    stay out unless `keep_test`: a Colab run has no kernel output, so its dataset is its store."""
    folder = work / "stash" / ref.replace("/", "--")
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    if kernel_log and kernel_log.exists():
        shutil.copy(kernel_log, run_dir / "kernel.log")
    with tarfile.open(folder / f"{run_dir.name}.tar", "w") as t:
        t.add(run_dir, arcname=run_dir.name, filter=None if keep_test else test_predictions)
    kaggle.dataset_upload(folder, ref, message or f"{run_dir.name} state")


def staged(kaggle, runs: list[dict], lanes: list[dict], user: str, work: Path, notes: list[str]) -> dict[str, str]:
    """{run_id: dataset} for every lane with saved state, the newest kept. State saved by another
    account is copied into `user`'s first. A copy that fails costs the progress, not the job."""
    out = {}
    for lane in lanes:
        saved = [r["stash"] for r in runs if r["run_id"] == lane["run_id"] and r.get("stash") and r.get("resumable")]
        if not saved:
            continue
        src, ref = saved[-1], stash_ref(user, lane["run_id"])
        try:
            if src.split("/")[0].lower() != user.lower():
                copy = work / "stash-copy" / lane["run_id"]
                shutil.rmtree(copy, ignore_errors=True)
                kaggle.dataset_download(src, copy)
                stash(kaggle, copy / lane["run_id"], ref, work)
                notes.append(f"copied {lane['run_id']}'s saved state from {src} to {ref}")
            out[lane["run_id"]] = ref
        except Exception as e:
            notes.append(f"could not stage {lane['run_id']}'s saved state from {src}; it starts fresh: {public(e)}")
    return out


def offline_until(runs: list[dict], user: str, hours: float = 6) -> dt.datetime | None:
    """When to try `user` again if its latest lane had no internet, else None."""
    mine = [r for r in runs if r.get("kernel", "").split("/")[0].lower() == user.lower() and r.get("finished")]
    if not mine:
        return None
    last = max(mine, key=lambda r: r["finished"])
    return parse_iso(last["finished"]) + dt.timedelta(hours=hours) if no_internet(last) else None


def public(e: Exception) -> str:
    """Error text that is safe for STATUS.md, which is public. Codabench errors end with the
    response body after the status code, e.g. "submission create failed (400): {...}". Keep the
    part up to the status code and drop the body. A known Codabench message (codabench.KNOWN_REASONS)
    is added back, since it names the cause and carries nothing else from the response."""
    text = str(e)
    m = re.match(r"(.*?\(\d{3}\))", text, re.S)
    reason = getattr(e, "reason", None)
    known = reason in KNOWN_REASONS  # exact match only: callers also pass other exception types
    return (m.group(1) if m else text) + (f": {reason}" if m and known else "")


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
    best = max((s["dev_weighted"] for s in used if s.get("dev_weighted") is not None), default=None)
    if best is not None and mine < best + C.get(cfg, "submit.min_gain"):
        return False, f"dev {mine:.4f} does not beat best submitted {best:.4f} by {C.get(cfg, 'submit.min_gain')}"
    return True, f"dev {mine:.4f}" + (f" vs best submitted {best:.4f}" if best is not None else " (first submission)")


def candidate(runs: list[dict], subs: list[dict], blocked: set[str] = frozenset()) -> dict | None:
    """Best finished full-test run by weighted dev accuracy that has no live submission yet and
    was not stopped by the pre-upload checks. A run without a fair dev score is never picked."""
    taken = {s["run_id"] for s in subs if s["status"] not in FAILED} | set(blocked)
    ok = [r for r in runs if r["status"] == "ok" and r.get("zip") and r["run_id"] not in taken
          and "weighted_accuracy" in (r.get("metrics") or {})]
    return max(ok, key=lambda r: r["metrics"]["weighted_accuracy"], default=None)


def rescore(kaggle, runs: list[dict], state: Path, work: Path, cfg: dict) -> dict:
    """Dev scores, on today's dev set, of the runs recorded before it (data.DEV_SET), from their
    private dev_probs.json, kept in dev_clean.json. Today's dev set is a subset of the old one, so
    nothing is predicted again. A run whose probabilities are gone (its kernel slug was reused
    before 8 Oct 2026) is marked missing: it has no fair score and is never picked or compared.
    Download errors raise, and the cycle tries again next time."""
    path = state / "dev_clean.json"
    done = json.loads(path.read_text()) if path.exists() else {}
    todo = [r for r in runs if r["status"] == "ok" and r.get("dev_set") != data.DEV_SET and r["run_id"] not in done]
    if not todo:
        return done
    ann = work / "annotations"
    data.fetch_annotations(ann, C.get(cfg, "data.hf_repo"))
    sp = data.make_splits(ann, C.get(cfg, "dev.holdout_frac"), C.get(cfg, "dev.seed"))
    for r in todo:
        try:
            probs = arena.mean([arena.load_probs(kaggle, m, work, "dev_probs.json") for m in r.get("members") or [r]])
        except FileNotFoundError:
            done[r["run_id"]] = {"missing": "dev_probs.json is no longer in the kernel output"}
            continue
        if any(q["qa_id"] not in probs for q in sp["dev"]):
            done[r["run_id"]] = {"missing": "scored on a dev set that does not cover today's"}
            continue
        preds = {q["qa_id"]: data.LETTERS[max(range(4), key=probs[q["qa_id"]].__getitem__)] for q in sp["dev"]}
        done[r["run_id"]] = {"metrics": score.summary(sp["dev"], preds, sp["test"], C.get(cfg, "dev.min_cell"))}
        path.write_text(json.dumps(done, indent=1))
    path.write_text(json.dumps(done, indent=1))
    return done


def fair(runs: list[dict], clean: dict) -> list[dict]:
    """runs with dev metrics on today's dev set. An older run takes its score from `clean`
    (rescore); one without a score there keeps its row with empty metrics: shown, never picked."""
    out = []
    for r in runs:
        if r.get("status") == "ok" and r.get("dev_set") != data.DEV_SET:
            r = {**r, "metrics": clean.get(r["run_id"], {}).get("metrics", {})}
        out.append(r)
    return out


def fair_subs(subs: list[dict], runs: list[dict]) -> list[dict]:
    """Submissions with dev_weighted from fair(runs): None when the run has no fair dev score."""
    w = {r["run_id"]: r["metrics"].get("weighted_accuracy") for r in runs if r.get("status") == "ok"}
    return [{**s, "dev_weighted": w.get(s["run_id"])} for s in subs]


def submission_zip(kaggle, run: dict, fmt: str, cfg: dict, work: Path) -> tuple[Path, list[dict], dict]:
    """Rebuild the run's zip in the chosen layout from its private test_probs.json (re-downloaded
    from the run's own kernel when this runner has not got it; averaged over the members for an
    ensemble), validated against test.json. Returns (zip, test rows, test probabilities) for the
    pre-upload checks."""
    ann = work / "annotations"
    data.fetch_annotations(ann, C.get(cfg, "data.hf_repo"))
    test = data.load_split(ann, "test")
    members = run.get("members") or [run]
    loaded = [arena.load_probs(kaggle, m, work, "test_probs.json") for m in members]
    for m, p in zip(members, loaded):  # each member, before anything is averaged or derived from them
        if problems := preflight.probs_problems(p, test):
            raise preflight.Blocked([f"{m['run_id']}: {x}" for x in problems] if run.get("members") else problems)
    probs = arena.mean(loaded)
    preds = {q: data.LETTERS[max(range(4), key=p.__getitem__)] for q, p in probs.items()}
    zip_path = package.write(work / "submit" / f"{run['run_id']}.zip", test, preds, fmt, data.load_metadata(ann, "test"))
    return zip_path, test, probs


def run_config(kaggle, run: dict, work: Path) -> dict:
    """The full config a run was made with, from the run.json in its private kernel output."""
    return json.loads(arena.fetch(kaggle, run, work, "run.json").read_text())["config"]


def checked_zip(kaggle, run: dict, fmt: str, cfg: dict, work: Path, fetch_columns=None) -> Path:
    """The run's zip, rebuilt and passed through every pre-upload check (reva.preflight).

    Raises preflight.Blocked when the run itself must never be uploaded, and CodabenchError when
    nothing may be uploaded right now (the board's metric changed, or could not be read)."""
    fetch_columns = fetch_columns or (lambda: preflight.live_columns(
        C.get(cfg, "competition.id"), C.get(cfg, "competition.base")))
    try:
        columns = fetch_columns()
    except Exception:  # a fixed note: STATUS.md is public and error text may hold response bodies
        raise CodabenchError("could not read the live leaderboard columns; trying again next cycle") from None
    if problem := preflight.metric_problem(columns):
        raise CodabenchError(problem)
    zip_path, test, probs = submission_zip(kaggle, run, fmt, cfg, work)
    configs = [run_config(kaggle, m, work) for m in run.get("members") or [run]]  # every member must comply
    if problems := preflight.check(zip_path, test, probs, fmt, configs):
        raise preflight.Blocked(problems)
    return zip_path


def summarize_config(cfg: dict) -> dict:
    keys = ["model.id", "model.load_4bit", "model.use_video", "frames.n", "frames.max_side", "train.enabled",
            "train.epochs", "train.max_samples", "train.refit", "train.refit_with_val", "infer.perms"]
    return {k: C.get(cfg, k) for k in keys}


def run_row(run_dir: Path, kernel: str, sha: str, now: dt.datetime, log: str = "") -> dict:
    """The runs.jsonl row for one lane's output folder: ok with its metrics, partial (a spanning
    run's session ended), or failed with its log tail."""
    run_id = run_dir.name
    rj = run_dir / "run.json"
    if rj.exists():
        r = json.loads(rj.read_text())
        return {"run_id": run_id, "status": "ok", "kernel": kernel, "metrics": r["metrics"], "dev_set": r.get("dev_set"),
                "hours": r["hours"], "zip": r["zip"], "n": r["n"], "train": r["train"],
                "timings": r["timings"], "versions": r["versions"], "why": r["config"].get("why", ""),
                "config": summarize_config(r["config"]), "finished": r["finished"], "sha": sha}
    lane_log = run_dir / "log.txt"
    text = lane_log.read_text(errors="replace") if lane_log.exists() else log
    partial = run_dir / "partial.json"
    row = {"run_id": run_id, "status": "partial" if partial.exists() else "failed", "kernel": kernel,
           "finished": iso(now), "sha": sha, "error": tail(text, 30)}
    if partial.exists():
        row["where"] = json.loads(partial.read_text()).get("where", "")
    if any((run_dir / n).exists() for n in RESUMABLE):
        row["resumable"] = True
    return row


def collect(kaggle, active: dict, work: Path, now: dt.datetime) -> tuple[list[dict], dict, str]:
    """Download a finished job. Returns (run rows, job row, log tail)."""
    dest = work / active["kernel"].split("/")[-1]
    files, log = kaggle.output(active["kernel"], dest)
    rows = [run_row(dest / run_id, active["kernel"], active["sha"], now, log) for run_id in active["runs"]]
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
        lines += ["", f"## Calibration (board minus local weighted dev, dev set {data.DEV_SET})", "",
                  "| run | dev weighted | board | diff |", "|---|---|---|---|"]
        for s in (s for s in fin if s.get("dev_weighted") is not None):
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


def colab_note(state: Path) -> str | None:
    """One line on the Colab session in progress, from colab_live.json (reva.colab)."""
    try:
        live = json.loads((state / "colab_live.json").read_text())
    except (OSError, ValueError):
        return None
    if live.get("stage") == "done":
        return None
    return (f"Colab: {live.get('run_id')} on {live.get('gpu')}, {live.get('hours_in', 0):.1f} h in at "
            f"{live.get('updated')}, {live.get('units_before')} units at start; last log line: {live.get('last') or '-'}")


def cycle(cfg: dict, queue: list[dict], state: Path, work: Path, kaggle, sha: str, kaggle_users: str | list[str],
          client=None, auto_submit: bool = False, now: dt.datetime | None = None,
          fetch_board=None, push: bool = True, notes: list[str] | None = None, fetch_columns=None) -> dict:
    """One step. `client` is a logged-in reva.codabench.Client or None (then nothing is submitted).
    `kaggle_users` are the Kaggle accounts in priority order; a job goes to the first with GPU quota left."""
    now = now or dt.datetime.now(dt.timezone.utc)
    state.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    notes = list(notes or [])  # e.g. the Codabench login result from the caller
    out = {"collected": [], "submitted": [], "pushed": None, "needs_fix": False, "waiting": False}

    # 1. leaderboard
    fetch_board = fetch_board or (lambda: board.fetch(C.get(cfg, "competition.phase"), C.get(cfg, "competition.base")))
    try:
        rows = fetch_board()
        new = board.snapshot(rows, state / "board.csv", iso(now))
        notes.append(f"leaderboard: {len(rows)} rows, {new} new; leader {rows[0]['owner']} "
                     f"{rows[0].get('overall_accuracy', 0):.4f}" if rows else "leaderboard empty")
    except Exception as e:  # the board is information, never a reason to stop the loop
        rows = []
        notes.append(f"leaderboard fetch failed: {public(e)}")

    # 2. open submissions
    subs = latest_submissions(registry.read(state / "submissions.jsonl"))
    if client:
        for s in subs:
            if s["status"] not in DONE | FAILED:
                try:
                    rec = client.submission(s["submission_id"])
                except CodabenchError as e:
                    notes.append(f"could not poll submission {s['submission_id']}: {public(e)}")
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
            kdest = work / active["kernel"].split("/")[-1]
            for r in fresh:
                if (kdest / r["run_id"]).is_dir():  # archive it; an unfinished run's next lane resumes from it
                    ref = stash_ref(active["kernel"].split("/")[0], r["run_id"])
                    try:
                        stash(kaggle, kdest / r["run_id"], ref, work, kdest / f"{kdest.name}.log",
                              f"{active['kernel']} {r['status']}")
                        r["stash"] = ref
                        notes.append(f"saved {r['run_id']} ({r['status']}) to private dataset {ref}")
                    except Exception as e:  # the kernel output still holds it; record the run regardless
                        notes.append(f"could not save {r['run_id']}'s state: {public(e)}")
                registry.append(state / "runs.jsonl", r)
            registry.append(state / "jobs.jsonl", job)
            active_path.unlink()
            active = None
            out["collected"] = [r["run_id"] for r in fresh]
            bad = [r["run_id"] for r in fresh if r["status"] not in ("ok", "partial")]
            if bad or kstate in K_FAILED:
                out["needs_fix"] = True
                out["log_tail"] = log
                notes.append(f"job {job['kernel']} {kstate}; failed lanes: {bad} {message}")
            notes.append(f"collected {out['collected']}")
        else:
            notes.append(f"job {active['kernel']} is {kstate}")
    live = colab_note(state)
    if live:
        notes.append(live)

    # 3a. every run on the same dev set: older runs are rescored once (data.unseen)
    clean_path = state / "dev_clean.json"
    try:
        clean, settled = rescore(kaggle, read_runs(state), state, work, cfg), True
    except Exception as e:  # never stops the loop; nothing is submitted until every run is comparable
        traceback.print_exc()
        clean, settled = json.loads(clean_path.read_text()) if clean_path.exists() else {}, False
        notes.append(f"dev rescoring incomplete this cycle: {type(e).__name__}")

    # 3b. arena: an average of the best runs competes with them on dev
    runs = fair(read_runs(state), clean)
    seen_path = state / "arena.json"
    seen = json.loads(seen_path.read_text())["considered"] if seen_path.exists() else []
    if len(arena.eligible(runs)) >= 2 and sorted(r["run_id"] for r in arena.eligible(runs)) != sorted(seen):
        try:
            ann = work / "annotations"
            data.fetch_annotations(ann, C.get(cfg, "data.hf_repo"))
            sp = data.make_splits(ann, C.get(cfg, "dev.holdout_frac"), C.get(cfg, "dev.seed"))
            row, considered, note = arena.step(kaggle, runs, sp["dev"], sp["test"], work, seen, iso(now),
                                               C.get(cfg, "dev.min_cell"))
            if row:
                registry.append(state / "runs.jsonl", row)
            seen_path.write_text(json.dumps({"considered": considered, "when": iso(now)}))
            if note:
                notes.append(note)
        except Exception as e:  # the arena adds a candidate; it never stops the loop
            traceback.print_exc()  # the full error goes to the Actions log; STATUS.md gets the type only
            notes.append(f"arena skipped this cycle: {type(e).__name__}")

    # 4. gated submission: the best run not yet submitted, from any cycle
    blocked = {b["run_id"] for b in registry.read(state / "blocked.jsonl")}
    runs = fair(read_runs(state), clean)
    r = candidate(runs, subs, blocked)
    if r:
        fsubs = fair_subs(subs, runs)
        notes.append(preflight.forecast(r, fsubs, rows[0].get("overall_accuracy") if rows else None))
        allowed, why = gate(r, fsubs, cfg, now) if settled else (False, "older runs' dev scores not settled yet")
        if not allowed or not auto_submit or not client:
            notes.append(f"not submitting {r['run_id']}: "
                         f"{why if not allowed else 'automatic submission off' if not auto_submit else 'no Codabench login'}")
        else:
            try:
                can, reason = client.can_submit(C.get(cfg, "competition.phase"))
                if not can:
                    raise CodabenchError(f"Codabench refuses submissions for this account: {reason}")
                fmt = pick_format(subs, cfg)
                try:
                    zip_path = checked_zip(kaggle, r, fmt, cfg, work, fetch_columns)
                except preflight.Blocked as b:
                    registry.append(state / "blocked.jsonl", {"run_id": r["run_id"], "when": iso(now), "problems": b.problems})
                    raise CodabenchError(f"pre-upload checks failed: {b}") from None
                org = submit_as(client, cfg)
                sid = client.submit(zip_path, C.get(cfg, "competition.id"), C.get(cfg, "competition.phase"),
                                    [C.get(cfg, "competition.task")], organization=org)
                rec = client.wait(sid, timeout_s=60 * C.get(cfg, "submit.wait_minutes"))
                row = {"run_id": r["run_id"], "kernel": r.get("kernel"), "submission_id": sid, "submitted": iso(now),
                       "status": rec.get("status"), "scores": scores(rec), "format": fmt,
                       "organization": C.get(cfg, "competition.organization"),
                       "dev_weighted": r["metrics"]["weighted_accuracy"], "dev_overall": r["metrics"]["overall_accuracy"]}
                if r.get("members"):
                    row["members"] = [m["run_id"] for m in r["members"]]
                registry.append(state / "submissions.jsonl", row)
                out["submitted"].append(row)
                notes.append(f"submitted {r['run_id']} as {sid} ({fmt}): {why}; status {row['status']}")
            except CodabenchError as e:
                out["submit_error"] = public(e)
                notes.append(f"submission of {r['run_id']} not made: {public(e)}")

    # 5. push the next lanes
    subs = latest_submissions(registry.read(state / "submissions.jsonl"))  # include this cycle's submission
    runs = read_runs(state)
    jobs = registry.read(state / "jobs.jsonl")
    if push and not active:
        done = {r["run_id"] for r in runs if r["status"] == "ok"}
        failed = failures(runs)  # an offline account or a broken image does not use up a lane's retries
        lanes = remote.pending(cfg, kaggle_queue(queue), done, failed)[: C.get(cfg, "remote.lanes")]
        users = [kaggle_users] if isinstance(kaggle_users, str) else list(kaggle_users)
        if not lanes:
            notes.append("queue empty: add experiments to configs/queue.yaml")
        for n, user in enumerate(users if lanes else [], 1):
            retry = offline_until(runs, user)
            if retry and retry > now:
                notes.append(f"Kaggle account {n} kernels have no internet (is its phone number verified?); "
                             f"next try after {iso(retry)}")
                out["resting"] = True  # keep cycling: nothing else starts the loop again at the retry time
                continue
            used = gpu_hours(jobs, now, user=user)
            # a shorter job on the hours left beats an idle account; training sizes itself to it
            hours = min(C.get(cfg, "remote.max_hours"), C.get(cfg, "remote.weekly_gpu_hours") - used - QUOTA_MARGIN_H)
            if hours < MIN_JOB_HOURS:
                notes.append(f"GPU quota pacing on Kaggle account {n}: {used:.1f} h used in 7 days")
                continue
            kdir = work / "kernel"
            resume = staged(kaggle, runs, lanes, user, work, notes)
            slug = remote.build(cfg, lanes, sha, kdir, user, hours=hours, stamp=now.strftime("%m%d%H%M"),
                                resume=resume)
            try:
                pushed = kaggle.push(kdir, timeout_s=int(3600 * hours),
                                     accelerator=C.get(cfg, "remote.accelerator"))
            except KaggleError as e:
                # The weekly GPU quota (hit on 7 Oct 2026) clears by itself: try the next account, and
                # if none is left keep the loop alive and retry each cycle. Any other refusal
                # (credentials, metadata) needs a fix: fail loudly.
                if "gpu quota" not in str(e).lower():
                    raise
                out["push_refused"] = True
                notes.append(f"Kaggle weekly GPU quota reached on account {n}")
                continue
            active = {"kernel": slug, "pushed": iso(now), "sha": sha, "runs": [c["run_id"] for c in lanes],
                      "url": pushed.url, "hours": round(hours, 2)}
            if resume:
                active["resume"] = resume
            active_path.write_text(json.dumps(active, indent=1))
            out["pushed"] = active
            out.pop("push_refused", None)
            out.pop("resting", None)
            notes.append(f"pushed {slug} with {active['runs']}"
                         + (f", resuming {sorted(resume)}" if resume else "")
                         + (f", sized to the {hours:.1f} h account {n} has left this week"
                            if hours < C.get(cfg, "remote.max_hours") else ""))
            break
        else:
            if lanes:
                notes.append("no Kaggle account can take the next job; it waits")

    # something will change without a push: a Kaggle job still running or a submission still being
    # scored. The workflow starts the next cycle itself while this holds.
    # A refused push also counts: nothing else would start the loop again once the quota resets.
    out["waiting"] = (bool(active) or out.get("push_refused", False) or out.get("resting", False)
                      or any(s["status"] not in DONE | FAILED for s in subs))
    shown = fair(runs, clean)
    (state / "STATUS.md").write_text(status_md(cfg, rows, shown, fair_subs(subs, shown), active, jobs, now, notes))
    out["notes"] = notes
    return out
