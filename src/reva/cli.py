"""Command line: `python -m reva.cli <command>` (or `reva <command>` after pip install).

    fetch                          download train/val/test annotations
    board                          live leaderboard and our gap (needs no login)
    smoke                          CPU dry run of the whole GPU job (tiny model, synthetic data)
    job --config C --out D         one experiment on this machine's GPU
    build --sha S --out D          write the Kaggle kernel for the next queued lanes, push nothing
    autopilot --state D            one unattended cycle (what GitHub Actions runs every hour)
    submit --run ID --state D      submit one finished run by hand (the owner's decision)
    validate ZIP                   check a submission zip against test.json

Codabench login comes from CODABENCH_USERNAME / CODABENCH_PASSWORD. Never pass it as an argument.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

from reva import config as C


def _cfg(a) -> dict:
    return C.override(C.load(a.config), C.parse_pairs(a.set or []))


def _client(cfg: dict):
    from reva.codabench import Client

    user, pw = os.environ.get("CODABENCH_USERNAME"), os.environ.get("CODABENCH_PASSWORD")
    return Client.login(user, pw, C.get(cfg, "competition.base")) if user and pw else None


def cmd_fetch(a) -> int:
    from reva import data

    cfg = _cfg(a)
    for p in data.fetch_annotations(C.get(cfg, "data.root"), C.get(cfg, "data.hf_repo")):
        print(p)
    return 0


def cmd_board(a) -> int:
    from reva import board

    cfg = _cfg(a)
    rows = board.fetch(C.get(cfg, "competition.phase"), C.get(cfg, "competition.base"))
    for i, r in enumerate(rows, 1):
        print(f"{i:>2} {r['owner'][:20]:<20} {r.get('overall_accuracy', 0):.4f} {r['created'][:16]}")
    return 0


def cmd_build(a) -> int:
    from reva import registry, remote

    cfg = _cfg(a)
    runs = registry.read(Path(a.state) / "runs.jsonl") if a.state else []
    lanes = remote.pending(cfg, remote.load_queue(a.queue), {r["run_id"] for r in runs if r["status"] == "ok"}, {})
    lanes = lanes[: C.get(cfg, "remote.lanes")]
    print(remote.build(cfg, lanes, a.sha, a.out, a.user), [c["run_id"] for c in lanes])
    return 0


def cmd_autopilot(a) -> int:
    from reva import autopilot, remote
    from reva.kaggle import from_env

    cfg = _cfg(a)
    auto = os.environ.get("AUTO_SUBMIT", "on").lower() != "off"  # on unless the owner sets it off
    try:
        client = _client(cfg)
        login = "Codabench login ok" if client else "Codabench secrets not set"
        if client:
            can, why = client.can_submit(C.get(cfg, "competition.phase"))
            login += "; account may submit" if can else f"; Codabench refuses submissions: {why}"
    except Exception as e:  # a bad login must not stop the GPU side of the loop
        client, login = None, f"Codabench login failed: {autopilot.public(e)}"  # STATUS.md is public
    if client and (org := C.get(cfg, "competition.organization")):
        try:
            login += f"; submits as {org} (id {client.organization_id(org)})"
        except Exception:  # a fixed note: error text may hold response bodies, and STATUS.md is public
            login += f"; could not confirm organization {org}, so nothing will be submitted"
    kaggle = from_env()
    out = autopilot.cycle(cfg, remote.load_queue(a.queue), Path(a.state), Path(a.work), kaggle, a.sha,
                          getattr(kaggle, "users", [a.user]),
                          client=client, auto_submit=auto, push=not a.no_push, notes=[login])
    print("\n".join(out["notes"]))
    if a.outcome:
        Path(a.outcome).write_text(json.dumps(out, indent=1, default=str))
    return 0


def cmd_colab(a) -> int:
    """One Colab session (reva.colab): the next Colab queue entry, or with --probe a short GPU check."""
    import datetime as dt

    from reva import autopilot, colab, remote
    from reva.kaggle import from_env

    if a.pending:  # exit 0 when a Colab entry waits; the autopilot then starts a session
        runs = autopilot.read_runs(Path(a.state))
        nxt = colab.next_lane(_cfg(a), remote.load_queue(a.queue), runs)
        print(f"Colab entry pending: {nxt[0]['run_id']} on {nxt[1]}" if nxt else "no Colab entry pending")
        return 0 if nxt else 1
    session = colab.Colab()
    if a.probe:
        print(colab.probe(session, a.gpu or "T4"))
        return 0
    notes: list[str] = []
    out = colab.run_next(_cfg(a), remote.load_queue(a.queue), Path(a.state), Path(a.work), session, from_env(), a.sha,
                         a.hours, dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), notes)
    print("\n".join(notes))
    if a.outcome:
        Path(a.outcome).write_text(json.dumps({"ran": (out["ran"] or {}).get("run_id"), "more": out["more"],
                                               "status": (out["ran"] or {}).get("status"), "notes": notes}))
    return 0


def cmd_submit(a) -> int:
    """Owner-triggered submission of a finished run: rebuild its zip from its private kernel output,
    run every pre-upload check, submit, wait, record. Skips the dev-gain gate (the owner decided),
    never the budget or the pre-upload checks (reva.preflight)."""
    from reva import preflight, registry
    from reva.autopilot import checked_zip, iso, latest_submissions, pick_format, submit_as
    from reva.codabench import CodabenchError, scores
    from reva.kaggle import from_env

    cfg = _cfg(a)
    state = Path(a.state)
    run = next((r for r in registry.read(state / "runs.jsonl") if r["run_id"] == a.run and r["status"] == "ok"), None)
    if not run or not run.get("zip"):
        print(f"no finished run {a.run} with a zip in {state}/runs.jsonl")
        return 1
    subs = latest_submissions(registry.read(state / "submissions.jsonl"))
    used = len([s for s in subs if s["status"] not in ("Failed", "Cancelled")])
    if used >= C.get(cfg, "submit.total_budget"):
        print("submission budget used up")
        return 1
    client = _client(cfg)
    if not client:
        print("set CODABENCH_USERNAME and CODABENCH_PASSWORD")
        return 1
    fmt = pick_format(subs, cfg)
    if fmt is None:
        print("Codabench failed every predictions.json layout; the scorer needs a code fix")
        return 1
    try:
        zip_path = checked_zip(from_env(), run, fmt, cfg, Path(a.work))
    except (preflight.Blocked, CodabenchError) as e:
        print(f"not submitting {run['run_id']}: {e}")
        return 1
    sid = client.submit(zip_path, C.get(cfg, "competition.id"), C.get(cfg, "competition.phase"),
                        [C.get(cfg, "competition.task")], organization=submit_as(client, cfg))
    rec = client.wait(sid, timeout_s=60 * C.get(cfg, "submit.wait_minutes"))
    row = {"run_id": run["run_id"], "kernel": run.get("kernel"), "submission_id": sid,
           "submitted": iso(dt.datetime.now(dt.timezone.utc)), "status": rec.get("status"), "scores": scores(rec),
           "format": fmt, "dev_weighted": run["metrics"]["weighted_accuracy"],
           "dev_overall": run["metrics"]["overall_accuracy"], "manual": True}
    if run.get("members"):  # an arena ensemble: its zip is the mean of these runs' test probabilities
        row["members"] = [m["run_id"] for m in run["members"]]
    registry.append(state / "submissions.jsonl", row)
    print(f"submission {sid}: {rec.get('status')} {scores(rec)}")
    return 0


def cmd_validate(a) -> int:
    from reva import data, package

    cfg = _cfg(a)
    data.fetch_annotations(C.get(cfg, "data.root"), C.get(cfg, "data.hf_repo"))
    n = package.validate(a.zip, data.load_split(C.get(cfg, "data.root"), "test"), C.get(cfg, "submit.format"))
    print(f"VALID {n} answers")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="reva")
    ap.add_argument("--config", default=str(C.DEFAULT))
    ap.add_argument("--set", action="append", help="one dotted override per flag, e.g. --set frames.n=32")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch")
    sub.add_parser("board")
    sub.add_parser("smoke").add_argument("--out", default="work/smoke")
    j = sub.add_parser("job")
    j.add_argument("--config-json", dest="job_config", required=True)
    j.add_argument("--out", required=True)
    j.add_argument("--device", default="cuda:0")
    queue = str(Path(C.DEFAULT).with_name("queue.yaml"))
    b = sub.add_parser("build")
    b.add_argument("--sha", required=True)
    b.add_argument("--out", default="work/kernel")
    b.add_argument("--user", default=os.environ.get("KAGGLE_USERNAME", "sidharthkumarpradhan"))
    b.add_argument("--queue", default=queue)
    b.add_argument("--state", default=None)
    p = sub.add_parser("autopilot")
    p.add_argument("--state", required=True)
    p.add_argument("--work", default="work/autopilot")
    p.add_argument("--sha", required=True)
    p.add_argument("--user", default=os.environ.get("KAGGLE_USERNAME", "sidharthkumarpradhan"))
    p.add_argument("--queue", default=queue)
    p.add_argument("--outcome", default=None)
    p.add_argument("--no-push", action="store_true", help="collect and report only")
    c = sub.add_parser("colab")
    c.add_argument("--state", default="state")
    c.add_argument("--work", default="work/colab")
    c.add_argument("--queue", default="configs/queue.yaml")
    c.add_argument("--sha", default="HEAD")
    c.add_argument("--hours", type=float, default=5.0, help="session length; an Actions job lasts at most 6 h")
    c.add_argument("--gpu", default="", help="GPU for --probe")
    c.add_argument("--probe", action="store_true")
    c.add_argument("--pending", action="store_true", help="exit 0 if a Colab entry is pending, else 1")
    c.add_argument("--outcome")
    s = sub.add_parser("submit")
    s.add_argument("--run", required=True)
    s.add_argument("--state", required=True)
    s.add_argument("--work", default="work/submit")
    v = sub.add_parser("validate")
    v.add_argument("zip")
    a = ap.parse_args(argv)
    if a.cmd == "smoke":
        from reva import smoke

        return smoke.main(["--out", a.out])
    if a.cmd == "job":
        from reva import job

        return job.main(["--config", a.job_config, "--out", a.out, "--device", a.device])
    return {"fetch": cmd_fetch, "board": cmd_board, "build": cmd_build, "autopilot": cmd_autopilot, "colab": cmd_colab,
            "submit": cmd_submit, "validate": cmd_validate}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
