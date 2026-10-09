"""Colab Pro as a second GPU backend, through Google's Colab CLI (google-colab-cli 0.7.4, read
8 Oct 2026; owner's go-ahead the same day).

One GitHub Actions job holds one Colab session from start to end (.github/workflows/colab.yml):

    pick the next Colab entry of configs/queue.yaml -> open a GPU session -> upload the run's
    saved state, if any -> start reva.job in the background -> read its log every few minutes
    -> download its output when it ends or the time is up -> archive it -> record the run

Colab Pro has no background execution, so a session lives only while this job holds it, and an
Actions job lasts at most 6 h. A run that needs longer spans sessions (train.span_sessions) the
same way it does on Kaggle: it checkpoints at its session's end and the next session resumes it.
While the job runs, its folder is also saved every hour (snapshot_code), so a session that is
lost midway (the VM drops, the units run out) resumes from the last hour, not from its start.

Login: the owner ran the CLI's copy-paste OAuth flow once and stored the token it saved as the
COLAB_TOKEN secret; the workflow writes it to ~/.config/colab-cli/token.json and the CLI refreshes
it. Every call goes through an injectable runner, so tests need neither the CLI nor the network.

Outputs: a Colab session leaves nothing behind, so the whole output folder, test probabilities
included, goes to the private Kaggle dataset reva-run-<run id> on the first Kaggle account. That
dataset is the run's private store, as a kernel output is for a Kaggle run; nothing of it enters
this public repo (CLAUDE.md compliance rules). State, on the `state` branch:
- colab_runs.jsonl   one row per Colab session's run, the same schema as runs.jsonl
- colab_jobs.jsonl   one row per Colab session: run, GPU, hours, compute units before and after
- colab_live.json    the session in progress: run, GPU, hours in, the job's last log line. Pushed
                     every 20 minutes, because Actions shows a job's log to the API only once it ends.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Callable

from reva import config as C

Runner = Callable[[list[str], float | None], tuple[int, str]]

PATHS = {"data.root": "/content/reva", "data.video_root": "/content/reva", "frames.cache": "/content/frames"}
T, OUT = "/content/reva-job", "/content/out"
SETUP_H = 1.0  # session start, pip installs, videos and frame cache, and the final download


class ColabError(RuntimeError):
    pass


def subprocess_runner(cmd: list[str], timeout: float | None = None) -> tuple[int, str]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired as e:
        return 124, f"timed out after {timeout} s: {(e.stdout or '')[-500:]}"
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def parse_usage(text: str) -> tuple[float, float]:
    """(compute-unit balance, units per hour) from `colab usage`."""
    bal = re.search(r"Current balance:\s*([\d.]+)", text)
    rate = re.search(r"Usage rate:\s*([\d.]+)", text)
    if not bal:
        raise ColabError(f"unexpected `colab usage` output: {text.strip()[-300:]}")
    return float(bal.group(1)), float(rate.group(1)) if rate else 0.0


def redact(text: str) -> str:
    """Drop the runtime's proxy token from a CLI error: the run rows that keep the error are public."""
    return re.sub(r"(token=)[^&\s'\"]+", r"\1***", text)


class Colab:
    """The CLI calls the backend needs. Each one fails loudly on a non-zero exit."""

    def __init__(self, cli: list[str] | None = None, runner: Runner | None = None):
        self.cli = cli or ["colab"]
        self.runner = runner or subprocess_runner

    def _run(self, *args: str, timeout: float | None = None) -> str:
        code, out = self.runner([*self.cli, *args], timeout)
        if code != 0:
            raise ColabError(f"`colab {args[0]}` exited {code}: {redact(out.strip())[-800:]}")
        return out

    def usage(self) -> tuple[float, float]:
        return parse_usage(self._run("usage", timeout=120))

    def new(self, name: str, gpu: str) -> None:
        out = self._run("new", "-s", name, "--gpu", gpu, timeout=900)
        if "READY" not in out:
            raise ColabError(f"session {name} on {gpu} did not start: {out.strip()[-500:]}")

    def exec(self, name: str, code: str, timeout: float = 120) -> str:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(code)
        try:
            return self._run("exec", "-s", name, "-f", f.name, "--timeout", str(timeout), timeout=timeout + 120)
        finally:
            Path(f.name).unlink(missing_ok=True)

    def upload(self, name: str, local: Path, remote: str) -> None:
        self._run("upload", "-s", name, str(local), remote, timeout=3600)

    def download(self, name: str, remote: str, local: Path) -> None:
        Path(local).parent.mkdir(parents=True, exist_ok=True)
        self._run("download", "-s", name, remote, str(local), timeout=3600)

    def stop(self, name: str) -> None:
        self._run("stop", "-s", name, timeout=300)


def lane_config(base: dict, lane: dict, hours: float, final: bool = False) -> dict:
    """The run's config for a Colab session of `hours`: Colab paths, and the job budget the
    session leaves after setup. In the `final` session the units allow, a run that spans sessions
    stops training in time to predict (train.final_session). The run id does not change."""
    over = {**PATHS, "job.max_hours": round(min(C.get(lane, "job.max_hours"), hours - SETUP_H), 2)}
    if final and C.get(lane, "train.span_sessions", False):
        over["train.final_session"] = True
    return C.override(lane, over)


def start_code(cfg: dict, sha: str, repo: str, pip: list[str]) -> str:
    """Python run in the session's kernel: start the whole job in the background and return."""
    run_id = cfg["run_id"]
    steps = " && ".join([
        f"rm -rf {T}/repo && git init -q {T}/repo && cd {T}/repo && git fetch -q --depth 1 {repo} {sha}"
        " && git checkout -q FETCH_HEAD",
        "pip install -q " + " ".join(f"'{p}'" for p in pip),
        # as on Kaggle: peft refuses LoRA next to an old torchao, and nothing here uses it
        "(pip uninstall -y -q torchao || true)",
        f"pip install -q --no-deps -e {T}/repo",
        f"cd {T}/repo && python -m reva.job --prepare-only --config {T}/run.json --out {OUT}/{run_id}",
        f"cd {T}/repo && python -m reva.job --config {T}/run.json --out {OUT}/{run_id}",
    ])
    shell = f"( set -ex; {steps} ) >> {OUT}/{run_id}/log.txt 2>&1; echo $? > {OUT}/{run_id}/exit_code"
    return f'''import json, os, subprocess
os.makedirs({T!r}, exist_ok=True)
os.makedirs({OUT + "/" + run_id!r}, exist_ok=True)
for name in ("exit_code",):
    p = os.path.join({OUT + "/" + run_id!r}, name)
    if os.path.exists(p):
        os.remove(p)
open({T + "/run.json"!r}, "w").write({json.dumps(json.dumps(cfg))})
env = {{**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "TOKENIZERS_PARALLELISM": "false"}}
proc = subprocess.Popen(["bash", "-c", {shell!r}], env=env, start_new_session=True)
print("STARTED", proc.pid)
'''


def poll_code(run_id: str) -> str:
    return f'''import os
d = {OUT + "/" + run_id!r}
log = os.path.join(d, "log.txt")
print("".join(open(log, errors="replace").readlines()[-25:]) if os.path.exists(log) else "(no log yet)")
code = os.path.join(d, "exit_code")
print("EXIT_CODE", open(code).read().strip() if os.path.exists(code) else "running")
'''


def tar_code(run_id: str) -> str:
    return f'''import subprocess
subprocess.run(["tar", "-cf", "/content/out.tar", "-C", {OUT!r}, {run_id!r}], check=True)
print("TARRED")
'''


def snapshot_code(run_id: str) -> str:
    """Tar a consistent copy of the running job's folder. A hard-link copy is instant and keeps
    the files as they were: a checkpoint save writes new files and renames folders, never edits
    one in place. Retried when a save swapped folders mid-copy. Nothing is sent before the job
    has a whole checkpoint, or has finished training."""
    return f'''import os, shutil, subprocess, time
d, snap = {OUT + "/" + run_id!r}, "/content/snap"
s = os.path.join(snap, {run_id!r})
def whole():
    return os.path.exists(os.path.join(s, "train.json")) or any(
        os.path.isfile(os.path.join(s, c, "state.pt")) and os.path.isdir(os.path.join(s, c, "adapter"))
        for c in ("ckpt", "ckpt.old"))
ok = False
for _ in range(3):
    shutil.rmtree(snap, ignore_errors=True)
    os.makedirs(snap)
    if subprocess.run(["cp", "-al", d, snap + "/"]).returncode != 0:
        subprocess.run(["cp", "-a", d, snap + "/"])
    if whole():
        ok = True
        break
    time.sleep(10)
if ok:
    shutil.rmtree(os.path.join(s, "ckpt.tmp"), ignore_errors=True)
    # exit 1 only means a file such as the log grew while it was read
    if subprocess.run(["tar", "-cf", "/content/snap.tar", "-C", snap, {run_id!r}]).returncode > 1:
        ok = False
shutil.rmtree(snap, ignore_errors=True)
print("SNAPPED" if ok else "NOTHING TO SAVE YET")
'''


def restore_code(run_id: str, parts: int, size: int) -> str:
    """Join the uploaded parts (upload_parts) into the saved state's tar and unpack it."""
    return f'''import os, tarfile
tar = "/content/restore.tar"
with open(tar, "wb") as out:
    for i in range({parts}):
        part = tar + ".part%04d" % i
        with open(part, "rb") as f:
            out.write(f.read())
        os.remove(part)
if os.path.getsize(tar) != {size}:
    raise SystemExit("restore.tar has %d bytes, not {size}" % os.path.getsize(tar))
os.makedirs({OUT!r}, exist_ok=True)
with tarfile.open(tar) as t:
    t.extractall({OUT!r}, filter="data")
os.remove(tar)
print("RESTORED", sorted(os.listdir({OUT + "/" + run_id!r})))
'''


PART_BYTES = 32 << 20


def upload_parts(colab: Colab, name: str, local: Path, remote: str, part_bytes: int = PART_BYTES,
                 tries: int = 3) -> int:
    """Upload `local` as `remote`.part0000, .part0001, ... and return how many parts it took.
    The CLI sends a file as one base64 JSON request, and the runtime dropped a 480 MB checkpoint
    sent that way twice (SSL EOF, 9 Oct 2026). Parts of 32 MiB stay well under any request limit,
    and each is tried `tries` times."""
    n = 0
    with open(local, "rb") as f:
        while chunk := f.read(part_bytes):
            part = Path(f"{local}.part{n:04d}")
            part.write_bytes(chunk)
            try:
                for attempt in range(tries):
                    try:
                        colab.upload(name, part, f"{remote}.part{n:04d}")
                        break
                    except ColabError:
                        if attempt == tries - 1:
                            raise
            finally:
                part.unlink(missing_ok=True)
            n += 1
    return n


LOST_AFTER = 3  # failed polls in a row (15 minutes) that mean the session is gone


def exit_code(poll: str) -> int | None:
    m = re.search(r"EXIT_CODE (\S+)", poll)
    return int(m.group(1)) if m and m.group(1).lstrip("-").isdigit() else None


def session(colab: Colab, cfg: dict, sha: str, repo: str, pip: list[str], gpu: str, hours: float, work: Path,
            restore: Path | None = None, poll_s: float = 300, sleep=time.sleep, clock=time.time,
            log=print, progress: Callable[..., None] | None = None, save: Callable[[Path], None] | None = None,
            save_every_s: float = 3600) -> tuple[Path | None, str, float, bool]:
    """Run one lane in one Colab session. Returns (the downloaded output folder or None, the last
    log tail, wall hours, whether the job ended by itself). The session is always stopped.

    Every `save_every_s` while the job runs, a snapshot of its folder is downloaded and handed to
    `save` (run_next archives it). A failed snapshot only warns: the job keeps running."""
    run_id, t0 = cfg["run_id"], clock()
    name = f"reva-{run_id}"[:40]
    report = progress or (lambda **kw: None)
    colab.new(name, gpu)
    tail, ended = "", False
    try:
        if restore is not None:  # the run's saved state, where reva.job looks for it
            parts = upload_parts(colab, name, restore, "/content/restore.tar")
            out = colab.exec(name, restore_code(run_id, parts, restore.stat().st_size), timeout=600).strip()
            if "RESTORED" not in out:
                raise ColabError(f"could not restore {run_id}'s saved state: {out[-500:]}")
            log(out)
        log(colab.exec(name, start_code(cfg, sha, repo, pip), timeout=120).strip())
        report(stage="started", hours_in=(clock() - t0) / 3600, last="")
        end = t0 + 3600 * hours - 1200  # leave 20 minutes to download and archive
        saved_at, misses = clock(), 0
        while True:
            sleep(poll_s)
            try:
                tail = colab.exec(name, poll_code(run_id), timeout=120)
                misses = 0
            except ColabError as e:  # one failed read is not a lost session; three in a row are
                log(f"poll failed: {e}")
                misses += 1
                if misses >= LOST_AFTER or clock() > end:
                    raise
                continue
            last = tail.strip().splitlines()[-2] if len(tail.strip().splitlines()) > 1 else "(no log yet)"
            log(last)
            ended = exit_code(tail) is not None
            report(stage="ended" if ended else "running", hours_in=(clock() - t0) / 3600, last=last)
            if ended or clock() > end:
                break
            if save and clock() - saved_at >= save_every_s:
                saved_at = clock()
                snapshot(colab, name, run_id, work, save, log)
        colab.exec(name, tar_code(run_id), timeout=1200)
        local = work / "colab" / f"{run_id}.tar"
        colab.download(name, "/content/out.tar", local)
        shutil.rmtree(work / "colab" / run_id, ignore_errors=True)
        with tarfile.open(local) as t:
            t.extractall(work / "colab", filter="data")
        return work / "colab" / run_id, tail, (clock() - t0) / 3600, ended
    except ColabError as e:
        log(f"session failed: {e}")
        return None, tail + f"\n{e}", (clock() - t0) / 3600, False
    finally:
        try:
            colab.stop(name)
        except ColabError as e:  # an unstopped session burns units: say so loudly
            log(f"WARNING could not stop {name}: {e}")


def snapshot(colab: Colab, name: str, run_id: str, work: Path, save: Callable[[Path], None], log=print) -> bool:
    """Download a snapshot of the running job's folder and hand it to `save`. Never raises."""
    try:
        if "SNAPPED" not in colab.exec(name, snapshot_code(run_id), timeout=1200):
            return False
        local, folder = work / "colab" / f"{run_id}.snap.tar", work / "colab" / "snap"
        colab.download(name, "/content/snap.tar", local)
        shutil.rmtree(folder, ignore_errors=True)
        with tarfile.open(local) as t:
            t.extractall(folder, filter="data")
        save(folder / run_id)
        log(f"saved a snapshot of {run_id}")
        return True
    except Exception as e:  # the job is still running; the next snapshot or the session's end saves it
        log(f"WARNING snapshot of {run_id} failed: {type(e).__name__}")
        return False


# Units per hour by GPU before a session has measured its own rate (`colab usage` reads it after
# the session starts). Community figures, not Google's (8 Oct 2026); the first session replaces them.
EST_RATE = {"T4": 2.0, "L4": 5.0, "G4": 5.0, "A100": 13.0, "H100": 20.0}
MIN_HOURS = 1.5  # less than this cannot set up, train a little and save


def items(queue: list[dict]) -> list[dict]:
    return [q for q in queue if q.get("backend") == "colab"]


def next_lane(base: dict, queue: list[dict], runs: list[dict]) -> tuple[dict, str] | None:
    """(config, GPU) of the first pending Colab entry, or None."""
    from reva import autopilot, remote

    done = {r["run_id"] for r in runs if r["status"] == "ok"}
    mine = items(queue)
    gpus = {remote.run_config(base, q)["run_id"]: q.get("gpu", "A100") for q in mine}
    lanes = remote.pending(base, mine, done, autopilot.failures(runs))
    return (lanes[0], gpus[lanes[0]["run_id"]]) if lanes else None


def last_rate(state: Path, gpu: str) -> float:
    """Units per hour this GPU burned in our last session on it, else the estimate."""
    from reva import registry

    seen = [j for j in registry.read(state / "colab_jobs.jsonl") if j.get("gpu") == gpu and j.get("rate")]
    return seen[-1]["rate"] if seen else EST_RATE.get(gpu, 13.0)


def run_next(base: dict, queue: list[dict], state: Path, work: Path, colab: Colab, kaggle, sha: str,
             max_hours: float, now_iso: str, notes: list[str], publish: Callable[[], None] | None = None,
             **session_kw) -> dict:
    """One Colab session for the next pending Colab entry. Returns {"ran", "more"}: the run row
    (or None) and whether another session has work and units to do it."""
    from reva import autopilot, registry

    runs = autopilot.read_runs(state)
    nxt = next_lane(base, queue, runs)
    if not nxt:
        notes.append("Colab: no Colab entry pending in configs/queue.yaml")
        return {"ran": None, "more": False}
    lane, gpu = nxt
    balance, _ = colab.usage()
    rate = last_rate(state, gpu)
    hours = min(max_hours, balance / rate - 0.25)
    if hours < MIN_HOURS:
        notes.append(f"Colab: {balance:.1f} compute units left, {hours:.1f} h of {gpu} at {rate:.1f}/h; waiting")
        return {"ran": None, "more": False}

    restore = None
    saved = [r["stash"] for r in runs if r["run_id"] == lane["run_id"] and r.get("stash") and r.get("resumable")]
    if saved:  # the newest saved state, from either backend
        copy = work / "colab-restore"
        shutil.rmtree(copy, ignore_errors=True)
        kaggle.dataset_download(saved[-1], copy)
        restore = work / f"{lane['run_id']}-restore.tar"
        with tarfile.open(restore, "w") as t:
            t.add(copy / lane["run_id"], arcname=lane["run_id"])
        notes.append(f"Colab: resuming {lane['run_id']} from {saved[-1]}")

    # units left after this session buy no other one: train only as long as prediction allows
    final = (balance - hours * rate) / rate - 0.25 < MIN_HOURS
    cfg = lane_config(base, lane, hours, final)

    def live(**kw):  # what the session is doing, for STATUS readers while it runs
        (state / "colab_live.json").write_text(json.dumps({
            "run_id": lane["run_id"], "gpu": gpu, "started": now_iso, "units_before": balance,
            "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **kw}, indent=1))
        if publish:
            publish()

    ref = autopilot.stash_ref(kaggle.users[0] if hasattr(kaggle, "users") else "me", lane["run_id"])
    snaps: list[str] = []

    def save(run_dir: Path) -> None:  # an hourly snapshot, a new version of the run's dataset
        autopilot.stash(kaggle, run_dir, ref, work / "snap", message=f"colab {gpu} snapshot", keep_test=True)
        snaps.append(time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime()))

    session_kw.setdefault("progress", live)
    session_kw.setdefault("save", save)
    notes.append(f"Colab: {lane['run_id']} on {gpu} for up to {hours:.1f} h ({balance:.1f} units, about {rate:.1f}/h)"
                 + ("; the last session the units allow, so it predicts" if C.get(cfg, "train.final_session", False) else ""))
    out_dir, log_tail, wall, ended = session(colab, cfg, sha, C.get(base, "remote.repo"), C.get(base, "remote.pip"), gpu,
                                      hours, work, restore, **session_kw)
    after, measured = colab.usage()
    row = autopilot.run_row(out_dir, "", sha, autopilot.parse_iso(now_iso), log_tail) if out_dir else {
        "run_id": lane["run_id"], "status": "failed", "kernel": "", "finished": now_iso, "sha": sha,
        "error": autopilot.tail(log_tail, 30)}
    row.update(backend="colab", gpu=gpu)
    if out_dir and not ended and row["status"] == "failed" and row.get("resumable"):
        # the session ran out before the job could stop itself; its last checkpoint carries on
        row.update(status="partial", where="the Colab session ended before the job")
    if out_dir:
        try:
            autopilot.stash(kaggle, out_dir, ref, work, message=f"colab {gpu} {row['status']}", keep_test=True)
            row.update(stash=ref, dataset=ref)
            notes.append(f"Colab: saved {lane['run_id']} ({row['status']}) to private dataset {ref}")
        except Exception as e:  # the row still records what happened
            notes.append(f"Colab: could not save {lane['run_id']}: {autopilot.public(e)}")
    if snaps and row["status"] != "ok" and "stash" not in row:
        # lost midway, or the final archive failed: the newest snapshot carries the run on
        row.update(status="partial", resumable=True, stash=ref, dataset=ref,
                   where=f"the Colab session was lost; resumes from the snapshot of {snaps[-1]}")
        notes.append(f"Colab: {lane['run_id']} resumes from the snapshot of {snaps[-1]}")
    used = max(0.0, balance - after)
    registry.append(state / "colab_runs.jsonl", row)
    registry.append(state / "colab_jobs.jsonl", {
        "run_id": lane["run_id"], "gpu": gpu, "started": now_iso, "hours": round(wall, 2), "units_before": balance,
        "units_after": after, "rate": round(used / wall, 2) if wall > 0.2 and used > 0 else None, "status": row["status"]})
    notes.append(f"Colab: {lane['run_id']} {row['status']} after {wall:.1f} h, {used:.1f} units used, {after:.1f} left")
    live(stage="done", status=row["status"], hours_in=round(wall, 2), units_after=after)
    more = next_lane(base, queue, autopilot.read_runs(state)) is not None and after / last_rate(state, gpu) - 0.25 >= MIN_HOURS
    return {"ran": row, "more": more}


class StatePusher:
    """Pushes colab_live.json to the state branch at most every `every_s` seconds. A failed push
    only warns: the session must never stop over its progress report."""

    def __init__(self, state: Path, every_s: float = 1200, git: Callable[[list[str]], int] | None = None,
                 clock=time.time):
        self.state, self.every_s, self.clock, self.last = Path(state), every_s, clock, None
        self.git = git or (lambda args: subprocess.run(["git", "-C", str(self.state), *args],
                                                       capture_output=True, timeout=120).returncode)

    def __call__(self) -> None:
        now = self.clock()
        if self.last is not None and now - self.last < self.every_s:
            return
        self.last = now
        who = ["-c", "user.name=Sidharth Pradhan", "-c", "user.email=sidharthp@assignall.ai"]
        steps = [("add", ["add", "--", "colab_live.json"]), ("commit", [*who, "commit", "-q", "-m", "colab: live progress"]),
                 ("pull", [*who, "pull", "-q", "--rebase", "origin", "state"]), ("push", ["push", "-q", "origin", "HEAD:state"])]
        for verb, args in steps:
            try:
                code = self.git(args)
            except Exception as e:  # a hung or missing git
                code = type(e).__name__
            if code != 0:
                print(f"WARNING live progress not pushed: git {verb} gave {code}", flush=True)
                if verb == "pull":
                    self.git(["rebase", "--abort"])
                return


TOKEN = Path("~/.config/colab-cli/token.json").expanduser()
TOKEN_FIELDS = ("refresh_token", "client_id", "client_secret")


def _google_refresh(info: dict) -> None:
    from google.auth.transport.requests import Request  # comes with the Colab CLI
    from google.oauth2.credentials import Credentials

    Credentials.from_authorized_user_info(info).refresh(Request())


def check_token(path: Path = TOKEN, refresh: Callable[[dict], None] | None = None) -> str:
    """Why the stored login would fail, before any session. The CLI hides it: on a token it cannot
    use it starts the browser login, which fails on a closed stdin (first session, 8 Oct 2026).
    Prints field names and Google's error only, never a value."""
    redo = "Log in again with `colab usage` and store the whole ~/.config/colab-cli/token.json as COLAB_TOKEN."
    try:
        info = json.loads(Path(path).read_text())
    except (OSError, ValueError) as e:
        raise ColabError(f"COLAB_TOKEN is not the JSON file the CLI saves ({type(e).__name__}). {redo}") from None
    if not isinstance(info, dict):
        raise ColabError(f"COLAB_TOKEN is JSON but not an object. {redo}")
    missing = [k for k in TOKEN_FIELDS if not info.get(k)]
    if missing:
        raise ColabError(f"COLAB_TOKEN lacks {', '.join(missing)} (it has {', '.join(sorted(info))}). {redo}")
    try:
        (refresh or _google_refresh)(info)
    except Exception as e:  # google.auth RefreshError and network errors alike
        raise ColabError(f"Google refused to refresh the Colab login: {e}. {redo}") from None
    return f"Colab login ok (fields: {', '.join(sorted(info))})"


def probe(colab: Colab, gpu: str = "T4") -> str:
    """A short check that the login, the subscription and a GPU session work."""
    balance, _ = colab.usage()
    name = "reva-probe"
    colab.new(name, gpu)
    try:
        out = colab.exec(name, "import subprocess, sys, torch\n"
                               "print(sys.version)\nprint('torch', torch.__version__, torch.cuda.get_device_name(0))\n"
                               "print(subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv'],"
                               " capture_output=True, text=True).stdout)", timeout=300)
    finally:
        colab.stop(name)
    after, _ = colab.usage()
    return f"units {balance:.2f} -> {after:.2f}\n{out.strip()}"
