"""The Colab backend (reva.colab) against a fake Colab CLI: no network, no GPU."""

import json
import tarfile
from pathlib import Path

import pytest

from reva import arena, autopilot, colab, registry, remote
from reva import config as C
from reva.data import DEV_SET

QUEUE = [{"name": "ft-k", "set": {"train.enabled": True}},
         {"name": "ft-c", "backend": "colab", "gpu": "A100", "set": {"train.enabled": True, "model.dtype": "bf16"}}]


def cfg():
    return C.load()


class FakeCLI:
    """Answers the colab commands reva.colab uses. `finish` is what the job leaves in its output."""

    def __init__(self, tmp: Path, finish="ok", balance=90.0, polls_until_done=2):
        self.tmp, self.finish, self.balance, self.left = tmp, finish, balance, polls_until_done
        self.calls, self.stopped, self.uploaded, self.started = [], [], [], None

    def __call__(self, cmd, timeout=None):
        args = cmd[1:]
        self.calls.append(args[0])
        if args[0] == "usage":
            out = f"Current balance: {self.balance:.2f} compute units\nUsage rate: 13.00/hr\nActive assignments: 0"
            self.balance -= 5  # what a session costs here
            return 0, out
        if args[0] == "new":
            return 0, "[colab] Creating session...\n[colab] Session READY."
        if args[0] == "stop":
            self.stopped.append(args[2])
            return 0, "[colab] Session terminated."
        if args[0] == "upload":
            self.uploaded.append(args[3:])
            return 0, ""
        if args[0] == "exec":
            code = Path(args[args.index("-f") + 1]).read_text()
            if "subprocess.Popen" in code:
                compile(code, "start", "exec")  # the code sent to the kernel must at least parse
                self.started = json.loads(json.loads(code.split('.write(')[1].split(')\n')[0]))
                return 0, "STARTED 42"
            if "EXIT_CODE" in code:
                self.left -= 1
                return 0, "train step 10\nEXIT_CODE " + ("running" if self.left > 0 else "0")
            if "RESTORED" in code:
                return 0, "RESTORED []"
            return 0, "TARRED"
        if args[0] == "download":
            run_id = self.started["run_id"]
            d = self.tmp / "vm" / run_id
            d.mkdir(parents=True, exist_ok=True)
            (d / "log.txt").write_text("DONE\n")
            (d / "test_probs.json").write_text("{}")
            if self.finish == "ok":
                (d / "run.json").write_text(json.dumps({
                    "metrics": {"weighted_accuracy": 0.9, "overall_accuracy": 0.9}, "dev_set": DEV_SET, "hours": 3.0,
                    "zip": f"{run_id}.zip", "n": {"test": 4000}, "train": {}, "timings": {}, "versions": {},
                    "config": {"why": "w"}, "finished": "2026-10-09T03:00:00Z"}))
            elif self.finish == "partial":
                (d / "ckpt").mkdir()
                (d / "partial.json").write_text(json.dumps({"where": "training: step 300"}))
            else:  # cut off mid-job: only the periodic checkpoint
                (d / "ckpt").mkdir()
            with tarfile.open(args[4], "w") as t:
                t.add(d, arcname=run_id)
            return 0, ""
        raise AssertionError(f"unexpected colab call {args}")


class FakeKaggle:
    users = ["first", "second"]

    def __init__(self):
        self.datasets, self.downloads = {}, []

    def dataset_upload(self, folder, ref, message, **kwargs):
        with tarfile.open(next(Path(folder).glob("*.tar"))) as t:
            self.datasets[ref] = sorted(t.getnames())

    def dataset_download(self, ref, dest):
        self.downloads.append(ref)
        run_id = ref.split("reva-run-")[1]
        (Path(dest) / run_id / "ckpt").mkdir(parents=True)
        return Path(dest)


def run(tmp_path, cli, kaggle=None, state=None):
    state = state or tmp_path / "state"
    state.mkdir(exist_ok=True)
    notes = []
    out = colab.run_next(cfg(), QUEUE, state, tmp_path / "work", colab.Colab(runner=cli), kaggle or FakeKaggle(), "abc",
                         5.0, "2026-10-09T00:00:00Z", notes, poll_s=0, sleep=lambda s: None)
    return out, notes, state


def test_a_colab_session_runs_the_colab_entry_and_archives_everything(tmp_path):
    cli, k = FakeCLI(tmp_path), FakeKaggle()
    out, notes, state = run(tmp_path, cli, k)
    row = out["ran"]
    assert row["run_id"].startswith("ft-c-") and row["status"] == "ok" and row["backend"] == "colab"
    assert cli.started["run_id"] == row["run_id"] and cli.started["data"]["root"] == "/content/reva"
    assert cli.started["job"]["max_hours"] <= 5.0 - colab.SETUP_H
    assert cli.stopped == [f"reva-{row['run_id']}"[:40]]  # the session never outlives the job
    # a Colab run has no kernel output: its dataset keeps the test probabilities, privately
    assert row["dataset"] == f"first/reva-run-{row['run_id']}" and f"{row['run_id']}/test_probs.json" in k.datasets[row["dataset"]]
    assert registry.read(state / "colab_runs.jsonl") == [row]
    job = registry.read(state / "colab_jobs.jsonl")[0]
    assert job["gpu"] == "A100" and job["units_before"] == 90.0 and job["units_after"] == 85.0
    assert out["more"] is False  # nothing else is queued for Colab


def test_the_kaggle_loop_sees_colab_runs_but_never_pushes_colab_entries(tmp_path):
    _, _, state = run(tmp_path, FakeCLI(tmp_path))
    runs = autopilot.read_runs(state)
    assert [r["backend"] for r in runs] == ["colab"]
    assert [c["run_id"].rsplit("-", 1)[0] for c in remote.pending(cfg(), autopilot.kaggle_queue(QUEUE), set(), {})] == ["ft-k"]


def test_an_unfinished_colab_run_resumes_from_its_archive(tmp_path):
    cli, k = FakeCLI(tmp_path, finish="partial"), FakeKaggle()
    first, _, state = run(tmp_path, cli, k)
    assert first["ran"]["status"] == "partial" and first["ran"]["resumable"] and first["more"]
    again = FakeCLI(tmp_path / "2")
    second, notes, _ = run(tmp_path, again, k, state)
    tar = str(tmp_path / "work" / f"{first['ran']['run_id']}-restore.tar")
    assert k.downloads == [first["ran"]["stash"]] and again.uploaded == [[tar + ".part0000", "/content/restore.tar.part0000"]]
    assert second["ran"]["status"] == "ok" and any("resuming" in n for n in notes)


def test_no_session_without_the_units_for_one(tmp_path):
    cli = FakeCLI(tmp_path, balance=10.0)  # under 1.5 h of A100
    out, notes, _ = run(tmp_path, cli)
    assert out == {"ran": None, "more": False} and "new" not in cli.calls and "waiting" in notes[-1]


def test_a_lost_session_is_still_stopped_and_recorded(tmp_path):
    cli = FakeCLI(tmp_path)

    def broken(cmd, timeout=None):
        if cmd[1] == "exec" and "subprocess.Popen" in Path(cmd[cmd.index("-f") + 1]).read_text():
            return 1, "[colab] Session 'x' appears to be lost (404/401). Cleaning up."
        return cli(cmd, timeout)

    out, _, state = run(tmp_path, broken)
    assert out["ran"]["status"] == "failed" and "lost" in out["ran"]["error"] and cli.stopped


def test_colab_runs_are_read_from_their_dataset(tmp_path):
    class K:
        def dataset_download(self, ref, dest):
            (Path(dest) / "r").mkdir(parents=True)
            (Path(dest) / "r" / "dev_probs.json").write_text('{"q": [1, 0, 0, 0]}')

    run_ = {"run_id": "r", "backend": "colab", "kernel": "", "dataset": "first/reva-run-r"}
    assert arena.load_probs(K(), run_, tmp_path, "dev_probs.json") == {"q": [1, 0, 0, 0]}


def test_queue_rejects_a_backend_typo_and_gpus_on_kaggle(tmp_path):
    for bad in ("- name: a-b\n  backend: colb\n", "- name: a-b\n  gpu: A100\n"):
        (tmp_path / "q.yaml").write_text(bad)
        with pytest.raises(ValueError):
            remote.load_queue(tmp_path / "q.yaml")


def test_usage_parses_the_cli_output():
    assert colab.parse_usage("Current balance: 87.25 compute units\nUsage rate: 11.70/hr\n") == (87.25, 11.7)
    with pytest.raises(colab.ColabError):
        colab.parse_usage("Error: not logged in")


def test_a_session_that_runs_out_mid_job_resumes_from_its_last_checkpoint(tmp_path):
    """The job's own deadline should stop it first; if setup ran long, the session's end comes
    first. Its periodic checkpoint is in the output, so the run is partial, not failed."""
    cli = FakeCLI(tmp_path, finish="crashed", polls_until_done=10**6)
    clock = iter(range(0, 10**9, 3600))  # every poll is an hour
    out, _, _ = run_with(tmp_path, cli, clock=lambda: next(clock))
    assert out["ran"]["status"] == "partial" and out["ran"]["resumable"] and "session ended" in out["ran"]["where"]


def run_with(tmp_path, cli, **kw):
    state = tmp_path / "state"
    state.mkdir(exist_ok=True)
    notes = []
    out = colab.run_next(cfg(), QUEUE, state, tmp_path / "work", colab.Colab(runner=cli), FakeKaggle(), "abc",
                         5.0, "2026-10-09T00:00:00Z", notes, poll_s=0, sleep=lambda s: None, **kw)
    return out, notes, state


def test_the_pending_check_the_autopilot_uses_to_start_sessions(tmp_path, capsys):
    from reva import cli

    state = tmp_path / "state"
    state.mkdir()
    q = tmp_path / "q.yaml"
    q.write_text("- name: ft-c\n  backend: colab\n  set: {train.enabled: true}\n")
    assert cli.main(["colab", "--pending", "--state", str(state), "--queue", str(q)]) == 0
    assert "ft-c-" in capsys.readouterr().out
    q.write_text("- name: ft-k\n  set: {train.enabled: true}\n")
    assert cli.main(["colab", "--pending", "--state", str(state), "--queue", str(q)]) == 1


def test_the_token_check_names_what_is_wrong_without_printing_values(tmp_path):
    """Regression (8 Oct 2026): the first session failed with only the CLI's login prompt, which
    hides whether the stored token was malformed or refused."""
    tok = tmp_path / "token.json"
    good = {"token": "SECRET-A", "refresh_token": "SECRET-R", "client_id": "id", "client_secret": "SECRET-C",
            "token_uri": "https://oauth2.googleapis.com/token"}

    def refused(info):
        raise RuntimeError("invalid_grant: Token has been expired or revoked.")

    for text, refresh, why in [("not json", None, "not the JSON"),
                               (json.dumps({"token": "SECRET-A"}), None, "lacks refresh_token, client_id, client_secret"),
                               (json.dumps(good), refused, "invalid_grant")]:
        tok.write_text(text)
        with pytest.raises(colab.ColabError, match=why) as e:
            colab.check_token(tok, refresh)
        assert "SECRET" not in str(e.value)
    tok.write_text(json.dumps(good))
    ok = colab.check_token(tok, lambda info: None)
    assert ok.startswith("Colab login ok") and "SECRET" not in ok


def test_a_session_reports_live_progress_and_pushes_it(tmp_path):
    """Actions shows a job's log to the API only after the job ends, so a 5 h session was a black
    box (8 Oct 2026). The session writes colab_live.json at every poll and pushes it."""
    pushes = []
    state = tmp_path / "state"
    state.mkdir()
    out = colab.run_next(cfg(), QUEUE, state, tmp_path / "work", colab.Colab(runner=FakeCLI(tmp_path)), FakeKaggle(),
                         "abc", 5.0, "2026-10-09T00:00:00Z", [], publish=lambda: pushes.append(
                             json.loads((state / "colab_live.json").read_text())["stage"]),
                         poll_s=0, sleep=lambda s: None)
    assert pushes[0] == "started" and "running" in pushes and pushes[-1] == "done"
    live = json.loads((state / "colab_live.json").read_text())
    assert live["run_id"] == out["ran"]["run_id"] and live["gpu"] == "A100" and live["status"] == "ok"
    assert live["units_before"] == 90.0 and live["units_after"] == 85.0


def test_live_progress_pushes_to_the_state_branch_at_most_every_20_minutes(tmp_path):
    import subprocess

    def git(*args, cwd=tmp_path):
        return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=cwd, check=True,
                              capture_output=True, text=True).stdout

    git("init", "-q", "--bare", "remote.git")
    git("clone", "-q", "remote.git", "seed")
    git("checkout", "-q", "-b", "state", cwd=tmp_path / "seed")
    git("commit", "-q", "--allow-empty", "-m", "s", cwd=tmp_path / "seed")
    git("push", "-q", "origin", "state", cwd=tmp_path / "seed")
    git("clone", "-q", "-b", "state", "remote.git", "state")
    now = [0.0]
    push = colab.StatePusher(tmp_path / "state", clock=lambda: now[0])
    for t, stage in [(0, "started"), (300, "running"), (1500, "running2")]:
        now[0] = t
        (tmp_path / "state" / "colab_live.json").write_text(json.dumps({"stage": stage}))
        push()
    shown = git("show", "state:colab_live.json", cwd=tmp_path / "remote.git")
    assert json.loads(shown)["stage"] == "running2"
    assert git("rev-list", "--count", "state", cwd=tmp_path / "remote.git").strip() == "3"  # seed + 2 pushes


def test_a_failed_progress_push_never_stops_the_session(capsys):
    calls = []

    def git(args):
        calls.append(args)
        return 1 if "pull" in args else 0

    colab.StatePusher(Path("."), git=git)()
    assert calls[-1] == ["rebase", "--abort"] and "not pushed" in capsys.readouterr().out


def test_status_shows_the_colab_session_in_progress(tmp_path):
    assert autopilot.colab_note(tmp_path) is None
    (tmp_path / "colab_live.json").write_text(json.dumps({"run_id": "ft-c-1", "gpu": "A100", "stage": "running",
                                                          "hours_in": 1.25, "updated": "t", "units_before": 90.0,
                                                          "last": "train step 120 loss 0.41"}))
    note = autopilot.colab_note(tmp_path)
    assert "ft-c-1 on A100, 1.2 h in" in note and "loss 0.41" in note
    (tmp_path / "colab_live.json").write_text(json.dumps({"stage": "done"}))
    assert autopilot.colab_note(tmp_path) is None


SPAN_QUEUE = [{"name": "ft-s", "backend": "colab", "gpu": "A100",
               "set": {"train.enabled": True, "model.dtype": "bf16", "train.span_sessions": True}}]


def test_the_last_session_the_units_pay_for_predicts_instead_of_only_training(tmp_path):
    def started(balance):
        home = tmp_path / str(int(balance))
        (home / "state").mkdir(parents=True)
        cli = FakeCLI(home, balance=balance)
        notes = []
        colab.run_next(cfg(), SPAN_QUEUE, home / "state", home / "work",
                       colab.Colab(runner=cli), FakeKaggle(), "abc", 5.0, "2026-10-09T00:00:00Z", notes,
                       poll_s=0, sleep=lambda s: None)
        return cli.started, notes

    last, notes = started(40.0)  # 2.8 h of A100 now, nothing after it
    assert C.get(last, "train.final_session") is True and any("predicts" in n for n in notes)
    more, _ = started(200.0)  # 5 h now and more sessions after it
    assert not C.get(more, "train.final_session", False)
    assert not C.get(colab.lane_config(cfg(), {**cfg(), "run_id": "r"}, 3.0, final=True), "train.final_session", False)


def test_a_session_lost_midway_resumes_from_its_hourly_snapshot(tmp_path):
    cli, k = FakeCLI(tmp_path, polls_until_done=10**6), FakeKaggle()
    snapped = []

    def runner(cmd, timeout=None):
        args = cmd[1:]
        code = Path(args[args.index("-f") + 1]).read_text() if args[0] == "exec" else ""
        if '"cp", "-al"' in code:
            snapped.append(True)
            return 0, "SNAPPED"
        if "EXIT_CODE" in code and snapped:
            return 1, "[colab] Session appears to be lost (404/401)."
        if args[0] == "download" and args[3] == "/content/snap.tar":
            d = tmp_path / "vm-snap" / cli.started["run_id"]
            (d / "ckpt" / "adapter").mkdir(parents=True)
            (d / "ckpt" / "state.pt").write_bytes(b"x")
            with tarfile.open(args[4], "w") as t:
                t.add(d, arcname=d.name)
            return 0, ""
        return cli(cmd, timeout)

    clock = iter(range(0, 10**9, 1200))  # 20 minutes pass between clock reads
    state = tmp_path / "state"
    state.mkdir()
    notes = []
    out = colab.run_next(cfg(), QUEUE, state, tmp_path / "work", colab.Colab(runner=runner), k, "abc", 5.0,
                         "2026-10-09T00:00:00Z", notes, poll_s=0, sleep=lambda s: None, clock=lambda: next(clock))
    row = out["ran"]
    assert snapped and cli.stopped and row["status"] == "partial" and row["resumable"]
    assert row["stash"] in k.datasets and f"{row['run_id']}/ckpt/state.pt" in k.datasets[row["stash"]]
    assert "snapshot" in row["where"] and any("resumes from the snapshot" in n for n in notes)
    assert autopilot.failures([row]) == {row["run_id"]: 0}  # a lost session is not held against the run


def test_a_failed_snapshot_never_stops_the_session(tmp_path):
    cli = FakeCLI(tmp_path)

    def runner(cmd, timeout=None):
        if cmd[1] == "exec" and '"cp", "-al"' in Path(cmd[cmd.index("-f") + 1]).read_text():
            return 1, "boom"
        return cli(cmd, timeout)

    lines = []
    assert not colab.snapshot(colab.Colab(runner=runner), "s", "r", tmp_path, lambda d: None, log=lines.append)
    assert "snapshot of r failed" in lines[-1]


def test_the_snapshot_code_parses():
    compile(colab.snapshot_code("ft-x-1"), "snap", "exec")


class FlakyUpload:
    """A colab CLI whose upload fails `fails` times before it works; keeps every part it got."""

    def __init__(self, fails=0):
        self.fails, self.got = fails, []

    def __call__(self, cmd, timeout=None):
        if self.fails:
            self.fails -= 1
            return 1, "[colab] Upload failed: SSLError(SSLEOFError(8, 'EOF occurred in violation of protocol'))"
        self.got.append((cmd[-1], Path(cmd[-2]).read_bytes()))
        return 0, ""


def test_a_large_saved_state_goes_up_in_parts_and_joins_back(tmp_path):
    # regression: the runtime dropped a 480 MB checkpoint sent as one request (9 Oct 2026)
    run_dir = tmp_path / "src" / "r1"
    (run_dir / "ckpt").mkdir(parents=True)
    (run_dir / "ckpt" / "state.pt").write_bytes(bytes(range(256)) * 400)
    tar = tmp_path / "r1-restore.tar"
    with tarfile.open(tar, "w") as t:
        t.add(run_dir, arcname="r1")
    cli = FlakyUpload(fails=1)  # the first part fails once and goes again
    n = colab.upload_parts(colab.Colab(runner=cli), "s", tar, "/content/restore.tar", part_bytes=10_000)
    assert n == len(cli.got) > 1 and [r for r, _ in cli.got] == [f"/content/restore.tar.part{i:04d}" for i in range(n)]
    assert not list(tmp_path.glob("*.part*"))  # local parts are cleaned up
    # run the join on "the runtime": /content mapped to a temp folder
    vm = tmp_path / "vm"
    vm.mkdir()
    for remote, data in cli.got:
        (vm / Path(remote).name).write_bytes(data)
    code = colab.restore_code("r1", n, tar.stat().st_size).replace("/content", str(vm))
    exec(compile(code, "restore", "exec"), {})
    assert (vm / "out" / "r1" / "ckpt" / "state.pt").read_bytes() == bytes(range(256)) * 400
    assert [p.name for p in vm.iterdir()] == ["out"]  # the parts and the joined tar are gone


def test_a_part_that_keeps_failing_fails_the_session_loudly(tmp_path):
    tar = tmp_path / "x.tar"
    tar.write_bytes(b"x" * 100)
    with pytest.raises(colab.ColabError, match="Upload failed"):
        colab.upload_parts(colab.Colab(runner=FlakyUpload(fails=3)), "s", tar, "/content/restore.tar", part_bytes=60)
    assert not list(tmp_path.glob("*.part*"))


def test_failed_colab_setups_do_not_use_up_the_runs_retries():
    # regression: two sessions died uploading ft-8b-32f-a100's checkpoint (9 Oct 2026); the run
    # dropped out of the queue although it was at step 1639 of 1686
    run_id = remote.run_config(cfg(), QUEUE[1])["run_id"]
    lost = {"run_id": run_id, "status": "failed", "backend": "colab",
            "error": "\n`colab upload` exited 1: [colab] Upload failed: SSLEOFError"}
    runs = [{"run_id": run_id, "status": "partial", "resumable": True, "stash": "first/reva-run-x"}, lost, lost]
    lane, gpu = colab.next_lane(cfg(), QUEUE, runs)
    assert lane["run_id"] == run_id and gpu == "A100"
    # a broken backend still cannot loop forever
    assert colab.next_lane(cfg(), QUEUE, runs[:1] + [lost] * 6) is None


def test_colab_errors_never_carry_the_runtime_proxy_token():
    out = ("[colab] Upload failed: Max retries exceeded with url: /api/contents/content/restore.tar"
           "?authuser=0&colab-runtime-proxy-token=eyJhbGciOi.secret.sig (Caused by SSLError)")
    with pytest.raises(colab.ColabError) as e:
        colab.Colab(runner=lambda cmd, timeout=None: (1, out)).upload("s", Path("x"), "/content/x")
    assert "eyJ" not in str(e.value) and "colab-runtime-proxy-token=***" in str(e.value)
