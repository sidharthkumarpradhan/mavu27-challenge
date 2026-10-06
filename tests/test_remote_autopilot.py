import datetime as dt
import json
from pathlib import Path

import pytest

from reva import autopilot, remote
from reva import config as C
from reva.kaggle import Push

NOW = dt.datetime(2026, 10, 7, 12, 0, tzinfo=dt.timezone.utc)
QUEUE = [{"name": "zs-4b", "set": {}}, {"name": "text-4b", "set": {"model.use_video": False}},
         {"name": "ft-4b", "set": {"train.enabled": True}}]


def cfg():
    return C.load()


def test_run_ids_follow_config():
    a = remote.run_config(cfg(), QUEUE[0])
    assert a["run_id"].startswith("zs-4b-") and a["run_id"] == remote.run_config(cfg(), QUEUE[0])["run_id"]
    b = remote.run_config(cfg(), {"name": "zs-4b", "set": {"frames.n": 32}})
    assert a["run_id"] != b["run_id"]


def test_pending_skips_done_and_repeat_failures():
    ids = [remote.run_config(cfg(), q)["run_id"] for q in QUEUE]
    left = remote.pending(cfg(), QUEUE, done={ids[0]}, failed={ids[1]: 2})
    assert [c["run_id"] for c in left] == [ids[2]]


def test_queue_names_are_checked(tmp_path):
    p = tmp_path / "q.yaml"
    p.write_text("- name: 'bad name; rm -rf /'\n  set: {}\n")
    with pytest.raises(ValueError):
        remote.load_queue(p)
    assert remote.load_queue(Path(C.DEFAULT).with_name("queue.yaml"))  # the real queue parses


def test_kernel_script_compiles_and_pins(tmp_path):
    lanes = remote.pending(cfg(), QUEUE, set(), {})[:2]
    slug = remote.build(cfg(), lanes, "abc123", tmp_path, "someone")
    src = (tmp_path / "job.py").read_text()
    compile(src, "job.py", "exec")
    assert "abc123" in src and "transformers==5.19.0" in src and "/kaggle/tmp/frames" in src
    meta = json.loads((tmp_path / "kernel-metadata.json").read_text())
    assert meta["id"] == slug and meta["is_private"] and meta["enable_gpu"] and meta["enable_internet"]
    assert slug.startswith("someone/reva-zs-4b-")


def run_row(run_id, w, test_n=4000):
    return {"run_id": run_id, "status": "ok", "zip": f"{run_id}.zip", "n": {"test": test_n},
            "metrics": {"weighted_accuracy": w, "overall_accuracy": w}, "kernel": "u/k"}


def sub_row(run_id, w, status="Finished", day="2026-10-07", fmt="fill_test", sid=1):
    return {"run_id": run_id, "submission_id": sid, "status": status, "submitted": f"{day}T01:00:00Z",
            "dev_weighted": w, "format": fmt, "scores": {}}


def test_gate_rules():
    c = cfg()
    assert autopilot.gate(run_row("a", 0.5), [], c, NOW)[0]
    assert not autopilot.gate(run_row("a", 0.5, test_n=10), [], c, NOW)[0]
    assert not autopilot.gate(run_row("b", 0.501), [sub_row("a", 0.5)], c, NOW)[0]  # below min_gain
    assert autopilot.gate(run_row("b", 0.51), [sub_row("a", 0.5)], c, NOW)[0]
    assert not autopilot.gate(run_row("a", 0.9), [sub_row("a", 0.5)], c, NOW)[0]  # already submitted
    failed = [sub_row("a", 0.5, status="Failed")]
    assert "format" in autopilot.gate(run_row("b", 0.9), failed, c, NOW)[1]
    four = [sub_row(f"r{i}", 0.1 * i, sid=i) for i in range(4)]
    assert "daily" in autopilot.gate(run_row("b", 0.9), four, c, NOW)[1]
    many = [sub_row(f"r{i}", 0.001 * i, day="2026-10-01", sid=i) for i in range(85)]
    assert "budget" in autopilot.gate(run_row("b", 0.9), many, c, NOW)[1]  # reserve of 15 held back
    late = NOW.replace(month=11, day=6)
    assert autopilot.gate(run_row("b", 0.9), many, c, late)[0]  # final week may spend the reserve


class FakeKaggle:
    def __init__(self, state="running", lanes_ok=True):
        self.state, self.lanes_ok, self.pushed = state, lanes_ok, []

    def status(self, slug):
        return self.state, ""

    def output(self, slug, dest):
        for run_id in self.active_runs:
            d = dest / run_id
            d.mkdir(parents=True, exist_ok=True)
            (d / "log.txt").write_text("Traceback: boom\n")
            if self.lanes_ok:
                (d / "run.json").write_text(json.dumps({
                    "metrics": {"weighted_accuracy": 0.6, "overall_accuracy": 0.61}, "hours": 2.0, "zip": f"{run_id}.zip",
                    "n": {"test": 4000}, "train": None, "timings": {}, "versions": {}, "config": {"why": "w"},
                    "finished": "2026-10-07T11:00:00Z"}))
                (d / f"{run_id}.zip").write_bytes(b"PK")
        return [], "EXIT 0"

    def push(self, kdir, timeout_s=None, accelerator=None):
        self.pushed.append(json.loads((kdir / "kernel-metadata.json").read_text())["id"])
        return Push(1, "https://kaggle/k")


class FakeClient:
    def __init__(self):
        self.submitted = []

    def submit(self, zip_path, comp, phase, tasks):
        assert Path(zip_path).exists()
        self.submitted.append(zip_path.name)
        return 99

    def wait(self, sid, timeout_s=0):
        return {"status": "Finished", "scores": [{"column_key": "overall_accuracy", "score": "0.62"}]}

    def submission(self, sid):
        return self.wait(sid)


def board_rows():
    return [{"owner": "leader", "submission": 1, "created": "", "overall_accuracy": 0.87}]


def test_cycle_push_collect_submit(tmp_path):
    c, state, work = cfg(), tmp_path / "state", tmp_path / "work"
    k = FakeKaggle()
    out = autopilot.cycle(c, QUEUE, state, work, k, "sha1", "me", now=NOW, fetch_board=board_rows)
    assert out["pushed"] and len(out["pushed"]["runs"]) == 2 and k.pushed
    assert (state / "active.json").exists() and "leader" in (state / "STATUS.md").read_text()

    out = autopilot.cycle(c, QUEUE, state, work, k, "sha1", "me", now=NOW, fetch_board=board_rows)
    assert out["pushed"] is None and any("running" in n for n in out["notes"])

    k.state, k.active_runs = "complete", out and json.loads((state / "active.json").read_text())["runs"]
    client = FakeClient()
    later = NOW + dt.timedelta(hours=3)
    out = autopilot.cycle(c, QUEUE, state, work, k, "sha1", "me", client=client, auto_submit=True, now=later,
                          fetch_board=board_rows)
    assert len(out["collected"]) == 2 and not out["needs_fix"]
    assert len(client.submitted) == 1 and out["submitted"][0]["scores"] == {"overall_accuracy": 0.62}
    assert out["pushed"]["runs"][0].startswith("ft-4b-")  # the next lane went out
    jobs = [json.loads(x) for x in (state / "jobs.jsonl").read_text().splitlines()]
    assert jobs[0]["hours"] == 3.0
    status = (state / "STATUS.md").read_text()
    assert "Calibration" in status and "Gap to the leader" in status


def test_cycle_failed_lanes_flag_a_fix_and_never_submit(tmp_path):
    c, state, work = cfg(), tmp_path / "state", tmp_path / "work"
    k = FakeKaggle(lanes_ok=False)
    autopilot.cycle(c, QUEUE, state, work, k, "sha1", "me", now=NOW, fetch_board=board_rows)
    k.state, k.active_runs = "error", json.loads((state / "active.json").read_text())["runs"]
    client = FakeClient()
    out = autopilot.cycle(c, QUEUE, state, work, k, "sha1", "me", client=client, auto_submit=True,
                          now=NOW + dt.timedelta(hours=1), fetch_board=board_rows, push=False)
    assert out["needs_fix"] and not client.submitted
    rows = [json.loads(x) for x in (state / "runs.jsonl").read_text().splitlines()]
    assert all(r["status"] == "failed" and "boom" in r["error"] for r in rows)


def test_cycle_respects_auto_submit_off_and_quota(tmp_path):
    c, state, work = cfg(), tmp_path / "state", tmp_path / "work"
    k = FakeKaggle()
    autopilot.cycle(c, QUEUE, state, work, k, "sha1", "me", now=NOW, fetch_board=board_rows)
    k.state, k.active_runs = "complete", json.loads((state / "active.json").read_text())["runs"]
    # a 25 h job leaves too little weekly quota for another 11.5 h session
    client = FakeClient()
    out = autopilot.cycle(c, QUEUE, state, work, k, "sha1", "me", client=client, auto_submit=False,
                          now=NOW + dt.timedelta(hours=25), fetch_board=board_rows)
    assert not client.submitted and any("AUTO_SUBMIT off" in n for n in out["notes"])
    assert out["pushed"] is None and any("quota" in n for n in out["notes"])


def test_board_failure_never_stops_the_loop(tmp_path):
    def broken():
        raise ConnectionError("down")
    out = autopilot.cycle(cfg(), QUEUE, tmp_path / "s", tmp_path / "w", FakeKaggle(), "sha", "me", now=NOW,
                          fetch_board=broken)
    assert out["pushed"] and any("leaderboard fetch failed" in n for n in out["notes"])
