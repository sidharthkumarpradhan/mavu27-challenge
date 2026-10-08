import datetime as dt
import json
import shutil
import tarfile
import zipfile
from pathlib import Path

import pytest

from reva import autopilot, data, package, registry, remote
from reva import config as C
from reva.codabench import CodabenchError
from reva.data import DEV_SET
from reva.kaggle import KaggleError, Push
from tests.conftest import make_rows

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


def test_run_ids_ignore_orchestration_settings():
    # regression: adding competition.organization re-queued the finished baseline under a new id
    a = remote.run_config(cfg(), QUEUE[0])
    for key, value in [("competition.organization", "Other"), ("remote.lanes", 1), ("submit.min_gain", 0.01)]:
        assert remote.run_config(C.override(cfg(), {key: value}), QUEUE[0])["run_id"] == a["run_id"]
    for key, value in [("frames.max_side", 320), ("remote.pip", ["transformers==9"])]:
        assert remote.run_config(C.override(cfg(), {key: value}), QUEUE[0])["run_id"] != a["run_id"]


def test_pending_skips_entries_done_under_an_older_id():
    now = remote.run_config(cfg(), QUEUE[0])["run_id"]
    queue = [{**QUEUE[0], "done_as": {now: "zs-4b-6254e8de"}}, *QUEUE[1:]]
    left = remote.pending(cfg(), queue, done={"zs-4b-6254e8de"}, failed={})
    assert [c["run_id"].rsplit("-", 1)[0] for c in left] == ["text-4b", "ft-4b"]
    # an edited experiment gets a new id, so the alias no longer matches and it runs again
    edited = [{**queue[0], "set": {"frames.n": 32}}]
    assert len(remote.pending(cfg(), edited, done={"zs-4b-6254e8de"}, failed={})) == 1


def test_shipped_queue_does_not_repeat_the_first_job():
    done = {"zs-4b-6254e8de", "text-4b-f571bdb5"}
    names = [c["run_id"].rsplit("-", 1)[0] for c in remote.pending(cfg(), remote.load_queue("configs/queue.yaml"), done, {})]
    assert "zs-4b" not in names and "text-4b" not in names


def test_pending_skips_done_and_repeat_failures():
    ids = [remote.run_config(cfg(), q)["run_id"] for q in QUEUE]
    left = remote.pending(cfg(), QUEUE, done={ids[0]}, failed={ids[1]: 2})
    assert [c["run_id"] for c in left] == [ids[2]]


def test_queue_edits_keep_the_running_job_ids():
    # job 2 was pushed as these ids on 7 Oct 2026. A new default in competition.yaml changes every
    # config hash, so the running lanes would look new and run a second time.
    left = remote.pending(cfg(), remote.load_queue("configs/queue.yaml"), done=set(), failed={})
    assert {"ft-4b-16f-830869b1", "zs-8b-4bit-029fe922"} <= {c["run_id"] for c in left}


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


def test_each_push_gets_its_own_kernel(tmp_path):
    """Regression (8 Oct 2026): a retry reused the first lane's slug, and the new version replaced
    the old one's output, so a finished run's probabilities and adapter were lost."""
    lanes = [{**remote.pending(cfg(), QUEUE, set(), {})[0], "run_id": "ft-8b-4bit-16f-62f77933" + "x" * 30}]
    a = remote.build(cfg(), lanes, "abc", tmp_path / "a", "someone", stamp="10082146")
    b = remote.build(cfg(), lanes, "abc", tmp_path / "b", "someone", stamp="10090311")
    assert a != b and a.endswith("-10082146") and b.endswith("-10090311")
    name = a.split("/", 1)[1]
    assert len(name) <= 50 and json.loads((tmp_path / "a" / "kernel-metadata.json").read_text())["title"] == name


def run_row(run_id, w, test_n=4000):
    return {"run_id": run_id, "status": "ok", "zip": f"{run_id}.zip", "n": {"test": test_n},
            "metrics": {"weighted_accuracy": w, "overall_accuracy": w}, "kernel": "u/k", "dev_set": DEV_SET}


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
    assert autopilot.gate(run_row("a", 0.5), failed, c, NOW)[0]  # a failed upload may be retried
    assert autopilot.pick_format(failed, c) == "id_map"
    all_failed = [sub_row("a", 0.5, status="Failed", fmt=f, sid=i) for i, f in enumerate(package.FORMATS)]
    assert "every" in autopilot.gate(run_row("a", 0.5), all_failed, c, NOW)[1]
    assert autopilot.pick_format(all_failed + [sub_row("z", 0.4, fmt="list", sid=9)], c) == "list"
    assert "still being scored" in autopilot.gate(run_row("b", 0.9), [sub_row("a", 0.5, status="Running")], c, NOW)[1]
    four = [sub_row(f"r{i}", 0.1 * i, sid=i) for i in range(4)]
    assert "daily" in autopilot.gate(run_row("b", 0.9), four, c, NOW)[1]
    many = [sub_row(f"r{i}", 0.001 * i, day="2026-10-01", sid=i) for i in range(85)]
    assert "budget" in autopilot.gate(run_row("b", 0.9), many, c, NOW)[1]  # reserve of 15 held back
    late = NOW.replace(month=11, day=6)
    assert autopilot.gate(run_row("b", 0.9), many, c, late)[0]  # final week may spend the reserve


TEST = make_rows("test", 1)
TEST_META = {"generated_at": "2026-04-25", "total_questions": len(TEST)}


def seed_annotations(work: Path) -> None:
    """The submit step reads test.json; tests give it a small one instead of the network."""
    ann = work / "annotations"
    ann.mkdir(parents=True, exist_ok=True)
    for split, rows in (("train", make_rows("train", 1)), ("val", make_rows("val", 1)), ("test", TEST)):
        (ann / f"{split}.json").write_text(json.dumps({"metadata": TEST_META, "QA": rows}))


RUN_CONFIG = {"why": "w", "model": {"use_video": True}, "train": {"refit_with_val": False}}


def probs_for(rows, letter=None):
    """A spread of answers like a real model's, or every answer on one letter."""
    out = {}
    for i, r in enumerate(rows):
        p = [0.1] * 4
        p[("ABCD".index(letter) if letter else i % 4)] = 0.7
        out[r["qa_id"]] = p
    return out


class FakeKaggle:
    def __init__(self, state="running", lanes_ok=True):
        self.state, self.lanes_ok, self.pushed = state, lanes_ok, []

    def status(self, slug):
        return self.state, ""

    def output(self, slug, dest, file_pattern=None):
        for run_id in self.active_runs:
            d = dest / run_id
            d.mkdir(parents=True, exist_ok=True)
            (d / "log.txt").write_text("Traceback: boom\n")
            if self.lanes_ok:
                (d / "run.json").write_text(json.dumps({
                    "metrics": {"weighted_accuracy": 0.6, "overall_accuracy": 0.61}, "dev_set": DEV_SET, "hours": 2.0,
                    "zip": f"{run_id}.zip",
                    "n": {"test": 4000}, "train": None, "timings": {}, "versions": {}, "config": RUN_CONFIG,
                    "finished": "2026-10-07T11:00:00Z"}))
                (d / f"{run_id}.zip").write_bytes(b"PK")
                (d / "test_probs.json").write_text(json.dumps(probs_for(TEST)))
        return [], "EXIT 0"

    def push(self, kdir, timeout_s=None, accelerator=None):
        self.pushed.append(json.loads((kdir / "kernel-metadata.json").read_text())["id"])
        return Push(1, "https://kaggle/k")


class FakeClient:
    def __init__(self, status="Finished", can=True, orgs=None):
        self.submitted, self.status, self.can = [], status, can
        self.orgs, self.organizations = {"StagAI": 5} if orgs is None else orgs, []

    def organization_id(self, name):
        if name not in self.orgs:
            raise CodabenchError(f"organization {name!r} not found for this account")
        return self.orgs[name]

    def can_submit(self, phase):
        return self.can, "" if self.can else "User not approved to participate in this competition"

    def submit(self, zip_path, comp, phase, tasks, organization=None):
        assert Path(zip_path).exists()
        self.submitted.append(zip_path)
        self.organizations.append(organization)
        return 99 + len(self.submitted)

    def wait(self, sid, timeout_s=0):
        scored = [{"column_key": "overall_accuracy", "score": "0.62"}] if self.status == "Finished" else []
        return {"status": self.status, "scores": scored}

    def submission(self, sid):
        return self.wait(sid)


def board_rows():
    return [{"owner": "leader", "submission": 1, "created": "", "overall_accuracy": 0.87}]


def test_cycle_push_collect_submit(tmp_path):
    c, state, work = C.override(cfg(), {"competition.organization": "StagAI"}), tmp_path / "state", tmp_path / "work"
    seed_annotations(work)
    k = FakeKaggle()
    out = autopilot.cycle(c, QUEUE, state, work, k, "sha1", "me", now=NOW, fetch_board=board_rows)
    assert out["pushed"] and len(out["pushed"]["runs"]) == 2 and k.pushed
    assert out["waiting"]  # the job runs on, so the workflow keeps cycling
    assert (state / "active.json").exists() and "leader" in (state / "STATUS.md").read_text()

    out = autopilot.cycle(c, QUEUE, state, work, k, "sha1", "me", now=NOW, fetch_board=board_rows)
    assert out["pushed"] is None and any("running" in n for n in out["notes"])

    k.state, k.active_runs = "complete", json.loads((state / "active.json").read_text())["runs"]
    client = FakeClient()
    later = NOW + dt.timedelta(hours=3)
    out = autopilot.cycle(c, QUEUE, state, work, k, "sha1", "me", client=client, auto_submit=True, now=later,
                          fetch_board=board_rows)
    assert len(out["collected"]) == 2 and not out["needs_fix"]
    assert len(client.submitted) == 1 and out["submitted"][0]["scores"] == {"overall_accuracy": 0.62}
    assert client.organizations == [5] and out["submitted"][0]["organization"] == "StagAI"
    assert package.validate(client.submitted[0], TEST, "fill_test") == len(TEST)  # rebuilt from test_probs
    with zipfile.ZipFile(client.submitted[0]) as z:  # same layout as test.json, metadata included
        assert json.loads(z.read("predictions.json"))["metadata"] == TEST_META
    assert out["pushed"]["runs"][0].startswith("ft-4b-")  # the next lane went out
    jobs = [json.loads(x) for x in (state / "jobs.jsonl").read_text().splitlines()]
    assert jobs[0]["hours"] == 3.0
    status = (state / "STATUS.md").read_text()
    assert "Calibration" in status and "Gap to the leader" in status


def test_cycle_waits_only_while_something_is_open(tmp_path):
    # GitHub never fired the cron (7 Oct 2026), so the workflow chains cycles while this is true
    state, work = tmp_path / "state", tmp_path / "work"
    out = autopilot.cycle(cfg(), QUEUE, state, work, FakeKaggle(), "sha", "me", now=NOW, fetch_board=board_rows,
                          push=False)
    assert out["waiting"] is False
    registry.append(state / "submissions.jsonl", sub_row("zs-4b-x", 0.6, status="Running"))
    out = autopilot.cycle(cfg(), QUEUE, state, work, FakeKaggle(), "sha", "me", now=NOW, fetch_board=board_rows,
                          push=False)
    assert out["waiting"] is True


class QuotaKaggle(FakeKaggle):
    def push(self, kdir, timeout_s=None, accelerator=None):
        raise KaggleError("kernel push failed: Maximum weekly GPU quota of 30.00 hours reached.")


def test_cycle_survives_a_refused_push_and_keeps_cycling(tmp_path):
    # regression: on 7 Oct 2026 the quota error crashed the cycle after collecting, so STATUS.md
    # was never written and the loop would have stopped for good
    state, work = tmp_path / "state", tmp_path / "work"
    out = autopilot.cycle(cfg(), QUEUE, state, work, QuotaKaggle(), "sha", "me", now=NOW, fetch_board=board_rows)
    assert out["pushed"] is None and out["waiting"] is True
    assert not (state / "active.json").exists()
    assert "GPU quota reached" in (state / "STATUS.md").read_text()


class FirstOutOfQuota(FakeKaggle):
    def push(self, kdir, timeout_s=None, accelerator=None):
        if json.loads((kdir / "kernel-metadata.json").read_text())["id"].startswith("first/"):
            raise KaggleError("kernel push failed: Maximum weekly GPU quota of 30.00 hours reached.")
        return super().push(kdir, timeout_s, accelerator)


def test_cycle_falls_back_to_the_second_account(tmp_path):
    # owner, 7 Oct 2026: the first account's weekly quota ran out, so the next job uses the second
    state, work, k = tmp_path / "state", tmp_path / "work", FirstOutOfQuota()
    out = autopilot.cycle(cfg(), QUEUE, state, work, k, "sha", ["first", "second"], now=NOW, fetch_board=board_rows)
    assert k.pushed[0].startswith("second/") and out["pushed"]["kernel"] == k.pushed[0]
    assert "push_refused" not in out and out["waiting"]  # the job runs on the second account
    assert json.loads((state / "active.json").read_text())["kernel"].startswith("second/")


def test_cycle_waits_when_every_account_is_out_of_quota(tmp_path):
    state, work = tmp_path / "state", tmp_path / "work"
    out = autopilot.cycle(cfg(), QUEUE, state, work, QuotaKaggle(), "sha", ["first", "second"], now=NOW,
                          fetch_board=board_rows)
    assert out["pushed"] is None and out["push_refused"] and out["waiting"]
    status = (state / "STATUS.md").read_text()
    assert "account 1" in status and "account 2" in status and "no Kaggle account can take the next job" in status


def test_gpu_pacing_counts_each_account_apart(tmp_path):
    state, work, k = tmp_path / "state", tmp_path / "work", FakeKaggle()
    registry.append(state / "jobs.jsonl", {"kernel": "first/reva-a", "hours": 26.0,
                                            "collected": autopilot.iso(NOW - dt.timedelta(days=1))})
    assert autopilot.gpu_hours(registry.read(state / "jobs.jsonl"), NOW, user="second") == 0
    autopilot.cycle(cfg(), QUEUE, state, work, k, "sha", ["first", "second"], now=NOW, fetch_board=board_rows)
    assert k.pushed[0].startswith("second/")  # the first account has 3.5 h left, too little for a job


def offline_row(run_id, finished, kernel="second/reva-a"):
    return {"run_id": run_id, "status": "failed", "kernel": kernel, "finished": autopilot.iso(finished), "sha": "s",
            "error": "fatal: unable to access 'https://github.com/x/': Could not resolve host: github.com"}


def test_no_internet_does_not_use_up_a_lanes_retries(tmp_path):
    # 7 Oct 2026: the second account had no internet, so both lanes failed in setup; two such
    # failures would have dropped the experiments from the queue for good
    state, work, k = tmp_path / "state", tmp_path / "work", FakeKaggle()
    lane = remote.pending(cfg(), QUEUE, set(), {})[0]["run_id"]
    for h in (3, 2):
        registry.append(state / "runs.jsonl", offline_row(lane, NOW - dt.timedelta(hours=h + 6)))
    out = autopilot.cycle(cfg(), QUEUE, state, work, k, "sha", ["first", "second"], now=NOW, fetch_board=board_rows)
    assert lane in out["pushed"]["runs"]


def test_offline_account_rests_then_gets_another_try(tmp_path):
    state, work, k = tmp_path / "state", tmp_path / "work", FakeKaggle()
    registry.append(state / "runs.jsonl", offline_row("x", NOW - dt.timedelta(hours=1)))
    out = autopilot.cycle(cfg(), QUEUE, state, work, k, "sha", ["second", "first"], now=NOW, fetch_board=board_rows)
    assert k.pushed[0].startswith("first/")  # the offline account is skipped
    assert "account 1 kernels have no internet" in (state / "STATUS.md").read_text()
    later = NOW + dt.timedelta(hours=6)

    # the only account is offline: the loop keeps cycling so the retry after 6 h happens
    state2 = tmp_path / "state2"
    registry.append(state2 / "runs.jsonl", offline_row("x", NOW - dt.timedelta(hours=1)))
    out = autopilot.cycle(cfg(), QUEUE, state2, work, FakeKaggle(), "sha", ["second"], now=NOW, fetch_board=board_rows)
    assert out["pushed"] is None and out["waiting"] is True
    assert autopilot.offline_until(registry.read(state / "runs.jsonl"), "second") <= later  # retried after 6 h


class BrokenKaggle(FakeKaggle):
    def push(self, kdir, timeout_s=None, accelerator=None):
        raise KaggleError("kernel push failed: Invalid credentials")


def test_cycle_fails_loudly_on_other_push_errors(tmp_path):
    with pytest.raises(KaggleError):
        autopilot.cycle(cfg(), QUEUE, tmp_path / "s", tmp_path / "w", BrokenKaggle(), "sha", "me", now=NOW,
                        fetch_board=board_rows)


def test_shipped_config_submits_as_the_personal_account():
    # owner, 7 Oct 2026: the account lacks participant rights in StagAI, so submit as the account itself
    assert C.get(cfg(), "competition.organization") == ""
    assert autopilot.submit_as(FakeClient(), cfg()) is None


def test_public_notes_drop_response_bodies():
    # STATUS.md is public; Codabench error text carries the response body after the status code
    e = CodabenchError('submission create failed (400): {"detail": "body"}')
    assert autopilot.public(e) == "submission create failed (400)"
    assert autopilot.public(CodabenchError("organization 'X' not found")) == "organization 'X' not found"


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
    # a 26 h job leaves 3.5 h of weekly quota, too little for even a short session
    client = FakeClient()
    out = autopilot.cycle(c, QUEUE, state, work, k, "sha1", "me", client=client, auto_submit=False,
                          now=NOW + dt.timedelta(hours=26), fetch_board=board_rows)
    assert not client.submitted and any("automatic submission off" in n for n in out["notes"])
    assert out["pushed"] is None and any("quota" in n for n in out["notes"])


def test_board_failure_never_stops_the_loop(tmp_path):
    def broken():
        raise ConnectionError("down")
    out = autopilot.cycle(cfg(), QUEUE, tmp_path / "s", tmp_path / "w", FakeKaggle(), "sha", "me", now=NOW,
                          fetch_board=broken)
    assert out["pushed"] and any("leaderboard fetch failed" in n for n in out["notes"])


def test_cycle_falls_back_to_the_next_format_and_reports_refusals(tmp_path):
    c, state, work = cfg(), tmp_path / "state", tmp_path / "work"
    seed_annotations(work)
    k = FakeKaggle()
    autopilot.cycle(c, QUEUE, state, work, k, "sha1", "me", now=NOW, fetch_board=board_rows)
    k.state, k.active_runs = "complete", json.loads((state / "active.json").read_text())["runs"]
    refused = FakeClient(can=False)
    out = autopilot.cycle(c, QUEUE, state, work, k, "sha1", "me", client=refused, auto_submit=True,
                          now=NOW + dt.timedelta(hours=1), fetch_board=board_rows, push=False)
    assert not refused.submitted and "not approved" in out["submit_error"]
    failing = FakeClient(status="Failed")
    out = autopilot.cycle(c, QUEUE, state, work, k, "sha1", "me", client=failing, auto_submit=True,
                          now=NOW + dt.timedelta(hours=2), fetch_board=board_rows, push=False)
    assert out["submitted"][0]["format"] == "fill_test" and out["submitted"][0]["status"] == "Failed"
    shutil.rmtree(work / k.pushed[0].split("/")[-1])  # a new runner: test_probs must be fetched again
    ok = FakeClient()
    out = autopilot.cycle(c, QUEUE, state, work, k, "sha1", "me", client=ok, auto_submit=True,
                          now=NOW + dt.timedelta(hours=3), fetch_board=board_rows, push=False)
    assert out["submitted"][0]["format"] == "id_map" and out["submitted"][0]["status"] == "Finished"
    assert package.validate(ok.submitted[0], TEST, "id_map") == len(TEST)


def test_unknown_organization_blocks_the_submission(tmp_path):
    c, state, work = C.override(cfg(), {"competition.organization": "StagAI"}), tmp_path / "state", tmp_path / "work"
    seed_annotations(work)
    k = FakeKaggle()
    autopilot.cycle(c, QUEUE, state, work, k, "sha1", "me", now=NOW, fetch_board=board_rows)
    k.state, k.active_runs = "complete", json.loads((state / "active.json").read_text())["runs"]
    outsider = FakeClient(orgs={})
    out = autopilot.cycle(c, QUEUE, state, work, k, "sha1", "me", client=outsider, auto_submit=True,
                          now=NOW + dt.timedelta(hours=1), fetch_board=board_rows, push=False)
    assert not outsider.submitted and "StagAI" in out["submit_error"]
    personal = FakeClient()
    out = autopilot.cycle(C.override(c, {"competition.organization": ""}), QUEUE, state, work, k, "sha1", "me",
                          client=personal, auto_submit=True, now=NOW + dt.timedelta(hours=2),
                          fetch_board=board_rows, push=False)
    assert personal.organizations == [None]


def test_caller_notes_reach_status(tmp_path):
    autopilot.cycle(cfg(), QUEUE, tmp_path / "s", tmp_path / "w", FakeKaggle(), "sha", "me", now=NOW,
                    fetch_board=board_rows, notes=["Codabench login ok"])
    assert "- Codabench login ok" in (tmp_path / "s" / "STATUS.md").read_text()


def test_a_broken_image_does_not_use_up_a_lanes_retries():
    # 8 Oct 2026: every fine-tune died adding LoRA because the Kaggle image's torchao was too old
    # for peft; two such failures dropped the 32-frame run from the queue
    torchao = "ImportError: Found an incompatible version of torchao. Found version 0.10.0"
    runs = [{"run_id": "a", "status": "failed", "error": torchao}] * 2 + [
        {"run_id": "b", "status": "failed", "error": "RuntimeError: inference alone needs 12.7 h"},
        {"run_id": "c", "status": "failed", "error": torchao}]
    assert autopilot.failures(runs) == {"a": 0, "b": 1, "c": 0}
    many = [{"run_id": "a", "status": "failed", "error": torchao}] * (autopilot.ENV_FREE_TRIES + 2)
    assert autopilot.failures(many) == {"a": 2}  # a fix that does not work cannot loop forever


def test_kernel_removes_the_images_torchao():
    script = remote.kernel_script([], "sha", "https://example.com/repo", ["peft==0.21.2"])
    assert script.index("pip uninstall -y -q torchao") > script.index("pip install -q ")


class TimedKaggle(FakeKaggle):
    def push(self, kdir, timeout_s=None, accelerator=None):
        self.timeout_s, self.script = timeout_s, (kdir / "job.py").read_text()
        return super().push(kdir, timeout_s, accelerator)


def test_a_job_shrinks_to_the_hours_an_account_has_left(tmp_path):
    # 8 Oct 2026: after one full job the second account had about 6 h left of its 30, and pacing
    # would have left it idle for days; a shorter job trains less but still runs
    state, work, k = tmp_path / "state", tmp_path / "work", TimedKaggle()
    registry.append(state / "jobs.jsonl", {"kernel": "second/reva-a", "hours": 24.0,
                                            "collected": autopilot.iso(NOW - dt.timedelta(hours=1))})
    out = autopilot.cycle(cfg(), QUEUE, state, work, k, "sha", ["second"], now=NOW, fetch_board=board_rows)
    assert out["pushed"]["hours"] == 5.5 and k.timeout_s == 5.5 * 3600  # 30 - 24 - 0.5
    lanes = json.loads(k.script.split("LANES = json.loads(")[1].split(")\n")[0])
    lanes = json.loads(lanes)
    assert {l["job"]["max_hours"] for l in lanes} == {4.5}  # the usual hour for setup
    assert out["pushed"]["runs"] == [l["run_id"] for l in lanes]  # same experiments, same ids
    assert "sized to the 5.5 h account 1 has left" in (state / "STATUS.md").read_text()


def test_an_account_with_too_little_left_still_waits(tmp_path):
    state, work, k = tmp_path / "state", tmp_path / "work", FakeKaggle()
    registry.append(state / "jobs.jsonl", {"kernel": "second/reva-a", "hours": 26.0,
                                            "collected": autopilot.iso(NOW - dt.timedelta(hours=1))})
    out = autopilot.cycle(cfg(), QUEUE, state, work, k, "sha", ["second"], now=NOW, fetch_board=board_rows)
    assert out["pushed"] is None and not k.pushed  # 3.5 h cannot train and score


class SavingKaggle(FakeKaggle):
    """Lanes that stop part way: one spanning run saved at a session's end, one crashed after a checkpoint."""

    def __init__(self):
        super().__init__(lanes_ok=False)
        self.datasets, self.downloads = {}, []

    def output(self, slug, dest, file_pattern=None):
        super().output(slug, dest, file_pattern)
        first, second = self.active_runs[:2]
        (dest / first / "ckpt" / "adapter").mkdir(parents=True)
        (dest / first / "ckpt" / "state.pt").write_bytes(b"state")
        (dest / first / "partial.json").write_text(json.dumps({"where": "training: step 40"}))
        (dest / second / "train.json").write_text("{}")
        (dest / second / "test_probs.part.json").write_text("{}")  # never leaves the kernel output
        (dest / f"{slug.split('/')[-1]}.log").write_text("kernel log")
        return [], "EXIT 3"

    def dataset_upload(self, folder, ref, message, **kwargs):
        with tarfile.open(next(Path(folder).glob("*.tar"))) as t:
            self.datasets[ref] = sorted(t.getnames())

    def dataset_download(self, ref, dest):
        self.downloads.append(ref)
        run_id = ref.split("reva-run-")[1]
        (Path(dest) / run_id / "ckpt").mkdir(parents=True)
        (Path(dest) / run_id / "ckpt" / "state.pt").write_bytes(b"state")
        return Path(dest)


def test_unfinished_runs_are_saved_and_the_next_push_resumes_them(tmp_path):
    """Owner, 8 Oct 2026: a quota cut or any other failure must not lose what a run got done."""
    c, state, work, k = cfg(), tmp_path / "state", tmp_path / "work", SavingKaggle()
    autopilot.cycle(c, QUEUE, state, work, k, "sha1", "first", now=NOW, fetch_board=board_rows)
    k.state, k.active_runs = "complete", json.loads((state / "active.json").read_text())["runs"]
    out = autopilot.cycle(c, QUEUE, state, work, k, "sha1", "first", now=NOW + dt.timedelta(hours=11),
                          fetch_board=board_rows, push=False)
    first, second = k.active_runs[:2]
    rows = {r["run_id"]: r for r in registry.read(state / "runs.jsonl")}
    assert rows[first]["status"] == "partial" and rows[first]["where"] == "training: step 40"
    assert rows[second]["status"] == "failed" and rows[second]["resumable"]
    assert rows[first]["stash"] == f"first/reva-run-{first}" and f"{first}/ckpt/state.pt" in k.datasets[rows[first]["stash"]]
    saved = k.datasets[rows[second]["stash"]]
    assert f"{second}/kernel.log" in saved and f"{second}/log.txt" in saved and f"{second}/train.json" in saved
    assert not [n for n in saved if "test_probs" in n]
    assert autopilot.failures(list(rows.values())) == {first: 0, second: 1}

    # the next push goes to the other account: the state is copied there and attached
    out = autopilot.cycle(c, QUEUE, state, work, k, "sha2", "second", now=NOW + dt.timedelta(hours=12),
                          fetch_board=board_rows)
    resume = out["pushed"]["resume"]
    assert resume == {first: f"second/reva-run-{first}", second: f"second/reva-run-{second}"}
    assert k.downloads == [f"first/reva-run-{first}", f"first/reva-run-{second}"]
    meta = json.loads((work / "kernel" / "kernel-metadata.json").read_text())
    assert meta["dataset_sources"] == sorted(resume.values())
    src = (work / "kernel" / "job.py").read_text()
    compile(src, "job.py", "exec")
    assert f"second/reva-run-{first}" in src and "restore(RESUME" in src


def test_partial_sessions_count_only_past_the_free_ones():
    rows = [{"run_id": "a", "status": "partial"}] * (autopilot.FREE_SESSIONS + 1)
    assert autopilot.failures(rows) == {"a": 1}


def test_restore_copies_saved_state_where_the_job_resumes(tmp_path, capsys):
    inp, work = tmp_path / "input", tmp_path / "working"
    unpacked = inp / "datasets" / "first" / "reva-run-a" / "a" / "ckpt"
    unpacked.mkdir(parents=True)
    (unpacked / "state.pt").write_bytes(b"s")
    packed = inp / "datasets" / "first" / "reva-run-b"
    packed.mkdir(parents=True)
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "train.json").write_text("{}")
    with tarfile.open(packed / "b.tar", "w") as t:
        t.add(tmp_path / "b", arcname="b")
    found = remote.restore({"a": "first/reva-run-a", "b": "first/reva-run-b", "c": "first/reva-run-c"}, inp, work)
    assert set(found) == {"a", "b"}
    assert (work / "a" / "ckpt" / "state.pt").read_bytes() == b"s" and (work / "b" / "train.json").exists()
    assert "no saved state for c" in capsys.readouterr().out  # a missing dataset starts fresh, never fails


def test_stash_names_fit_kaggle():
    ref = autopilot.stash_ref("Second", "ft-8b-4bit-16f-62f77933" + "x" * 40)
    assert ref.startswith("second/reva-run-ft-8b") and len(ref.split("/")[1]) <= 50


def test_older_runs_are_rescored_on_the_clean_dev_set_before_any_comparison(tmp_path, ann):
    """Runs recorded before dev dropped the train copies get their dev score again from their own
    dev_probs.json; a run whose probabilities are gone has no fair score and is never picked."""
    c, state, work = C.override(cfg(), {"dev.holdout_frac": 0.1, "dev.seed": 0, "dev.min_cell": 1}), tmp_path / "s", tmp_path / "w"
    shutil.copytree(ann, work / "annotations")
    old_dev = data.load_split(ann, "val") + data.load_split(ann, "train")  # covers any dev the splits make
    letters = {r["qa_id"]: r.get("correct_answer") or "A" for r in old_dev}

    class K(FakeKaggle):
        def output(self, slug, dest, file_pattern=None):
            if "gone" in slug:
                return [], ""
            (dest / "old").mkdir(parents=True, exist_ok=True)
            (dest / "old" / "dev_probs.json").write_text(json.dumps(
                {q: [1.0 if L == a else 0.0 for L in "ABCD"] for q, a in letters.items()}))
            return [], ""

    old = {**run_row("old", 0.99), "kernel": "u/k-old"}
    gone = {**run_row("gone", 0.98), "kernel": "u/k-gone"}
    del old["dev_set"], gone["dev_set"]
    state.mkdir()
    for r in (old, gone):
        registry.append(state / "runs.jsonl", r)
    clean = autopilot.rescore(K(), registry.read(state / "runs.jsonl"), state, work, c)
    assert clean["old"]["metrics"]["weighted_accuracy"] == 1.0 and "missing" in clean["gone"]
    runs = autopilot.fair(registry.read(state / "runs.jsonl"), clean)
    assert autopilot.candidate(runs, [])["run_id"] == "old"
    assert autopilot.candidate(runs, [sub_row("old", 0.99)]) is None  # "gone" is shown but never picked
    subs = autopilot.fair_subs([sub_row("gone", 0.98)], runs)
    assert subs[0]["dev_weighted"] is None and autopilot.gate(run_row("new", 0.5), subs, cfg(), NOW)[0]
