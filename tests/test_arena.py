import datetime as dt
import json
import zipfile

import pytest

from reva import arena, autopilot, data
from reva import config as C
from reva.data import DEV_SET
from tests.conftest import make_rows
from tests.test_remote_autopilot import (NOW, QUEUE, TEST, FakeClient, FakeKaggle, board_rows, cfg, probs_for,
                                         seed_annotations)

DEV = make_rows("val", 2)
TEST_ROWS = make_rows("test", 1)


def onehot(letter, conf):
    p = [(1 - conf) / 3] * 4
    p["ABCD".index(letter)] = conf
    return p


def wrong(letter):
    return "ABCD"[("ABCD".index(letter) + 1) % 4]


def split_probs(rows, right_first_half):
    """Confidently right on one half of the questions and unsure-and-wrong on the other."""
    half = len(rows) // 2
    out = {}
    for i, r in enumerate(rows):
        right = (i < half) == right_first_half
        out[r["qa_id"]] = onehot(r["correct_answer"], 0.9) if right else onehot(wrong(r["correct_answer"]), 0.4)
    return out


def run(run_id, w, **cfg):
    return {"run_id": run_id, "status": "ok", "zip": "z", "n": {"test": 4000}, "kernel": f"u/{run_id}",
            "metrics": {"weighted_accuracy": w}, "dev_set": DEV_SET, "config": {"model.use_video": True, "train.refit_with_val": False, **cfg}}


def test_only_fair_single_runs_compete():
    runs = [run("a", 0.5), run("b", 0.7), run("text", 0.9, **{"model.use_video": False}),
            run("refit", 0.95, **{"train.refit": True}), run("val", 0.96, **{"train.refit_with_val": True}),
            {**run("ens", 0.99), "members": [{}]},
            {**run("part", 0.8), "n": {"test": 10}}, {**run("bad", 0.8), "status": "failed"},
            run("old", 0.97, **{"train.refit_with_val": None}), {**run("bare", 0.98), "config": {}}]
    assert [r["run_id"] for r in arena.eligible(runs)] == ["b", "a"]


def test_mean_and_its_guard():
    assert arena.mean([{"q": [1, 0, 0, 0]}, {"q": [0, 1, 0, 0]}]) == {"q": [0.5, 0.5, 0, 0]}
    with pytest.raises(ValueError):
        arena.mean([{"q": [1, 0, 0, 0]}, {"r": [1, 0, 0, 0]}])


def test_paired_chance_of_beating():
    right = {r["qa_id"]: r["correct_answer"] for r in DEV}
    half = {r["qa_id"]: (r["correct_answer"] if i % 2 else wrong(r["correct_answer"])) for i, r in enumerate(DEV)}
    assert arena.p_better(DEV, right, half) > 0.99 and arena.p_better(DEV, half, right) < 0.01
    assert arena.p_better(DEV, half, half) == 0.5


def test_singles_are_rescored_before_they_are_compared():
    a, b = run("a", 0.99), run("b", 0.1)  # stored scores from another scoring setting
    probs = {"a": split_probs(DEV, True), "b": split_probs(DEV, False)}
    found = arena.best_ensemble([a, b], probs, DEV, TEST_ROWS, min_cell=1)
    assert found["best_weighted"] == 0.5 and found["metrics"]["weighted_accuracy"] == 1.0


def test_complementary_runs_make_a_better_ensemble():
    a, b = run("a", 0.5), run("b", 0.5)
    probs = {"a": split_probs(DEV, True), "b": split_probs(DEV, False)}
    found = arena.best_ensemble([a, b], probs, DEV, TEST_ROWS, min_cell=1)
    assert [m["run_id"] for m in found["members"]] == ["a", "b"] and found["metrics"]["weighted_accuracy"] == 1.0
    assert found["p"] > 0.99
    same = {"a": probs["a"], "b": probs["a"]}
    assert arena.best_ensemble([a, b], same, DEV, TEST_ROWS, min_cell=1) is None  # no gain, no candidate


class ProbsKaggle:
    def __init__(self, probs):
        self.probs, self.calls = probs, 0

    def output(self, slug, dest, file_pattern=None):
        self.calls += 1
        run_id = slug.split("/")[-1]
        (dest / run_id).mkdir(parents=True, exist_ok=True)
        (dest / run_id / "dev_probs.json").write_text(json.dumps(self.probs[run_id]))


def test_step_reruns_only_when_a_new_run_arrives(tmp_path):
    runs = [run("a", 0.5), run("b", 0.5)]
    k = ProbsKaggle({"a": split_probs(DEV, True), "b": split_probs(DEV, False)})
    row, seen, note = arena.step(k, runs, DEV, TEST_ROWS, tmp_path, [], "t", min_cell=1)
    assert row["run_id"] == arena.ensemble_id(["a", "b"]) and row["metrics"]["weighted_accuracy"] == 1.0
    assert seen == ["a", "b"] and "mean of 2 runs" in note and k.calls == 2
    assert arena.step(k, runs + [row], DEV, TEST_ROWS, tmp_path, seen, "t", min_cell=1)[0] is None
    assert k.calls == 2  # nothing new, nothing downloaded


def test_a_run_on_another_dev_split_is_left_out(tmp_path):
    runs = [run("a", 0.6), run("b", 0.5), run("c", 0.4)]
    other = {f"x{i}": [0.25] * 4 for i in range(3)}
    k = ProbsKaggle({"a": split_probs(DEV, True), "b": split_probs(DEV, False), "c": other})
    row, _, _ = arena.step(k, runs, DEV, TEST_ROWS, tmp_path, [], "t", min_cell=1)
    assert [m["run_id"] for m in row["members"]] == ["a", "b"]


def member_test_probs(i, rows=TEST):
    """Each member alone answers a wrong letter; their mean answers probs_for's letter."""
    out = {}
    for q, p in probs_for(rows).items():
        t = max(range(4), key=p.__getitem__)
        lean = (t + 1 + i) % 4
        v = [0.05] * 4
        v[t], v[lean] = 0.4, 0.5
        out[q] = v
    return out


def ensemble_cycle(tmp_path, member_probs=member_test_probs):
    c, state, work = cfg(), tmp_path / "state", tmp_path / "work"
    seed_annotations(work)
    c = C.override(c, {"dev.min_cell": 1})
    dev = data.make_splits(work / "annotations", C.get(c, "dev.holdout_frac"), C.get(c, "dev.seed"))["dev"]
    k = FakeKaggle()
    autopilot.cycle(c, QUEUE, state, work, k, "sha1", "me", now=NOW, fetch_board=board_rows)
    k.state, k.active_runs = "complete", json.loads((state / "active.json").read_text())["runs"]
    real = k.output

    def lanes(slug, dest, file_pattern=None):  # right on opposite halves of dev
        out = real(slug, dest)
        for i, run_id in enumerate(k.active_runs):
            (dest / run_id / "dev_probs.json").write_text(json.dumps(split_probs(dev, i == 0)))
            (dest / run_id / "test_probs.json").write_text(json.dumps(member_probs(i)))
        return out
    k.output = lanes
    client = FakeClient()
    out = autopilot.cycle(c, QUEUE, state, work, k, "sha1", "me", client=client, auto_submit=True,
                          now=NOW + dt.timedelta(hours=1), fetch_board=board_rows, push=False)
    return out, client, state, k


def test_a_broken_member_blocks_the_ensemble_instead_of_crashing(tmp_path):
    out, client, state, k = ensemble_cycle(tmp_path, lambda i: member_test_probs(i, TEST[i:]))
    ens = arena.ensemble_id(k.active_runs)
    assert not client.submitted and f"{k.active_runs[1]}: probabilities: 1 test questions have none" in out["submit_error"]
    assert json.loads((state / "blocked.jsonl").read_text().splitlines()[0])["run_id"] == ens


def test_cycle_submits_the_ensemble_when_it_wins_on_dev(tmp_path):
    out, client, state, k = ensemble_cycle(tmp_path)
    ens = arena.ensemble_id(k.active_runs)
    assert out["submitted"][0]["run_id"] == ens and out["submitted"][0]["dev_weighted"] == 1.0
    assert sorted(out["submitted"][0]["members"]) == sorted(k.active_runs) and out["submitted"][0]["kernel"] is None
    with zipfile.ZipFile(client.submitted[0]) as z:
        got = {r["qa_id"]: r["correct_answer"] for r in json.loads(z.read("predictions.json"))["QA"]}
    assert got == arena.argmax(probs_for(TEST))  # the mean: either member alone answers otherwise
    assert all(got != arena.argmax(member_test_probs(i)) for i in range(2))
    assert json.loads((state / "arena.json").read_text())["considered"] == sorted(k.active_runs)
    status = (state / "STATUS.md").read_text()
    assert ens in status and "mean of top 2" in status


def test_only_the_needed_file_is_downloaded(tmp_path):
    import re

    calls = []

    class K:
        def output(self, slug, dest, file_pattern=None):
            calls.append(file_pattern)
            (dest / "a").mkdir(parents=True)
            (dest / "a" / "run.json").write_text("{}")
    arena.fetch(K(), {"run_id": "a", "kernel": "u/k"}, tmp_path, "run.json")
    pattern = re.compile(calls[0])
    assert pattern.search("a/run.json") and not pattern.search("a/a.zip") and not pattern.search("ba/run.json")
