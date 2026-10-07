import datetime as dt
import json
import math

from reva import autopilot, package, preflight
from reva.preflight import live_columns  # the real one: conftest swaps preflight.live_columns for a fake
from reva.score import TASK_COLUMNS
from tests.conftest import BOARD_COLUMNS, make_rows
from tests.test_remote_autopilot import (NOW, QUEUE, RUN_CONFIG, TEST, FakeClient, FakeKaggle, board_rows, cfg,
                                         probs_for, seed_annotations)


ROWS = make_rows("test", 2)


def zip_of(tmp_path, probs, rows=ROWS, fmt="fill_test"):
    preds = {q: "ABCD"[max(range(4), key=p.__getitem__)] for q, p in probs.items()}
    return package.write(tmp_path / "s.zip", rows, preds, fmt, {"total_questions": len(rows)})


def test_a_healthy_run_is_ready(tmp_path):
    probs = probs_for(ROWS)
    assert preflight.check(zip_of(tmp_path, probs), ROWS, probs, "fill_test", [RUN_CONFIG]) == []


def test_zip_must_hold_the_runs_own_answers(tmp_path):
    probs = probs_for(ROWS)
    z = zip_of(tmp_path, probs)
    other = {**probs, ROWS[0]["qa_id"]: [0.0, 0.0, 0.0, 1.0] if probs[ROWS[0]["qa_id"]][3] < 0.5 else [1.0, 0, 0, 0]}
    assert any("round trip: 1 answers" in p for p in preflight.check(z, ROWS, other, "fill_test", [RUN_CONFIG]))


def test_one_letter_for_almost_everything_is_a_broken_run(tmp_path):
    probs = probs_for(ROWS, letter="A")
    problems = preflight.check(zip_of(tmp_path, probs), ROWS, probs, "fill_test", [RUN_CONFIG])
    assert problems == ["letter balance: A 100.0%, B 0.0%, C 0.0%, D 0.0% of answers (expected 5% to 60% each)"]
    assert not any(r["qa_id"] in problems[0] for r in ROWS)  # STATUS.md is public: no ids, no answers


def test_compliance_rules_block_the_test_set(tmp_path):
    probs = probs_for(ROWS)
    z = zip_of(tmp_path, probs)

    def problems(model, train):
        return preflight.check(z, ROWS, probs, "fill_test", [{"model": model, "train": train}])
    assert problems({"use_video": False}, {"refit_with_val": False}) == [
        "compliance: a text-only run never goes to the test set"]
    assert problems({"use_video": True}, {"refit_with_val": True}) == [
        "compliance: train.refit_with_val is the owner's call and stays false"]
    # no evidence, no upload: a run whose config lacks a flag is not trusted
    assert problems({}, {}) == ["compliance: the run does not record model.use_video",
                                "compliance: the run does not record train.refit_with_val"]
    ensemble = [RUN_CONFIG, {"model": {"use_video": False}, "train": {"refit_with_val": False}}]
    assert "text-only" in preflight.check(z, ROWS, probs, "fill_test", ensemble)[0]


def test_bad_probabilities_and_bad_files_are_caught(tmp_path):
    probs = probs_for(ROWS)
    z = zip_of(tmp_path, probs)
    nan = {**probs, ROWS[0]["qa_id"]: [math.nan, 0.1, 0.1, 0.1]}
    assert "probabilities: 1 rows" in preflight.check(z, ROWS, nan, "fill_test", [RUN_CONFIG])[0]
    short_probs = {q: p for q, p in probs.items() if q != ROWS[0]["qa_id"]}
    assert preflight.probs_problems(short_probs, ROWS) == ["probabilities: 1 test questions have none"]
    assert preflight.probs_problems({**probs, "x": [1, 0, 0, 0], ROWS[1]["qa_id"]: []}, ROWS) == [
        "probabilities: 1 are for questions not in test.json", "probabilities: 1 rows are not 4 finite non-negative numbers"]
    short = zip_of(tmp_path, probs_for(ROWS[1:]), rows=ROWS[1:])
    assert preflight.check(short, ROWS, probs, "fill_test", [RUN_CONFIG])[0].startswith("format: 1 test qa_ids missing")


def test_metric_must_match_the_live_board():
    assert preflight.metric_problem(BOARD_COLUMNS) is None
    assert "update reva.score" in preflight.metric_problem(("temporalgrounding_accuracy", BOARD_COLUMNS[1]))
    assert preflight.metric_problem(("overall_accuracy", {"overall_accuracy"}))


def test_live_columns_reads_the_competition_api(monkeypatch):
    cols = [{"key": k, "index": i} for i, k in enumerate(["overall_accuracy", *TASK_COLUMNS.values()])][::-1]

    class R:
        def raise_for_status(self):
            pass

        def json(self):
            return {"leaderboards": [{"primary_index": 0, "columns": cols}]}

    monkeypatch.setattr(preflight.requests, "get", lambda url, timeout: R())
    assert live_columns(18274, "https://x") == BOARD_COLUMNS


def test_projection_uses_scored_submissions_only():
    subs = [{"status": "Finished", "scores": {"overall_accuracy": 0.80}, "dev_weighted": 0.78},
            {"status": "Finished", "scores": {"overall_accuracy": 0.84}, "dev_weighted": 0.80},
            {"status": "Failed", "scores": {}, "dev_weighted": 0.90}]
    p, k = preflight.projected(0.85, subs)
    assert k == 2 and abs(p - 0.88) < 1e-9
    assert preflight.projected(0.85, []) == (0.85, 0)
    run = {"run_id": "r", "metrics": {"weighted_accuracy": 0.85}, "n": {"dev": 3000}}
    line = preflight.forecast(run, subs, 0.8735)
    assert "projected board 0.8800 (2 calibration pairs)" in line and "above by 0.0065" in line
    assert "+/- 0.0128" in line  # 1.96 * sqrt(0.85 * 0.15 / 3000)


def collected(tmp_path, letter=None):
    c, state, work = cfg(), tmp_path / "state", tmp_path / "work"
    seed_annotations(work)
    k = FakeKaggle()
    autopilot.cycle(c, QUEUE, state, work, k, "sha1", "me", now=NOW, fetch_board=board_rows)
    k.state, k.active_runs = "complete", json.loads((state / "active.json").read_text())["runs"]
    if letter:
        real = k.output

        def one_letter(slug, dest, file_pattern=None):
            out = real(slug, dest)
            for run_id in k.active_runs:
                (dest / run_id / "test_probs.json").write_text(json.dumps(probs_for(TEST, letter)))
            return out
        k.output = one_letter
    return c, state, work, k


def test_cycle_blocks_a_broken_run_once_and_moves_on(tmp_path):
    c, state, work, k = collected(tmp_path, letter="B")
    client = FakeClient()
    out = autopilot.cycle(c, QUEUE, state, work, k, "sha1", "me", client=client, auto_submit=True,
                          now=NOW + dt.timedelta(hours=1), fetch_board=board_rows, push=False)
    assert not client.submitted and "pre-upload checks failed: letter balance" in out["submit_error"]
    blocked = [json.loads(x) for x in (state / "blocked.jsonl").read_text().splitlines()]
    assert len(blocked) == 1 and "projected board" in (state / "STATUS.md").read_text()
    out = autopilot.cycle(c, QUEUE, state, work, k, "sha1", "me", client=client, auto_submit=True,
                          now=NOW + dt.timedelta(hours=2), fetch_board=board_rows, push=False)
    assert "letter balance" in out["submit_error"]  # the other lane of the same broken job
    assert {b["run_id"] for b in map(json.loads, (state / "blocked.jsonl").read_text().splitlines())} == set(k.active_runs)
    out = autopilot.cycle(c, QUEUE, state, work, k, "sha1", "me", client=client, auto_submit=True,
                          now=NOW + dt.timedelta(hours=3), fetch_board=board_rows, push=False)
    assert "submit_error" not in out and not client.submitted  # nothing left to try


def test_cycle_holds_every_upload_when_the_board_metric_changes(tmp_path):
    c, state, work, k = collected(tmp_path)
    client = FakeClient()
    out = autopilot.cycle(c, QUEUE, state, work, k, "sha1", "me", client=client, auto_submit=True,
                          now=NOW + dt.timedelta(hours=1), fetch_board=board_rows, push=False,
                          fetch_columns=lambda: ("weighted_score", {"weighted_score"}))
    assert not client.submitted and "update reva.score" in out["submit_error"]
    assert not (state / "blocked.jsonl").exists()  # not the run's fault, so it stays a candidate

    def down():
        raise ConnectionError("<html>secret body</html>")
    out = autopilot.cycle(c, QUEUE, state, work, k, "sha1", "me", client=client, auto_submit=True,
                          now=NOW + dt.timedelta(hours=2), fetch_board=board_rows, push=False, fetch_columns=down)
    assert "could not read the live leaderboard columns" in out["submit_error"]
    assert "secret" not in (state / "STATUS.md").read_text()


def test_cycle_blocks_a_run_with_missing_probabilities_instead_of_crashing(tmp_path):
    c, state, work, k = collected(tmp_path)
    real = k.output

    def short(slug, dest, file_pattern=None):
        out = real(slug, dest)
        for run_id in k.active_runs:
            (dest / run_id / "test_probs.json").write_text(json.dumps(probs_for(TEST[1:])))
        return out
    k.output = short
    client = FakeClient()
    out = autopilot.cycle(c, QUEUE, state, work, k, "sha1", "me", client=client, auto_submit=True,
                          now=NOW + dt.timedelta(hours=1), fetch_board=board_rows, push=False)
    assert not client.submitted and "1 test questions have none" in out["submit_error"]
    assert (state / "blocked.jsonl").exists() and (state / "STATUS.md").exists()


def test_cycle_blocks_a_run_whose_config_lacks_the_compliance_flags(tmp_path):
    c, state, work, k = collected(tmp_path)
    real = k.output

    def old_run(slug, dest, file_pattern=None):
        out = real(slug, dest)
        for run_id in k.active_runs:
            rj = dest / run_id / "run.json"
            rj.write_text(json.dumps({**json.loads(rj.read_text()), "config": {"why": "w"}}))
        return out
    k.output = old_run
    client = FakeClient()
    out = autopilot.cycle(c, QUEUE, state, work, k, "sha1", "me", client=client, auto_submit=True,
                          now=NOW + dt.timedelta(hours=1), fetch_board=board_rows, push=False)
    assert not client.submitted and "does not record model.use_video" in out["submit_error"]


def test_every_test_module_answers_for_the_live_board():
    # regression: the fake lived in one module's fixture, so this module's cycle tests called Codabench
    assert preflight.live_columns(18274, "http://unreachable.invalid") == BOARD_COLUMNS
