from reva import autopilot, cli
from reva.codabench import CodabenchError


class LeakyClient:
    def can_submit(self, phase):
        return True, ""

    def organization_id(self, name):
        raise CodabenchError("organizations failed (500): <html>OtherTeam secret page</html>")


def test_failed_organization_lookup_keeps_error_text_out_of_status(tmp_path, monkeypatch):
    seen = {}

    def fake_cycle(*args, notes=None, **kw):
        seen["notes"] = notes
        return {"notes": notes}

    monkeypatch.setattr(cli, "_client", lambda cfg: LeakyClient())
    monkeypatch.setattr(autopilot, "cycle", fake_cycle)
    assert cli.main(["--set", "competition.organization=StagAI", "autopilot", "--state", str(tmp_path / "s"),
                     "--work", str(tmp_path / "w"), "--sha", "x"]) == 0
    note = seen["notes"][0]
    assert "could not confirm organization StagAI" in note
    assert "OtherTeam" not in note and "500" not in note  # STATUS.md is public


class LeakyLogin(LeakyClient):
    def can_submit(self, phase):
        raise CodabenchError("phase failed (502): <html>OtherTeam secret page</html>")


def test_failed_login_check_keeps_the_response_body_out_of_status(tmp_path, monkeypatch):
    seen = {}

    def fake_cycle(*args, notes=None, **kw):
        seen["notes"] = notes
        return {"notes": notes}

    monkeypatch.setattr(cli, "_client", lambda cfg: LeakyLogin())
    monkeypatch.setattr(autopilot, "cycle", fake_cycle)
    assert cli.main(["autopilot", "--state", str(tmp_path / "s"), "--work", str(tmp_path / "w"), "--sha", "x"]) == 0
    assert seen["notes"][0] == "Codabench login failed: phase failed (502)"


def test_manual_submit_runs_the_pre_upload_checks(tmp_path, monkeypatch):
    import json

    import reva.kaggle
    from reva import preflight, registry
    from tests.conftest import BOARD_COLUMNS
    from tests.test_remote_autopilot import TEST, FakeClient, FakeKaggle, probs_for, seed_annotations

    state, work = tmp_path / "state", tmp_path / "work"
    seed_annotations(work)
    k = FakeKaggle()
    k.active_runs = ["zs-4b-x"]
    registry.append(state / "runs.jsonl", {"run_id": "zs-4b-x", "status": "ok", "zip": "zs-4b-x.zip",
                                           "kernel": "u/reva-zs-4b-x", "metrics": {"weighted_accuracy": 0.5,
                                                                                  "overall_accuracy": 0.5}})
    real = k.output

    def one_letter(slug, dest, file_pattern=None):
        out = real(slug, dest)
        (dest / "zs-4b-x" / "test_probs.json").write_text(json.dumps(probs_for(TEST, "C")))
        return out
    k.output = one_letter
    client = FakeClient()
    monkeypatch.setattr(reva.kaggle, "Kaggle", lambda: k)
    monkeypatch.setattr(cli, "_client", lambda cfg: client)
    monkeypatch.setattr(preflight, "live_columns", lambda *a, **kw: BOARD_COLUMNS)
    argv = ["submit", "--run", "zs-4b-x", "--state", str(state), "--work", str(work)]
    assert cli.main(argv) == 1 and not client.submitted  # every answer on one letter: blocked

    k.output = real
    for f in (work / "reva-zs-4b-x" / "zs-4b-x").glob("*_probs.json"):
        f.unlink()
    assert cli.main(argv) == 0 and len(client.submitted) == 1


def test_manual_submit_takes_an_arena_ensemble(tmp_path, monkeypatch):
    import json

    import reva.kaggle
    from reva import preflight, registry
    from tests.conftest import BOARD_COLUMNS
    from tests.test_remote_autopilot import FakeClient, FakeKaggle, seed_annotations

    state, work = tmp_path / "state", tmp_path / "work"
    seed_annotations(work)
    k = FakeKaggle()
    k.active_runs = ["a", "b"]
    members = [{"run_id": "a", "kernel": "u/reva-ab"}, {"run_id": "b", "kernel": "u/reva-ab"}]
    registry.append(state / "runs.jsonl", {"run_id": "ens-1", "status": "ok", "zip": "rebuilt from members",
                                           "members": members, "metrics": {"weighted_accuracy": 0.7,
                                                                           "overall_accuracy": 0.7}})
    client = FakeClient()
    monkeypatch.setattr(reva.kaggle, "Kaggle", lambda: k)
    monkeypatch.setattr(cli, "_client", lambda cfg: client)
    monkeypatch.setattr(preflight, "live_columns", lambda *a, **kw: BOARD_COLUMNS)
    assert cli.main(["submit", "--run", "ens-1", "--state", str(state), "--work", str(work)]) == 0
    row = json.loads((state / "submissions.jsonl").read_text().splitlines()[-1])
    assert len(client.submitted) == 1 and row["members"] == ["a", "b"] and row["kernel"] is None
