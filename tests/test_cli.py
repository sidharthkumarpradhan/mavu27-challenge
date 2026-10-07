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
    assert cli.main(["autopilot", "--state", str(tmp_path / "s"), "--work", str(tmp_path / "w"), "--sha", "x"]) == 0
    note = seen["notes"][0]
    assert "could not confirm organization StagAI" in note
    assert "OtherTeam" not in note and "500" not in note  # STATUS.md is public


def test_manual_submit_runs_the_pre_upload_checks(tmp_path, monkeypatch):
    import json

    import reva.kaggle
    from reva import preflight, registry
    from tests.test_remote_autopilot import BOARD_COLUMNS, TEST, FakeClient, FakeKaggle, probs_for, seed_annotations

    state, work = tmp_path / "state", tmp_path / "work"
    seed_annotations(work)
    k = FakeKaggle()
    k.active_runs = ["zs-4b-x"]
    registry.append(state / "runs.jsonl", {"run_id": "zs-4b-x", "status": "ok", "zip": "zs-4b-x.zip",
                                           "kernel": "u/reva-zs-4b-x", "metrics": {"weighted_accuracy": 0.5,
                                                                                  "overall_accuracy": 0.5}})
    real = k.output

    def one_letter(slug, dest):
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
