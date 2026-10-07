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
