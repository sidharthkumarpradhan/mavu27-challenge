import json

import pytest
from pathlib import Path

from reva import board, codabench


def payload():
    def sub(sid, owner, overall, task):
        return {"id": sid, "owner": owner, "created_when": "2026-10-06T00:00:00Z",
                "scores": [{"column_key": "overall_accuracy", "score": str(overall)},
                           {"column_key": "temporalgrounding_accuracy", "score": str(task)}]}
    return {"submissions": [sub(2, "b", 0.80, 0.70), sub(1, "a", 0.87, 0.84)]}


def test_parse_sorts_by_overall():
    rows = board.parse(payload())
    assert [r["owner"] for r in rows] == ["a", "b"]
    assert rows[0]["temporalgrounding_accuracy"] == 0.84
    assert board.rank_of(rows, "B") == 2


def test_snapshot_dedupes(tmp_path: Path):
    rows = board.parse(payload())
    assert board.snapshot(rows, tmp_path / "b.csv", "t1") == 2
    assert board.snapshot(rows, tmp_path / "b.csv", "t2") == 0


def test_gap_weights_by_test_count():
    rows = board.parse(payload())
    g = board.gap(rows, {"temporalgrounding_accuracy": 0.64}, {"Temporal Grounding": 640, "Change Detection": 3360})
    tg = next(x for x in g if x["task"] == "Temporal Grounding")
    assert abs(tg["lost"] - 0.20 * 640 / 4000) < 1e-9
    assert g[0]["task"] == "Temporal Grounding"


class FakeResp:
    def __init__(self, code=200, body=None):
        self.status_code, self._body = code, body or {}
        self.content = json.dumps(self._body).encode()
        self.text = self.content.decode()

    def json(self):
        return self._body


class FakeSession:
    def __init__(self):
        self.headers, self.calls = {}, []

    def post(self, url, json=None, data=None, timeout=None):
        self.calls.append(("POST", url, json or data))
        if url.endswith("/api-token-auth/"):
            return FakeResp(200, {"token": "t0k"})
        if url.endswith("/datasets/"):
            return FakeResp(201, {"key": "k1", "sassy_url": "https://minio/put"})
        return FakeResp(201, {"id": 77})

    def put(self, url, timeout=None):
        self.calls.append(("PUT", url, None))
        return FakeResp(200, {"key": "k1"})

    def get(self, url, timeout=None):
        self.calls.append(("GET", url, None))
        if url.endswith("/participant_organizations/"):
            return FakeResp(200, [{"id": 5, "name": "StagAI", "url": ""}, {"id": 6, "name": "Other", "url": ""}])
        return FakeResp(200, {"status": "Finished", "scores": [{"column_key": "overall_accuracy", "score": "0.5"}]})


def test_submit_follows_the_web_client(tmp_path, monkeypatch):
    s = FakeSession()
    puts = []
    monkeypatch.setattr(codabench.requests, "put", lambda url, data, headers, timeout: puts.append((url, headers)) or FakeResp())
    c = codabench.Client.login("u", "p", session=s)
    assert s.headers["Authorization"] == "Token t0k"
    z = tmp_path / "run1.zip"
    z.write_bytes(b"PK")
    assert c.submit(z, 18274, 30831, [36510]) == 77
    create = next(body for m, url, body in s.calls if url.endswith("/datasets/"))
    assert create == {"type": "submission", "competition": 18274, "request_sassy_file_name": "run1.zip",
                      "file_name": "run1.zip", "file_size": 2}
    assert puts == [("https://minio/put", {"Content-Type": "application/zip"})]
    assert any(m == "PUT" and url.endswith("/datasets/completed/k1/") for m, url, _ in s.calls)
    sub = s.calls[-1]
    assert sub[1].endswith("/api/submissions/") and sub[2] == {"data": "k1", "phase": 30831, "tasks": [36510]}
    rec = c.wait(77, sleep=lambda s: None)
    assert codabench.scores(rec) == {"overall_accuracy": 0.5}


def test_submit_as_organization(tmp_path, monkeypatch):
    s = FakeSession()
    monkeypatch.setattr(codabench.requests, "put", lambda url, data, headers, timeout: FakeResp())
    c = codabench.Client.login("u", "p", session=s)
    z = tmp_path / "run1.zip"
    z.write_bytes(b"PK")
    assert c.organization_id("StagAI") == 5
    c.submit(z, 18274, 30831, [36510], organization=5)
    assert s.calls[-1][2] == {"data": "k1", "phase": 30831, "tasks": [36510], "organization": 5}
    with pytest.raises(codabench.CodabenchError, match="not found") as err:
        c.organization_id("stagai")  # exact name only: never guess which team gets the score
    assert "Other" not in str(err.value)  # other memberships stay out of the public status page


class RefusingSession(FakeSession):
    def post(self, url, json=None, data=None, timeout=None):
        if url.endswith("/api/submissions/"):
            return FakeResp(400, ["You do not have participant permissions for this group"])
        return super().post(url, json, data, timeout)


def test_a_refused_submission_names_a_known_reason(tmp_path, monkeypatch):
    # 7 Oct 2026: "submission create failed (400)" with the body kept out of public STATUS.md told us
    # nothing. Codabench's own fixed messages name the cause and are safe to show.
    from reva import autopilot

    monkeypatch.setattr(codabench.requests, "put", lambda url, data, headers, timeout: FakeResp())
    c = codabench.Client.login("u", "p", session=RefusingSession())
    z = tmp_path / "run1.zip"
    z.write_bytes(b"PK")
    with pytest.raises(codabench.CodabenchError) as err:
        c.submit(z, 18274, 30831, [36510], organization=5)
    assert autopilot.public(err.value) == \
        "submission create failed (400): You do not have participant permissions for this group"


def test_an_unknown_refusal_keeps_only_the_status_code():
    from reva import autopilot

    r = FakeResp(400, {"detail": "OtherTeam secret page"})
    with pytest.raises(codabench.CodabenchError) as err:
        codabench.Client("t", session=FakeSession())._ok(r, "submission create")
    assert err.value.reason is None and autopilot.public(err.value) == "submission create failed (400)"
