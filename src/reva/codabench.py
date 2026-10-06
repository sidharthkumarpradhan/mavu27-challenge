"""Codabench client: log in, upload a submission zip, start it, poll the result.

Mirrors the official web client (codalab/codabench, develop branch, read 6 Oct 2026):
- POST /api/api-token-auth/ {username, password} -> {"token"}; then "Authorization: Token <t>".
- POST /api/datasets/ {type: "submission", competition, request_sassy_file_name, file_name,
  file_size} -> {"key", "sassy_url"} (src/apps/api/views/datasets.py, DataViewSet.create).
- PUT <sassy_url> with the zip bytes, Content-Type application/zip (static/js/ours/client.js).
- PUT /api/datasets/completed/<key>/ marks the upload done.
- POST /api/submissions/ {data: key, phase, tasks: [task ids], organization} (submission_upload.tag).
  `organization` is the id behind the page's "Submit as" box; leaving it out submits as yourself.
- GET /api/users/participant_organizations/ -> [{id, name, url}] for that box (views/profiles.py).
- GET /api/submissions/<id>/ -> status in Submitting, Submitted, Preparing, Running, Scoring,
  Finished, Failed, Cancelled, plus "scores" when finished (competitions/models.py).
Limits come from Phase.can_user_make_submissions: per-day and per-person totals; Failed
submissions do not count.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import requests

BASE = "https://www.codabench.org"
DONE = {"Finished"}
FAILED = {"Failed", "Cancelled"}


class CodabenchError(RuntimeError):
    pass


class Client:
    def __init__(self, token: str, base: str = BASE, session: requests.Session | None = None):
        self.base = base.rstrip("/")
        self.s = session or requests.Session()
        self.s.headers["Authorization"] = f"Token {token}"

    @classmethod
    def login(cls, username: str, password: str, base: str = BASE,
              session: requests.Session | None = None) -> "Client":
        s = session or requests.Session()
        r = s.post(f"{base.rstrip('/')}/api/api-token-auth/", data={"username": username, "password": password},
                   timeout=60)
        if r.status_code != 200 or "token" not in r.json():
            raise CodabenchError(f"login failed ({r.status_code})")  # never echo the response: it may hold secrets
        return cls(r.json()["token"], base, s)

    def _ok(self, r: requests.Response, what: str) -> dict:
        if r.status_code >= 300:
            raise CodabenchError(f"{what} failed ({r.status_code}): {r.text[:500]}")
        return r.json() if r.content else {}

    def can_submit(self, phase: int) -> tuple[bool, str]:
        d = self._ok(self.s.get(f"{self.base}/api/can_make_submission/{phase}/", timeout=60), "can_make_submission")
        return bool(d.get("can")), d.get("reason") or ""

    def organization_id(self, name: str) -> int:
        """Id of the organization this account submits for. Raises if the account is not in it, so
        a run never goes up under the wrong name."""
        orgs = self._ok(self.s.get(f"{self.base}/api/users/participant_organizations/", timeout=60), "organizations")
        match = [o for o in orgs if o.get("name") == name]
        if len(match) != 1:
            # name only the one we asked for: this message lands in the public STATUS.md
            raise CodabenchError(f"organization {name!r} not found for this account")
        return int(match[0]["id"])

    def submit(self, zip_path: str | Path, competition: int, phase: int, tasks: list[int],
               organization: int | None = None) -> int:
        """Upload and start one submission, as `organization` when given. The zip's file name shows
        up on Codabench, so name it after the run id. Returns the submission id."""
        zip_path = Path(zip_path)
        meta = {"type": "submission", "competition": competition, "request_sassy_file_name": zip_path.name,
                "file_name": zip_path.name, "file_size": zip_path.stat().st_size}
        ds = self._ok(self.s.post(f"{self.base}/api/datasets/", json=meta, timeout=60), "dataset create")
        with open(zip_path, "rb") as f:  # presigned URL: plain request, no auth header
            up = requests.put(ds["sassy_url"], data=f, headers={"Content-Type": "application/zip"}, timeout=600)
        if up.status_code >= 300:
            raise CodabenchError(f"zip upload failed ({up.status_code}): {up.text[:300]}")
        self._ok(self.s.put(f"{self.base}/api/datasets/completed/{ds['key']}/", timeout=60), "upload completed")
        body = {"data": ds["key"], "phase": phase, "tasks": tasks}
        if organization is not None:
            body["organization"] = organization
        sub = self._ok(self.s.post(f"{self.base}/api/submissions/", json=body, timeout=60), "submission create")
        return int(sub["id"])

    def submission(self, sid: int) -> dict:
        return self._ok(self.s.get(f"{self.base}/api/submissions/{sid}/", timeout=60), "submission status")

    def wait(self, sid: int, timeout_s: float = 1800, poll_s: float = 20,
             sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic) -> dict:
        """Poll until Finished or Failed. Returns the last submission record (status may still be
        running when the timeout hits; the next autopilot cycle polls again)."""
        end = clock() + timeout_s
        while True:
            d = self.submission(sid)
            if d.get("status") in DONE | FAILED or clock() >= end:
                return d
            sleep(poll_s)


def scores(record: dict) -> dict[str, float]:
    """{column_key: value} from a finished submission record."""
    return {s["column_key"]: float(s["score"]) for s in record.get("scores") or [] if "column_key" in s}
