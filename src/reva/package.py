"""Write and check submission.zip.

Codabench's Data page (read 6 Oct 2026) asks for `submission.zip` holding `predictions.json`, with
every one of the 4,000 test qa_ids answered by exactly one of A, B, C or D. It says to fill the
`correct_answer` field, so the default format is test.json itself with that field filled. The
scoring program is not public, so the format is a config knob (`submit.format`) and the first real
submission doubles as the format check:

- fill_test: {"metadata": ..., "QA": [test rows with correct_answer set]}, with test.json's own
  metadata. The hidden reference data zips to 335,929 bytes and test.json filled with answers to
  about 336,100, so the answer key is very likely this same layout (measured 6 Oct 2026).
- id_map:    {"test_000001": "A", ...}
- list:      [{"qa_id": "test_000001", "answer": "A"}, ...]

`validate` applies every rule the Data page lists. Nothing is uploaded unless it passes.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from reva.data import LETTERS

FORMATS = ("fill_test", "id_map", "list")
MEMBER = "predictions.json"


def build(test: list[dict], preds: dict[str, str], fmt: str = "fill_test", metadata: dict | None = None):
    """The predictions.json payload for one run."""
    if fmt == "fill_test":
        return {"metadata": metadata or {}, "QA": [{**r, "correct_answer": preds[r["qa_id"]]} for r in test]}
    if fmt == "id_map":
        return {r["qa_id"]: preds[r["qa_id"]] for r in test}
    if fmt == "list":
        return [{"qa_id": r["qa_id"], "answer": preds[r["qa_id"]]} for r in test]
    raise ValueError(f"unknown submission format {fmt!r}, expected one of {FORMATS}")


def parse(payload, fmt: str) -> list[tuple[str, str]]:
    """(qa_id, answer) pairs from a payload, keeping duplicates so validate can see them."""
    if fmt == "fill_test":
        return [(r["qa_id"], r["correct_answer"]) for r in payload["QA"]]
    if fmt == "id_map":
        return list(payload.items())
    if fmt == "list":
        return [(r["qa_id"], r["answer"]) for r in payload]
    raise ValueError(f"unknown submission format {fmt!r}")


def write(path: str | Path, test: list[dict], preds: dict[str, str], fmt: str = "fill_test",
          metadata: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = json.dumps(build(test, preds, fmt, metadata), ensure_ascii=False)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(MEMBER, blob)
    validate(path, test, fmt)
    return path


def validate(path: str | Path, test: list[dict], fmt: str = "fill_test") -> int:
    """Raise ValueError on any rule Codabench would reject. Returns the number of answers."""
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if not n.endswith("/")]
        if names != [MEMBER]:
            raise ValueError(f"zip must hold only {MEMBER} at its root, found {names}")
        pairs = parse(json.loads(z.read(MEMBER)), fmt)
    ids = [qa for qa, _ in pairs]
    want = {r["qa_id"] for r in test}
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate qa_id in predictions")
    if missing := want - set(ids):
        raise ValueError(f"{len(missing)} test qa_ids missing, e.g. {sorted(missing)[:3]}")
    if unknown := set(ids) - want:
        raise ValueError(f"{len(unknown)} unknown qa_ids, e.g. {sorted(unknown)[:3]}")
    if bad := [(qa, a) for qa, a in pairs if a not in LETTERS]:
        raise ValueError(f"{len(bad)} answers outside A-D, e.g. {bad[:3]}")
    return len(pairs)
