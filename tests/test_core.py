import json
import zipfile

import pytest

from reva import config as C
from reva import data, package, score
from tests.conftest import make_rows


def test_override_parses_yaml_values():
    cfg = C.override({"a": {"b": 1}}, {"a.b": "16", "c.d": "true", "e": [1]})
    assert cfg == {"a": {"b": 16}, "c": {"d": True}, "e": [1]}
    assert C.parse_pairs(["x.y=1", "z = a=b"]) == {"x.y": "1", "z": "a=b"}
    with pytest.raises(ValueError):
        C.parse_pairs(["novalue"])


def test_fingerprint_is_stable_and_sensitive():
    assert C.fingerprint({"a": 1, "b": 2}) == C.fingerprint({"b": 2, "a": 1})
    assert C.fingerprint({"a": 1}) != C.fingerprint({"a": 2})


def test_default_config_loads():
    cfg = C.load()
    assert C.get(cfg, "competition.phase") == 30831
    assert C.get(cfg, "submit.format") in package.FORMATS


def test_holdout_is_stratified_and_disjoint():
    rows = make_rows("train", 10)
    keep, held = data.stratified_holdout(rows, 0.2, seed=1)
    assert len(held) == 3 * 11 * 2  # 2 of 10 from every (source, task) cell
    assert not {r["qa_id"] for r in keep} & {r["qa_id"] for r in held}
    assert data.stratified_holdout(rows, 0.2, seed=1)[1] == held  # deterministic


def test_make_splits_refit_rules(ann):
    sp = data.make_splits(ann, 0.1, 0)
    assert len(sp["dev"]) == len(make_rows("val", 3)) + 33
    assert len(sp["fit"]) + 33 == len(make_rows("train", 10))
    refit = data.make_splits(ann, 0.1, 0, refit=True)
    assert len(refit["fit"]) == len(make_rows("train", 10))  # holdout returns, val stays out
    with_val = data.make_splits(ann, 0.1, 0, refit=True, refit_with_val=True)
    assert len(with_val["fit"]) == len(make_rows("train", 10)) + len(make_rows("val", 3))


def test_check_rows_rejects_bad_schema():
    rows = make_rows("val", 1)
    rows[0]["correct_answer"] = "E"
    with pytest.raises(ValueError):
        data.check_rows(rows, labeled=True)
    rows = make_rows("val", 1)
    rows[1]["qa_id"] = rows[0]["qa_id"]
    with pytest.raises(ValueError):
        data.check_rows(rows, labeled=True)


def test_columns_match_codabench_keys():
    rows = make_rows("val", 2)
    perfect = {r["qa_id"]: r["correct_answer"] for r in rows}
    cols = score.columns(rows, perfect)
    assert set(cols) == {score.OVERALL, *score.TASK_COLUMNS.values()}
    assert all(v == 1.0 for v in cols.values())


def test_weighted_uses_test_mix_and_falls_back():
    dev = make_rows("val", 2)
    test = [r for r in make_rows("test", 1) if r["dataset_name"] == "ERA_Tra"]
    # right only on ERA questions: the test set here is all ERA, so weighted accuracy is 1.0
    preds = {r["qa_id"]: (r["correct_answer"] if r["dataset_name"] == "ERA_Tra" else "X") for r in dev}
    assert score.weighted(dev, preds, test, min_cell=1) == pytest.approx(1.0)
    # with cells too small, it falls back to task level: 1/3 of each task is ERA
    assert score.weighted(dev, preds, test, min_cell=50) == pytest.approx(1 / 3)


@pytest.mark.parametrize("fmt", package.FORMATS)
def test_package_roundtrip(tmp_path, fmt):
    test = make_rows("test", 1)
    preds = {r["qa_id"]: "ABCD"[i % 4] for i, r in enumerate(test)}
    z = package.write(tmp_path / "s.zip", test, preds, fmt)
    assert package.validate(z, test, fmt) == len(test)
    assert dict(package.parse(json.loads(zipfile.ZipFile(z).read("predictions.json")), fmt)) == preds


def _zip(tmp_path, payload, name="predictions.json"):
    p = tmp_path / "bad.zip"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr(name, json.dumps(payload))
    return p


def test_validate_catches_every_codabench_rule(tmp_path):
    test = make_rows("test", 1)
    ok = {r["qa_id"]: "A" for r in test}
    first = test[0]["qa_id"]
    with pytest.raises(ValueError, match="missing"):
        package.validate(_zip(tmp_path, {k: v for k, v in ok.items() if k != first}), test, "id_map")
    with pytest.raises(ValueError, match="unknown"):
        package.validate(_zip(tmp_path, {**ok, "test_999999": "A"}), test, "id_map")
    with pytest.raises(ValueError, match="outside"):
        package.validate(_zip(tmp_path, {**ok, first: "E"}), test, "id_map")
    dup = [{"qa_id": k, "answer": v} for k, v in ok.items()] + [{"qa_id": first, "answer": "B"}]
    with pytest.raises(ValueError, match="duplicate"):
        package.validate(_zip(tmp_path, dup), test, "list")
    with pytest.raises(ValueError, match="only predictions.json"):
        package.validate(_zip(tmp_path, ok, name="sub/predictions.json"), test, "id_map")
