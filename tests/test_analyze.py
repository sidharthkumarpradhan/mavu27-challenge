import json

import pytest

from reva import analyze, cli, score
from reva.arena import argmax
from tests.conftest import make_rows

DEV = make_rows("val", 25)
TEST = make_rows("test", 4)


def onehot(rows, right_every: int, conf: float = 0.7) -> dict[str, list[float]]:
    """Probabilities that pick the right letter on every right_every-th question, a wrong one otherwise."""
    out = {}
    for n, r in enumerate(rows):
        k = "ABCD".index(r["correct_answer"])
        k = k if n % right_every else (k + 1) % 4
        p = [(1 - conf) / 3] * 4
        p[k] = conf
        out[r["qa_id"]] = p
    return out


def test_losses_add_up_to_the_weighted_gap():
    probs = onehot(DEV, 3)
    preds = argmax(probs)
    lost = analyze.losses(DEV, preds, TEST)
    assert sum(x["points"] for x in lost) == pytest.approx(100 * (1 - score.weighted(DEV, preds, TEST)))
    assert [x["points"] for x in lost] == sorted((x["points"] for x in lost), reverse=True)


def test_losses_fall_back_to_the_task_for_a_small_cell():
    preds = argmax(onehot(DEV, 3))
    small = analyze.losses(DEV, preds, TEST, min_cell=1000)
    assert sum(x["points"] for x in small) == pytest.approx(100 * (1 - score.weighted(DEV, preds, TEST, 1000)))


def test_confidence_bins_cover_every_question():
    bins = analyze.confidence(DEV, onehot(DEV, 3, conf=0.95))
    assert sum(b["n"] for b in bins) == len(DEV)
    assert bins[-1]["n"] == len(DEV) and bins[0]["accuracy"] is None


def test_oracle_and_blend():
    a, b = onehot(DEV, 2), onehot(DEV[1:] + DEV[:1], 2)  # wrong on different questions
    b = {r["qa_id"]: b[r["qa_id"]] for r in DEV}
    assert analyze.oracle(DEV, {"a": a}) == pytest.approx(score.columns(DEV, argmax(a))["overall_accuracy"])
    assert analyze.oracle(DEV, {"a": a, "b": b}) >= analyze.oracle(DEV, {"a": a})
    q = DEV[0]["qa_id"]
    assert analyze.blend(a, b, 0.0)[q] == a[q] and analyze.blend(a, b, 1.0)[q] == b[q]
    assert [w for w, _ in analyze.blend_search(DEV, TEST, a, b, (0.1, 0.2))] == [0.1, 0.2]


def test_load_refuses_probabilities_missing_dev_questions(tmp_path):
    p = tmp_path / "dev_probs.json"
    p.write_text(json.dumps({DEV[0]["qa_id"]: [1, 0, 0, 0]}))
    with pytest.raises(ValueError, match="no probabilities"):
        analyze.load(p, DEV)


def test_report_names_the_best_run_first():
    text = analyze.report(DEV, TEST, {"weak": onehot(DEV, 2), "strong": onehot(DEV, 5)})
    assert text.index("strong") < text.index("weak")
    assert "oracle" in text and "points lost by strong" in text


def test_cli_analyze_reads_named_probability_files(ann, tmp_path, capsys):
    from reva import data

    dev = data.make_splits(ann, 0.08, 0)["dev"]
    p = tmp_path / "a.json"
    p.write_text(json.dumps(onehot(dev, 3)))
    assert cli.main(["--set", f"data.root={ann}", "--set", "dev.min_cell=1", "analyze", "--probs", f"a={p}"]) == 0
    assert "weighted dev accuracy" in capsys.readouterr().out
