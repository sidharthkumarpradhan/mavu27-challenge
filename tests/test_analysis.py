from reva import analysis


def q(qid, task, src, gold):
    return {"qa_id": qid, "task": task, "dataset_name": src, "correct_answer": gold}


DEV = [q("1", "TG", "Hawk", "A"), q("2", "TG", "Hawk", "B"), q("3", "TG", "ERA", "C"), q("4", "GU", "Hawk", "D")]
# hits: 1 right, 2 wrong, 3 right, 4 right
PROBS = {"1": [0.9, 0.1, 0, 0], "2": [0.6, 0.4, 0, 0], "3": [0, 0, 0.7, 0.3], "4": [0, 0, 0.1, 0.9]}
TEST = [q(f"t{i}", "TG", "Hawk", "") for i in range(6)] + [q("t6", "GU", "Hawk", ""), q("t7", "CD", "ERA", ""),
                                                         q("t8", "TG", "ERA", ""), q("t9", "GU", "Hawk", "")]


def test_points_lost_weights_dev_errors_by_the_test_mix():
    rows = analysis.breakdown(DEV, PROBS, TEST)
    tg = next(r for r in rows if r["cell"] == ("TG",))
    assert tg["dev_n"] == 3 and abs(tg["dev_acc"] - 2 / 3) < 1e-9 and tg["test_share"] == 0.7
    assert abs(tg["test_points_lost"] - 0.7 * (1 / 3) * 100) < 1e-9
    gu = next(r for r in rows if r["cell"] == ("GU",))
    assert gu["test_points_lost"] == 0
    # a test task with no dev questions is flagged, not hidden
    cd = next(r for r in rows if r["cell"] == ("CD",))
    assert cd["dev_acc"] is None and cd["test_points_lost"] is None
    assert rows[0]["cell"] == ("CD",) and rows[1]["cell"] == ("TG",)  # unmeasured first, then the biggest loss


def test_cells_split_by_task_and_source():
    rows = analysis.breakdown(DEV, PROBS, TEST, ("task", "dataset_name"))
    hawk = next(r for r in rows if r["cell"] == ("TG", "Hawk"))
    assert hawk["dev_n"] == 2 and hawk["dev_acc"] == 0.5 and hawk["test_n"] == 6


def test_agreement_and_oracle():
    other = {"1": [0, 1, 0, 0], "2": [0, 1, 0, 0], "3": [0, 0, 1, 0], "4": [1, 0, 0, 0]}
    g = analysis.agreement(DEV, PROBS, other)
    assert g == {"both": 1, "only_a": 2, "only_b": 1, "neither": 0, "oracle": 1.0}


def test_report_reads_as_markdown_and_never_needs_test_labels():
    text = analysis.report(DEV, {"run-a": PROBS}, TEST, top=3)
    assert "## run-a" in text and "| TG | 0.667 | 3 | 0.700 | 23.33 |" in text and "not measured" in text
    assert "Predicted letters: A 2, B 0, C 1, D 1." in text
