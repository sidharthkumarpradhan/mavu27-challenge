import pytest

from reva.kaggle import Kaggle, KaggleError


def client(codes, clock_step=60.0):
    """A Kaggle client whose CLI calls exit with the given codes in turn, on a fake clock."""
    calls, now = [], [0.0]

    def runner(cmd):
        calls.append(cmd)
        return codes[min(len(calls), len(codes)) - 1], "404 Client Error: Not Found"

    def sleep(s):
        now[0] += clock_step

    return Kaggle(cli=["kaggle"], runner=runner, sleep=sleep, clock=lambda: now[0]), calls


def test_download_retries_a_fresh_version_that_still_404s(tmp_path):
    # regression: a Colab session failed when the checkpoint it resumed from 404'd a minute
    # after the previous session uploaded it (9 Oct 2026)
    k, calls = client([1, 1, 0])
    assert k.dataset_download("me/reva-run-x", tmp_path / "d", poll_s=30) == tmp_path / "d"
    assert len(calls) == 3
    assert calls[0][:3] == ["kaggle", "datasets", "download"]


def test_download_gives_up_after_the_timeout(tmp_path):
    k, calls = client([1])
    with pytest.raises(KaggleError, match="404"):
        k.dataset_download("me/reva-run-x", tmp_path / "d", timeout_s=300, poll_s=30)
    assert len(calls) == 6  # tries at 0, 60, ..., 300 s on the fake clock
