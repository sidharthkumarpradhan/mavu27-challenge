from pathlib import Path

import pytest

from reva.kaggle import Kaggle, KaggleError

FILES = {"r1/run.json": b"{}", "r1/ckpt/state.pt": b"weights", "r1/ckpt/adapter/adapter_config.json": b"{}"}


class FakeCLI:
    """The kaggle CLI on a dataset whose archive is not built yet: the whole-dataset download
    404s, single files download (saved under their base name, as the real CLI does)."""

    def __init__(self, file_fails=0):
        self.calls, self.file_fails = [], file_fails

    def __call__(self, cmd):
        self.calls.append(cmd)
        args = cmd[1:]
        if args[:2] == ["datasets", "files"]:
            rows = "\n".join(f"{p},{len(b)},2026-10-09 07:27:45" for p, b in FILES.items())
            return 0, "name,size,creationDate\n" + rows + "\n"
        if args[:2] == ["datasets", "download"] and "-f" in args:
            if self.file_fails:
                self.file_fails -= 1
                return 1, "404 Client Error: Not Found"
            path = args[args.index("-f") + 1]
            folder = Path(args[args.index("-p") + 1])
            (folder / Path(path).name).write_bytes(FILES[path])
            return 0, "Dataset URL: https://www.kaggle.com/datasets/me/reva-run-r1"
        if args[:2] == ["datasets", "download"]:
            return 1, "404 Client Error: Not Found for url: .../DownloadDataset"
        raise AssertionError(f"unexpected call {args}")


def client(cli):
    now = [0.0]

    def sleep(s):
        now[0] += 60

    return Kaggle(cli=["kaggle"], runner=cli, sleep=sleep, clock=lambda: now[0])


def test_a_dataset_whose_archive_404s_downloads_file_by_file(tmp_path):
    # regression: a fresh Colab run's dataset 404'd as a whole for over an hour (9 Oct 2026),
    # which stalled the autopilot cycle that wanted its dev and test probabilities
    cli = FakeCLI()
    client(cli).dataset_download("me/reva-run-r1", tmp_path / "d")
    for path, data in FILES.items():
        assert (tmp_path / "d" / path).read_bytes() == data
    assert sum("--unzip" in c for c in cli.calls) == 1  # the archive is tried once, not retried


def test_one_file_lands_at_its_path_and_a_404_is_retried(tmp_path):
    cli = FakeCLI(file_fails=2)
    got = client(cli).dataset_file("me/reva-run-r1", "r1/ckpt/state.pt", tmp_path)
    assert got == tmp_path / "r1" / "ckpt" / "state.pt" and got.read_bytes() == b"weights"


def test_a_file_that_keeps_failing_raises_after_the_timeout(tmp_path):
    cli = FakeCLI(file_fails=99)
    with pytest.raises(KaggleError, match="404"):
        client(cli).dataset_file("me/reva-run-r1", "r1/run.json", tmp_path, timeout_s=300)
    assert len(cli.calls) == 6  # tries at 0, 60, ..., 300 s on the fake clock
