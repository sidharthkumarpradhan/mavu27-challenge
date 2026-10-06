"""Frame sampling and prompts without a GPU. The full job runs in `reva smoke` (CI)."""

import pytest

from reva import frames
from reva.data import LETTERS


def test_sample_indices_are_uniform_and_in_range():
    assert frames.sample_indices(100, 4) == [12, 37, 62, 87]
    assert frames.sample_indices(2, 4) == [0, 0, 1, 1]  # short clips repeat frames
    with pytest.raises(ValueError):
        frames.sample_indices(0, 4)


def test_decode_and_cache_roundtrip(tmp_path):
    pytest.importorskip("av")
    from reva.smoke import write_video

    write_video(tmp_path / "v" / "a.mp4", n=20, size=(128, 64))
    d = frames.decode(tmp_path / "v" / "a.mp4", 5, max_side=64)
    assert d["frames"].shape == (5, 32, 64, 3) and d["total"] == 20 and d["fps"] == 8
    frames.build_cache(tmp_path / "v", ["a.mp4", "a.mp4"], tmp_path / "c", 5, 64, workers=1)
    assert (frames.load(tmp_path / "c", "a.mp4", 5, 64)["indices"] == d["indices"]).all()


def test_prompt_and_option_shifts():
    pytest.importorskip("torch")
    from reva.model import prompt_text, shift

    row = {"question": "Q?", "options": {"A": "a", "B": "b", "C": "c", "D": "d"}}
    assert shift(0) == list(LETTERS) and shift(1) == ["B", "C", "D", "A"]
    text = prompt_text(row, shift(1), times=[0.0, 1.5])
    assert text.splitlines() == ["Frames are sampled at 0.0s, 1.5s.", "Q?", "A. b", "B. c", "C. d", "D. a",
                                 "Answer with the option's letter from the given choices directly."]
