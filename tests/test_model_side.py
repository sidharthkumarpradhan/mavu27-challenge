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


class ThinkingProcessor:
    """A chat template like Qwen3.5's: it opens a <think> block unless enable_thinking is False."""

    def __init__(self, honours_switch=True):
        self.honours_switch = honours_switch

    def apply_chat_template(self, messages, add_generation_prompt, tokenize, **kw):
        closed = self.honours_switch and kw.get("enable_thinking") is False
        return "<|im_start|>assistant\n" + ("<think>\n\n</think>\n\n" if closed else "<think>\n")

    def __call__(self, **kw):
        import torch

        self.text = kw["text"][0]
        return {"input_ids": torch.zeros(1, 3, dtype=torch.long)}


def test_the_answer_letter_comes_before_any_reasoning():
    pytest.importorskip("torch")
    from reva.model import VLM

    row = {"question": "Q?", "options": {"A": "a", "B": "b", "C": "c", "D": "d"}}
    vlm = object.__new__(VLM)
    vlm.cfg, vlm.device, vlm.processor = {"id": "hybrid"}, "cpu", ThinkingProcessor()
    vlm.inputs(row, None, list(LETTERS))
    assert vlm.processor.text.endswith("</think>\n\n")
    vlm.processor = ThinkingProcessor(honours_switch=False)  # would score the letters inside a reasoning block
    with pytest.raises(ValueError, match="reasoning block"):
        vlm.inputs(row, None, list(LETTERS))
