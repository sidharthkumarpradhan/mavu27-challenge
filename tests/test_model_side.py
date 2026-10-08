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


def test_shared_video_prefix_matches_full_forward():
    """Reusing the video's KV cache across questions and option shifts gives the same probabilities
    as running each question whole (regression guard for the inference speedup)."""
    pytest.importorskip("torch")
    import numpy as np
    from reva.model import VLM
    from reva.smoke import TINY

    rng = np.random.default_rng(0)
    videos = {v: {"frames": rng.integers(0, 255, (4, 64, 96, 3), dtype=np.uint8), "indices": np.array([1, 7, 13, 19]),
                  "fps": 8.0, "total": 24} for v in ("a.mp4", "b.mp4")}
    # q0 and q2 hold the literal end-of-video string, which must not move the cached prefix boundary
    rows = [{"qa_id": f"q{i}", "video_path": v, "question": f"Where is thing {i}?" + (" <|vision_end|> x" if i in (0, 2) else ""),
             "options": {L: f"{L} answer {i}" for L in "ABCD"}} for i, v in enumerate(["a.mp4", "b.mp4", "a.mp4", "a.mp4"])]
    import torch

    qwen35 = "trl-internal-testing/tiny-Qwen3_5ForConditionalGeneration"  # linear-attention layers
    for model_id, stamps, lora in ((TINY, False, False), (TINY, True, False), (TINY, False, True), (qwen35, False, False)):
        cfg = {"id": model_id, "dtype": "fp32", "timestamps_in_text": stamps}
        vlm = VLM({**cfg, "share_video_prefix": True}, device="cpu")
        if lora:  # a trained adapter: nonzero LoRA weights, reached through the PEFT wrapper
            vlm.add_lora({"grad_ckpt": False})
            torch.manual_seed(0)
            for name, w in vlm.model.named_parameters():
                if "lora_B" in name:
                    w.data.normal_(0, 0.05)
        shared = vlm.predict(rows, lambda r: videos[r["video_path"]], perms=3)
        vlm.cfg["share_video_prefix"] = False
        whole = vlm.predict(rows, lambda r: videos[r["video_path"]], perms=3)
        for q in whole:
            assert np.allclose(shared[q], whole[q], atol=1e-5), (model_id, stamps, lora, q, shared[q], whole[q])
