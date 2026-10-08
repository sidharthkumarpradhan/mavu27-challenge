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


def test_groups_cover_every_question_once_and_never_mix_videos():
    pytest.importorskip("torch")
    import random

    from reva.model import groups_by_video

    rows = [{"qa_id": f"q{i}", "video_path": f"v{i % 3}"} for i in range(20)]
    groups = groups_by_video(rows, pack=4, epochs=1, limit=len(rows), rng=random.Random(0))
    assert sorted(r["qa_id"] for g in groups for r in g) == sorted(r["qa_id"] for r in rows)
    assert all(1 <= len(g) <= 4 and len({r["video_path"] for r in g}) == 1 for g in groups)
    two = groups_by_video(rows, pack=4, epochs=1.5, limit=len(rows), rng=random.Random(0))
    assert sum(map(len, two)) == 30  # a fractional epoch stops at the sample count


def test_packed_training_gradients_equal_separate_passes():
    """Several questions about one video in one sequence give the same LoRA gradients as one pass
    per question (regression guard for packed training)."""
    torch = pytest.importorskip("torch")
    import numpy as np

    from reva.model import VLM, shift
    from reva.smoke import TINY

    rng = np.random.default_rng(0)
    video = {"frames": rng.integers(0, 255, (4, 64, 96, 3), dtype=np.uint8), "indices": np.array([1, 7, 13, 19]),
             "fps": 8.0, "total": 24}
    rows = [{"qa_id": f"q{i}", "video_path": "a.mp4", "question": "Where is it" + "?" * (i + 1),
             "options": {L: f"{L} {i}" for L in "ABCD"}, "correct_answer": "ABCD"[i]} for i in range(3)]
    orders = [shift(i) for i in range(3)]
    labels = torch.tensor([o.index(r["correct_answer"]) for r, o in zip(rows, orders)])
    vlm = VLM({"id": TINY, "dtype": "fp32", "timestamps_in_text": True}, device="cpu")
    vlm.add_lora({"grad_ckpt": False, "lora_dropout": 0.0})
    torch.manual_seed(0)
    for name, w in vlm.model.named_parameters():
        if "lora_B" in name:
            w.data.normal_(0, 0.05)
    params = [w for w in vlm.model.parameters() if w.requires_grad]

    def grads(loss):
        vlm.model.zero_grad()
        loss.backward()
        return [w.grad.clone() for w in params]

    assert vlm.can_pack()
    packed = grads(torch.nn.functional.cross_entropy(vlm.packed_logits(rows, video, orders), labels, reduction="sum"))
    alone = grads(sum(torch.nn.functional.cross_entropy(vlm.letter_logits(vlm.inputs(r, video, o))[None], y[None])
                      for r, o, y in zip(rows, orders, labels)))
    assert all(torch.allclose(a, b, atol=1e-5) for a, b in zip(packed, alone))
    assert any(a.abs().sum() > 0 for a in packed)


def test_linear_attention_models_train_one_question_at_a_time():
    pytest.importorskip("torch")
    from reva.model import VLM

    assert not VLM({"id": "trl-internal-testing/tiny-Qwen3_5ForConditionalGeneration", "dtype": "fp32"}, device="cpu").can_pack()
