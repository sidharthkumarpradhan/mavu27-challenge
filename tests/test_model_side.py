"""Frame sampling and prompts without a GPU. The full job runs in `reva smoke` (CI)."""

import json

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


def test_chat_template_kwargs_close_the_think_block():
    """Qwen3.5 opens <think> by default, so the next token would not be the answer letter."""
    pytest.importorskip("torch")
    from reva.model import VLM

    qwen35 = "trl-internal-testing/tiny-Qwen3_5ForConditionalGeneration"
    row = {"question": "Q?", "options": {L: L.lower() for L in "ABCD"}}
    vlm = VLM({"id": qwen35, "dtype": "fp32", "chat_template_kwargs": {"enable_thinking": False}}, device="cpu")
    assert vlm.chat_text(row, None, list(LETTERS)).endswith("<think>\n\n</think>\n\n")


def test_step_count_matches_the_windows_train_takes():
    pytest.importorskip("torch")
    from reva.model import count_steps

    assert count_steps([4, 4, 4, 4, 3], accum=8) == 3  # windows of 8, 8 and a short last 3
    assert count_steps([4, 3, 3, 1], accum=8) == 2  # 10 (groups never split), then 1
    assert count_steps([1] * 16, accum=8) == 2
    assert count_steps([2], accum=8, since=6) == 1


def _tiny_training_set(n: int = 14):
    import numpy as np

    rng = np.random.default_rng(0)
    videos = {f"v{k}": {"frames": rng.integers(0, 255, (4, 64, 96, 3), dtype=np.uint8),
                        "indices": np.array([1, 7, 13, 19]), "fps": 8.0, "total": 24} for k in range(3)}
    rows = [{"qa_id": f"q{i}", "video_path": f"v{i % 3}", "question": f"What {i}?",
             "options": {L: f"{L}{i}" for L in "ABCD"}, "correct_answer": "ABCD"[i % 4]} for i in range(n)]
    return rows, lambda r: videos[r["video_path"]]


def test_training_takes_every_planned_step(tmp_path):
    """Groups of up to 4 never split a window, so the schedule must count the windows train() really
    takes, short last one included, or the learning rate never reaches the end of its cosine."""
    pytest.importorskip("torch")
    from reva.model import VLM, train

    rows, video_of = _tiny_training_set()
    vlm = VLM({"id": "trl-internal-testing/tiny-Qwen3VLForConditionalGeneration", "dtype": "fp32"}, device="cpu")
    vlm.add_lora({"grad_ckpt": False})
    stats = train(vlm, rows, video_of, {"grad_accum": 8, "pack": 4, "lr": 1e-4}, tmp_path, deadline=None)
    assert stats["pack"] == 4 and stats["samples"] == len(rows)
    assert stats["steps"] == stats["planned_steps"] >= 2 and stats["loss"] is not None


def test_a_prompt_that_does_not_split_falls_back_instead_of_failing(tmp_path, monkeypatch):
    """One odd prompt must not cost a whole GPU job: inference scores it whole, training unpacked."""
    torch = pytest.importorskip("torch")
    import numpy as np

    from reva.model import VLM, train

    rows, video_of = _tiny_training_set(6)
    vlm = VLM({"id": "trl-internal-testing/tiny-Qwen3VLForConditionalGeneration", "dtype": "fp32"}, device="cpu")
    vlm.cfg["share_video_prefix"] = False
    whole = vlm.predict(rows, video_of, perms=2)

    def no_split(*a, **k):
        raise ValueError("prompt does not split cleanly after the video")

    monkeypatch.setattr(VLM, "encode_video", no_split)
    vlm.cfg["share_video_prefix"] = True
    shared = vlm.predict(rows, video_of, perms=2)
    assert all(np.allclose(shared[q], whole[q], atol=1e-6) for q in whole)

    monkeypatch.setattr(VLM, "packed_logits", no_split)
    vlm.add_lora({"grad_ckpt": False})
    stats = train(vlm, rows, video_of, {"grad_accum": 4, "pack": 4}, tmp_path, deadline=None)
    assert stats["samples"] == len(rows) and torch.isfinite(torch.tensor(stats["loss"]))


def test_lora_keeps_the_frozen_weights_in_half_precision():
    """The 4-bit 8B shares a 16 GB T4 with its activations: adding LoRA must not upcast the frozen
    vision tower or lm_head to fp32 (peft's prepare_model_for_kbit_training does)."""
    torch = pytest.importorskip("torch")
    from reva.model import VLM

    vlm = VLM({"id": "trl-internal-testing/tiny-Qwen3VLForConditionalGeneration", "dtype": "fp16"}, device="cpu")
    vlm.cfg["load_4bit"] = True  # the branch the 8B QLoRA takes; bitsandbytes itself needs a GPU
    vlm.add_lora({})
    frozen = {w.dtype for w in vlm.model.parameters() if not w.requires_grad}
    assert frozen == {torch.float16}
    assert any(w.requires_grad for w in vlm.model.parameters())


def _lora_weights(vlm):
    return {k: v.detach().clone() for k, v in vlm.model.named_parameters() if "lora_" in k}


def _fresh_lora(init=None):
    import torch

    from reva.model import VLM

    torch.manual_seed(0)
    vlm = VLM({"id": "trl-internal-testing/tiny-Qwen3VLForConditionalGeneration", "dtype": "fp32"}, device="cpu")
    vlm.add_lora({"grad_ckpt": False}, init)
    return vlm


def test_a_crashed_run_resumes_to_the_same_weights(tmp_path):
    """A session that dies mid-training loses nothing: resuming from the last checkpoint gives the
    same adapter as a run that never stopped (optimizer, scheduler and RNG states all restored)."""
    torch = pytest.importorskip("torch")
    from reva.model import find_checkpoint, train

    rows, video_of = _tiny_training_set()
    tcfg = {"grad_accum": 4, "pack": 2, "lr": 1e-3, "ckpt_minutes": 0}
    whole = _fresh_lora()
    full = train(whole, rows, video_of, tcfg, tmp_path / "a", deadline=None)

    calls = []

    def dies_on_the_fifth_video(r):
        calls.append(r["qa_id"])
        if len(calls) == 5:
            raise RuntimeError("session killed")
        return video_of(r)

    with pytest.raises(RuntimeError):
        train(_fresh_lora(), rows, dies_on_the_fifth_video, tcfg, tmp_path / "b", deadline=None)
    ckpt = find_checkpoint(tmp_path / "b")
    assert ckpt is not None
    resumed = _fresh_lora(ckpt / "adapter")
    stats = train(resumed, rows, video_of, tcfg, tmp_path / "b", deadline=None, resume=ckpt)
    assert stats["steps"] == full["steps"] and stats["samples"] == full["samples"] and stats["sessions"] == 2
    a, b = _lora_weights(whole), _lora_weights(resumed)
    assert a.keys() == b.keys() and all(torch.allclose(a[k], b[k], atol=1e-6) for k in a)


def test_a_spanning_run_trains_to_the_session_end_and_finishes_in_the_next(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from types import SimpleNamespace

    from reva import model
    from reva.model import find_checkpoint, train

    rows, video_of = _tiny_training_set()
    clock = [1000.0]  # every group takes 10 s of fake time
    monkeypatch.setattr(model, "time", SimpleNamespace(time=lambda: clock[0]))

    def slow(r):
        clock[0] += 10
        return video_of(r)

    tcfg = {"grad_accum": 4, "pack": 2, "calib_samples": 4}
    # the session has 35 s for training and no time for inference (the deadline has passed)
    first = train(_fresh_lora(), rows, slow, tcfg, tmp_path, deadline=999.0, stop_at=1035.0)
    assert first["stopped"] == "session" and 0 < first["samples"] < len(rows)
    assert not (tmp_path / "adapter").exists()  # only a finished run writes the final adapter
    ckpt = find_checkpoint(tmp_path)
    second = train(_fresh_lora(ckpt / "adapter"), rows, video_of, tcfg, tmp_path, deadline=None, resume=ckpt)
    assert second["stopped"] == "done" and second["samples"] == len(rows) and second["sessions"] == 2
    assert (tmp_path / "adapter" / "adapter_config.json").exists()


def test_a_half_written_checkpoint_falls_back_to_the_last_whole_one(tmp_path):
    from reva.model import find_checkpoint

    (tmp_path / "ckpt.old" / "adapter").mkdir(parents=True)
    (tmp_path / "ckpt.old" / "state.pt").write_bytes(b"x")
    (tmp_path / "ckpt.tmp").mkdir()  # the save that was cut off
    assert find_checkpoint(tmp_path) == tmp_path / "ckpt.old"
    assert find_checkpoint(tmp_path / "empty") is None


def test_inference_carries_on_where_an_earlier_session_stopped(tmp_path):
    from reva import job

    rows = [{"qa_id": f"q{i}", "video_path": f"v{i % 3}"} for i in range(9)]

    class Fake:
        def __init__(self, die_after=None):
            self.seen, self.die_after = [], die_after

        def predict(self, part, video_of, perms, log_every=None):
            if self.die_after is not None and len(self.seen) >= self.die_after:
                raise RuntimeError("session killed")
            self.seen += [r["qa_id"] for r in part]
            return {r["qa_id"]: [0.1, 0.2, 0.3, 0.4] for r in part}

    chunks = job.video_chunks(rows, 2)
    assert sorted(r["qa_id"] for c in chunks for r in c) == sorted(r["qa_id"] for r in rows)
    assert all(len({r["video_path"] for r in c}) == 1 for c in chunks)  # 3 per video, chunks of 2: one video each

    path = tmp_path / "dev_probs.part.json"
    with pytest.raises(RuntimeError):
        job.predict_saved(Fake(die_after=3), rows, None, 1, path, chunk=2)
    first = json.loads(path.read_text())
    assert len(first) == 3
    again = Fake()
    probs = job.predict_saved(again, rows, None, 1, path, chunk=2)
    assert list(probs) == [r["qa_id"] for r in rows] and not set(again.seen) & set(first)
    with pytest.raises(job.SessionOver):  # a spanning run past its session's end stops before scoring
        job.predict_saved(Fake(), rows, None, 1, tmp_path / "other.part.json", stop_at=0.0, chunk=2)
