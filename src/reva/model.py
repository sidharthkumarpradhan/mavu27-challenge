"""The video LLM: prompt, answer scoring by letter logits, and LoRA fine-tuning.

Scoring reads the logits of the tokens "A", "B", "C", "D" at the first answer position, in one
forward pass. That is faster than generating text, and it cannot fail to parse. Training uses the
same four logits with a cross-entropy loss on the correct letter, so training and inference see the
exact same input.

Option-order test-time augmentation: with `infer.perms: k`, the options are shown in k cyclic
shifts and the probabilities are mapped back and averaged. This cancels any letter-position bias.
The same idea in training: every step shows the options in a random order.

Default backbone: Qwen3-VL (transformers native). It takes frames plus their real timestamps.
"""

from __future__ import annotations

import copy
import math
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from reva.data import LETTERS

INSTRUCTION = "Answer with the option's letter from the given choices directly."


def prompt_text(row: dict, order: list[str], times: list[float] | None = None) -> str:
    """Question plus options. Shown letter LETTERS[j] carries original option order[j]."""
    lines = [row["question"]]
    lines += [f"{LETTERS[j]}. {row['options'][o]}" for j, o in enumerate(order)]
    if times is not None:
        lines.insert(0, "Frames are sampled at " + ", ".join(f"{t:.1f}s" for t in times) + ".")
    lines.append(INSTRUCTION)
    return "\n".join(lines)


def shift(k: int) -> list[str]:
    """k-th cyclic shift of the options: shift(1) shows B, C, D, A as A, B, C, D."""
    return [LETTERS[(j + k) % 4] for j in range(4)]


def dtype_of(name: str) -> torch.dtype:
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[name]


class VLM:
    def __init__(self, mcfg: dict, device: str = "cuda"):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.cfg = mcfg
        self.device = device
        kwargs = {"dtype": dtype_of(mcfg.get("dtype", "fp16"))}
        if mcfg.get("load_4bit"):
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=kwargs["dtype"],
                bnb_4bit_use_double_quant=True, llm_int8_skip_modules=["visual", "lm_head"])
        if device.startswith("cuda"):
            kwargs["device_map"] = {"": device}
            kwargs["attn_implementation"] = mcfg.get("attn", "sdpa")
        self.processor = AutoProcessor.from_pretrained(mcfg["id"])
        self.model = AutoModelForImageTextToText.from_pretrained(mcfg["id"], **kwargs)
        if not device.startswith("cuda"):
            self.model.to(device)
        tok = self.processor.tokenizer
        ids = [tok.encode(L, add_special_tokens=False) for L in LETTERS]
        if any(len(i) != 1 for i in ids):
            raise ValueError(f"answer letters are not single tokens for {mcfg['id']}: {ids}")
        self.letter_ids = [i[0] for i in ids]
        self.vision_end = getattr(self.processor, "vision_end_token", "<|vision_end|>")

    def chat_text(self, row: dict, video: dict | None, order: list[str]) -> str:
        """The templated prompt: video placeholder (if any), question, options, answer cue."""
        content = [{"type": "video"}] if video is not None else []
        times = None
        if video is not None and self.cfg.get("timestamps_in_text"):
            times = [float(i) / float(video["fps"]) for i in video["indices"]]
        content.append({"type": "text", "text": prompt_text(row, order, times)})
        # chat_template_kwargs: Qwen3.5 opens a <think> block unless enable_thinking is false, and the
        # answer letter must be the very next token
        return self.processor.apply_chat_template([{"role": "user", "content": content}],
                                                  add_generation_prompt=True, tokenize=False,
                                                  **self.cfg.get("chat_template_kwargs", {}))

    def inputs(self, row: dict, video: dict | None, order: list[str]) -> dict:
        """Tokenized model input for one question. video=None gives a text-only probe."""
        from transformers.video_utils import VideoMetadata

        kw = {"text": [self.chat_text(row, video, order)], "return_tensors": "pt"}
        if video is not None:
            fps = float(video["fps"])
            kw["videos"] = [video["frames"]]
            kw["video_metadata"] = [VideoMetadata(total_num_frames=int(video["total"]), fps=fps,
                                                  frames_indices=[int(i) for i in video["indices"]],
                                                  duration=float(video["total"]) / fps)]
            kw["do_sample_frames"] = False
            kw.update(self.cfg.get("processor_kwargs", {}))
        batch = self.processor(**kw)
        return {k: v.to(self.device) for k, v in batch.items()}

    def letter_logits(self, batch: dict) -> torch.Tensor:
        """(4,) float32 logits of A-D at the answer position."""
        out = self.model(**batch, logits_to_keep=1)
        return out.logits[0, -1, self.letter_ids].float()

    # Inference shares the video between questions. The prompt is [video][question and options], so
    # the KV cache after the video is the same for every question about that video and every option
    # shift. On a T4 the video prefix (vision encoder plus about 2,000 tokens) is nearly all of the
    # 3 to 4 s a question cost when each was run whole (job 1, 7 Oct 2026). Videos have about 4
    # test questions each, so encoding once per video cuts inference several fold and makes
    # option-shift TTA almost free.
    def _base(self):
        """The CausalLM under any PEFT wrapper. LoRA layers sit inside it, so they still apply."""
        return self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model

    def tail_ids(self, row: dict, video: dict, order: list[str]) -> torch.Tensor:
        """(1, L) token ids of the prompt after the video: question, options, answer cue. The video
        comes first in the template, so its first end-of-video marker is the split, whatever the
        question text holds."""
        text = self.chat_text(row, video, order)
        tail = text[text.index(self.vision_end) + len(self.vision_end):]
        return self.processor.tokenizer(tail, add_special_tokens=False, return_tensors="pt").input_ids

    def encode_video(self, row: dict, video: dict):
        """KV cache of the prompt up to and including the end of the video."""
        batch = self.inputs(row, video, list(LETTERS))
        ids, tail = batch["input_ids"], self.tail_ids(row, video, list(LETTERS)).to(self.device)
        n = ids.shape[1] - tail.shape[1]
        # the processor expands the video into several marked frame groups, so the boundary comes from
        # the question's own tokens; if they do not line up exactly, sharing would score other text
        if n <= 0 or not torch.equal(ids[:, n:], tail):
            raise ValueError(f"prompt does not split cleanly after the video for {row['qa_id']}")
        kw = {k: (v[:, :n] if v.dim() >= 2 and v.shape[:2] == ids.shape else v) for k, v in batch.items()}
        return self._base().model(**kw, use_cache=True).past_key_values

    def suffix_logits(self, cache, row: dict, video: dict, order: list[str]) -> torch.Tensor:
        """(4,) letter logits for one question on top of the video prefix cache (left unchanged)."""
        ids = self.tail_ids(row, video, order).to(self.device)
        base = self._base()
        # each question runs on a copy: linear-attention layers (Qwen3.5) keep a recurrent state that
        # cannot be cropped back, and a copy of a few hundred MB on the GPU takes about a millisecond
        h = base.model(input_ids=ids, past_key_values=copy.deepcopy(cache), use_cache=True).last_hidden_state
        return base.lm_head(h[:, -1])[0, self.letter_ids].float()

    # Training packs several questions about one video into one sequence: [video][q1][q2]...[qk].
    # Each question attends to the video and to itself only, at the positions it would have alone,
    # so the k losses and their gradients equal k separate passes, but the video is encoded once.
    def can_pack(self) -> bool:
        """Block masks need attention in every layer; linear-attention layers (Qwen3.5) carry a
        recurrent state from one question into the next. Only sdpa and eager read a custom 4D mask."""
        cfg = self._base().config
        types = getattr(getattr(cfg, "text_config", cfg), "layer_types", None) or []
        return all(t == "full_attention" for t in types) and cfg._attn_implementation in ("sdpa", "eager")

    def packed_logits(self, rows: list[dict], video: dict, orders: list[list[str]]) -> torch.Tensor:
        """(k, 4) letter logits for k questions about one video, from one forward pass."""
        batch = self.inputs(rows[0], video, orders[0])
        ids = batch["input_ids"]
        tails = [self.tail_ids(r, video, o).to(self.device) for r, o in zip(rows, orders)]
        n = ids.shape[1] - tails[0].shape[1]
        if n <= 0 or not torch.equal(ids[:, n:], tails[0]):
            raise ValueError(f"prompt does not split cleanly after the video for {rows[0]['qa_id']}")
        base = self._base()
        pos, _ = base.model.get_rope_index(ids, video_grid_thw=batch.get("video_grid_thw"),
                                           attention_mask=batch["attention_mask"],
                                           mm_token_type_ids=batch["mm_token_type_ids"])
        start = pos[:, :, n:n + 1]  # text after the video moves on all three rope axes together
        lens = [t.shape[1] for t in tails]
        seq = torch.cat([ids[:, :n], *tails], dim=1)
        positions = torch.cat([pos[:, :, :n], *(start + torch.arange(m, device=pos.device) for m in lens)], dim=2)
        seg = torch.cat([torch.zeros(n, dtype=torch.long)] + [torch.full((m,), j + 1) for j, m in enumerate(lens)])
        seg = seg.to(self.device)
        size = seq.shape[1]
        causal = torch.ones(size, size, dtype=torch.bool, device=self.device).tril()
        mask = causal & ((seg[None, :] == 0) | (seg[:, None] == seg[None, :]))
        if base.config._attn_implementation == "eager":  # eager adds the mask to the scores
            mask = torch.zeros(mask.shape, dtype=self.model.dtype, device=self.device).masked_fill(
                ~mask, torch.finfo(self.model.dtype).min)
        kw = {k: v for k, v in batch.items() if k not in ("input_ids", "attention_mask", "mm_token_type_ids")}
        mm = torch.cat([batch["mm_token_type_ids"][:, :n], torch.zeros_like(seq[:, n:])], dim=1)
        h = base.model(input_ids=seq, attention_mask=mask[None, None], position_ids=positions, mm_token_type_ids=mm,
                       use_cache=False, **kw).last_hidden_state
        ends = torch.tensor(lens, device=self.device).cumsum(0) + n - 1
        return base.lm_head(h[0, ends])[:, self.letter_ids].float()

    @torch.no_grad()
    def predict(self, rows: list[dict], video_of, perms: int = 1, log_every: int = 200) -> dict[str, list[float]]:
        """{qa_id: [pA, pB, pC, pD]} in original option order, averaged over `perms` shifts."""
        self.model.eval()
        share = self.cfg.get("share_video_prefix", True)
        out, t0, prefix = {}, time.time(), (None, None)
        for n, row in enumerate(sorted(rows, key=lambda r: (r["video_path"], r["qa_id"])), 1):
            video = video_of(row)
            shared = share and video is not None
            if shared and prefix[0] != row["video_path"]:
                prefix = (None, None)  # free the last video's cache before building the next
                try:
                    prefix = (row["video_path"], self.encode_video(row, video))
                except ValueError as e:  # score this video's questions whole rather than lose the job
                    print(f"WARNING {e}; scoring {row['video_path']} without the shared prefix", flush=True)
                    prefix = (row["video_path"], None)
            probs = np.zeros(4)
            for k in range(perms):
                order = shift(k)
                if shared and prefix[1] is not None:
                    logits = self.suffix_logits(prefix[1], row, video, order)
                else:
                    logits = self.letter_logits(self.inputs(row, video, order))
                if not torch.isfinite(logits).all():  # fp16 overflow: fail loudly, never score garbage
                    raise FloatingPointError(f"non-finite logits on {row['qa_id']}; use dtype fp32 or bf16")
                p = torch.softmax(logits, -1).cpu().numpy()
                for j, o in enumerate(order):
                    probs[LETTERS.index(o)] += p[j] / perms
            out[row["qa_id"]] = [round(float(x), 6) for x in probs]
            if n % log_every == 0:
                print(f"predict {n}/{len(rows)} {(time.time() - t0) / n:.2f} s/q", flush=True)
        return out

    # training
    def add_lora(self, tcfg: dict, init: str | Path | None = None) -> None:
        from peft import LoraConfig, PeftModel, get_peft_model

        # 4-bit takes the same path as fp16. peft's prepare_model_for_kbit_training would upcast every
        # unquantized weight (vision tower, lm_head, norms) to fp32: 3 to 4 GB more for the 8B on a
        # 16 GB T4, and an fp32 residual stream and KV cache for the inference after training.
        # get_peft_model freezes the base weights either way, and autocast covers the compute.
        if tcfg.get("grad_ckpt", True):
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            self.model.enable_input_require_grads()
        if init:
            self.model = PeftModel.from_pretrained(self.model, str(init), is_trainable=True)
        else:
            self.model = get_peft_model(self.model, LoraConfig(
                r=tcfg.get("lora_r", 16), lora_alpha=tcfg.get("lora_alpha", 32), lora_dropout=tcfg.get("lora_dropout", 0.05),
                target_modules=tcfg.get("lora_targets",
                                        r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"),
                task_type="CAUSAL_LM"))
        self.model.print_trainable_parameters()


def groups_by_video(rows: list[dict], pack: int, epochs: float, limit: int, rng: random.Random) -> list[list[dict]]:
    """Training order: groups of up to `pack` questions about the same video, in random order, over
    `epochs` passes of min(limit, len(rows)) questions. Every pass reshuffles which questions share a
    group, so a question meets different neighbours."""
    want = int(min(limit, len(rows)) * epochs)
    out, n = [], 0
    while n < want:
        by_video: dict[str, list[dict]] = {}
        for r in rng.sample(rows, len(rows)):
            by_video.setdefault(r["video_path"], []).append(r)
        chunk = [g[i:i + pack] for g in by_video.values() for i in range(0, len(g), pack)]
        for g in rng.sample(chunk, len(chunk)):
            if n >= want:
                break
            out.append(g[:want - n])
            n += len(out[-1])
    return out


def count_steps(sizes, accum: int, since: int = 0) -> int:
    """Optimizer steps train() takes for groups of these sizes: one whenever `accum` samples have
    gathered (groups do not split, so a window holds accum to accum + pack - 1), plus a last one for
    any remainder."""
    steps = 0
    for n in sizes:
        since += n
        if since >= accum:
            steps, since = steps + 1, 0
    return steps + (since > 0)


def save_checkpoint(model, out_dir: str | Path, state: dict) -> Path:
    """Everything train() needs to carry on: the adapter plus optimizer, scheduler, scaler, RNG
    states and the position in the training order. Written next to the last checkpoint and then
    swapped in, so a session that dies while saving still leaves one whole checkpoint behind."""
    out_dir = Path(out_dir)
    tmp, cur, old = out_dir / "ckpt.tmp", out_dir / "ckpt", out_dir / "ckpt.old"
    shutil.rmtree(tmp, ignore_errors=True)
    model.save_pretrained(tmp / "adapter")
    torch.save(state, tmp / "state.pt")
    shutil.rmtree(old, ignore_errors=True)
    if cur.exists():
        cur.rename(old)
    tmp.rename(cur)
    shutil.rmtree(old, ignore_errors=True)
    return cur


def find_checkpoint(out_dir: str | Path) -> Path | None:
    """The newest whole checkpoint in out_dir: ckpt, or ckpt.old when a save was cut off mid-swap."""
    for name in ("ckpt", "ckpt.old"):
        p = Path(out_dir) / name
        if (p / "state.pt").is_file() and (p / "adapter").is_dir():
            return p
    return None


def train(vlm: VLM, rows: list[dict], video_of, tcfg: dict, out_dir: str | Path, deadline: float | None,
          seed: int = 0, stop_at: float | None = None, resume: str | Path | None = None) -> dict:
    """LoRA fine-tune on letter cross-entropy. Stops at the end of the epochs or at `deadline`
    (time.time()), whichever is first, and saves the adapter to out_dir/adapter.

    Questions about the same video train together, `train.pack` at a time (VLM.packed_logits), so
    the video is encoded once per group. Models with linear-attention layers train one at a time.
    fp16 (T4, V100) uses a GradScaler; steps with a non-finite loss are skipped and counted.

    A full checkpoint goes to out_dir/ckpt every `train.ckpt_minutes`. `resume` (a checkpoint
    directory, whose adapter the caller already loaded) carries on from it: the training order
    comes from `seed` again, and the RNG states make the rest of the run the same as if it had
    never stopped. With `stop_at` (the session's end), training that cannot finish before
    `deadline` uses the whole session instead of shrinking, checkpoints, and returns with
    stopped == "session"; the next session resumes it (train.span_sessions).
    """
    model = vlm.model
    deadline = float("inf") if deadline is None else deadline
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=tcfg.get("lr", 2e-4), weight_decay=tcfg.get("weight_decay", 0.0))
    accum = tcfg.get("grad_accum", 8)
    pack = tcfg.get("pack", 4) if vlm.can_pack() else 1
    rng = random.Random(seed)
    order = groups_by_video(rows, pack, tcfg.get("epochs", 1), tcfg.get("max_samples") or len(rows), rng)
    plan = {"total": count_steps(map(len, order), accum)}
    plan["warm"] = max(1, int(plan["total"] * tcfg.get("warmup", 0.03)))
    # the cosine reads plan["total"] at every step, so resizing below reshapes the schedule
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / plan["warm"]) * 0.5 * (
        1 + math.cos(math.pi * min(1.0, s / max(1, plan["total"])))))
    fp16 = vlm.cfg.get("dtype", "fp16") == "fp16" and vlm.device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=fp16)
    stats = {"steps": 0, "samples": 0, "skipped": 0, "loss": None, "stopped": "done", "pack": pack, "sessions": 1}
    i = 0
    if resume:
        state = torch.load(Path(resume) / "state.pt", map_location="cpu", weights_only=False)
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        scaler.load_state_dict(state["scaler"])
        rng.setstate(state["rng"])
        torch.set_rng_state(state["torch_rng"])
        if state.get("cuda_rng") and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        i, stats = state["i"], {**state["stats"], "stopped": "done"}
        stats["sessions"] = stats.get("sessions", 1) + 1
        plan["total"] = stats["steps"] + count_steps(map(len, order[i:]), accum)
        plan["warm"] = state["plan"]["warm"]
        print(f"resumed from {resume}: step {stats['steps']}, {stats['samples']} samples, "
              f"group {i} of {len(order)}", flush=True)
    model.train()
    run_loss, run_n, since, seen, t0 = 0.0, 0, 0, 0, time.time()
    calib, sized = tcfg.get("calib_samples", 2 * accum), False
    every_s, saved = 60 * tcfg.get("ckpt_minutes", 20), time.time()

    def checkpoint() -> None:
        save_checkpoint(model, out_dir, {
            "i": i, "stats": stats, "plan": dict(plan), "rng": rng.getstate(), "opt": opt.state_dict(),
            "sched": sched.state_dict(), "scaler": scaler.state_dict(), "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None})

    def step(window: int) -> None:
        """One optimizer step over `window` samples. The loss was divided by accum, so rescale to a
        true mean over the window: groups do not split, and the last window can be short."""
        scaler.unscale_(opt)
        if window != accum:
            for w in params:
                if w.grad is not None:
                    w.grad.mul_(accum / window)
        torch.nn.utils.clip_grad_norm_(params, tcfg.get("clip", 1.0))
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        sched.step()
        stats["steps"] += 1

    spanning = False
    while i < len(order):
        if not sized and seen >= calib and deadline < float("inf"):
            sized, rate = True, (time.time() - t0) / seen
            left = sum(map(len, order[i:]))
            if stop_at is not None and time.time() + rate * left > deadline:
                # too much for this session: train until it ends, and the next one carries on
                spanning, deadline = True, stop_at
                print(f"train needs {rate * left / 3600:.2f} h more at {rate:.2f} s/sample; "
                      f"training to the session's end, the next session resumes", flush=True)
            else:
                # self-sizing: keep only as many samples as fit before the deadline
                fit, kept = int(0.95 * (deadline - time.time()) / rate), i
                while kept < len(order) and fit >= len(order[kept]):
                    fit -= len(order[kept])
                    kept += 1
                if kept < len(order):
                    order = order[:max(kept, i + 1)]
                    plan["total"] = stats["steps"] + count_steps(map(len, order[i:]), accum, since)
                    plan["warm"] = max(1, int(plan["total"] * tcfg.get("warmup", 0.03)))
                stats["planned_samples"] = stats["samples"] + sum(map(len, order[i:]))
                print(f"train sized to {stats['planned_samples']} samples at {rate:.2f} s/sample", flush=True)
        if time.time() > deadline:
            if stop_at is not None and not spanning:  # no time left for inference here: train on, infer next session
                spanning, deadline = True, stop_at
                continue
            stats["stopped"] = "session" if spanning else "deadline"
            break
        group = order[i]
        i += 1
        seen += len(group)
        shown = [rng.sample(LETTERS, 4) if tcfg.get("shuffle_options", True) else list(LETTERS) for _ in group]
        label = torch.tensor([s.index(r["correct_answer"]) for r, s in zip(group, shown)], device=vlm.device)
        video = video_of(group[0])
        with torch.autocast("cuda", dtype=torch.float16, enabled=fp16):
            try:
                logits = vlm.packed_logits(group, video, shown) if len(group) > 1 else None
            except ValueError as e:  # train this group one question at a time rather than lose the job
                print(f"WARNING {e}; training the group unpacked", flush=True)
                logits = None
            if logits is None:
                logits = torch.stack([vlm.letter_logits(vlm.inputs(r, video, o)) for r, o in zip(group, shown)])
        loss = torch.nn.functional.cross_entropy(logits, label, reduction="sum") / accum
        if not torch.isfinite(loss):
            stats["skipped"] += 1
            opt.zero_grad(set_to_none=True)
            since = 0
            if stats["skipped"] > tcfg.get("max_skipped", 50):
                raise RuntimeError("too many non-finite losses; try dtype fp32 compute or a lower lr")
            continue
        scaler.scale(loss).backward()
        run_loss += loss.item() * accum
        run_n += len(group)
        stats["samples"] += len(group)
        since += len(group)
        if since >= accum:
            step(since)
            since = 0
            if stats["steps"] % tcfg.get("log_every", 25) == 0:
                stats["loss"] = run_loss / run_n
                print(f"train step {stats['steps']}/{plan['total']} loss {stats['loss']:.4f} "
                      f"{(time.time() - t0) / seen:.2f} s/sample", flush=True)
                run_loss, run_n = 0.0, 0
            if time.time() - saved >= every_s:  # only between optimizer steps: no gradients in flight
                checkpoint()
                saved = time.time()
    if since:  # the last short window was backpropagated; step on it too
        step(since)
        since = 0
    if stats["loss"] is None and run_n:  # a run shorter than one logging interval
        stats["loss"] = run_loss / run_n
    stats["planned_steps"] = plan["total"]
    stats["seconds"] = stats.get("seconds", 0) + round(time.time() - t0)
    if stats["stopped"] == "session":
        checkpoint()
        return stats
    model.save_pretrained(Path(out_dir) / "adapter")
    model.eval()
    return stats
