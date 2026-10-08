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
        return self.processor.apply_chat_template([{"role": "user", "content": content}],
                                                  add_generation_prompt=True, tokenize=False)

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
        recurrent state from one question into the next."""
        cfg = self._base().config
        types = getattr(getattr(cfg, "text_config", cfg), "layer_types", None) or []
        return all(t == "full_attention" for t in types)

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
                prefix = (row["video_path"], self.encode_video(row, video))
            probs = np.zeros(4)
            for k in range(perms):
                order = shift(k)
                if shared:
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
        from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

        if self.cfg.get("load_4bit"):
            self.model = prepare_model_for_kbit_training(self.model, use_gradient_checkpointing=True)
        elif tcfg.get("grad_ckpt", True):
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


def train(vlm: VLM, rows: list[dict], video_of, tcfg: dict, out_dir: str | Path, deadline: float | None,
          seed: int = 0) -> dict:
    """LoRA fine-tune on letter cross-entropy. Stops at the end of the epochs or at `deadline`
    (time.time()), whichever is first, and saves the adapter to out_dir/adapter.

    Questions about the same video train together, `train.pack` at a time (VLM.packed_logits), so
    the video is encoded once per group. Models with linear-attention layers train one at a time.
    fp16 (T4, V100) uses a GradScaler; steps with a non-finite loss are skipped and counted.
    """
    model = vlm.model
    deadline = float("inf") if deadline is None else deadline
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=tcfg.get("lr", 2e-4), weight_decay=tcfg.get("weight_decay", 0.0))
    accum = tcfg.get("grad_accum", 8)
    pack = tcfg.get("pack", 4) if vlm.can_pack() else 1
    rng = random.Random(seed)
    order = groups_by_video(rows, pack, tcfg.get("epochs", 1), tcfg.get("max_samples") or len(rows), rng)
    plan = {"total": math.ceil(sum(map(len, order)) / accum)}
    plan["warm"] = max(1, int(plan["total"] * tcfg.get("warmup", 0.03)))
    # the cosine reads plan["total"] at every step, so resizing below reshapes the schedule
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / plan["warm"]) * 0.5 * (
        1 + math.cos(math.pi * min(1.0, s / max(1, plan["total"])))))
    fp16 = vlm.cfg.get("dtype", "fp16") == "fp16" and vlm.device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=fp16)
    model.train()
    stats = {"steps": 0, "samples": 0, "skipped": 0, "loss": None, "stopped": "done", "pack": pack}
    run_loss, run_n, since, seen, t0 = 0.0, 0, 0, 0, time.time()
    calib, sized = tcfg.get("calib_samples", 2 * accum), False
    i = 0
    while i < len(order):
        if not sized and seen >= calib and deadline < float("inf"):
            # self-sizing: after `calib` samples, keep only as many as fit before the deadline
            sized, rate = True, (time.time() - t0) / seen
            fit, kept = int(0.95 * (deadline - time.time()) / rate), i
            while kept < len(order) and fit >= len(order[kept]):
                fit -= len(order[kept])
                kept += 1
            if kept < len(order):
                order = order[:max(kept, i + 1)]
                plan["total"] = math.ceil((seen + sum(map(len, order[i:]))) / accum)
                plan["warm"] = max(1, int(plan["total"] * tcfg.get("warmup", 0.03)))
            stats["planned_samples"] = seen + sum(map(len, order[i:]))
            print(f"train sized to {stats['planned_samples']} samples at {rate:.2f} s/sample", flush=True)
        if time.time() > deadline:
            stats["stopped"] = "deadline"
            break
        group = order[i]
        i += 1
        seen += len(group)
        shown = [rng.sample(LETTERS, 4) if tcfg.get("shuffle_options", True) else list(LETTERS) for _ in group]
        label = torch.tensor([s.index(r["correct_answer"]) for r, s in zip(group, shown)], device=vlm.device)
        video = video_of(group[0])
        with torch.autocast("cuda", dtype=torch.float16, enabled=fp16):
            if len(group) > 1:
                logits = vlm.packed_logits(group, video, shown)
            else:
                logits = vlm.letter_logits(vlm.inputs(group[0], video, shown[0]))[None]
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
            since = 0
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, tcfg.get("clip", 1.0))
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            stats["steps"] += 1
            if stats["steps"] % tcfg.get("log_every", 25) == 0:
                stats["loss"] = run_loss / run_n
                print(f"train step {stats['steps']}/{plan['total']} loss {stats['loss']:.4f} "
                      f"{(time.time() - t0) / seen:.2f} s/sample", flush=True)
                run_loss, run_n = 0.0, 0
            if stats["steps"] % tcfg.get("save_every", 200) == 0:
                model.save_pretrained(Path(out_dir) / "adapter")
    model.save_pretrained(Path(out_dir) / "adapter")
    model.eval()
    stats["seconds"] = round(time.time() - t0)
    return stats
