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

    def inputs(self, row: dict, video: dict | None, order: list[str]) -> dict:
        """Tokenized model input for one question. video=None gives a text-only probe."""
        from transformers.video_utils import VideoMetadata

        content = [{"type": "video"}] if video is not None else []
        times = None
        if video is not None and self.cfg.get("timestamps_in_text"):
            times = [float(i) / float(video["fps"]) for i in video["indices"]]
        content.append({"type": "text", "text": prompt_text(row, order, times)})
        text = self.processor.apply_chat_template([{"role": "user", "content": content}],
                                                  add_generation_prompt=True, tokenize=False)
        kw = {"text": [text], "return_tensors": "pt"}
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

    @torch.no_grad()
    def predict(self, rows: list[dict], video_of, perms: int = 1, log_every: int = 200) -> dict[str, list[float]]:
        """{qa_id: [pA, pB, pC, pD]} in original option order, averaged over `perms` shifts."""
        self.model.eval()
        out, t0 = {}, time.time()
        for n, row in enumerate(sorted(rows, key=lambda r: (r["video_path"], r["qa_id"])), 1):
            video = video_of(row)
            probs = np.zeros(4)
            for k in range(perms):
                order = shift(k)
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


def train(vlm: VLM, rows: list[dict], video_of, tcfg: dict, out_dir: str | Path, deadline: float,
          seed: int = 0) -> dict:
    """LoRA fine-tune on letter cross-entropy. Stops at the end of the epochs or at `deadline`
    (time.time()), whichever is first, and saves the adapter to out_dir/adapter.

    fp16 (T4, V100) uses a GradScaler; steps with a non-finite loss are skipped and counted.
    """
    model = vlm.model
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=tcfg.get("lr", 2e-4), weight_decay=tcfg.get("weight_decay", 0.0))
    accum = tcfg.get("grad_accum", 8)
    limit = tcfg.get("max_samples") or len(rows)
    total = math.ceil(min(limit, len(rows)) * tcfg.get("epochs", 1) / accum)
    warm = max(1, int(total * tcfg.get("warmup", 0.03)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (
        1 + math.cos(math.pi * min(1.0, s / max(1, total)))))
    fp16 = vlm.cfg.get("dtype", "fp16") == "fp16" and vlm.device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=fp16)
    rng = random.Random(seed)
    order = [r for _ in range(math.ceil(tcfg.get("epochs", 1))) for r in rng.sample(rows, len(rows))]
    order = order[: int(min(limit, len(rows)) * tcfg.get("epochs", 1))]
    model.train()
    stats = {"steps": 0, "samples": 0, "skipped": 0, "loss": None, "stopped": "done"}
    run_loss, t0 = 0.0, time.time()
    for i, row in enumerate(order, 1):
        if time.time() > deadline:
            stats["stopped"] = "deadline"
            break
        shown = rng.sample(LETTERS, 4) if tcfg.get("shuffle_options", True) else list(LETTERS)
        label = torch.tensor([shown.index(row["correct_answer"])], device=vlm.device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=fp16):
            logits = vlm.letter_logits(vlm.inputs(row, video_of(row), shown))
        loss = torch.nn.functional.cross_entropy(logits[None], label) / accum
        if not torch.isfinite(loss):
            stats["skipped"] += 1
            opt.zero_grad(set_to_none=True)
            if stats["skipped"] > tcfg.get("max_skipped", 50):
                raise RuntimeError("too many non-finite losses; try dtype fp32 compute or a lower lr")
            continue
        scaler.scale(loss).backward()
        run_loss += loss.item() * accum
        stats["samples"] += 1
        if i % accum == 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, tcfg.get("clip", 1.0))
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            stats["steps"] += 1
            if stats["steps"] % tcfg.get("log_every", 25) == 0:
                stats["loss"] = run_loss / (accum * tcfg.get("log_every", 25))
                print(f"train step {stats['steps']}/{total} loss {stats['loss']:.4f} "
                      f"{(time.time() - t0) / i:.2f} s/sample", flush=True)
                run_loss = 0.0
            if stats["steps"] % tcfg.get("save_every", 200) == 0:
                model.save_pretrained(Path(out_dir) / "adapter")
    model.save_pretrained(Path(out_dir) / "adapter")
    model.eval()
    stats["seconds"] = round(time.time() - t0)
    return stats
