"""One experiment, start to finish, on one GPU: data, frames, (train), dev score, test, zip.

    python -m reva.job --config run.json --out /kaggle/working/<run_id> [--device cuda:0]

Writes into --out:
- config.json     the run's full config, written first (a resumed session checks it is the same run)
- run.json        config, dev metrics (Codabench columns + weighted_accuracy), timings, versions
- dev_probs.json  dev probabilities (labeled data only, safe to keep)
- test_probs.json and <run_id>.zip   test predictions; these never enter the public repo
- adapter/        LoRA weights when the run trains
- ckpt/           the last full training checkpoint (adapter, optimizer, RNG, position)
- train.json      training stats, written when training is complete
- dev_probs.part.json, test_probs.part.json   predictions so far, while inference runs

Every one of these lets a later session carry on: start it with the same --out holding a copy of
this output, and finished stages are skipped, training resumes from ckpt/, inference from the
.part files. The last stdout line is "DONE <run_id>" on success and "PARTIAL <run_id>" (exit 3) when
a run that spans sessions stopped at a session's end. Any failure exits non-zero.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

from reva import config as C
from reva import data, frames, package, score


def ensure_videos(cfg: dict, rows: list[dict]) -> Path:
    """The video root. Downloads only the missing videos from Hugging Face (about 1.8 GB in all)."""
    root = Path(C.get(cfg, "data.video_root"))
    need = sorted({r["video_path"] for r in rows if not (root / r["video_path"]).exists()})
    if need:
        from huggingface_hub import snapshot_download

        print(f"downloading {len(need)} videos", flush=True)
        snapshot_download(C.get(cfg, "data.hf_repo"), repo_type="dataset", local_dir=str(root),
                          allow_patterns=need, max_workers=8)
    still = [p for p in need if not (root / p).exists()]
    if still:
        raise FileNotFoundError(f"{len(still)} videos missing after download, e.g. {still[:3]}")
    return root


def video_reader(cfg: dict, cache_dir: Path):
    n, side = C.get(cfg, "frames.n"), C.get(cfg, "frames.max_side")
    if not C.get(cfg, "model.use_video", True):
        return lambda row: None
    return lambda row: frames.load(cache_dir, row["video_path"], n, side)


def argmax(probs: dict[str, list[float]]) -> dict[str, str]:
    return {q: data.LETTERS[max(range(4), key=p.__getitem__)] for q, p in probs.items()}


def prepare(cfg: dict) -> tuple[dict[str, list[dict]], Path]:
    """Annotations, splits, videos and the frame cache. Idempotent, so a multi-lane job runs it
    once per lane before the lanes start, and the lanes never race on the cache."""
    ann = Path(C.get(cfg, "data.root"))
    data.fetch_annotations(ann, C.get(cfg, "data.hf_repo"))
    sp = data.make_splits(ann, C.get(cfg, "dev.holdout_frac"), C.get(cfg, "dev.seed"),
                          refit=C.get(cfg, "train.refit", False), refit_with_val=C.get(cfg, "train.refit_with_val", False))
    for k in ("fit", "dev", "test"):
        if lim := C.get(cfg, f"limit.{k}"):  # smoke runs and quick probes only
            sp[k] = sp[k][:lim]
    used = sp["dev"] + sp["test"] + (sp["fit"] if C.get(cfg, "train.enabled", False) else [])
    cache_dir = Path(C.get(cfg, "frames.cache"))
    if C.get(cfg, "model.use_video", True):
        root = ensure_videos(cfg, used)
        frames.build_cache(root, [r["video_path"] for r in used], cache_dir, C.get(cfg, "frames.n"),
                           C.get(cfg, "frames.max_side"), workers=C.get(cfg, "frames.workers", 4))
    return sp, cache_dir


PARTIAL_EXIT = 3
MARGIN_S = 900  # what a spanning run keeps at the session's end to save its state


class SessionOver(Exception):
    """A run that spans sessions (train.span_sessions) reached this session's end; its state is saved."""


def write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj))
    tmp.replace(path)  # atomic: a session killed mid-write never leaves half a file


def video_chunks(rows: list[dict], size: int) -> list[list[dict]]:
    """Rows in chunks of about `size`, never splitting a video, so each chunk keeps the shared prefix."""
    by_video: dict[str, list[dict]] = {}
    for r in rows:
        by_video.setdefault(r["video_path"], []).append(r)
    out, cur = [], []
    for group in by_video.values():
        cur += group
        if len(cur) >= size:
            out, cur = out + [cur], []
    return out + ([cur] if cur else [])


def predict_saved(vlm, rows: list[dict], video_of, perms: int, path: Path, stop_at: float | None = None,
                  chunk: int = 100) -> dict[str, list[float]]:
    """vlm.predict over rows, saved to `path` after every chunk. Questions already in `path` (from an
    earlier session) are not scored again. Raises SessionOver once `stop_at` has passed."""
    done = json.loads(path.read_text()) if path.exists() else {}
    todo = [r for r in rows if r["qa_id"] not in done]
    if done:
        print(f"resuming {path.name}: {len(rows) - len(todo)} of {len(rows)} already scored", flush=True)
    t0, n = time.time(), 0
    for part in video_chunks(todo, chunk):
        if stop_at is not None and time.time() > stop_at:
            raise SessionOver(f"{path.name}: {len(rows) - len(todo) + n} of {len(rows)} scored")
        done.update(vlm.predict(part, video_of, perms, log_every=10**9))
        n += len(part)
        write_json(path, done)
        print(f"predict {len(rows) - len(todo) + n}/{len(rows)} {(time.time() - t0) / n:.2f} s/q", flush=True)
    return {r["qa_id"]: done[r["qa_id"]] for r in rows}


def train_deadline(cfg: dict, vlm, sp: dict, video_of, perms: int, end: float, probe: int = 12,
                   strict: bool = True) -> float:
    """When training must stop so dev and test inference still finish inside the session.

    Times `probe` dev questions first (warm-up included, so the estimate errs high), then reserves
    that rate for every dev and test question plus 15% and a 10 minute margin. Training sizes
    itself to what is left (reva.model.train), so nobody has to guess sample counts per model.
    """
    rows = sp["dev"][:probe]
    t = time.time()
    vlm.predict(rows, video_of, perms, log_every=10**9)
    rate = (time.time() - t) / max(1, len(rows))
    n = len(sp["dev"]) + (0 if C.get(cfg, "infer.skip_test", False) else len(sp["test"]))
    infer_s = rate * n * 1.15 + 600
    cap = time.time() + 3600 * C.get(cfg, "train.max_hours", 1e9)
    deadline = min(cap, end - infer_s)
    print(f"BUDGET {rate:.2f} s/q, inference needs {infer_s / 3600:.2f} h, "
          f"training gets {max(0, deadline - time.time()) / 3600:.2f} h", flush=True)
    if strict and end - time.time() < infer_s:  # a spanning run finishes inference in a later session
        raise RuntimeError(f"inference alone needs {infer_s / 3600:.1f} h, more than the session has left; "
                           "use fewer frames, a smaller model, or infer.perms 1")
    return deadline


def run(cfg: dict, out: Path, device: str) -> dict:
    t0 = time.time()
    run_id = cfg["run_id"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "partial.json").unlink(missing_ok=True)  # an earlier session's; this one writes its own
    saved = out / "config.json"
    if saved.exists() and json.loads(saved.read_text()).get("run_id") != run_id:  # never resume someone else's state
        raise RuntimeError(f"{out} holds the state of {json.loads(saved.read_text()).get('run_id')}, not {run_id}")
    write_json(saved, cfg)  # with the checkpoint, everything a later session needs to carry on
    ann = Path(C.get(cfg, "data.root"))
    sp, cache_dir = prepare(cfg)
    train_on = C.get(cfg, "train.enabled", False)
    timings = {"prep_s": round(time.time() - t0)}
    video_of = video_reader(cfg, cache_dir)

    from reva.model import VLM, find_checkpoint, train

    vlm = VLM(cfg["model"], device)
    perms = C.get(cfg, "infer.perms", 1)
    end = t0 + 3600 * C.get(cfg, "job.max_hours", 1e9)
    span = C.get(cfg, "train.span_sessions", False)
    stop_at = end - MARGIN_S if span else None
    stats = None
    if train_on and (out / "train.json").exists():  # trained in an earlier session
        stats = json.loads((out / "train.json").read_text())
        vlm.add_lora(cfg["train"], out / "adapter")
        vlm.model.eval()
        print(f"training finished in an earlier session ({stats['samples']} samples); inference only", flush=True)
    elif train_on:
        deadline = train_deadline(cfg, vlm, sp, video_of, perms, end, strict=not span)
        ckpt = find_checkpoint(out)
        vlm.add_lora(cfg["train"], ckpt / "adapter" if ckpt else C.get(cfg, "train.init_adapter"))
        stats = train(vlm, sp["fit"], video_of, cfg["train"], out, deadline, seed=C.get(cfg, "train.seed", 0),
                      stop_at=stop_at, resume=ckpt)
        if stats["stopped"] == "session":
            raise SessionOver(f"training: step {stats['steps']}, {stats['samples']} samples")
        write_json(out / "train.json", stats)
    timings["train_s"] = round(time.time() - t0) - sum(timings.values())

    dev_probs = predict_saved(vlm, sp["dev"], video_of, perms, out / "dev_probs.part.json", stop_at)
    metrics = score.summary(sp["dev"], argmax(dev_probs), sp["test"], C.get(cfg, "dev.min_cell", 20))
    write_json(out / "dev_probs.json", dev_probs)
    print("DEV " + json.dumps({k: round(v, 4) for k, v in metrics.items()}), flush=True)
    timings["dev_s"] = round(time.time() - t0) - sum(timings.values())

    zip_path = None
    if not C.get(cfg, "infer.skip_test", False):
        test_probs = predict_saved(vlm, sp["test"], video_of, perms, out / "test_probs.part.json", stop_at)
        write_json(out / "test_probs.json", test_probs)
        if len(test_probs) == 4000 or C.get(cfg, "limit.test"):
            zip_path = out / f"{run_id}.zip"
            meta = data.load_metadata(ann, "test")
            if C.get(cfg, "limit.test"):  # a smoke run checks the zip against the rows it predicted
                package.write(zip_path, sp["test"], argmax(test_probs), C.get(cfg, "submit.format"), meta)
            else:
                package.write(zip_path, data.load_split(ann, "test"), argmax(test_probs), C.get(cfg, "submit.format"), meta)
    timings["test_s"] = round(time.time() - t0) - sum(timings.values())

    import torch
    import transformers

    result = {"run_id": run_id, "config": cfg, "metrics": metrics, "dev_set": data.DEV_SET, "train": stats, "timings": timings,
              "hours": round((time.time() - t0) / 3600, 3), "zip": zip_path.name if zip_path else None,
              "n": {k: len(v) for k, v in sp.items()},
              "versions": {"python": platform.python_version(), "torch": torch.__version__,
                           "transformers": transformers.__version__,
                           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"},
              "finished": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (out / "run.json").write_text(json.dumps(result, indent=1))
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--prepare-only", action="store_true", help="download and cache, then exit")
    a = ap.parse_args(argv)
    cfg = json.loads(Path(a.config).read_text())
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if a.prepare_only:
        prepare(cfg)
        print(f"PREPARED {cfg['run_id']}", flush=True)
        return 0
    try:
        run(cfg, Path(a.out), a.device)
    except SessionOver as e:
        write_json(Path(a.out) / "partial.json", {"run_id": cfg["run_id"], "where": str(e),
                                                  "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        print(f"PARTIAL {cfg['run_id']}: {e}", flush=True)
        return PARTIAL_EXIT
    print(f"DONE {cfg['run_id']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
