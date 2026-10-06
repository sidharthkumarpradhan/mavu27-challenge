"""One experiment, start to finish, on one GPU: data, frames, (train), dev score, test, zip.

    python -m reva.job --config run.json --out /kaggle/working/<run_id> [--device cuda:0]

Writes into --out:
- run.json        config, dev metrics (Codabench columns + weighted_accuracy), timings, versions
- dev_probs.json  dev probabilities (labeled data only, safe to keep)
- test_probs.json and <run_id>.zip   test predictions; these never enter the public repo
- adapter/        LoRA weights when the run trains
The last stdout line is "DONE <run_id>" on success. Any failure exits non-zero.
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


def run(cfg: dict, out: Path, device: str) -> dict:
    t0 = time.time()
    run_id = cfg["run_id"]
    out.mkdir(parents=True, exist_ok=True)
    ann = Path(C.get(cfg, "data.root"))
    sp, cache_dir = prepare(cfg)
    train_on = C.get(cfg, "train.enabled", False)
    timings = {"prep_s": round(time.time() - t0)}
    video_of = video_reader(cfg, cache_dir)

    from reva.model import VLM, train

    vlm = VLM(cfg["model"], device)
    stats = None
    if train_on:
        vlm.add_lora(cfg["train"], C.get(cfg, "train.init_adapter"))
        deadline = t0 + 3600 * C.get(cfg, "train.max_hours", 1e9)
        stats = train(vlm, sp["fit"], video_of, cfg["train"], out, deadline, seed=C.get(cfg, "train.seed", 0))
    timings["train_s"] = round(time.time() - t0) - sum(timings.values())

    perms = C.get(cfg, "infer.perms", 1)
    dev_probs = vlm.predict(sp["dev"], video_of, perms)
    metrics = score.summary(sp["dev"], argmax(dev_probs), sp["test"], C.get(cfg, "dev.min_cell", 20))
    (out / "dev_probs.json").write_text(json.dumps(dev_probs))
    print("DEV " + json.dumps({k: round(v, 4) for k, v in metrics.items()}), flush=True)
    timings["dev_s"] = round(time.time() - t0) - sum(timings.values())

    zip_path = None
    if not C.get(cfg, "infer.skip_test", False):
        test_probs = vlm.predict(sp["test"], video_of, perms)
        (out / "test_probs.json").write_text(json.dumps(test_probs))
        if len(test_probs) == 4000 or C.get(cfg, "limit.test"):
            zip_path = out / f"{run_id}.zip"
            meta = {"run_id": run_id}
            if C.get(cfg, "limit.test"):  # a smoke run checks the zip against the rows it predicted
                package.write(zip_path, sp["test"], argmax(test_probs), C.get(cfg, "submit.format"), meta)
            else:
                package.write(zip_path, data.load_split(ann, "test"), argmax(test_probs), C.get(cfg, "submit.format"), meta)
    timings["test_s"] = round(time.time() - t0) - sum(timings.values())

    import torch
    import transformers

    result = {"run_id": run_id, "config": cfg, "metrics": metrics, "train": stats, "timings": timings,
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
    run(cfg, Path(a.out), a.device)
    print(f"DONE {cfg['run_id']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
