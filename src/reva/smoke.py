"""CPU dry run of the whole GPU job on synthetic data with a tiny random Qwen3-VL.

    python -m reva.smoke [--out work/smoke]

It writes a few short synthetic videos and annotations in the ReVA schema, then runs reva.job with
training on: frames, LoRA steps, dev scoring, test prediction and a validated submission zip. It
prints READY only when every stage produced its file. Needs the `gpu` extras (CPU torch is fine)
and network access to Hugging Face for the 10 MB test model.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np

from reva import config as C
from reva import job
from reva.score import TASK_COLUMNS

TINY = "trl-internal-testing/tiny-Qwen3VLForConditionalGeneration"


def write_video(path: Path, n: int = 24, size: tuple[int, int] = (96, 64), seed: int = 0) -> None:
    import av

    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    with av.open(str(path), "w") as c:
        s = c.add_stream("mpeg4", rate=8)
        s.width, s.height, s.pix_fmt = size[0], size[1], "yuv420p"
        for _ in range(n):
            img = rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
            for p in s.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                c.mux(p)
        for p in s.encode():
            c.mux(p)


def synthetic(root: Path) -> None:
    """Three videos; train/val/test rows over every task and two sources."""
    videos = ["VisDrone/a.mp4", "Hawk_UAV/NJ/b.mp4", "ERA_Select/Baseball/c.mp4"]
    src = {"VisDrone": ("VisDrone", "VisDrone"), "Hawk_UAV": ("Hawk_UAV", "NJ"), "ERA_Select": ("ERA_Tra", "Baseball")}
    for i, v in enumerate(videos):
        write_video(root / v, seed=i)
    rng = np.random.default_rng(0)

    def rows(split: str, per_task: int) -> list[dict]:
        out = []
        for t in TASK_COLUMNS:
            for k in range(per_task):
                v = videos[(len(out) + k) % 3]
                name, sub = src[v.split("/")[0]]
                out.append({"qa_id": f"{split}_{len(out) + 1:06d}", "video_path": v, "subdir": sub, "dataset_name": name,
                            "category": "c", "task": t, "question": f"What is shown ({t})?",
                            "options": {L: f"option {L}" for L in "ABCD"},
                            "correct_answer": "" if split == "test" else "ABCD"[int(rng.integers(4))]})
        return out

    for split, n in (("train", 2), ("val", 1), ("test", 1)):
        qa = rows(split, n)
        (root / f"{split}.json").write_text(json.dumps({"metadata": {"total_questions": len(qa)}, "QA": qa}))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="work/smoke")
    a = ap.parse_args(argv)
    out = Path(a.out)
    shutil.rmtree(out, ignore_errors=True)
    synthetic(out / "data")
    cfg = C.override(C.load(), {
        "run_id": "smoke", "data.root": str(out / "data"), "data.video_root": str(out / "data"),
        "frames.cache": str(out / "frames"), "frames.n": 4, "frames.max_side": 64, "frames.workers": 1,
        "model.id": TINY, "model.dtype": "fp32", "train.enabled": True, "train.grad_accum": 2,
        "train.log_every": 1, "dev.holdout_frac": 0.0, "dev.min_cell": 1, "infer.perms": 2,
        "limit.test": 1000})
    result = job.run(cfg, out / "run", "cpu")
    need = ["run.json", "dev_probs.json", "test_probs.json", "smoke.zip", "adapter/adapter_config.json"]
    missing = [n for n in need if not (out / "run" / n).exists()]
    if missing:
        print(f"NOT READY: missing {missing}")
        return 1
    with zipfile.ZipFile(out / "run" / "smoke.zip") as z:  # the zip must keep test.json's own metadata
        meta = json.loads(z.read("predictions.json"))["metadata"]
    if meta != json.loads((out / "data" / "test.json").read_text())["metadata"]:
        print(f"NOT READY: zip metadata {meta} differs from test.json")
        return 1
    print("DEV", json.dumps(result["metrics"]))
    print("READY")
    return 0


if __name__ == "__main__":
    sys.exit(main())
