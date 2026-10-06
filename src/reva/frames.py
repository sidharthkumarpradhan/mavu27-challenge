"""Decode each video once into a small cache of uniformly sampled frames.

Every video has about 15 questions in train and 4 in test, so decoding once per video and reading
a .npz afterwards saves most of the CPU time. The cache keeps the true frame indices and fps, so
Qwen3-VL can print real timestamps ("<0.5 seconds>") for questions like "at the 00:03 mark".

Uniform sampling takes the center of n equal segments, the common choice (the ReVA paper samples
32 frames uniformly at 640x360, Appendix B).
"""

from __future__ import annotations

import hashlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


def sample_indices(total: int, n: int) -> list[int]:
    """Centers of n equal segments of [0, total). Short videos repeat frames rather than fail."""
    if total <= 0:
        raise ValueError("video has no frames")
    return [min(total - 1, int((i + 0.5) * total / n)) for i in range(n)]


def resize(img, max_side: int):
    """PIL image scaled so its longer side is at most max_side (aspect kept)."""
    w, h = img.size
    s = max_side / max(w, h)
    return img if s >= 1 else img.resize((max(1, round(w * s)), max(1, round(h * s))))


def decode(path: str | Path, n: int, max_side: int) -> dict:
    """{"frames": uint8 (n, H, W, 3), "indices", "fps", "total"} for one video.

    Two passes: count packets (no decoding), then decode and keep only the chosen frames. This
    stays fast and small even for 4K clips, and never trusts the container's frame count.
    """
    import av

    with av.open(str(path)) as c:
        total = sum(1 for p in c.demux(video=0) if p.size)
    with av.open(str(path)) as c:
        stream = c.streams.video[0]
        fps = float(stream.average_rate or stream.guessed_rate or 30)
        want = sample_indices(total, n)
        keep = {}
        for i, frame in enumerate(c.decode(video=0)):
            if i in want and i not in keep:
                keep[i] = np.asarray(resize(frame.to_image(), max_side))
            if i >= want[-1]:
                break
    last = keep[max(keep)]  # a short decode (broken tail) reuses the last good frame
    frames = np.stack([keep.get(i, last) for i in want])
    return {"frames": frames, "indices": np.array(want), "fps": fps, "total": total}


def cache_name(video_path: str, n: int, max_side: int) -> str:
    return hashlib.sha1(f"{video_path}|{n}|{max_side}".encode()).hexdigest()[:16] + ".npz"


def _one(args) -> str:
    root, rel, out, n, max_side = args
    dest = Path(out) / cache_name(rel, n, max_side)
    if not dest.exists():
        d = decode(Path(root) / rel, n, max_side)
        tmp = dest.with_suffix(".tmp.npz")
        np.savez(tmp, **d)
        tmp.replace(dest)
    return rel


def build_cache(video_root: str | Path, rel_paths: list[str], out: str | Path, n: int, max_side: int,
                workers: int = 4) -> Path:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(str(video_root), p, str(out), n, max_side) for p in sorted(set(rel_paths))]
    if workers <= 1:
        for j in jobs:
            _one(j)
    else:
        with ProcessPoolExecutor(workers) as pool:
            list(pool.map(_one, jobs, chunksize=4))
    return out


def load(out: str | Path, rel: str, n: int, max_side: int) -> dict:
    with np.load(Path(out) / cache_name(rel, n, max_side)) as z:
        return {k: z[k] for k in z.files}
