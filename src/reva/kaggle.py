"""Thin wrapper over the ``kaggle`` CLI. Copied from the EURS pipeline (macvi.remote.kaggle,
checked against kaggle 2.2.4 on 2 Oct 2026) and trimmed to the kernel calls used here.

Every call goes through an injectable ``runner(cmd) -> (returncode, output)``, so tests use a
fake and nothing here needs the network. Output formats relied on (kaggle_api_extended.py):

- ``kernels push`` prints ``Kernel version N successfully pushed.  Please check progress at URL``
  or ``Kernel push error: ...`` and exits 0 in both cases, so the text is what counts.
- ``kernels status`` prints ``<slug> has status "KernelWorkerStatus.COMPLETE"``, plus
  ``Failure message: "..."`` on failure. States: QUEUED, RUNNING, COMPLETE, ERROR,
  CANCEL_REQUESTED, CANCEL_ACKNOWLEDGED, NEW_SCRIPT.
- ``kernels output -p DIR -o`` downloads every output file and writes the run log to
  ``DIR/<kernel-slug>.log`` (a JSON list of ``{"stream_name", "time", "data"}`` records).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

Runner = Callable[[list[str]], tuple[int, str]]

DONE = {"complete"}
FAILED = {"error", "cancel_requested", "cancel_acknowledged"}


class KaggleError(RuntimeError):
    pass


class KaggleTimeout(KaggleError):
    pass


def subprocess_runner(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                          # PYTHONUTF8: the CLI writes <slug>.log with open()'s default encoding, which is
                          # cp1252 on Windows and crashed `kernels output` on the first real job (3 Oct 2026)
                          env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"})
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


@dataclass(frozen=True)
class Push:
    version: int | None
    url: str


def parse_push(text: str) -> Push:
    if m := re.search(r"Kernel push error:\s*(.*)", text):
        raise KaggleError(f"kernel push failed: {m.group(1).strip() or text.strip()}")
    m = re.search(r"Kernel version\s*(\d+)?\s*successfully pushed.*?progress at\s+(\S+)", text, re.S)
    if not m:
        raise KaggleError(f"unexpected `kaggle kernels push` output: {text.strip()[-500:]}")
    return Push(int(m.group(1)) if m.group(1) else None, m.group(2))


def parse_status(text: str) -> tuple[str, str]:
    """(state, failure message): state is lower case without the enum prefix, e.g. 'complete'."""
    m = re.search(r'has status "([^"]+)"', text)
    if not m:
        raise KaggleError(f"unexpected `kaggle kernels status` output: {text.strip()[-500:]}")
    state = m.group(1).split(".")[-1].strip().lower()
    fail = re.search(r'Failure message: "(.*)"', text, re.S)
    return state, fail.group(1).strip() if fail else ""


def log_text(raw: str) -> str:
    """Plain text of a Kaggle run log (JSON records, or already plain)."""
    try:
        records = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw
    if isinstance(records, list):
        return "".join(str(r.get("data", "")) for r in records if isinstance(r, dict))
    return raw


def tail(text: str, lines: int = 200) -> str:
    return "\n".join(text.splitlines()[-lines:])


def kernel_metadata(slug: str, code_file: str, *, gpu: bool, accelerator: str | None = None,
                    dataset_sources: list[str] = (), kernel_sources: list[str] = ()) -> dict:
    """kernel-metadata.json for a private Python script kernel (title slugifies to ``slug``)."""
    return {"id": slug, "title": slug.split("/", 1)[1], "code_file": code_file, "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": gpu, "enable_tpu": False,
            "enable_internet": True, "machine_shape": accelerator if gpu and accelerator else "",
            "dataset_sources": list(dataset_sources), "competition_sources": [], "kernel_sources": list(kernel_sources),
            "model_sources": []}


class Kaggle:
    def __init__(self, cli: list[str] | None = None, runner: Runner = subprocess_runner,
                 sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic):
        self.cli = cli or [sys.executable, "-m", "kaggle"]
        self.runner, self.sleep, self.clock = runner, sleep, clock

    def _run(self, *args: str, check: bool = True) -> str:
        code, out = self.runner([*self.cli, *args])
        if check and code != 0:
            raise KaggleError(f"`kaggle {' '.join(args)}` exited {code}: {out.strip()[-800:]}")
        return out

    # kernels
    def push(self, kernel_dir: Path, timeout_s: int | None = None, accelerator: str | None = None) -> Push:
        args = ["kernels", "push", "-p", str(kernel_dir)]
        if timeout_s:
            args += ["-t", str(int(timeout_s))]
        if accelerator:
            args += ["--accelerator", accelerator]
        return parse_push(self._run(*args))

    def status(self, slug: str) -> tuple[str, str]:
        return parse_status(self._run("kernels", "status", slug))

    def wait(self, slug: str, timeout_s: float, poll_s: float = 60.0,
             on_poll: Callable[[str], None] | None = None) -> tuple[str, str]:
        """Poll until complete/error/cancelled; raises KaggleTimeout after ``timeout_s``."""
        end = self.clock() + timeout_s
        while True:
            state, message = self.status(slug)
            if on_poll:
                on_poll(state)
            if state in DONE | FAILED:
                return state, message
            if self.clock() >= end:
                raise KaggleTimeout(f"{slug} still '{state}' after {timeout_s / 3600:.1f} h")
            self.sleep(poll_s)

    def output(self, slug: str, dest: Path, file_pattern: str | None = None) -> tuple[list[Path], str]:
        """Download every output file (or those whose name matches the regex ``file_pattern``);
        returns (files, plain-text run log). The CLI holds each file in memory while saving it."""
        dest.mkdir(parents=True, exist_ok=True)
        self._run("kernels", "output", slug, "-p", str(dest), "-o", "-q",
                  *(["--file-pattern", file_pattern] if file_pattern else []))
        log = dest / f"{slug.split('/')[-1]}.log"
        text = log_text(log.read_text(encoding="utf-8", errors="replace")) if log.exists() else ""
        return sorted(p for p in dest.rglob("*") if p.is_file() and p != log), text

    def logs(self, slug: str) -> str:
        return log_text(self._run("kernels", "logs", slug, check=False))
