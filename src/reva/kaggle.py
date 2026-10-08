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


def subprocess_runner(cmd: list[str], env: dict[str, str] | None = None) -> tuple[int, str]:
    """Run the CLI. `env` replaces the process environment (see account_env), else it is inherited."""
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                          # PYTHONUTF8: the CLI writes <slug>.log with open()'s default encoding, which is
                          # cp1252 on Windows and crashed `kernels output` on the first real job (3 Oct 2026)
                          env={**(os.environ if env is None else env), "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"})
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


# Secret pairs in priority order. The second account takes over when the first one's weekly GPU quota
# is used up (owner's call, 7 Oct 2026).
ACCOUNT_VARS = [("KAGGLE_USERNAME", "KAGGLE_KEY"), ("KAGGLE_USERNAME_NEW", "KAGGLE_KEY_NEW")]
CRED_VARS = ("KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_CONFIG_DIR")


def account_env(user: str, key: str, home: Path, base: dict[str, str] | None = None) -> dict[str, str]:
    """Environment for a CLI call as one account. The CLI tries KAGGLE_API_TOKEN, then
    ~/.kaggle/access_token, then the username and key (kaggle 2.2.4, KaggleApi.authenticate), so a
    token left by another account would win. A fresh HOME and config dir per account rule that out."""
    env = {k: v for k, v in (os.environ if base is None else base).items() if k not in CRED_VARS}
    home.mkdir(parents=True, exist_ok=True)
    env.update(HOME=str(home), KAGGLE_CONFIG_DIR=str(home / ".kaggle"))
    key = "".join(key.split())
    if key.startswith("KGAT"):  # the new access token format; the classic key is 32 hex chars
        env["KAGGLE_API_TOKEN"] = key
    else:
        env.update(KAGGLE_USERNAME=user.strip(), KAGGLE_KEY=key)
    return env


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
    def __init__(self, cli: list[str] | None = None, runner: Runner | None = None,
                 sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic,
                 env: dict[str, str] | None = None):
        self.cli = cli or [sys.executable, "-m", "kaggle"]
        self.runner = runner or (lambda cmd: subprocess_runner(cmd, env))
        self.sleep, self.clock = sleep, clock

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

    # datasets: private stores for unfinished runs' checkpoints (reva.autopilot.stash). Checked
    # against kaggle 2.2.4 on 8 Oct 2026: a missing dataset's status prints a 403 and exits 0, an
    # uploaded .tar arrives unpacked, and a kernel mounts the dataset at
    # /kaggle/input/datasets/<owner>/<name>/.
    def dataset_status(self, ref: str) -> str | None:
        """'ready' and so on, or None when the dataset does not exist (or is not ours)."""
        out = self._run("datasets", "status", ref, "--format", "json", check=False)
        try:
            return json.loads(out.strip().splitlines()[-1])["status"]
        except (ValueError, KeyError, IndexError, TypeError):
            return None

    def dataset_upload(self, folder: Path, ref: str, message: str, timeout_s: float = 900, poll_s: float = 15) -> None:
        """Make `folder` the private dataset `ref`: create it, or add a version (the earlier ones
        stay). Returns once Kaggle has processed it."""
        meta = {"title": ref.split("/", 1)[1], "id": ref, "licenses": [{"name": "CC0-1.0"}]}
        (Path(folder) / "dataset-metadata.json").write_text(json.dumps(meta))
        if self.dataset_status(ref) is None:
            out = self._run("datasets", "create", "-p", str(folder), "-q")
        else:
            out = self._run("datasets", "version", "-p", str(folder), "-m", message, "-q")
        if "error" in out.lower():
            raise KaggleError(f"dataset upload to {ref} failed: {out.strip()[-500:]}")
        end = self.clock() + timeout_s
        while (state := self.dataset_status(ref)) != "ready":
            if self.clock() >= end:
                raise KaggleTimeout(f"dataset {ref} still '{state}' after {timeout_s / 60:.0f} min")
            self.sleep(poll_s)

    def dataset_download(self, ref: str, dest: Path) -> Path:
        Path(dest).mkdir(parents=True, exist_ok=True)
        self._run("datasets", "download", ref, "-p", str(dest), "--unzip", "-q")
        return Path(dest)


def owner(slug: str) -> str:
    return slug.split("/", 1)[0].lower()


class Accounts:
    """Several Kaggle accounts behind the Kaggle interface. A call on a kernel goes to the account
    that owns it: a private kernel's status and outputs are visible to its owner only."""

    def __init__(self, clients: dict[str, Kaggle]):
        if not clients:
            raise KaggleError("no Kaggle account configured")
        self.clients = {u.lower(): k for u, k in clients.items()}
        self.users = list(self.clients)  # priority order

    def _for(self, slug: str) -> Kaggle:
        if owner(slug) not in self.clients:
            raise KaggleError(f"no credentials for the owner of {slug}")
        return self.clients[owner(slug)]

    def push(self, kernel_dir: Path, timeout_s: int | None = None, accelerator: str | None = None) -> Push:
        slug = json.loads((Path(kernel_dir) / "kernel-metadata.json").read_text())["id"]
        return self._for(slug).push(kernel_dir, timeout_s, accelerator)

    def status(self, slug: str) -> tuple[str, str]:
        return self._for(slug).status(slug)

    def wait(self, slug: str, *args, **kwargs) -> tuple[str, str]:
        return self._for(slug).wait(slug, *args, **kwargs)

    def output(self, slug: str, dest: Path, file_pattern: str | None = None) -> tuple[list[Path], str]:
        return self._for(slug).output(slug, dest, file_pattern)

    def logs(self, slug: str) -> str:
        return self._for(slug).logs(slug)

    def dataset_status(self, ref: str) -> str | None:
        return self._for(ref).dataset_status(ref)

    def dataset_upload(self, folder: Path, ref: str, message: str, **kwargs) -> None:
        self._for(ref).dataset_upload(folder, ref, message, **kwargs)

    def dataset_download(self, ref: str, dest: Path) -> Path:
        return self._for(ref).dataset_download(ref, dest)


def from_env(home_root: str | Path | None = None, environ: dict[str, str] | None = None) -> Accounts | Kaggle:
    """Every account whose secrets are set (ACCOUNT_VARS), or the CLI's own login when none is."""
    environ = os.environ if environ is None else environ
    root = Path(home_root or Path.home() / ".reva-kaggle")
    clients = {}
    for i, (uvar, kvar) in enumerate(ACCOUNT_VARS):
        user, key = environ.get(uvar, "").strip(), environ.get(kvar, "").strip()
        if user and key and user.lower() not in clients:
            clients[user.lower()] = Kaggle(env=account_env(user, key, root / str(i), environ))
    return Accounts(clients) if clients else Kaggle()
