"""YAML configs with dotted-key overrides, e.g. `model.frames=16`."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

DEFAULT = Path(__file__).resolve().parents[2] / "configs" / "competition.yaml"


def load(path: str | Path = DEFAULT) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get(cfg: dict[str, Any], dotted: str, default: Any = None) -> Any:
    node: Any = cfg
    for key in dotted.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def override(cfg: dict[str, Any], pairs: dict[str, Any]) -> dict[str, Any]:
    """Copy of cfg with {"a.b": value} applied. Values given as strings are parsed as YAML,
    so "16" becomes 16 and "true" becomes True."""
    out = copy.deepcopy(cfg)
    for dotted, value in pairs.items():
        if isinstance(value, str):
            value = yaml.safe_load(value)
        node = out
        *parents, leaf = dotted.split(".")
        for key in parents:
            node = node.setdefault(key, {})
        node[leaf] = value
    return out


def parse_pairs(items: list[str]) -> dict[str, str]:
    """["a.b=1", "c=x"] -> {"a.b": "1", "c": "x"}."""
    out = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"override must look like key=value, got {item!r}")
        out[key.strip()] = value.strip()
    return out


def fingerprint(cfg: dict[str, Any]) -> str:
    """Short stable hash of a config. The run id, so the same experiment is never run twice."""
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:10]


# Keys that only steer the pipeline: where we submit, how the GPU job is scheduled. Changing them
# does not change what a run predicts, so they stay out of the run id. Without this, adding the
# Codabench organization queued the finished baseline again as a new run. remote.pip and
# remote.accelerator stay in: package versions and the GPU can change the outputs.
ORCHESTRATION = ("competition", "submit", "remote.kernel", "remote.lanes", "remote.max_hours",
                 "remote.weekly_gpu_hours", "remote.repo")


def experiment_fingerprint(cfg: dict[str, Any]) -> str:
    """Fingerprint of the parts of a config that change a run's predictions."""
    cfg = copy.deepcopy(cfg)
    for dotted in ORCHESTRATION:
        *parents, leaf = dotted.split(".")
        node = cfg
        for key in parents:
            node = node.get(key) if isinstance(node, dict) else None
        if isinstance(node, dict):
            node.pop(leaf, None)
    return fingerprint(cfg)
