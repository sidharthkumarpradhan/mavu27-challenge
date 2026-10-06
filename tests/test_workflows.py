"""The workflow files must parse. A step name with ": " once broke autopilot.yml (6 Oct 2026),
and GitHub rejects the whole workflow without running any of it."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_workflows_parse_and_have_steps():
    for f in sorted((ROOT / ".github" / "workflows").glob("*.yml")):
        d = yaml.safe_load(f.read_text())
        assert d.get("jobs"), f"{f.name}: no jobs"
        for name, job in d["jobs"].items():
            assert job.get("steps"), f"{f.name}:{name}: no steps"


def test_autopilot_is_on_by_default():
    d = yaml.safe_load((ROOT / ".github" / "workflows" / "autopilot.yml").read_text())
    assert d["jobs"]["cycle"]["if"] == "vars.AUTOPILOT != 'off'"


def test_autopilot_starts_on_queue_pushes_but_not_on_state():
    d = yaml.safe_load((ROOT / ".github" / "workflows" / "autopilot.yml").read_text())
    on = d[True]  # PyYAML reads the key `on` as True
    assert "configs/**" in on["push"]["paths"] and on["push"]["branches-ignore"] == ["state"]
