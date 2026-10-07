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


def test_autopilot_starts_on_merged_queue_pushes():
    d = yaml.safe_load((ROOT / ".github" / "workflows" / "autopilot.yml").read_text())
    on = d[True]  # PyYAML reads the key `on` as True
    assert "configs/**" in on["push"]["paths"] and on["push"]["branches"] == ["main"]


def test_autopilot_runs_only_main():
    text = (ROOT / ".github" / "workflows" / "autopilot.yml").read_text()
    d = yaml.safe_load(text)
    checkout = d["jobs"]["cycle"]["steps"][0]
    assert checkout["uses"].startswith("actions/checkout") and checkout["with"]["ref"] == "main"
    assert "GITHUB_SHA" not in text  # the commit sent to Kaggle is main's HEAD, not the trigger's


def test_autopilot_runs_twice_an_hour():
    # a finished job waits at most half an hour to be collected and submitted
    d = yaml.safe_load((ROOT / ".github" / "workflows" / "autopilot.yml").read_text())
    assert [s["cron"] for s in d[True]["schedule"]] == ["23,53 * * * *"]


def test_every_state_file_is_saved():
    # a state file the save step does not add is lost when the runner ends (blocked.jsonl, 7 Oct 2026)
    import re

    from reva import autopilot
    named = set(re.findall(r"^- (\S+\.(?:json|jsonl|csv|md)) ", autopilot.__doc__, re.M))
    text = (ROOT / ".github" / "workflows" / "autopilot.yml").read_text()
    saved = set(re.search(r"for f in ([^;]+); do", text).group(1).split())
    assert named and named <= saved, named - saved


def test_autopilot_chains_itself_while_waiting():
    # the cron never fired (7 Oct 2026); a finished Kaggle job must not wait for a push to main
    d = yaml.safe_load((ROOT / ".github" / "workflows" / "autopilot.yml").read_text())
    assert d["permissions"]["actions"] == "write"
    last = d["jobs"]["cycle"]["steps"][-1]
    assert last["if"] == "always()" and "waiting" in last["run"]
    # the dispatch must not depend on a checked-out repo: an always() step runs after a failed checkout
    assert "gh workflow run autopilot.yml --ref main --repo \"$GITHUB_REPOSITORY\"" in last["run"]
    assert d["jobs"]["cycle"]["timeout-minutes"] > 20 + 8 + 10  # the wait, the retries and a cycle fit


def test_next_cycle_dispatch_is_retried():
    # GitHub answered one dispatch with HTTP 500 and the loop stopped (run 56, 7 Oct 2026)
    import re

    d = yaml.safe_load((ROOT / ".github" / "workflows" / "autopilot.yml").read_text())
    run = d["jobs"]["cycle"]["steps"][-1]["run"]
    waits = re.search(r"for wait in ([\d ]+); do", run)
    assert waits and len(waits.group(1).split()) >= 3
    assert "&& exit 0" in run and run.rstrip().endswith("exit 1")  # a dispatch that never lands fails loudly
