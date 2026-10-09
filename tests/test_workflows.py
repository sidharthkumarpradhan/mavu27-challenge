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
    assert last["if"] == "always()" and "[ -f idle ]" in last["run"]  # the loop marks idle when nothing is open
    # the dispatch must not depend on a checked-out repo: an always() step runs after a failed checkout
    assert "gh workflow run autopilot.yml --ref main --repo \"$GITHUB_REPOSITORY\"" in last["run"]
    # 4 h of cycles, then the last wait, a cycle that waits up to 20 minutes on a submission, the retries
    assert d["jobs"]["cycle"]["timeout-minutes"] >= 240 + 20 + 25 + 8 + 10


def test_next_cycle_dispatch_is_retried():
    # GitHub answered one dispatch with HTTP 500 and the loop stopped (run 56, 7 Oct 2026)
    import re

    d = yaml.safe_load((ROOT / ".github" / "workflows" / "autopilot.yml").read_text())
    run = d["jobs"]["cycle"]["steps"][-1]["run"]
    waits = re.search(r"for wait in ([\d ]+); do", run)
    assert waits and len(waits.group(1).split()) >= 3
    assert "&& exit 0" in run and run.rstrip().endswith("exit 1")  # a dispatch that never lands fails loudly


def test_autopilot_gets_both_kaggle_accounts():
    # reva.kaggle.from_env reads these; a secret missing from the job env leaves the account unused
    from reva.kaggle import ACCOUNT_VARS

    env = yaml.safe_load((ROOT / ".github" / "workflows" / "autopilot.yml").read_text())["jobs"]["cycle"]["env"]
    for user, key in ACCOUNT_VARS:
        assert env[user] == f"${{{{ secrets.{user} }}}}" and env[key] == f"${{{{ secrets.{key} }}}}"


def cycles_step():
    d = yaml.safe_load((ROOT / ".github" / "workflows" / "autopilot.yml").read_text())
    return next(s for s in d["jobs"]["cycle"]["steps"] if s.get("name", "").startswith("Cycles"))


def test_one_job_runs_many_cycles():
    # the dispatch API failed for 8 minutes straight (7 Oct 2026), so one job loops for hours and
    # dispatches the next job a few times a day instead of after every cycle
    run = cycles_step()["run"]
    assert "while :; do" in run and "-ge 14400" in run and "seq 20" in run
    assert "|| crashed=1" in run and run.rstrip().endswith("exit $crashed")  # a crash cycles on, then shows red
    assert "touch idle" in run


def test_loop_hands_over_when_main_moves_but_not_when_github_is_unreachable():
    run = cycles_step()["run"]
    assert "git ls-remote origin refs/heads/main" in run
    assert '[ -n "$main" ]' in run  # an empty answer is a network failure, not a new main


def test_tests_that_need_torch_skip_without_it():
    """Regression (8 Oct 2026): the autopilot runner installs no torch, and one test imported
    reva.model unguarded. Its red suite blocked every GPU push. A test that imports a module
    needing torch must call pytest.importorskip("torch") first."""
    import ast

    for path in (ROOT / "tests").glob("test_*.py"):
        for fn in ast.walk(ast.parse(path.read_text())):
            if not isinstance(fn, ast.FunctionDef) or not fn.name.startswith("test_"):
                continue
            imports = any(isinstance(n, ast.ImportFrom) and n.module == "reva.model" for n in ast.walk(fn))
            skips = any(isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "importorskip" for n in ast.walk(fn))
            assert skips or not imports, f"{path.name}::{fn.name} imports reva.model without importorskip('torch')"


def test_the_colab_workflow_saves_its_state_files():
    import re

    from reva import colab
    named = set(re.findall(r"^- (\S+\.jsonl) ", colab.__doc__, re.M))
    text = (ROOT / ".github" / "workflows" / "colab.yml").read_text()
    saved = set(re.search(r"for f in ([^;]+); do", text).group(1).split())
    assert named and named <= saved, named - saved
    assert "secrets.COLAB_TOKEN" in text and "chmod 600" in text  # the token never lands world-readable


def test_the_autopilot_starts_colab_sessions_every_cycle(tmp_path):
    """Regression (8 Oct 2026): the Colab start ran after the 4 h cycle loop, so a pending Colab
    entry waited up to 4 h for its session. It runs after every cycle now."""
    import os
    import subprocess

    assert "bash .github/autopilot/start_colab.sh" in cycles_step()["run"]
    script = ROOT / ".github" / "autopilot" / "start_colab.sh"
    (tmp_path / "state").mkdir()
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "queue.yaml").write_text("- name: ft-c\n  backend: colab\n  set: {train.enabled: true}\n")
    gh = tmp_path / "bin" / "gh"
    gh.parent.mkdir()
    gh.write_text('#!/bin/sh\necho "$*" >> "$GH_LOG"\n[ "$1" = run ] && cat "$GH_RUNS"\nexit 0\n')
    gh.chmod(0o755)
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run([*git, "commit", "-q", "--allow-empty", "-m", "m"], cwd=tmp_path, check=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=tmp_path, check=True, capture_output=True, text=True).stdout.strip()

    def start(runs, colab=""):
        (tmp_path / "runs.json").write_text(runs)
        log = tmp_path / "gh.log"
        log.unlink(missing_ok=True)
        env = {**os.environ, "PATH": f"{gh.parent}:{os.environ['PATH']}", "GH_LOG": str(log),
               "GH_RUNS": str(tmp_path / "runs.json"), "GITHUB_REPOSITORY": "o/r", "COLAB": colab}
        subprocess.run(["bash", str(script)], cwd=tmp_path, env=env, check=True, capture_output=True)
        return "workflow run colab.yml" in (log.read_text() if log.exists() else "")

    assert start("[]")  # pending entry, nothing running: start one
    assert not start('[{"status": "in_progress", "conclusion": ""}]')  # one session at a time
    failed = '[{"status": "completed", "conclusion": "failure", "headSha": "%s"}]'
    assert not start(failed % head)  # a failure needs a fix first
    assert start(failed % "0ld")  # the fix moved main: try the new code once
    assert not start("[]", colab="off")  # kill switch


def test_each_autopilot_cycle_reads_the_latest_state():
    # regression: the 07:44Z cycle on 9 Oct 2026 missed a Colab run committed at 07:27Z
    text = (ROOT / ".github" / "workflows" / "autopilot.yml").read_text()
    loop = text[text.index("while :; do"):]
    assert loop.index("git -C state pull") < loop.index("python -m reva.cli autopilot")
