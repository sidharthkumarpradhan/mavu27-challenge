You are fixing the ReVA autopilot after a failed Kaggle job. Work in this repo checkout.

1. Read CLAUDE.md and docs/HANDOFF.md. Follow every rule there, including the writing rules and the
   commit identity (no AI trailers).
2. Read outcome.json (`log_tail`, `notes`) and state/runs.jsonl (the `error` field of failed runs).
3. Find the root cause. Reproduce it with a unit test or `python -m reva.cli smoke` when you can.
4. Make the smallest fix. Add a regression test. Run `python -m pytest -q` until it passes.
5. Work on a branch made from the latest main: `git fetch origin main && git switch -c
   hotfix/<short-name> origin/main`. Commit only the files you changed (never `git add -A`), as
   the owner, with no AI trailers. Push the branch, open a PR with `gh pr create` (a plain,
   specific summary of the cause and the fix, no AI footer), and merge it with
   `gh pr merge --merge` once `python -m pytest -q` and `python -m reva.cli smoke` pass.
   The next cycle retries the failed runs (each run gets two attempts).
6. If the failure is outside the code (Kaggle quota, an account problem, a Codabench outage),
   change nothing and stop. The autopilot's issue already records it.

Never commit test predictions, probabilities or secrets. Never change the compliance rules.
