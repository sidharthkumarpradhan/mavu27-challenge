You are fixing the ReVA autopilot after a failed Kaggle job. Work in this repo checkout.

1. Read CLAUDE.md and docs/HANDOFF.md. Follow every rule there, including the writing rules and the
   commit identity (no AI trailers).
2. Read outcome.json (`log_tail`, `notes`) and state/runs.jsonl (the `error` field of failed runs).
3. Find the root cause. Reproduce it with a unit test or `python -m reva.cli smoke` when you can.
4. Make the smallest fix. Add a regression test. Run `python -m pytest -q` until it passes.
5. Commit only the files you changed (never `git add -A`) and push to the default branch.
   The next hourly cycle retries the failed runs (each run gets two attempts).
6. If the failure is outside the code (Kaggle quota, an account problem, a Codabench outage),
   change nothing and stop. The autopilot's issue already records it.

Never commit test predictions, probabilities or secrets. Never change the compliance rules.
