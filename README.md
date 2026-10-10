# ReVA challenge pipeline (MAVU @ WACV 2027)

An end-to-end, unattended pipeline for the
[ReVA Drone Video Understanding Challenge](https://www.codabench.org/competitions/18274/):
4-option multiple-choice questions about drone videos, scored by accuracy on 4,000 hidden answers.
Challenge deadline Nov 10, 2026.

It follows the playbook that took rank 1 at MaCVi 2027 EURS
(`sidharthkumarpradhan/macvi27-challenge`, `docs/WORKFLOW.md`): measure first, a local scorer that
mirrors the board, compliance gates before any upload, and a half-hourly loop that needs no laptop.

```
live leaderboard -> per-task gap ----------------------------------------------+
configs/queue.yaml -> Kaggle job (2 lanes, one per T4) -> dev score + test zip -> gate -> Codabench
                                                     \-> runs.jsonl, STATUS.md on the `state` branch
```

## Modules (`src/reva`)

| module | job |
|---|---|
| `data` | annotations, schema checks, the dev split (val plus a stratified train holdout) |
| `score` | Codabench's 12 columns locally, plus accuracy re-weighted to the test mix |
| `package` | writes `submission.zip` and checks every rule on the Data page |
| `frames` | decodes each video once into uniformly sampled frames with true timestamps |
| `model` | Qwen3-VL prompt, A-D letter scoring in one pass, option-shift TTA, LoRA training |
| `job` | one experiment start to finish on one GPU |
| `remote`, `kaggle` | queue entries to a Kaggle kernel; the Kaggle CLI wrapper |
| `board`, `codabench` | public leaderboard; login, upload, submit, poll |
| `preflight` | last checks before an upload, and the projected board score |
| `arena` | averages the best runs' probabilities; the best top-k mix on dev becomes one more candidate |
| `analyze` | error analysis on dev probabilities: points lost per source and task, confidence, oracle, blends |
| `autopilot` | one cycle (twice an hour): board, collect, gated submit, push next, STATUS.md |
| `smoke` | the whole GPU job on CPU with a tiny model and synthetic videos |

## Use

```bash
make install        # package + CPU torch + test deps
make test           # unit tests (no GPU)
make smoke          # full job on CPU, prints READY
python -m reva.cli board                     # live leaderboard
python -m reva.cli build --sha <commit>      # the Kaggle kernel for the next queued lanes (pushes nothing)
python -m reva.cli job --config-json run.json --out out/   # one experiment on a local GPU
python -m reva.cli submit --run <run_id> --state state/      # owner-triggered submission, same checks
python -m reva.cli analyze --probs a=dev_probs.json --probs b=other.json   # where runs lose points, on dev only
```

## Automation

`.github/workflows/autopilot.yml` runs twice an hour on the default branch, on every merged
code or config change, and, while a Kaggle job or a submission is open, about 20 minutes after
the last cycle (each cycle starts the next):

1. Snapshot the leaderboard.
2. Poll open submissions.
3. Collect a finished Kaggle job: dev metrics into `runs.jsonl`, the zip kept in the kernel's
   private output.
   Then the arena: when a new run arrives, the mean of the top k runs (k from 2 to 5, picked on
   dev) joins `runs.jsonl` as one more candidate if it beats the best single run on dev.
4. Submit the best new run if `reva.autopilot.gate` allows it, `AUTO_SUBMIT` is `on`, and the
   rebuilt zip passes `reva.preflight` (file rules, round trip to the run's own probabilities,
   compliance flags, letter balance, live board metric). A run that fails is recorded in
   `blocked.jsonl` and never retried.
5. Push the next pending queue entries as a new kernel, within the weekly GPU quota.
6. Write `STATUS.md` (board, the next candidate's projected board score, our gap per task,
   calibration, runs, budgets) on the `state` branch.

Repo settings it needs: variables `AUTOPILOT=on` and, when the owner decides, `AUTO_SUBMIT=on`;
secrets `KAGGLE_USERNAME`, `KAGGLE_KEY`, `CODABENCH_USERNAME`, `CODABENCH_PASSWORD`.

The repo is public. Test predictions never enter it: they stay in private Kaggle outputs and
go straight to Codabench.

To add an experiment, append an entry to `configs/queue.yaml`. See `CLAUDE.md` for the rules
every change follows and `docs/HANDOFF.md` for the current state.
