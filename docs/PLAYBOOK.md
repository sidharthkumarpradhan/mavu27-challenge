# Competition playbook (from MAVU 2027 ReVA, Oct 2026)

What worked on ReVA, in the order we built it. Paste this into the new repo as `docs/PLAYBOOK.md`
and point the new session at it. Swap names (Codabench, Kaggle, video QA) for the new
competition's equivalents. Reference code: `sidharthkumarpradhan/mavu27-challenge` (`src/reva`).

## 0. Session setup

1. New public repo with an empty `main`. Root `CLAUDE.md` holds four blocks: goal and deadlines,
   the owner's standing rules (below), compliance rules, and "where things are".
2. `docs/HANDOFF.md`: current state, what is unverified, next steps. Dated sections, appended,
   never rewritten. `docs/research.md`: every fact with source URL and date.
3. Session protocol: read HANDOFF, then CLAUDE.md, plan, small verified steps, update both
   before ending, push.
4. Owner's standing rules to copy verbatim into CLAUDE.md:
   - Commit as `git -c user.name="Sidharth Pradhan" -c user.email="sidharthp@assignall.ai"`.
     No Co-Authored-By, no "Generated with", no AI footprint in code, commits or PRs.
   - Branch from latest `origin/main` as `feature/<name>` or `hotfix/<name>`. One change per PR.
   - `make test` and `make smoke` before every PR. Every module has tests. Every bug gets a
     regression test. Never `git add -A`. Secrets never enter the repo.
   - US English, short sentences, no bold in prose, no em dashes.
   - Haiku subagents for research and token-heavy reads; the main model for design.
   - Fail loudly. Long jobs end with a sentinel line (`DONE` or `EXIT <code>`).
5. Automation grants to record in CLAUDE.md with the owner's quote and date: auto submit,
   auto merge once CI is green ("bot reviews never block"), which GPU accounts may be used.

## 1. Research before any code (day 1)

- Read every competition page in full: Overview, Data, Terms, phase settings, leaderboard, forum.
  The platform's public API often serves these without login (Codabench:
  `GET /api/competitions/<id>/`, `GET /api/phases/<phase>/get_leaderboard/`).
- Record: submission caps (daily and total; on Codabench failed submissions are free), execution
  limit, leaderboard columns and sort key, file format, end date.
- Read the dataset card and the paper. Note the baseline recipe and its score.
- Measure the data: split sizes, label balance, source mix per split, and overlap between test
  and train/val (videos, questions). This decides how the dev split must look.
- Write the compliance rules into CLAUDE.md before the first score. ReVA's set, reusable:
  - Predictions must use the real input (video), not text-only. Probes never go to test.
  - Never redistribute data or labels. Public repo holds code, configs, metrics only.
  - Never commit test predictions. They live in private kernel outputs / private datasets.
  - No per-question tuning on board feedback (label probing). Submit only when local dev improves.
  - No lookup tables from overlapping questions. Refit on val is the owner's call, default off.
  - Only official data and public pretrained weights; disclose and cite them.

## 2. Repo skeleton (one module per concern)

| Module | Job |
| --- | --- |
| `config.py` | YAML plus dotted overrides (`model.frames=16`). |
| `data.py` | Fetch annotations, build the dev split. |
| `score.py` | Local scorer that mirrors the leaderboard columns exactly (keys copied from the API). |
| `frames.py` / input cache | Decode each input once, cache it. |
| `model.py` | Prompt, scoring, LoRA fine-tune, checkpoints. |
| `job.py` | One experiment end to end on one GPU: data, train, dev score, test predict, zip. |
| `package.py` | Write and `validate` the submission zip (all ids, one valid answer each). |
| `preflight.py` | Last checks before upload (below). |
| `board.py` | Live leaderboard snapshot and per-task gap to the leader. |
| `codabench.py` | Login, upload, poll (mirror the platform's own web client). |
| `kaggle.py` / `remote.py` | Generate and push one private Kaggle kernel per queue job. |
| `colab.py` | Second GPU backend via the Colab CLI. |
| `registry.py` | Append-only JSONL on a `state` branch: runs, submissions, board. |
| `arena.py` | Rank runs and averaged ensembles on dev. |
| `autopilot.py` | One unattended cycle (below). |
| `smoke.py` | CPU dry run of the whole job with a tiny random model and synthetic data. |
| `cli.py` | `python -m <pkg>.cli <command>` for all of the above. |

`configs/competition.yaml` holds defaults. `configs/queue.yaml` lists experiments as overrides
plus `backend` (kaggle or colab). Makefile: `install`, `test`, `smoke`, `board`.

## 3. Dev split that predicts the board

- Mirror how the test was drawn. ReVA's test reused train videos, so dev holds out questions, not
  videos. Add a stratified train holdout so every test source is measured.
- Drop dev questions that copy train questions (ReVA: 1,456 of 2,000 val). After that, dev was
  0.813 against 0.8125 on the board.
- Report dev per leaderboard column plus a test-weighted overall, with a 95% interval.

## 4. Model recipe that worked (video QA)

- Score answers by the logits of the option letter tokens at the first answer position.
  No generation, no parsing.
- Encode each input once and reuse its KV cache for every question and option shift. Option-shift
  TTA (`infer.perms`) becomes nearly free.
- LoRA fine-tune beats zero-shot by a wide margin (0.7410 to 0.8125). Training time was the lever,
  so train a full epoch across GPU sessions.
- Pack several questions about one input into one sequence with a block mask (halve the pack on
  OOM). 8B in 4-bit on a T4; bf16 on an A100.
- Self-size training to the session: reserve inference time from a 12-question timing probe,
  reshape the cosine schedule to the deadline, print a BUDGET line.
- Run ids hash the whole config. Never add a default to competition.yaml while a job runs; read
  new keys with a fallback in code. A test pins the ids of running jobs.

## 5. Checkpoints and resume (never lose compute)

- Training writes a full checkpoint every 20 minutes (adapter plus optimizer state), via
  `ckpt.tmp` then rename, keeping `ckpt.old` until the swap. Resume reads `ckpt`, then `ckpt.old`.
  Inference saves partial predictions every 100 questions.
- After each session every lane folder becomes a new version of a private Kaggle dataset
  `<prefix>-run-<run id>` (logs and config included). The next session restores from it.
- `train.span_sessions`: train to the session end, stop, resume next session.
- Colab: snapshot the run folder to the private dataset every hour, not only at the end. Three
  failed polls in a row end the session; it is recorded "partial, resumable" and the next session
  resumes from the snapshot. Partial sessions do not count as failures.
- `train.final_session`: when the units left buy no further session, shrink training so dev and
  test prediction fit, so the run still reaches the board.

## 6. GPU backends

- Kaggle: one generated private kernel per push. It git-fetches the public repo at the exact
  commit and runs one lane per GPU (T4 x2). Every push gets its own kernel slug (a reused slug
  overwrote outputs). Two accounts in order of remaining weekly quota; every call on a kernel uses
  its owner's credentials. GPU pacing per account.
- Colab Pro via `google-colab-cli`: one session at a time, `colab.yml` workflow, token in the
  `COLAB_TOKEN` secret. Set it with `gh secret set COLAB_TOKEN < ~/.config/colab-cli/token.json`
  (the file has no trailing newline; a terminal paste breaks the JSON). A `--check-token` command
  names the problem without printing the secret. Check the unit balance before each session
  (A100 about 13 units/hour, keep a 1.5 h minimum).
- Kill switches as repo variables: `AUTOPILOT=off`, `AUTO_SUBMIT=off`, `COLAB=off`.

## 7. Autopilot (GitHub Actions)

One cycle: board snapshot, poll open submissions, collect finished runs (score dev, stash, rescore),
arena, gate, preflight, submit, launch the next queue job, write STATUS.md to the `state` branch.

- The cron schedule did not fire reliably. While anything is open (job running, submission
  scoring), each cycle dispatches the next one about 20 minutes later. A push to main restarts it.
- Submission gate, all must hold: zip passes `validate`; dev beats the best submitted run by
  `submit.min_gain`; nothing still scoring; daily cap and total budget reserve allow it
  (total cap / days left; ReVA: 100 total, about 3 a day).
- Preflight before every upload, automatic or manual: the zip holds exactly the run's answers,
  run.json records real input and no val training (missing flag fails), no answer letter under 5%
  or over 60%, and the board still ranks by the columns we score. Failures go to `blocked.jsonl`.
- Probe the predictions file layout automatically: a failed layout tries the next one next cycle;
  a layout that scored once is kept.
- Name each submission with the run id so board scores pair with local runs (calibration).
- STATUS.md shows the projected board score (dev plus mean board-minus-dev gap). It informs, it
  never gates.
- Arena: once two fair runs exist, average the top k on dev and add the winner as an `ens-` run.
  It goes through the same gate and preflight; every member must be compliant.

## 8. Git and GitHub mechanics in the cloud session

- `gh` GraphQL returned 403; REST works. Create a PR with
  `gh api repos/<owner>/<repo>/pulls --input pr.json`; merge with
  `gh api -X PUT repos/<owner>/<repo>/pulls/<n>/merge -f merge_method=squash -f sha=<head sha>`
  (head sha from `git ls-remote origin refs/heads/<branch>`).
- Strip any auto footer from PR bodies. Merge as soon as CI is green on the latest commit.
- No foreground `sleep`; wait on CI with a background poll and react to its completion.
- Scheduled check-ins (`send_later`, an hourly watch routine) read job logs and message the owner
  only on a new board score, a fix, or a blocker.
- Secrets live in GitHub secrets and a scratchpad env file. Never print them. STATUS.md and
  Actions logs are public.

## 9. Order of work for the new competition

1. Day 1: research (section 1), CLAUDE.md, compliance rules, data measurements.
2. Scorer, dev split, packaging with validate, smoke on CPU. CI runs `make test`.
3. Board fetch and the gap table. Codabench (or platform) client, preflight.
4. Remote job on one GPU: zero-shot baseline plus a text-only probe (never submitted).
5. Autopilot with submission gate. First submission confirms the format and gives the first
   calibration pair. Failed submissions are free, so submit early.
6. Fine-tune with checkpoints and multi-session resume. Then the second GPU backend.
7. Work the largest (test weight x gap to leader) column. Arena ensembles once runs diversify.
8. Paper outline once the first fine-tune is scored.
