# Project memory: MAVU 2027 ReVA challenge pipeline

## Goal

Win the ReVA Drone Video Understanding Challenge of the MAVU workshop at WACV 2027.
Codabench competition 18274, phase 30831. Challenge deadline Nov 10, 2026. Workshop paper deadline
Oct 20, 2026 (OpenReview group `thecvf.com/WACV/2027/Workshop/MAVU`).

The repo is an end-to-end automated pipeline. It runs experiments on remote GPUs, scores them with a
local scorer that mirrors the Codabench leaderboard columns, compares against the live leaderboard,
and submits the best gated candidate. The method follows the EURS playbook that took rank 1 on MaCVi
2027 (`sidharthkumarpradhan/macvi27-challenge`, `docs/WORKFLOW.md`).

## Owner's standing rules (always follow, no exceptions)

Writing (docs, commit messages, comments, reports, chat):
- US English.
- Natural tone. No AI-sounding phrases.
- No bold within prose.
- Short, specific sentences.
- No em dashes.
- SQL curriculum outputs use Jupyter notebook-style markdown.
- Explain concepts inline while building code, not as standalone theory.

Engineering:
- Always do thorough research before code changes. Favor simple, efficient code design.
- Model selection for agent work: a lower tier model (preferably Haiku 4.5) for research and
  token-heavy tasks; a higher tier model for system design that needs more judgment. This gets the
  best work for the fewest tokens. Always do thorough analysis before proceeding.
- No AI author on any commit. Commit as the owner:
  `git -c user.name="Sidharth Pradhan" -c user.email="sidharthp@assignall.ai" commit ...`.
  No `Co-Authored-By` or other AI trailers. GitHub account: `sidharthkumarpradhan`.
- One feature per branch. Never `git add -A`. Secrets never enter the repo.
- Every module has tests. Every bug gets a regression test. `make test` passes before a merge.
- Fail loudly. Long jobs end with a sentinel line (`DONE` or `EXIT <code>`).

Decisions (from the EURS playbook, `docs/WORKFLOW.md` there):
- The owner decides. The agent prepares. Irreversible or outward-facing steps need the owner's OK,
  unless the owner has switched on the automated path for them (see "Submission policy").
- Compliance before score. Evidence over intuition. Every number has a source and a date.
- Measure before building. Find the largest (test weight x gap to leader) and work only that.
- Ceilings before architecture. Run an oracle or probe before a big build.
- Simple first. Reproducible from the remote alone.

## Compliance rules for ReVA (decided before any score)

- Use only official data and pretrained public weights. Disclose every pretrained model.
- Never tune on leaderboard feedback per question. Codabench shows per-task accuracy and allows
  100 submissions a day. Using that to infer test labels is label probing. We do not do it. A
  submission is only made when the local dev score improves; the board result only calibrates.
- Never commit test predictions or probabilities to this repo. It is public. Predictions stay in
  private Kaggle kernel outputs and go straight to Codabench.
- 392 test questions share video, question text and option set with the labeled val split (letters
  shuffled; measured 6 Oct 2026). A lookup table is not a model. It stays off. Whether the final
  refit may train on val is the owner's call (`train.refit_with_val`, default false).

## Submission policy

- Codabench allows 100 a day but also 100 in total per person for the only phase. Failed
  submissions do not count (`Phase.can_user_make_submissions` in the Codabench source). That is
  about 3 a day until Nov 10. The leaderboard keeps each user's best (`Force_Best`).
- The autopilot submits only when all hold: the zip passes `reva.package.validate`, the run's
  weighted dev accuracy beats the best submitted run by `submit.min_gain`, the daily cap leaves room,
  and the repo variable `AUTO_SUBMIT` is `on`. The owner flips that variable. Default off.
- Name each submission with the run id. Calibration pairs board scores with local runs by that id.

## Verified facts (6 Oct 2026)

See `docs/research.md` for sources. Key ones:
- Codabench API is public for reads: `GET /api/competitions/18274/` (pages, phases, leaderboard
  columns) and `GET /api/phases/30831/get_leaderboard/` (all rows). Scoring program and public data
  need login (403).
- One phase, "Test Phase", final, started 29 Sep 2026, no end date set, 100 submissions a day and
  100 in total per person. Execution limit 600 s.
- Leaderboard on 6 Oct 2026: leader 0.8735 (mkhlystun), then 0.8620, 0.8313, 0.8260. The paper's
  best model scores 80.04 on test. Change Detection and Temporal Grounding are weakest for all.
- Paper (arXiv 2609.35507): Qwen2.5-VL-7B + LoRA r16 a32, lr 2e-4, 32 frames at 640x360.
  Text-only input scores 29.95, so the options alone give little away.
- Videos: 1,139 mp4 files, 1.95 GB. Default backbone Qwen3-VL-4B-Instruct (fp16 fits a T4);
  8B needs 4-bit on a T4. A-D are single tokens; Qwen3-VL gets true frame timestamps.

## Where things are

- `src/reva`: one module per concern (see README). Config: `configs/competition.yaml` plus
  dotted overrides. Experiments: `configs/queue.yaml`.
- Run state: the `state` branch (STATUS.md, runs.jsonl, submissions.jsonl, board.csv).
- Current state and next steps: `docs/HANDOFF.md`. Facts with sources: `docs/research.md`.
- Session protocol: read HANDOFF, then this file, plan, small verified steps, update HANDOFF and
  this file before ending, push.
- Leaderboard columns: `overall_accuracy` plus 11 per-task accuracies. Sorted desc, 4 decimals.
- Data: Hugging Face `ReVA-Benchmark/ReVA` (Apache-2.0, not gated). `train.json` 15,773 QA,
  `val.json` 2,000 QA, `test.json` 4,000 QA with `correct_answer` empty. Every question has 4
  options A-D. Labels are balanced (about 25% per letter).
- Test is not a held-out video set. 1,012 of 1,014 test videos also appear in train; 366 in val.
  So local validation holds out questions, not videos. That mirrors the test.
- Source mix differs. Test: Hawk_UAV 1,856, ERA_Tra 1,285, VisDrone 733, UAVDT 126. Val: VisDrone
  1,115, Hawk_UAV 880, ERA_Tra 5. The dev set adds a stratified train holdout so ERA is measured.
- Submission: `submission.zip` holding `predictions.json`; all 4,000 `qa_id`s, each exactly one of
  A, B, C, D. Missing, duplicate or unknown ids fail evaluation.
