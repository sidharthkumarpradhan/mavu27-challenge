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
- Dig deep into the best practices of the tech stack before any code change.
- No AI author on any commit. Commit as the owner:
  `git -c user.name="Sidharth Pradhan" -c user.email="sidharthp@assignall.ai" commit ...`.
  No `Co-Authored-By` or other AI trailers. GitHub account: `sidharthkumarpradhan`.
- No AI footprint anywhere: simple, modular code that follows the stack's conventions, comments
  written the way a careful engineer writes them, PR titles and summaries that read as the
  owner's own. No "Generated with" lines in PRs, commits or code.
- Branches: always start from the latest `main` (`git fetch origin main`, branch from
  `origin/main`). Name it `feature/<name>` for new work or `hotfix/<name>` for a fix. One change
  per branch. Merge into `main` through a PR.
- Before raising a PR: run the unit tests (`make test`) and run the app locally end to end
  (`make smoke` here; for apps with a UI, drive the screen and check every change by hand).
  Nothing goes up untested.
- If stuck, ask the owner. Do not guess around a blocker.
- Never `git add -A`. Secrets never enter the repo.
- Every module has tests. Every bug gets a regression test.
- Fail loudly. Long jobs end with a sentinel line (`DONE` or `EXIT <code>`).

These engineering and branch rules apply to every project, not only this one.

Decisions (from the EURS playbook, `docs/WORKFLOW.md` there):
- The owner decides. The agent prepares. Irreversible or outward-facing steps need the owner's OK,
  unless the owner has switched on the automated path for them. Submissions are on that path.
- Merging is on that path too (owner, 7 Oct 2026: "you have the complete control, as long as
  everything adheres to guidelines, please merge it, don't wait for me"). The agent merges its
  own PRs once CI is green on the latest commit, every review finding is handled, and the change
  follows the compliance rules below. Anything that touches compliance is the owner's call.
- Compliance before score. Evidence over intuition. Every number has a source and a date.
- Measure before building. Find the largest (test weight x gap to leader) and work only that.
- Ceilings before architecture. Run an oracle or probe before a big build.
- Simple first. Reproducible from the remote alone.

## Compliance rules for ReVA (decided before any score)

Owner's rules (6 Oct 2026): always follow the competition guidelines, because we will publish a
paper on this work. The official dataset (Hugging Face `ReVA-Benchmark/ReVA`) and the Codabench
competition 18274 pages are the source of truth. When anything goes wrong, check the dataset and
every Codabench section (Overview, Data, Terms, phase settings, leaderboard, forum) before trusting
our own notes. Leave no stone unturned.

From the Codabench pages (read in full 6 Oct 2026):
- Predictions must come from the video together with the question and the options (Data page).
  A text-only or option-only model is a probe for our own analysis. It never goes to the test set.
- Academic, non-commercial use only. The data may be used for this challenge and related research.
- Never redistribute or publicly release the dataset or its annotations (Terms). The public repo
  holds code, configs and metrics only. Kaggle kernels stay private.
- Cite the ReVA paper (arXiv 2609.35507) in the workshop paper and any report.
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
- Fully automatic by the owner's instruction (6 Oct 2026: "there should not be any human
  intervention"). The autopilot submits when all hold: the zip passes `reva.package.validate`, the
  run's weighted dev accuracy beats the best submitted run by `submit.min_gain`, no submission is
  still being scored, the daily cap and the budget reserve leave room. Kill switches are repo
  variables: `AUTO_SUBMIT=off` stops submissions, `AUTOPILOT=off` stops the loop.
- The predictions.json layout is probed automatically: if Codabench fails one layout, the next
  cycle tries the next (Failed submissions are free). A layout that scored once is kept.
- `train.refit_with_val` stays false. Agent's call under the owner's "handle everything", made for
  compliance (the val overlap above). Revisit only with the owner.
- Every upload, automatic or manual, passes `reva.preflight` first: the zip holds exactly the run's
  own answers, the run's own run.json records video input and no val training (a missing flag
  fails), no letter takes under 5% or over 60% of answers, and the live
  board still ranks by overall accuracy over our 11 columns. A failed run goes to `blocked.jsonl`.
- The test labels are hidden, so no local check proves a board score. STATUS.md shows a projection
  (dev weighted plus the mean board-minus-dev gap of our scored submissions). It is not a gate:
  holding back until we project above the leader would skip the format check and calibration.
- Arena ensembles (`reva.arena`) are candidates like any run: picked on dev only, never on the
  board. Only single video runs without refit or val training take part. Preflight reads every
  member's run.json, so one non-compliant member blocks the ensemble.
- Name each submission with the run id. Calibration pairs board scores with local runs by that id.
- Submit as the owner's personal Codabench account (owner, 7 Oct 2026: "use my personal account,
  make the organization empty"). `competition.organization` is empty. The earlier choice, StagAI,
  failed: the account lacks participant rights there ("You do not have participant permissions for
  this group"). If an organization is set again and the account cannot submit for it, the loop
  submits nothing.

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
