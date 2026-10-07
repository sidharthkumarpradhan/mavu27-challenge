# Handoff (6 Oct 2026)

Read this first, then `CLAUDE.md`, then the relevant part of `docs/research.md`.

## State

- Code lives on `main`. Changes go through `feature/<name>` or `hotfix/<name>` branches and PRs.
  The autopilot always runs `main`. `claude/serene-euler-g5spyi` is the old bootstrap branch.
- 31 unit tests pass. `make smoke` prints READY on CPU (tiny Qwen3-VL, synthetic videos): frames,
  LoRA steps, dev scoring, test prediction with 2-shift TTA, validated zip.
- A zip for the real 4,000 test ids passes `reva.package.validate`.
- No GPU job has run yet. No submission has been made. Leaderboard leader: 0.8735.

## What has not been verified on real hardware

These are the first things the first job will tell us. Check its log before trusting the loop.
1. The generated Kaggle kernel itself (git fetch of the commit, pip pins, two lanes on a T4 x2).
2. Qwen3-VL-4B in fp16 on a T4: no overflow (the job fails loudly on non-finite logits), memory,
   seconds per question, and the self-sizing training on real speeds (the job log prints BUDGET).
3. The 4-bit path for the 8B model (bitsandbytes on Kaggle).
4. The Codabench upload path and the `predictions.json` layout (`submit.format`). The first
   submission confirms both. A failed submission does not count against the budget.

## Switched on (6 Oct 2026)

- The owner added the four secrets and asked for no human intervention. The loop now runs by
  default every hour; `AUTOPILOT=off` and `AUTO_SUBMIT=off` are the kill switches.
- Training sizes itself to the session (`reva.job.train_deadline`), and the submission layout is
  probed automatically, so no step waits on a person.
- Still unknown from here: whether the Codabench account has joined the competition. If not,
  STATUS.md shows "Codabench refuses submissions for this account" and the runs wait in runs.jsonl
  until it has; the loop then submits them on the next cycle.
- GitHub made `claude/serene-euler-g5spyi` the default branch (first push to an empty repo).
  The owner should switch the default to `main` in Settings > General. Until then the hourly
  schedule fires from that branch, but the job checks out and runs `main` anyway.

## Added 7 Oct 2026

- The loop runs twice an hour (:23 and :53).
- Pre-upload checks (`reva.preflight`) run on every rebuilt zip before it goes to Codabench.
  STATUS.md shows the next candidate's dev score with a 95% interval and a projected board score.
- Account approved: STATUS shows "submits as StagAI (id 2763)".
- Model arena (`reva.arena`): once two or more fair runs exist, the loop averages the top k on
  dev and adds the winner as an `ens-` run. It goes through the same gate and preflight.
  `arena.json` on the state branch records which runs were last compared.

## Next steps, in order

1. Job 1 (`zs-4b` + `text-4b`): read seconds per question and dev accuracy per task. The loop
   submits `zs-4b` itself, which confirms the format and gives the first calibration pair.
2. Jobs 2 and 3 are queued (fine-tunes, 8B, 32 frames). Extend `configs/queue.yaml` from the gap.
3. Work the largest test-weighted gap in STATUS.md. Today that is likely Temporal Grounding and
   Change Detection (weakest for every team, 1,140 test questions).
4. Paper: workshop deadline Oct 20. Start the outline once the first fine-tune is scored.

## Ideas queue (not built; measure first)

- 32 frames (the paper's setting) once the 16-frame speed is known.
- Option-shift TTA with `infer.perms: 4`.
- Refit on train plus holdout for the final submission (`train.refit: true`).
- More backbones in the arena (Qwen2.5-VL-7B, InternVL3.5, LLaVA-OneVision) so the mix has
  diverse members, not only Qwen3-VL variants.
- The owner's V100 box can run `reva.job` directly for longer fine-tunes.
