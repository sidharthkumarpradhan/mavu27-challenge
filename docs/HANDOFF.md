# Handoff (6 Oct 2026)

Read this first, then `CLAUDE.md`, then the relevant part of `docs/research.md`.

## State

- Pipeline built on branch `claude/serene-euler-g5spyi`. Not yet merged to `master`.
- 28 unit tests pass. `make smoke` prints READY on CPU (tiny Qwen3-VL, synthetic videos): frames,
  LoRA steps, dev scoring, test prediction with 2-shift TTA, validated zip.
- A zip for the real 4,000 test ids passes `reva.package.validate`.
- No GPU job has run yet. No submission has been made. Leaderboard leader: 0.8735.

## What has not been verified on real hardware

These are the first things the first job will tell us. Check its log before trusting the loop.
1. The generated Kaggle kernel itself (git fetch of the commit, pip pins, two lanes on a T4 x2).
2. Qwen3-VL-4B in fp16 on a T4: no overflow (the job fails loudly on non-finite logits), memory,
   seconds per question. The training-time estimate (about 2 s per 8-frame sample) is a guess.
3. The 4-bit path for the 8B model (bitsandbytes on Kaggle).
4. The Codabench upload path and the `predictions.json` layout (`submit.format`). The first
   submission confirms both. A failed submission does not count against the budget.

## Owner actions to switch it on

1. Merge the branch into `master` (scheduled workflows run on the default branch only).
2. Join the competition on Codabench and accept its terms (submissions need an approved
   participant). Note the Codabench username in `competition.owner` in the config.
3. Repo secrets: `KAGGLE_USERNAME`, `KAGGLE_KEY`, `CODABENCH_USERNAME`, `CODABENCH_PASSWORD`.
4. Repo variable `AUTOPILOT=on`. Leave `AUTO_SUBMIT` unset until the first job's numbers look sane.
5. Decide `train.refit_with_val` (see CLAUDE.md compliance rules).

## Next steps, in order

1. Run job 1 (`zs-4b` + `text-4b`). Read seconds per question and dev accuracy per task.
2. Submit `zs-4b` by hand (`reva submit`) to confirm the format and get the first calibration pair.
3. Size `ft-4b-8f` from the measured speed (`train.max_samples`), then let the loop run.
4. Work the largest test-weighted gap in STATUS.md. Today that is likely Temporal Grounding and
   Change Detection (weakest for every team, 1,140 test questions).
5. Paper: workshop deadline Oct 20. Start the outline once the first fine-tune is scored.

## Ideas queue (not built; measure first)

- 32 frames (the paper's setting) once the 16-frame speed is known.
- Option-shift TTA with `infer.perms: 4`.
- Refit on train plus holdout for the final submission (`train.refit: true`).
- Ensembling lanes by averaging `test_probs.json` (needs a small `reva` command).
- The owner's V100 box can run `reva.job` directly for longer fine-tunes.
