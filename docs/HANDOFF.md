# Handoff (8 Oct 2026)

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
- GitHub had not fired that schedule once by 01:00Z on 7 Oct. Every cycle came from a push to
  main. So while a Kaggle job runs or a submission is being scored, each cycle now dispatches the
  next one about 20 minutes later. When nothing is open the chain stops and a push restarts it.
- Pre-upload checks (`reva.preflight`) run on every rebuilt zip before it goes to Codabench.
  STATUS.md shows the next candidate's dev score with a 95% interval and a projected board score.
- Account approved: STATUS shows "submits as StagAI (id 2763)".
- Model arena (`reva.arena`): once two or more fair runs exist, the loop averages the top k on
  dev and adds the winner as an `ens-` run. It goes through the same gate and preflight.
  `arena.json` on the state branch records which runs were last compared.

## Added 8 Oct 2026

- Board: leader mkhlystun 0.8795, ours 0.7177 (zs-4b, 5th). Biggest gaps: Temporal Grounding and
  Perspective and Viewpoint. Most Temporal Grounding questions ask for a time in seconds at 0.5 s
  steps on videos of mostly 2 to 8 s, so frame density matters.
- Job 1 spent 7 of 10.5 GPU hours on inference (3.0 s per dev question, 3.9 s per test question).
  Two speedups are merged and not yet measured on a T4:
  - Inference encodes each video once and reuses its KV cache for every question and option
    shift (#21). Option-shift TTA (`infer.perms`) is now nearly free.
  - Training packs up to `train.pack` (4) questions about one video into one sequence with a
    block mask (#23). Qwen3.5 has linear-attention layers and trains one question at a time.
- Qwen3.5-4B joined the arena (#22). Job 3: ft-4b-32f and ft-q35-4b-16f. Job 4: ft-8b-4bit-16f.
- A second Kaggle account takes jobs when the first is out of weekly quota (see CLAUDE.md).
- Run ids hash the whole config. Never add a default to competition.yaml while a job runs; read new
  keys with a fallback in code. A test pins job 2's ids.
- The owner has an OpenReview account (8 Oct 2026). Workshop paper due Oct 20.

## Added late 8 Oct 2026

- ft-8b-4bit-16f (a third of an epoch) took the board from 0.7410 to 0.8125 (6th). Training time is
  the lever, so fine-tunes now train a full epoch across sessions (`train.span_sessions`).
- Checkpoints: training saves a full checkpoint every 20 minutes, inference every 100 questions.
  After each job every lane's folder goes to the private Kaggle dataset `reva-run-<run id>` (one
  version per session, logs and config included) and an unfinished run resumes from it (#35, #36).
- Every push gets its own kernel slug (#34); a reused slug had overwritten zs-8b-4bit's outputs.
- Dev leaves out the 1,456 val questions that copy train questions (#38): 1,805 questions.
  ft-8b-4bit-16f scores 0.813 on it against 0.8125 on the board.
- The fp16 4B ran out of memory on a T4 with 4-question packs; training now halves the pack (#40).
- Colab Pro backend (`reva.colab`, `colab.yml`): one A100 session at a time for `backend: colab`
  queue entries. The autopilot starts a session after any cycle in which a Colab entry is pending
  and no Colab run is active (#42, #43). After a failed Colab run it waits for main to move, so a
  merged fix gets one try; a changed secret alone does not move main (start colab.yml by hand).
- `COLAB_TOKEN` must be the whole `~/.config/colab-cli/token.json`. Set it with
  `gh secret set COLAB_TOKEN < ~/.config/colab-cli/token.json`: the file has no trailing newline,
  so a terminal copy picks up the shell prompt and the JSON breaks (first two runs, 8 Oct).
  `reva.cli colab --check-token` names the problem without printing the token (#44).
- Colab checkpoints leave the VM every hour, not only at the session's end: a snapshot of the
  run folder becomes a new version of its private dataset `reva-run-<id>`. A session lost midway
  (VM dropped, units ran out) is recorded as partial and the next one resumes from that snapshot.
  Three failed polls in a row end a session. Drive is not used: mounting it needs a browser click
  in every session, and the private dataset already serves resume.
- When the units left after a session buy no other one, that session is the run's last
  (`train.final_session`): training shrinks so dev and test prediction fit, and the run reaches
  the board instead of stopping mid-epoch with no predictions.

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
