# Handoff (10 Oct 2026)

Read this first, then `CLAUDE.md`, then the relevant part of `docs/research.md`. The sections
from "Added 9 Oct" down are the current state. The dated sections above them are history.

## State

- Code lives on `main`. Changes go through `feature/<name>` or `hotfix/<name>` branches and PRs.
  The autopilot always runs `main`. Run state lives on the `state` branch.
- 155 unit tests pass. `make smoke` prints READY on CPU.
- Board (10 Oct): 0xyuan 0.9375, mkhlystun 0.8802, h 0.8638, T.H 0.8438, am 0.84,
  Vincente 0.8355, amirmazaheri 0.8313, StagAI (us) 0.8297, 8th.
- 4 of 100 submissions used. Kaggle GPU: about 36 of 30 weekly hours used on the first account.
- No GPU job runs without a written plan the owner has approved (see "Added 10 Oct").

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

## Added 9 Oct 2026

- ft-8b-32f-a100 (Colab A100, bf16, 32 frames, full epoch) reached 0.821 weighted dev, our best
  single run. zs-32b-4bit-a100 (zero-shot 32B probe) reached 0.768 against 0.744 for the 8B
  zero-shot under the same setup (#54). So a 32B fine-tune is queued on Colab (#58).
- The arena's ens-79d0d049 (ft-8b-32f-a100 with ft-8b-4bit-16f) scored 0.82975 on the board,
  0.8288 on dev. It sat Running on Codabench for hours; a submission Running 3 h after upload
  is now Stalled and stops blocking the gate (#57).
- Colab robustness: checkpoints upload in 32 MiB parts (#51), a Colab CLI failure counts as an
  environment failure and proxy tokens are redacted from run rows (#52, #53), Kaggle dataset
  downloads retry a fresh 404 for 15 minutes (#49) and fall back to file by file (#56).
- The autopilot pulls the state branch before each cycle (#55). Kaggle and Colab jobs fetch code
  from the repo running the workflow (#60). `docs/PLAYBOOK.md` is the reusable workflow (#48).
- The state branch history holds Colab proxy tokens from before #52. If it ever moves to a public
  repo, push it as one fresh commit, never with its history.

## Added 10 Oct 2026

- Owner's policy: no training until the local evidence says it can win. Fix what a local check can
  find first. A GPU run goes out with its hypothesis, the evidence and the expected dev written down.
- Upload gate (#61): a run goes up only when its projection (dev plus the mean board-minus-dev gap
  of our scored runs, within 0.002 so far) beats the best rival on the live board by 0.01. Today
  that needs about 0.947 weighted dev. Our own row (`competition.owner: StagAI`) is not a rival.
- Local error analysis on dev only (unseen-v1, 1,805 questions), scored to the test mix:
  - Weighted dev: ft-8b-32f 0.8210, zs-32b 0.7676, zs-4b 0.7159. A probability blend of
    ft-8b-32f with zs-32b at weight 0.2 gives 0.8311 (0.1: 0.8302, 0.3: 0.826, 0.5: 0.8088).
    An oracle that takes any run's right answer reaches 0.9069, so these runs alone cannot win.
  - ft-8b-32f per task: Temporal Grounding 0.742, Geometric Relation 0.766, Change Detection
    0.798, Perspective and Viewpoint 0.812, Object and Land Cover 0.813, Trend and Pattern 0.822,
    Structural Layout 0.860. zs-32b is better on Object and Land Cover (0.841) and Hypothetical.
  - Test-weighted points lost by (source, task), blend: Hawk_UAV Temporal Grounding 3.03 (dev
    0.709), Hawk Perspective 1.24, ERA_Tra Object/Land Cover 1.07, ERA Change Detection 1.00,
    Hawk Geometric 0.98, VisDrone Object/Land Cover 0.93. 16.94 points lost in total.
  - Hawk Temporal Grounding options sit 0.5 s apart, so frame timing is the limit there.
  - Confidence: at max probability 0.9 or more, 96.7% right (1,013 questions). Below 0.6,
    about half right. No pipeline bug was found.
- The analysis is `reva.analyze` (`python -m reva.cli analyze --probs NAME=PATH ...`). It reproduces
  every number above from the runs' dev probabilities and runs from the remote alone.
- Running: Colab ft-32b-4bit-a100-5f522e72 (32B QLoRA, resumed 16:31Z). It goes through the gate
  like any run; score it per task and source and in the blend before any plan.
- Stopped by the owner: Kaggle ft-8b-4bit-16f-1ep (8B 4-bit, 16 frames, one epoch over sessions).
  It, ft-4b-32f-1ep (out of GPU memory on a T4 at 32 frames) and ft-q35-4b-16f are commented out
  of the queue (#62) and return only with an approved plan.

## Rebuild from scratch

1. Fork or clone the repo. `make install`, `make test`, `make smoke` (prints READY).
2. Secrets: `KAGGLE_USERNAME`, `KAGGLE_KEY` (and `_NEW` for the second account),
   `CODABENCH_USERNAME`, `CODABENCH_PASSWORD`, `COLAB_TOKEN` (the whole
   `~/.config/colab-cli/token.json`, see 8 Oct). Variables: `AUTOPILOT`, `AUTO_SUBMIT`, `COLAB`.
3. The first autopilot cycle creates the `state` branch. Data comes from Hugging Face
   `ReVA-Benchmark/ReVA` inside each job. Nothing else lives outside the repo except the private
   Kaggle datasets `reva-run-<run id>` (checkpoints, logs, dev and test probabilities).
4. To redo the local analysis: download each run's `dev_probs.json` from its private dataset
   (`kaggle datasets download sidharthkumarpradhan/reva-run-<run id>`), then run
   `python -m reva.cli analyze --probs <name>=<path> ...`. It fetches the annotations itself.

## Next steps, in order

1. When ft-32b-4bit-a100 finishes: run `reva.cli analyze` on it with ft-8b-32f and zs-32b.
2. Write one training plan (hypothesis, evidence, expected dev) aimed at the largest losses above,
   Hawk Temporal Grounding first. Launch only with the owner's approval.
3. Paper: workshop deadline Oct 20. Cite ReVA (arXiv 2609.35507) and disclose every model.

## Ideas queue (not built; measure first)

- Dense frames around the asked time window for Temporal Grounding (options 0.5 s apart).
- Per-task routing between models, picked on dev only.
- Temperature calibration of each member before blending.
- More backbones in the arena (Qwen2.5-VL-7B, InternVL3.5, LLaVA-OneVision) for diversity.
- Refit on train plus holdout for the final submission (`train.refit: true`; never val).
