# Research notes: ReVA challenge (MAVU @ WACV 2027)

Every number has a source and a date. Read on 6 Oct 2026 unless noted.

## 1. The competition

Source: Codabench API, `GET https://www.codabench.org/api/competitions/18274/` (public, no login).

- Title: "ReVA: Drone Video Understanding Challenge". Part of the MAVU workshop at WACV 2027.
- One phase: id 30831, "Test Phase", `is_final_phase: true`, start 2026-09-29 12:00 UTC, no end set.
- `max_submissions_per_day: 100`, `max_submissions_per_person: 100`, `execution_time_limit: 600`.
- One task: id 36510, "ReVA VideoQA". The scoring program and public data need a login (HTTP 403).
- Leaderboard 20415, `submission_rule: Force_Best`, primary column `overall_accuracy`, plus 11
  per-task columns (keys in `src/reva/score.py`). Four decimals.
- Prizes: $1,000 / $500 / $200, sponsored by Qualcomm. Top teams present at the workshop.
- Challenge deadline Nov 10, 2026 (workshop site, https://zyaocoder.github.io/MAVU-workshop-2027/).
  Workshop paper deadline Oct 20, 2026.

What the per-person limit means. In the Codabench source (`src/apps/competitions/models.py`,
`Phase.can_user_make_submissions`), the per-person count covers the whole phase and excludes
Failed submissions. So we have 100 scored submissions in total, about 3 a day until Nov 10. A
failed upload (for example a wrong file format) costs nothing.

### Submission format

Source: the competition's Data page (in the API response above).

- `submission.zip` holding `predictions.json`.
- All 4,000 test `qa_id`s, each answered by exactly one of A, B, C, D.
- Missing, duplicate or unknown ids, or other values, fail the evaluation.
- The page says to fill the `correct_answer` field. The exact JSON layout is not shown, and the
  scoring program is private. `reva.package` writes three layouts; `fill_test` (test.json with
  the field filled) is the default. The first real submission confirms it.

### Submission API

Source: codalab/codabench `develop` branch, read 6 Oct 2026. Implemented in `src/reva/codabench.py`.

1. `POST /api/api-token-auth/` with username and password returns a token
   (`src/apps/api/urls.py`, DRF `obtain_auth_token`). Header: `Authorization: Token <token>`.
2. `POST /api/datasets/` with `type: submission`, `competition`, `request_sassy_file_name`,
   `file_name`, `file_size` returns `key` and `sassy_url` (`views/datasets.py`, `DataViewSet.create`).
3. `PUT <sassy_url>` with the zip bytes and `Content-Type: application/zip`
   (`static/js/ours/client.js`, `create_dataset`). Codabench.org stores on MinIO (S3).
4. `PUT /api/datasets/completed/<key>/`.
5. `POST /api/submissions/` with `data: <key>`, `phase: 30831`, `tasks: [36510]`
   (`static/riot/competitions/detail/submission_upload.tag`).
6. `GET /api/submissions/<id>/` returns `status` (Submitting, Submitted, Preparing, Running,
   Scoring, Finished, Failed, Cancelled) and `scores` with `column_key`.

## 2. The data

Source: Hugging Face `ReVA-Benchmark/ReVA` (Apache-2.0, not gated), README and JSON files.

| split | questions | videos | labels |
|---|---:|---:|---|
| train | 15,773 | 1,045 | yes |
| val | 2,000 | 379 | yes |
| test | 4,000 | 1,014 | empty |

- Every question has 4 options. Labels are balanced (train: 3,943 or 3,944 per letter).
- Sources in test: Hawk_UAV 1,856, ERA_Tra 1,285, VisDrone 733, UAVDT 126.
  In val: VisDrone 1,115, Hawk_UAV 880, ERA_Tra 5.
- Test task counts: Object and Land Cover 660, Temporal Grounding 640, Change Detection 500,
  Perspective and Viewpoint 480, Geometric Relation 400, Trend and Pattern 360, Structural
  Layout 340, General Understanding 180, Causation 180, Hypothetical 160, Consequence 100.
- Videos: 1,139 mp4 files, 1.95 GB in all (HF tree API). The annotations use 1,047 of them.
  About 15 s each (workshop site). Small enough to download inside every GPU job.

### Overlap between splits (our measurement, 6 Oct 2026)

- 1,012 of 1,014 test videos also appear in train. 366 appear in val.
- So the hidden test asks new questions about videos the model may have trained on. Local
  validation must hold out questions, not videos. Holding out videos would underestimate.
- 392 test questions match a val question on video, question text and option set, with the
  letters shuffled. 23 also keep the same letter order. This is a leak through official labels.
  Policy in CLAUDE.md: no lookup table; training on val in the final refit is the owner's call.
- 37 train (video, question) pairs repeat inside train.

### Dev set design

`reva.data.make_splits`: dev = val plus 8% of train held out per (source, task) cell. The model
does not train on dev during selection. `reva.score.weighted` re-weights dev accuracy to the
test mix of (source, task), falling back to task level for cells under 20 questions. This is the
analog of the EURS calibrated scorer. Board results then measure the offset (STATUS.md table).

## 3. The paper

Source: Yao et al., "ReVA: A Scene-Centric Dataset Beyond Repetition for Remote Sensing Video
Question Answering", arXiv 2609.35507, ICLR 2027. PDF text extracted 6 Oct 2026.

- Test accuracy (Table 3). Fine-tuned on ReVA: ReMoSense 80.04, VideoLLaMA2 76.33, BIMBA 73.50,
  VideoChat-Flash 72.60. Zero-shot style (trained on LLaVA-Video-178K): ReMoSense 76.45,
  BIMBA 72.85, Nemotron3-Nano-Omni 30B 67.00, Gemma3 27B 66.72, InternVL3 7B 66.28.
  Proprietary LLoVi 64.73.
- ReMoSense: Qwen2.5-VL-7B-Instruct plus a motion-aware alignment module, then LoRA r16, alpha 32.
  AdamW lr 2e-4, cosine, warmup 0.03, batch 1 per device, grad clip 1.0. Baselines fine-tuned for
  24K steps at global batch 1. 32 frames sampled uniformly, resized to 640x360. 2 A100 80 GB.
  (Section 5.1 and Appendix B.)
- Fine-tuning gains are largest on Temporal Grounding and Perspective and Viewpoint, 10 to 20
  points. Causal tasks gain little (Observation 3).
- Input ablation (Appendix C.2, Table 6): text only 29.95, first frame 41.80, middle frame 42.00,
  full video 76.45. The options alone give little away.
- The GitHub repo `zyaocoder/ReVA` returned 403 on 6 Oct 2026 (private or not yet public).

## 4. The leaderboard on 6 Oct 2026

Source: `GET /api/phases/30831/get_leaderboard/`.

| # | owner | overall | weakest tasks |
|---|---|---:|---|
| 1 | mkhlystun | 0.8735 | Change Detection 0.806, Temporal Grounding 0.842 |
| 2 | h | 0.8620 | Temporal Grounding 0.792, Change Detection 0.794 |
| 3 | amirmazaheri | 0.8313 | Change Detection 0.772, Geometric 0.782 |
| 4 | minh_leduc | 0.8260 | Change Detection 0.750, Temporal Grounding 0.758 |
| 5 | Hoang Bui | 0.6720 | Temporal Grounding 0.428 |
| 6 | chrisathy (organizer) | 0.2520 | random baseline |

The top two are already 7 points above the paper's best (80.04). Change Detection and Temporal
Grounding are the weakest tasks for everyone, and they carry 1,140 of 4,000 test questions.

## 5. Models and compute

Sources: Hugging Face model API (6 Oct 2026), our CPU probes, Kaggle limits from the EURS repo.

- Qwen3-VL Instruct: 2B, 4B (4.44B params), 8B (8.77B), all Apache-2.0, released Oct 2025,
  native in transformers. Qwen2.5-VL-7B (8.29B) is the paper's backbone.
- On a T4 (16 GB, no bf16): Qwen3-VL-4B fits in fp16 (about 9 GB of weights). The 8B needs
  4-bit (bitsandbytes nf4).
- Token cost, measured with the Qwen3-VL processor on CPU (transformers 5.19):
  8 frames at 640x360 give 924 tokens, 16 frames 1,837, 32 frames 3,664.
- Qwen3-VL prints each frame's real timestamp ("<0.5 seconds>") when it gets the true frame
  indices and fps. That helps the "at the 00:03 mark" questions, so the frame cache keeps them.
- "A", "B", "C", "D" are single tokens in the Qwen3 tokenizer, so one forward pass scores all four.
- Kaggle: 2x T4 per session, 12 h per session, about 30 GPU hours a week (EURS `docs/research.md`).

## 6. How the EURS playbook maps here

| EURS practice | ReVA version |
|---|---|
| Leaderboard-calibrated scorer (0.001) | dev set mirroring test (question holdout, test-mix weights), calibrated against our board scores |
| Hold out whole scenes | hold out questions; test shares videos with train |
| Official scripts are the gate | `reva.package.validate` applies every Data-page rule before any upload |
| Two tracks, two models | one track; lanes run two experiments in parallel, one per T4 |
| Ceilings before architecture | text-only probe, frame-count probes, backbone size probe |
| Use allowed metadata | real frame timestamps; `dataset_name` and `task` are given per question |
| Kaggle autopilot, state in a private dataset | same Kaggle jobs; state on a git branch (public repo, metrics only) |
| One upload a day, owner confirms | 100 in total; gate plus `AUTO_SUBMIT` switch the owner controls |
