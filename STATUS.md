# ReVA autopilot status (2026-10-09T12:32:37Z)

- Codabench login ok; account may submit
- leaderboard: 9 rows, 0 new; leader mkhlystun 0.8795
- submission 971508 (ens-79d0d049) still Running after 3 h; it no longer holds back the next submission
- Colab: ft-32b-4bit-a100-5f522e72 on A100, 0.3 h in at 2026-10-09T12:12:47Z, 72.67 units at start; last log line: -
- next candidate ens-79d0d049: dev weighted 0.8288 +/- 0.0174, projected board 0.8295 (2 calibration pairs); leader 0.8795, below by 0.0500
- submitted ens-79d0d049 as 972027 (fill_test): dev 0.8288 vs best submitted 0.8130; status Finished
- copied ft-8b-4bit-16f-1ep-2d94b189's saved state from tubu9938/reva-run-ft-8b-4bit-16f-1ep-2d94b189 to sidharthkumarpradhan/reva-run-ft-8b-4bit-16f-1ep-2d94b189
- Kaggle weekly GPU quota reached on account 1
- GPU quota pacing on Kaggle account 2: 28.4 h used in 7 days
- no Kaggle account can take the next job; it waits

## Leaderboard (top 8)

| # | owner | overall |
|---|---|---|
| 1 | mkhlystun | 0.8795 |
| 2 | h | 0.8625 |
| 3 | T.H | 0.8438 |
| 4 | am | 0.8400 |
| 5 | Vincente | 0.8355 |
| 6 | amirmazaheri | 0.8313 |
| 7 | StagAI | 0.8125 |
| 8 | Hoang Bui | 0.6793 |

## Gap to the leader (best submission ens-79d0d049)

| task | leader | ours | overall points lost |
|---|---|---|---|
| Temporal Grounding | 0.834 | 0.742 | 1.48 |
| Perspective and Viewpoint | 0.900 | 0.819 | 0.98 |
| Object and Land Cover Recognition | 0.882 | 0.833 | 0.80 |
| Change Detection | 0.816 | 0.774 | 0.52 |
| Trend and Pattern | 0.883 | 0.825 | 0.52 |
| Geometric Relation | 0.892 | 0.853 | 0.40 |
| Structural Layout | 0.882 | 0.856 | 0.23 |
| Consequence Reasoning | 0.980 | 0.950 | 0.08 |
| General Understanding | 0.983 | 0.983 | -0.00 |
| Causation Reasoning | 0.944 | 0.944 | -0.00 |
| Hypothetical Reasoning | 0.887 | 0.894 | -0.03 |

## Calibration (board minus local weighted dev, dev set unseen-v1)

| run | dev weighted | board | diff |
|---|---|---|---|
| zs-4b-6254e8de | 0.7159 | 0.7177 | +0.0018 |
| ft-8b-4bit-16f-62f77933 | 0.8130 | 0.8125 | -0.0005 |
| ens-79d0d049 | 0.8288 | 0.8297 | +0.0009 |

## Runs (newest first)

| run | status | dev weighted | dev overall | hours | why |
|---|---|---|---|---|---|
| ens-79d0d049 | ok | 0.8288 | 0.8249 | 0 | mean of top 2 on dev: ft-8b-32f-a100-20321a8a, ft-8b-4bit-16f-62f77933; P(beats ft-8b-32f-a100-20321a8a) 0.63 |
| zs-32b-4bit-a100-127cb717 | ok | 0.7676 | 0.7634 | 0.953 | zero-shot Qwen3-VL-32B in 4-bit on a Colab A100, set up like zs-8b-4bit. A ceiling probe before a 32B fine-tune |
| zs-32b-4bit-a100-127cb717 | failed | nan | nan |  |  |
| ft-8b-32f-a100-20321a8a | ok | 0.8210 | 0.8233 | 1.416 | Qwen3-VL-8B in bf16 on a Colab A100, 32 frames, a full epoch. The strongest model the compute allows, aimed at Temporal Grounding |
| ft-8b-32f-a100-20321a8a | failed | nan | nan |  |  |
| ft-8b-32f-a100-20321a8a | failed | nan | nan |  |  |
| ft-8b-4bit-16f-1ep-2d94b189 | partial | nan | nan |  |  |
| ft-4b-32f-1ep-9bd3414d | failed | nan | nan |  |  |
| ft-8b-32f-a100-20321a8a | partial | nan | nan |  |  |
| ft-4b-16f-830869b1 | failed | nan | nan |  |  |
| ft-4b-32f-527078f0 | failed | nan | nan |  |  |
| ft-q35-4b-16f-e4ba68c0 | failed | nan | nan |  |  |
| ft-8b-4bit-16f-62f77933 | ok | 0.8130 | 0.8089 | 10.058 | QLoRA fine-tune of Qwen3-VL-8B, 16 frames |
| ft-4b-32f-527078f0 | failed | nan | nan |  |  |
| ft-q35-4b-16f-e4ba68c0 | failed | nan | nan |  |  |
| ft-4b-16f-830869b1 | failed | nan | nan |  |  |
| ft-4b-32f-527078f0 | failed | nan | nan |  |  |
| ft-4b-16f-830869b1 | failed | nan | nan |  |  |
| zs-8b-4bit-029fe922 | ok | nan | nan | 10.889 | zero-shot Qwen3-VL-8B in 4-bit. Does the bigger backbone pay for its speed on a T4 |
| ft-4b-16f-830869b1 | failed | nan | nan |  |  |

Submissions used: 5 of 100. GPU hours, last 7 days: 35.7 of 30.
Active job: none
