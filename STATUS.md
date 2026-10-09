# ReVA autopilot status (2026-10-09T04:27:21Z)

- Codabench login ok; account may submit
- leaderboard: 8 rows, 0 new; leader mkhlystun 0.8795
- copied ft-8b-4bit-16f-1ep-2d94b189's saved state from tubu9938/reva-run-ft-8b-4bit-16f-1ep-2d94b189 to sidharthkumarpradhan/reva-run-ft-8b-4bit-16f-1ep-2d94b189
- Kaggle weekly GPU quota reached on account 1
- GPU quota pacing on Kaggle account 2: 28.4 h used in 7 days
- no Kaggle account can take the next job; it waits

## Leaderboard (top 8)

| # | owner | overall |
|---|---|---|
| 1 | mkhlystun | 0.8795 |
| 2 | h | 0.8625 |
| 3 | Vincente | 0.8355 |
| 4 | am | 0.8340 |
| 5 | amirmazaheri | 0.8313 |
| 6 | StagAI | 0.8125 |
| 7 | Hoang Bui | 0.6793 |
| 8 | chrisathy | 0.2515 |

## Gap to the leader (best submission ft-8b-4bit-16f-62f77933)

| task | leader | ours | overall points lost |
|---|---|---|---|
| Temporal Grounding | 0.834 | 0.731 | 1.65 |
| Perspective and Viewpoint | 0.900 | 0.796 | 1.25 |
| Object and Land Cover Recognition | 0.882 | 0.818 | 1.05 |
| Change Detection | 0.816 | 0.748 | 0.85 |
| Geometric Relation | 0.892 | 0.815 | 0.78 |
| Trend and Pattern | 0.883 | 0.811 | 0.65 |
| Structural Layout | 0.882 | 0.844 | 0.33 |
| General Understanding | 0.983 | 0.961 | 0.10 |
| Hypothetical Reasoning | 0.887 | 0.881 | 0.02 |
| Causation Reasoning | 0.944 | 0.939 | 0.02 |
| Consequence Reasoning | 0.980 | 0.980 | 0.00 |

## Calibration (board minus local weighted dev, dev set unseen-v1)

| run | dev weighted | board | diff |
|---|---|---|---|
| zs-4b-6254e8de | 0.7159 | 0.7177 | +0.0018 |
| ft-8b-4bit-16f-62f77933 | 0.8130 | 0.8125 | -0.0005 |

## Runs (newest first)

| run | status | dev weighted | dev overall | hours | why |
|---|---|---|---|---|---|
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
| zs-8b-4bit-029fe922 | failed | nan | nan |  |  |
| ft-4b-16f-830869b1 | failed | nan | nan |  |  |
| zs-8b-4bit-029fe922 | failed | nan | nan |  |  |
| zs-4b-6254e8de | ok | 0.7159 | 0.7191 | 7.076 | zero-shot Qwen3-VL-4B, 16 frames. Baseline, T4 seconds per question, first submission (confirms the format) |
| text-4b-f571bdb5 | ok | 0.4835 | 0.4953 | 0.099 | text-only probe. How much the options alone give away (paper's text-only result is 29.95%) |

Submissions used: 3 of 100. GPU hours, last 7 days: 35.7 of 30.
Active job: none
