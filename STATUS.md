# ReVA autopilot status (2026-10-07T23:16:19Z)

- Codabench login ok; account may submit
- leaderboard: 7 rows, 0 new; leader mkhlystun 0.8795
- Kaggle weekly GPU quota reached on account 1
- Kaggle account 2 kernels have no internet (is its phone number verified?); next try after 2026-10-07T23:32:23Z
- no Kaggle account can take the next job; it waits

## Leaderboard (top 8)

| # | owner | overall |
|---|---|---|
| 1 | mkhlystun | 0.8795 |
| 2 | h | 0.8625 |
| 3 | amirmazaheri | 0.8313 |
| 4 | Vincente | 0.8257 |
| 5 | StagAI | 0.7177 |
| 6 | Hoang Bui | 0.6793 |
| 7 | chrisathy | 0.2515 |

## Gap to the leader (best submission zs-4b-6254e8de)

| task | leader | ours | overall points lost |
|---|---|---|---|
| Temporal Grounding | 0.834 | 0.503 | 5.30 |
| Perspective and Viewpoint | 0.900 | 0.571 | 3.95 |
| Object and Land Cover Recognition | 0.882 | 0.783 | 1.62 |
| Change Detection | 0.816 | 0.688 | 1.60 |
| Geometric Relation | 0.892 | 0.762 | 1.30 |
| Trend and Pattern | 0.883 | 0.767 | 1.05 |
| Structural Layout | 0.882 | 0.771 | 0.95 |
| General Understanding | 0.983 | 0.950 | 0.15 |
| Hypothetical Reasoning | 0.887 | 0.863 | 0.10 |
| Causation Reasoning | 0.944 | 0.922 | 0.10 |
| Consequence Reasoning | 0.980 | 0.960 | 0.05 |

## Calibration (board minus local weighted dev)

| run | dev weighted | board | diff |
|---|---|---|---|
| zs-4b-6254e8de | 0.7130 | 0.7177 | +0.0047 |

## Runs (newest first)

| run | status | dev weighted | dev overall | hours | why |
|---|---|---|---|---|---|
| ft-4b-16f-830869b1 | failed | nan | nan |  |  |
| zs-8b-4bit-029fe922 | failed | nan | nan |  |  |
| ft-4b-16f-830869b1 | failed | nan | nan |  |  |
| zs-8b-4bit-029fe922 | failed | nan | nan |  |  |
| zs-4b-6254e8de | ok | 0.7130 | 0.7020 | 7.076 | zero-shot Qwen3-VL-4B, 16 frames. Baseline, T4 seconds per question, first submission (confirms the format) |
| text-4b-f571bdb5 | ok | 0.4773 | 0.4807 | 0.099 | text-only probe. How much the options alone give away (paper's text-only result is 29.95%) |

Submissions used: 1 of 100. GPU hours, last 7 days: 7.8 of 30.
Active job: none
