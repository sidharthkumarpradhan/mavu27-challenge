# ReVA autopilot status (2026-10-08T19:04:44Z)

- Codabench login ok; account may submit
- leaderboard: 8 rows, 0 new; leader mkhlystun 0.8795
- job tubu9938/reva-ft-q35-4b-16f-e4ba68c0 is running

## Leaderboard (top 8)

| # | owner | overall |
|---|---|---|
| 1 | mkhlystun | 0.8795 |
| 2 | h | 0.8625 |
| 3 | Vincente | 0.8355 |
| 4 | am | 0.8340 |
| 5 | amirmazaheri | 0.8313 |
| 6 | StagAI | 0.7410 |
| 7 | Hoang Bui | 0.6793 |
| 8 | chrisathy | 0.2515 |

## Gap to the leader (best submission zs-8b-4bit-029fe922)

| task | leader | ours | overall points lost |
|---|---|---|---|
| Temporal Grounding | 0.834 | 0.548 | 4.58 |
| Perspective and Viewpoint | 0.900 | 0.698 | 2.42 |
| Object and Land Cover Recognition | 0.882 | 0.780 | 1.67 |
| Change Detection | 0.816 | 0.686 | 1.62 |
| Geometric Relation | 0.892 | 0.740 | 1.52 |
| Trend and Pattern | 0.883 | 0.789 | 0.85 |
| Structural Layout | 0.882 | 0.806 | 0.65 |
| Consequence Reasoning | 0.980 | 0.890 | 0.22 |
| General Understanding | 0.983 | 0.950 | 0.15 |
| Hypothetical Reasoning | 0.887 | 0.869 | 0.07 |
| Causation Reasoning | 0.944 | 0.928 | 0.07 |

## Calibration (board minus local weighted dev)

| run | dev weighted | board | diff |
|---|---|---|---|
| zs-4b-6254e8de | 0.7130 | 0.7177 | +0.0047 |
| zs-8b-4bit-029fe922 | 0.7442 | 0.7410 | -0.0032 |

## Runs (newest first)

| run | status | dev weighted | dev overall | hours | why |
|---|---|---|---|---|---|
| ft-4b-32f-527078f0 | failed | nan | nan |  |  |
| ft-q35-4b-16f-e4ba68c0 | failed | nan | nan |  |  |
| ft-4b-16f-830869b1 | failed | nan | nan |  |  |
| ft-4b-32f-527078f0 | failed | nan | nan |  |  |
| ft-4b-16f-830869b1 | failed | nan | nan |  |  |
| zs-8b-4bit-029fe922 | ok | 0.7442 | 0.7354 | 10.889 | zero-shot Qwen3-VL-8B in 4-bit. Does the bigger backbone pay for its speed on a T4 |
| ft-4b-16f-830869b1 | failed | nan | nan |  |  |
| zs-8b-4bit-029fe922 | failed | nan | nan |  |  |
| ft-4b-16f-830869b1 | failed | nan | nan |  |  |
| zs-8b-4bit-029fe922 | failed | nan | nan |  |  |
| zs-4b-6254e8de | ok | 0.7130 | 0.7020 | 7.076 | zero-shot Qwen3-VL-4B, 16 frames. Baseline, T4 seconds per question, first submission (confirms the format) |
| text-4b-f571bdb5 | ok | 0.4773 | 0.4807 | 0.099 | text-only probe. How much the options alone give away (paper's text-only result is 29.95%) |

Submissions used: 2 of 100. GPU hours, last 7 days: 19.7 of 30.
Active job: tubu9938/reva-ft-q35-4b-16f-e4ba68c0 ['ft-q35-4b-16f-e4ba68c0', 'ft-8b-4bit-16f-62f77933']
