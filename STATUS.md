# ReVA autopilot status (2026-10-08T21:56:37Z)

- Codabench login ok; account may submit
- leaderboard: 8 rows, 0 new; leader mkhlystun 0.8795
- job tubu9938/reva-ft-4b-16f-830869b1 is running
- arena skipped this cycle: FileNotFoundError

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

## Calibration (board minus local weighted dev)

| run | dev weighted | board | diff |
|---|---|---|---|
| zs-4b-6254e8de | 0.7130 | 0.7177 | +0.0047 |
| zs-8b-4bit-029fe922 | 0.7442 | 0.7410 | -0.0032 |
| ft-8b-4bit-16f-62f77933 | 0.8175 | 0.8125 | -0.0050 |

## Runs (newest first)

| run | status | dev weighted | dev overall | hours | why |
|---|---|---|---|---|---|
| ft-q35-4b-16f-e4ba68c0 | failed | nan | nan |  |  |
| ft-8b-4bit-16f-62f77933 | ok | 0.8175 | 0.8115 | 10.058 | QLoRA fine-tune of Qwen3-VL-8B, 16 frames |
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

Submissions used: 3 of 100. GPU hours, last 7 days: 29.9 of 30.
Active job: tubu9938/reva-ft-4b-16f-830869b1 ['ft-4b-16f-830869b1', 'ft-4b-32f-527078f0']
