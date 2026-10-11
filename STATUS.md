# ReVA autopilot status (2026-10-11T01:59:32Z)

- Codabench login ok; account may submit
- leaderboard: 10 rows, 0 new; leader 0xyuan 0.9375
- next candidate ens-ebbfe5d8: dev weighted 0.8361 +/- 0.0171, projected board 0.8369 (3 calibration pairs); leader 0.9375, below by 0.1006
- not submitting ens-ebbfe5d8: projected board 0.8369 does not beat the best rival 0.9375 by 0.01
- queue empty: add experiments to configs/queue.yaml

## Leaderboard (top 8)

| # | owner | overall |
|---|---|---|
| 1 | 0xyuan | 0.9375 |
| 2 | mkhlystun | 0.8802 |
| 3 | h | 0.8638 |
| 4 | T.H | 0.8438 |
| 5 | am | 0.8400 |
| 6 | Vincente | 0.8355 |
| 7 | amirmazaheri | 0.8313 |
| 8 | StagAI | 0.8297 |

Our rank: 8

## Gap to the leader (best submission ens-79d0d049)

| task | leader | ours | overall points lost |
|---|---|---|---|
| Temporal Grounding | 0.912 | 0.742 | 2.72 |
| Object and Land Cover Recognition | 0.955 | 0.833 | 2.00 |
| Perspective and Viewpoint | 0.950 | 0.819 | 1.57 |
| Change Detection | 0.896 | 0.774 | 1.52 |
| Trend and Pattern | 0.950 | 0.825 | 1.12 |
| Structural Layout | 0.941 | 0.856 | 0.73 |
| Geometric Relation | 0.920 | 0.853 | 0.68 |
| Hypothetical Reasoning | 0.950 | 0.894 | 0.22 |
| Causation Reasoning | 0.967 | 0.944 | 0.10 |
| Consequence Reasoning | 0.970 | 0.950 | 0.05 |
| General Understanding | 0.994 | 0.983 | 0.05 |

## Calibration (board minus local weighted dev, dev set unseen-v1)

| run | dev weighted | board | diff |
|---|---|---|---|
| zs-4b-6254e8de | 0.7159 | 0.7177 | +0.0018 |
| ft-8b-4bit-16f-62f77933 | 0.8130 | 0.8125 | -0.0005 |
| ens-79d0d049 | 0.8288 | 0.8297 | +0.0009 |

## Runs (newest first)

| run | status | dev weighted | dev overall | hours | why |
|---|---|---|---|---|---|
| ens-ebbfe5d8 | ok | 0.8361 | 0.8360 | 0 | mean of top 3 on dev: ft-32b-4bit-a100-5f522e72, ft-8b-32f-a100-20321a8a, ft-8b-4bit-16f-62f77933; P(beats ft-32b-4bit-a100-5f522e72) 0.70 |
| ft-32b-4bit-a100-5f522e72 | ok | 0.8304 | 0.8332 | 2.408 | QLoRA of Qwen3-VL-32B on a Colab A100, 16 frames, a full epoch across sessions. Its zero-shot beat the 8B's by 2.4 points |
| ft-8b-4bit-16f-1ep-2d94b189 | failed | nan | nan |  |  |
| ft-4b-32f-1ep-9bd3414d | failed | nan | nan |  |  |
| ft-32b-4bit-a100-5f522e72 | partial | nan | nan |  |  |
| ft-32b-4bit-a100-5f522e72 | partial | nan | nan |  |  |
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

Submissions used: 4 of 100. GPU hours, last 7 days: 37.3 of 30.
Active job: none
