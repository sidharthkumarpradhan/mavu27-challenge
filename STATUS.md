# ReVA autopilot status (2026-10-09T18:25:24Z)

- Codabench login ok; account may submit
- leaderboard fetch failed: 500 Server Error: Internal Server Error for url: https://www.codabench.org/api/phases/30831/get_leaderboard/
- Colab: ft-32b-4bit-a100-5f522e72 on A100, 2.1 h in at 2026-10-09T18:04:16Z, 51.23 units at start; last log line: -
- next candidate ft-8b-32f-a100-20321a8a: dev weighted 0.8210 +/- 0.0177, projected board 0.8217 (3 calibration pairs)
- not submitting ft-8b-32f-a100-20321a8a: dev 0.8210 does not beat best submitted 0.8288 by 0.003
- copied ft-8b-4bit-16f-1ep-2d94b189's saved state from tubu9938/reva-run-ft-8b-4bit-16f-1ep-2d94b189 to sidharthkumarpradhan/reva-run-ft-8b-4bit-16f-1ep-2d94b189
- Kaggle weekly GPU quota reached on account 1
- GPU quota pacing on Kaggle account 2: 28.4 h used in 7 days
- no Kaggle account can take the next job; it waits

## Leaderboard (top 8)

| # | owner | overall |
|---|---|---|

## Runs (newest first)

| run | status | dev weighted | dev overall | hours | why |
|---|---|---|---|---|---|
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
| ft-q35-4b-16f-e4ba68c0 | failed | nan | nan |  |  |
| ft-4b-16f-830869b1 | failed | nan | nan |  |  |
| ft-4b-32f-527078f0 | failed | nan | nan |  |  |
| ft-4b-16f-830869b1 | failed | nan | nan |  |  |
| zs-8b-4bit-029fe922 | ok | nan | nan | 10.889 | zero-shot Qwen3-VL-8B in 4-bit. Does the bigger backbone pay for its speed on a T4 |

Submissions used: 5 of 100. GPU hours, last 7 days: 35.7 of 30.
Active job: none
