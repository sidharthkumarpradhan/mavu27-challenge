# ReVA autopilot status (2026-10-10T18:07:18Z)

- Codabench login ok; account may submit
- leaderboard fetch failed: HTTPSConnectionPool(host='www.codabench.org', port=443): Read timed out. (read timeout=60)
- saved ft-8b-4bit-16f-1ep-2d94b189 (failed) to private dataset sidharthkumarpradhan/reva-run-ft-8b-4bit-16f-1ep-2d94b189
- saved ft-4b-32f-1ep-9bd3414d (failed) to private dataset sidharthkumarpradhan/reva-run-ft-4b-32f-1ep-9bd3414d
- job sidharthkumarpradhan/reva-ft-8b-4bit-16f-1ep-2d94b189-10101631 cancel_acknowledged; failed lanes: ['ft-8b-4bit-16f-1ep-2d94b189', 'ft-4b-32f-1ep-9bd3414d'] 
- collected ['ft-8b-4bit-16f-1ep-2d94b189', 'ft-4b-32f-1ep-9bd3414d']
- Colab: ft-32b-4bit-a100-5f522e72 on A100, 1.4 h in at 2026-10-10T17:59:18Z, 32.81 units at start; last log line: -
- next candidate ft-8b-32f-a100-20321a8a: dev weighted 0.8210 +/- 0.0177, projected board 0.8217 (3 calibration pairs)
- not submitting ft-8b-32f-a100-20321a8a: dev 0.8210 does not beat best submitted 0.8288 by 0.003
- queue empty: add experiments to configs/queue.yaml

## Leaderboard (top 8)

| # | owner | overall |
|---|---|---|

Our rank: not on the board

## Runs (newest first)

| run | status | dev weighted | dev overall | hours | why |
|---|---|---|---|---|---|
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
| ft-q35-4b-16f-e4ba68c0 | failed | nan | nan |  |  |
| ft-4b-16f-830869b1 | failed | nan | nan |  |  |

Submissions used: 4 of 100. GPU hours, last 7 days: 37.3 of 30.
Active job: none
