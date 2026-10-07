# ReVA autopilot status (2026-10-07T11:06:27Z)

- Codabench login ok; account may submit; submits as StagAI (id 2763)
- leaderboard fetch failed: 500 Server Error: Internal Server Error for url: https://www.codabench.org/api/phases/30831/get_leaderboard/
- next candidate zs-4b-6254e8de: dev weighted 0.7130 +/- 0.0155, projected board 0.7130 (0 calibration pairs)
- submission of zs-4b-6254e8de not made: could not read the live leaderboard columns; trying again next cycle
- Kaggle weekly GPU quota reached; the next job waits for it to reset

## Leaderboard (top 8)

| # | owner | overall |
|---|---|---|

## Runs (newest first)

| run | status | dev weighted | dev overall | hours | why |
|---|---|---|---|---|---|
| zs-4b-6254e8de | ok | 0.7130 | 0.7020 | 7.076 | zero-shot Qwen3-VL-4B, 16 frames. Baseline, T4 seconds per question, first submission (confirms the format) |
| text-4b-f571bdb5 | ok | 0.4773 | 0.4807 | 0.099 | text-only probe. How much the options alone give away (paper's text-only result is 29.95%) |

Submissions used: 0 of 100. GPU hours, last 7 days: 7.3 of 30.
Active job: none
