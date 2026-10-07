# ReVA autopilot status (2026-10-07T09:23:15Z)

- Codabench login ok; account may submit; submits as StagAI (id 2763)
- leaderboard: 6 rows, 0 new; leader mkhlystun 0.8735
- next candidate zs-4b-6254e8de: dev weighted 0.7130 +/- 0.0155, projected board 0.7130 (0 calibration pairs); leader 0.8735, below by 0.1605
- submission of zs-4b-6254e8de not made: submission create failed (400): You do not have participant permissions for this group
- Kaggle weekly GPU quota reached; the next job waits for it to reset

## Leaderboard (top 8)

| # | owner | overall |
|---|---|---|
| 1 | mkhlystun | 0.8735 |
| 2 | h | 0.8625 |
| 3 | amirmazaheri | 0.8313 |
| 4 | Vincente | 0.8257 |
| 5 | Hoang Bui | 0.6793 |
| 6 | chrisathy | 0.2515 |

## Runs (newest first)

| run | status | dev weighted | dev overall | hours | why |
|---|---|---|---|---|---|
| zs-4b-6254e8de | ok | 0.7130 | 0.7020 | 7.076 | zero-shot Qwen3-VL-4B, 16 frames. Baseline, T4 seconds per question, first submission (confirms the format) |
| text-4b-f571bdb5 | ok | 0.4773 | 0.4807 | 0.099 | text-only probe. How much the options alone give away (paper's text-only result is 29.95%) |

Submissions used: 0 of 100. GPU hours, last 7 days: 7.3 of 30.
Active job: none
