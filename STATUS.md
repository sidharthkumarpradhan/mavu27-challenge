# ReVA autopilot status (2026-10-07T22:15:53Z)

- Codabench login ok; account may submit
- leaderboard fetch failed: 500 Server Error: Internal Server Error for url: https://www.codabench.org/api/phases/30831/get_leaderboard/
- Kaggle weekly GPU quota reached on account 1
- Kaggle account 2 kernels have no internet (is its phone number verified?); next try after 2026-10-07T23:32:23Z
- no Kaggle account can take the next job; it waits

## Leaderboard (top 8)

| # | owner | overall |
|---|---|---|

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
