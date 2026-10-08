# ReVA autopilot status (2026-10-08T18:44:22Z)

- Codabench login failed: login failed (500)
- leaderboard fetch failed: 500 Server Error: Internal Server Error for url: https://www.codabench.org/api/phases/30831/get_leaderboard/
- job tubu9938/reva-ft-q35-4b-16f-e4ba68c0 is running

## Leaderboard (top 8)

| # | owner | overall |
|---|---|---|

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
