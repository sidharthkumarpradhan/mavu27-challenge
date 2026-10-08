#!/usr/bin/env bash
# Start a Colab session (colab.yml) when a Colab entry is pending and none is running or queued.
# The autopilot calls this after every cycle, so a new Colab entry starts within about 20 minutes.
# Kill switch: the repo variable COLAB=off (passed in as $COLAB).
set -u
[ "${COLAB:-}" = "off" ] && exit 0
[ -d state ] || exit 0
python -m reva.cli colab --pending --state state >/dev/null || exit 0
runs=$(gh run list --workflow colab.yml --repo "$GITHUB_REPOSITORY" --limit 5 --json status,conclusion,headSha) || exit 0
if [ "$(echo "$runs" | jq '[.[] | select(.status != "completed")] | length')" != "0" ]; then
  echo "a Colab session is already running or queued"; exit 0
fi
# a failed session needs a fix, not a retry every 20 minutes (a broken login uses no units but
# loops). A fix moves main, so the same code is never retried and new code is tried once.
if [ "$(echo "$runs" | jq -r '.[0].conclusion // ""')" = "failure" ] \
   && [ "$(echo "$runs" | jq -r '.[0].headSha // ""')" = "$(git rev-parse HEAD)" ]; then
  echo "::warning::the last Colab session failed on this commit; not starting another until a fix is merged"; exit 0
fi
gh workflow run colab.yml --ref main --repo "$GITHUB_REPOSITORY" && echo "started a Colab session" \
  || echo "::warning::could not start a Colab session"
