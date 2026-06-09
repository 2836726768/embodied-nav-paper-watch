#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TZ_NAME="${PAPER_WATCH_TZ:-Asia/Shanghai}"
START_DATE="${PAPER_WATCH_BACKFILL_START:-2026-06-01}"
END_DATE="${PAPER_WATCH_BACKFILL_END:-$(TZ="$TZ_NAME" date +%F)}"

if [ "$#" -ge 1 ] && [[ "$1" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
  START_DATE="$1"
  shift
fi

if [ "$#" -ge 1 ] && [[ "$1" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
  END_DATE="$1"
  shift
fi

MAX_RESULTS="${PAPER_WATCH_BACKFILL_MAX_RESULTS:-300}"
DRY_RUN=0
for arg in "$@"; do
  if [ "$arg" = "--dry-run" ]; then
    DRY_RUN=1
  fi
done

current="$START_DATE"
end_epoch="$(date -d "$END_DATE" +%s)"
while [ "$(date -d "$current" +%s)" -le "$end_epoch" ]; do
  echo "[paper-watch] backfill ${current}" >&2
  python3 "$ROOT_DIR/src/daily_paper_watch.py" \
    --run-date "$current" \
    --days 1 \
    --max-results "$MAX_RESULTS" \
    "$@"
  current="$(date -d "$current +1 day" +%F)"
done

if [ -d "$ROOT_DIR/site" ]; then
  rm -rf "$ROOT_DIR/docs"
  cp -a "$ROOT_DIR/site" "$ROOT_DIR/docs"
  touch "$ROOT_DIR/docs/.nojekyll"
fi

if [ "${PAPER_WATCH_GIT_PUSH:-0}" = "1" ] && [ "$DRY_RUN" = "0" ]; then
  cd "$ROOT_DIR"
  git add docs site out
  if git diff --cached --quiet; then
    echo "[paper-watch] no archive changes to commit" >&2
  else
    git commit -m "Backfill paper watch ${START_DATE} to ${END_DATE}"
  fi
  git push "${PAPER_WATCH_GIT_REMOTE:-origin}" "HEAD:${PAPER_WATCH_GIT_BRANCH:-main}"
fi
