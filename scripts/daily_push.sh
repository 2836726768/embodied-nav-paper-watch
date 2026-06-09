#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NO_SITE=0
DRY_RUN=0
for arg in "$@"; do
  if [ "$arg" = "--no-site" ]; then
    NO_SITE=1
  fi
  if [ "$arg" = "--dry-run" ]; then
    DRY_RUN=1
  fi
done

python3 "$ROOT_DIR/src/daily_paper_watch.py" "$@"

if [ "$NO_SITE" = "0" ] && [ -d "$ROOT_DIR/site" ]; then
  rm -rf "$ROOT_DIR/docs"
  cp -a "$ROOT_DIR/site" "$ROOT_DIR/docs"
  touch "$ROOT_DIR/docs/.nojekyll"
fi

if [ "${PAPER_WATCH_GIT_PUSH:-0}" = "1" ] && [ "$DRY_RUN" = "0" ]; then
  cd "$ROOT_DIR"
  git add docs site
  if git diff --cached --quiet; then
    echo "[paper-watch] no website changes to commit" >&2
  else
    report_date="$(TZ="${PAPER_WATCH_TZ:-Asia/Shanghai}" date +%F)"
    git commit -m "Update paper watch for ${report_date}"
  fi
  git push "${PAPER_WATCH_GIT_REMOTE:-origin}" "HEAD:${PAPER_WATCH_GIT_BRANCH:-main}"
fi
