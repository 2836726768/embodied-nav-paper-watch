#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NO_SITE=0
for arg in "$@"; do
  if [ "$arg" = "--no-site" ]; then
    NO_SITE=1
  fi
done

python3 "$ROOT_DIR/src/daily_paper_watch.py" "$@"

if [ "$NO_SITE" = "0" ] && [ -d "$ROOT_DIR/site" ]; then
  rm -rf "$ROOT_DIR/docs"
  cp -a "$ROOT_DIR/site" "$ROOT_DIR/docs"
  touch "$ROOT_DIR/docs/.nojekyll"
fi
