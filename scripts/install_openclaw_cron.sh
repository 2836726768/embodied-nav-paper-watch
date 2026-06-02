#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JOB_NAME="${OPENCLAW_JOB_NAME:-embodied-nav-paper-watch}"
SCHEDULE="${OPENCLAW_CRON_SCHEDULE:-0 9 * * *}"
TIMEZONE="${OPENCLAW_CRON_TZ:-Asia/Shanghai}"
PROMPT="${OPENCLAW_CRON_PROMPT:-Run this local command and send the generated Markdown report as the final reply: cd \"$ROOT_DIR\" && ./scripts/daily_push.sh}"

usage() {
  cat <<'USAGE'
Usage: scripts/install_openclaw_cron.sh [--dry-run]

Environment overrides:
  OPENCLAW_JOB_NAME        default: embodied-nav-paper-watch
  OPENCLAW_CRON_SCHEDULE   default: 0 9 * * *
  OPENCLAW_CRON_TZ         default: Asia/Shanghai
  OPENCLAW_CRON_PROMPT     default: run scripts/daily_push.sh and deliver output
USAGE
}

DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $arg" >&2; usage >&2; exit 2 ;;
  esac
done

cmd=(
  openclaw cron add "$SCHEDULE" "$PROMPT"
  --name "$JOB_NAME"
  --session isolated
  --tz "$TIMEZONE"
  --announce
  --best-effort-deliver
  --expect-final
  --timeout-seconds 900
)

printf 'OpenClaw cron command:\n'
printf '  %q' "${cmd[@]}"
printf '\n'

if [ "$DRY_RUN" = "1" ]; then
  exit 0
fi

if ! command -v openclaw >/dev/null 2>&1; then
  echo "ERROR: openclaw command not found in PATH. Install OpenClaw or add its CLI to PATH, then rerun this script." >&2
  exit 127
fi

exec "${cmd[@]}"
