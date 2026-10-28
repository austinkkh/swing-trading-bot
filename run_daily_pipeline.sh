#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
cd "$REPO_ROOT"

# Optional modes:
#   ./run_daily_pipeline.sh
#   ./run_daily_pipeline.sh --submit-paper DUXXXXXXX 7497
#
# Or use environment variables (useful for launchd):
#   TRADING_MODE=dry-run|paper-submit
#   TRADING_ACCOUNT=DUXXXXXXX
#   TRADING_TWS_PORT=7497
#   TRADING_CLIENT_ID=11   (the watcher holds the config CLIENT_ID)
#   PYTHON=/path/to/python (launchd's PATH finds macOS python3, which has no pandas)
#   TRADING_SCAN_ATTEMPTS=3        scanner runs before giving up (only exit 1, a TWS/data failure, retries)
#   TRADING_SCAN_RETRY_SLEEP=120   seconds between attempts
#   TRADING_SCAN_DEADLINE=0945     ET HHMM: no retry starts at or after this (keeps the plan premarket-ish)
#   TRADING_FARM_WAIT_SEC=600      cap on the scanner's total data-farm wait per attempt; each attempt
#                                  also gets no more than the time left before the deadline
#   TRADING_NOTIFY=1               macOS notifications via notify.py: scan / pipeline failure, snapshot
#                                  failure, engine BUY + MANUAL_REVIEW counts (0 to silence; config
#                                  NOTIFY_ENABLED = False also silences)
PYTHON="${PYTHON:-python3}"
SUBMIT_MODE="${1:-}"
ACCOUNT="${2:-${TRADING_ACCOUNT:-}}"
TWS_PORT="${3:-${TRADING_TWS_PORT:-7497}}"
TRADING_MODE="${TRADING_MODE:-dry-run}"
if [[ "$SUBMIT_MODE" == "--submit-paper" ]]; then
  TRADING_MODE="paper-submit"
fi

if [[ "$(date +%u)" -ge 6 ]]; then
  echo "==> Weekend detected; premarket pipeline skipped"
  exit 0
fi

OUTPUT_DIR=$("$PYTHON" - <<'PY'
import os, config
print(os.path.expanduser(config.OUTPUT_DIR))
PY
)

mkdir -p "$OUTPUT_DIR" "$OUTPUT_DIR/paper_runs"
LOG_DIR="$REPO_ROOT/Logs/scheduler"
mkdir -p "$LOG_DIR"
STAMP=$(date +"%Y-%m-%d_%H%M")
LOG_FILE="$LOG_DIR/premarket_pipeline_${STAMP}.log"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "==> Repo root: $REPO_ROOT"
echo "==> Mode: $TRADING_MODE"
echo "==> Output dir: $OUTPUT_DIR"

# Hold off idle/system sleep while this pipeline runs: the 2026-10-26 scan failed because the
# Mac dropped back to sleep mid-connect. (Doesn't stop lid-closed sleep on battery.)
if command -v caffeinate >/dev/null 2>&1; then
  caffeinate -i -s -w $$ >/dev/null 2>&1 &
fi

# notify.py is rate-limited and always exits 0, so a notification can never stop the pipeline.
# Its output stays in this log: it prints one line, only when a post fails.
notify() {
  if [[ "${TRADING_NOTIFY:-1}" == "1" ]]; then
    "$PYTHON" notify.py --title "$1" --level "$2" "$3" || true
  fi
}

NOTIFIED=0
notify_failure() {
  echo "ERROR: $1"
  NOTIFIED=1
  notify "Premarket scan FAILED" error "$1"
}

SCAN_ATTEMPTS="${TRADING_SCAN_ATTEMPTS:-3}"
SCAN_RETRY_SLEEP="${TRADING_SCAN_RETRY_SLEEP:-120}"
SCAN_DEADLINE="${TRADING_SCAN_DEADLINE:-0945}"
FARM_WAIT_CAP="${TRADING_FARM_WAIT_SEC:-600}"

# Seconds from now (ET) to the deadline, 0 if past it. Bounds an attempt's data-farm waiting so
# one attempt can't keep waiting until after the open (review M1).
seconds_to_deadline() {
  local now_hhmm now_min dl_min
  now_hhmm=$(TZ=America/New_York date +%H%M)
  now_min=$(( 10#${now_hhmm:0:2} * 60 + 10#${now_hhmm:2:2} ))
  dl_min=$(( 10#${SCAN_DEADLINE:0:2} * 60 + 10#${SCAN_DEADLINE:2:2} ))
  echo $(( dl_min > now_min ? (dl_min - now_min) * 60 : 0 ))
}

echo "==> Running momentum scanner"
# Only this run's scan may feed the engine: never last night's file or the watcher's
# intraday *_watch_COMPLETE.csv (a failed scan exits non-zero and the pipeline stops here).
RUN_MARKER=$(mktemp)
# Any other non-zero exit (set -e: engine, executor, missing COMPLETE file) also gets one notification
on_exit() {
  local rc=$?
  rm -f "$RUN_MARKER"
  if [[ $rc -ne 0 && $NOTIFIED -eq 0 ]]; then
    notify "Premarket pipeline FAILED" error "run_daily_pipeline.sh exited $rc; see Logs/scheduler/$(basename "$LOG_FILE")"
  fi
}
trap on_exit EXIT
attempt=1
while true; do
  echo "==> scan attempt $attempt/$SCAN_ATTEMPTS"
  rc=0
  farm_wait="$FARM_WAIT_CAP"
  left=$(seconds_to_deadline)
  # Started before the deadline: wait no longer than the time left. A manual run after the
  # deadline keeps the full cap (its first attempt is the only one).
  if [[ $left -gt 0 && $left -lt $farm_wait ]]; then
    farm_wait=$left
  fi
  echo "==> data-farm wait budget for this attempt: ${farm_wait}s"
  TRADING_FARM_WAIT_SEC="$farm_wait" "$PYTHON" momentum_scanner.py || rc=$?
  if [[ $rc -eq 0 ]]; then
    break
  fi
  # Exit 1 = couldn't connect / data farm / SPY: worth another try. Anything else (2 = no
  # universe or survivors) won't change on a rerun.
  if [[ $rc -ne 1 ]]; then
    notify_failure "momentum scanner exited $rc; no plan today"
    exit "$rc"
  fi
  if [[ $attempt -ge $SCAN_ATTEMPTS ]]; then
    notify_failure "momentum scanner failed $attempt time(s) (TWS / data farm); no plan today"
    exit 1
  fi
  sleep "$SCAN_RETRY_SLEEP"
  now_hhmm=$(TZ=America/New_York date +%H%M)
  if [[ 10#$now_hhmm -ge 10#$SCAN_DEADLINE ]]; then
    notify_failure "momentum scanner failed; past the ${SCAN_DEADLINE} ET retry deadline, no plan today"
    exit 1
  fi
  attempt=$((attempt + 1))
done

LATEST_COMPLETE=$(find "$OUTPUT_DIR" -maxdepth 1 -name 'scan_*_afterhours_COMPLETE.csv' -newer "$RUN_MARKER" \
  -print 2>/dev/null | sort | tail -n 1 || true)
rm -f "$RUN_MARKER"
if [[ -z "${LATEST_COMPLETE}" ]]; then
  echo "ERROR: this run's scan wrote no COMPLETE file in $OUTPUT_DIR"
  exit 1
fi
echo "==> Latest COMPLETE file: $LATEST_COMPLETE"

DECISIONS_CSV="$OUTPUT_DIR/scan_${STAMP}_execution_decisions.csv"

echo "==> Reading paper account positions (read-only)"
# This run's own snapshot file: if it fails, there's no file and the engine blocks new entries
# (an older snapshot from a manual run is never picked up instead).
SNAPSHOT_JSON="$OUTPUT_DIR/paper_runs/account_snapshot_${STAMP}.json"
rm -f "$SNAPSHOT_JSON"
"$PYTHON" account_snapshot.py --client-id 13 --port "$TWS_PORT" ${ACCOUNT:+--account "$ACCOUNT"} \
  --out "$SNAPSHOT_JSON" || { echo "WARN: account snapshot failed; new entries will be blocked"
  notify "Account snapshot failed" warning "no snapshot this run: the engine blocks new entries"; }

echo "==> Running execution engine"
"$PYTHON" execution_engine.py --input "$LATEST_COMPLETE" --output "$DECISIONS_CSV" --mode premarket \
  --account-snapshot "$SNAPSHOT_JSON" --require-account-snapshot ${ACCOUNT:+--expect-account "$ACCOUNT"}

echo "==> Running paper executor"
if [[ "$TRADING_MODE" == "paper-submit" ]]; then
  if [[ -z "$ACCOUNT" ]]; then
    echo "ERROR: account required. Example: ./run_daily_pipeline.sh --submit-paper DUXXXXXXX 7497"
    exit 1
  fi
  "$PYTHON" ibkr_paper_executor.py \
    --input "$DECISIONS_CSV" \
    --output-dir "$OUTPUT_DIR/paper_runs" \
    --client-id "${TRADING_CLIENT_ID:-11}" \
    --submit-paper \
    --account "$ACCOUNT" \
    --tws-port "$TWS_PORT"
else
  "$PYTHON" ibkr_paper_executor.py \
    --input "$DECISIONS_CSV" \
    --output-dir "$OUTPUT_DIR/paper_runs"
fi

# Dry-run bracket order plan (Data/orders/plan_<date>.csv). --plan-only never connects to TWS and
# never transmits; this pipeline NEVER passes --submit to any order module (enable steps: README
# "Automated paper trading (disabled by default)"). Off with config ORDER_PLAN_ENABLED = False.
# A failure here is non-fatal: the decisions and paper artifacts above are already written.
ORDER_PLAN_ON=$("$PYTHON" -c 'import config; print(getattr(config, "ORDER_PLAN_ENABLED", True) is True)' \
  2>/dev/null || echo "False")
if [[ "$ORDER_PLAN_ON" == "True" ]]; then
  echo "==> Writing dry-run order plan (no orders, no TWS connection)"
  plan_rc=0
  "$PYTHON" order_entry.py --decisions "$DECISIONS_CSV" --account-snapshot "$SNAPSHOT_JSON" --plan-only \
    ${ACCOUNT:+--account "$ACCOUNT"} || plan_rc=$?
  if [[ $plan_rc -eq 2 ]]; then
    # 2 = a guard refused the file or halted the plan (stale scan, no snapshot, ...); the reason is logged above
    echo "WARN: order plan halted or refused (exit 2); no orders were affected"
  elif [[ $plan_rc -ne 0 ]]; then
    echo "WARN: order plan failed (exit $plan_rc); no orders were affected"
  fi
else
  echo "==> Order plan step disabled (config ORDER_PLAN_ENABLED is not True)"
fi

echo "==> Done"
echo "Log: $LOG_FILE"
echo "Decisions: $DECISIONS_CSV"
echo "Paper artifacts: $OUTPUT_DIR/paper_runs"
