"""
Config template — copy to config.py and edit. config.py is git-ignored.

    cp config.template.py config.py

Strategy thresholds (tiers, stop bands, sizing, exits) are EXAMPLE values in
strategy_params.py; the scanner's tier settings can also be overridden here.
"""

import os

# ─── IBKR connection (TWS paper: 7497, TWS live: 7496; Gateway paper: 4002) ───
TWS_HOST = "127.0.0.1"
TWS_PORT = 7497
CLIENT_ID = 1
MAX_CONCURRENT_REQUESTS = 50
REQUEST_DELAY = 0.1
# Seconds (awake time) to wait for IBKR's historical data farm after connecting before a scan
# gives up; env TRADING_FARM_WAIT_SEC overrides. Data reliability only, no trading rule.
FARM_WAIT_SEC = 600

# ─── Universe ───
SCAN_UNIVERSE = "DYNAMIC"          # DYNAMIC (Data/ files) or STAGE2_WATCHLIST
STAGE2_WATCHLIST = []              # manual symbols always included
CUSTOM_SYMBOLS = []
DEBUG_SYMBOLS = []                 # symbols to trace in debug output

# Symbol hygiene: quarantine a ticker after this many distinct scan days of IBKR Error 200
# ("no security definition"). Hand-kept skips/renames live in symbol_overrides.csv (repo root).
SYMBOL_QUARANTINE_DAYS = 3

# ─── Base filters ───
MIN_PRICE = 10.0
MAX_PRICE = 2000.0
MIN_AVG_VOLUME = 300_000
MIN_MONTHLY_TURNOVER = 500_000_000  # example liquidity floor, $ traded per month
REQUIRE_ABOVE_50MA = True
REQUIRE_ABOVE_200MA = True
MAX_DISTANCE_FROM_52W_HIGH = 25.0  # % below the 52-week high
MIN_DISTANCE_FROM_52W_LOW = 30.0   # % above the 52-week low

# ─── Indicators ───
ATR_PERIOD = 14
RS_LOOKBACK_PERIODS = [20, 50, 250]
VCP_CONTRACTION_REQUIRED = 3

# ─── IBD data (files in Data/; raw IBD exports are git-ignored) ───
# EPS growth = IBD "Curr Yr EPS Est. % Chg." from the newest ibd_universe_*.csv (automatic).
# True: a stock with no IBD EPS Rating needs that estimate >= TIER_BUY_MIN_EPS for BUY.
USE_IBD_EPS_FILTERS = True
# IBD_UNIVERSE_GLOB = os.path.expanduser("~/swing-trading-bot/Data/ibd_universe_*.csv")

# ─── Output ───
ENABLE_PRIORITY_RATING = True
ENABLE_CSV_EXPORT = True
ENABLE_CONSOLE_OUTPUT = True
# True sets the scan log to DEBUG level (per-symbol fetch/contract lines; one log reached 413 MB)
# and prints the DEBUG_SYMBOLS trace. Warnings, scan-health issues and order alerts log at
# INFO/WARNING, so they stay in the log with it off.
DEBUG_MODE = False
OUTPUT_DIR = os.path.expanduser("~/swing-trading-bot/Scans")
LOG_DIR = os.path.expanduser("~/swing-trading-bot/Logs")
# Scan log size cap (momentum_scan_<date>.log): rolls to .1 ... .5 at LOG_MAX_MB each, so one day
# tops out near 300 MB. launchd's own stdout/stderr logs are not covered: scripts/rotate_launchd_logs.py
LOG_MAX_MB = 50
LOG_BACKUP_COUNT = 5
# macOS notifications (notify.py): scan/pipeline failures, unprotected positions, order-module
# alerts and exit code 2, engine BUY / MANUAL_REVIEW counts. Same message at most once per 30 min.
NOTIFY_ENABLED = True

# ─── Bot order path (order_broker / order_entry) ───
# Paper only. An order is transmitted ONLY if AUTO_SUBMIT_ENABLED is the bool True AND the CLI
# gets --submit; anything else is a dry run that writes the planned orders. Turning this on
# is a deliberate, manual decision.
AUTO_SUBMIT_ENABLED = False
ORDER_PLAN_ENABLED = True            # pipeline writes a dry-run order plan after the engine
MAX_AUTO_ENTRIES_PER_DAY = 1         # bot entries per session (separate from the engine's daily plan cap)
import strategy_params as _sp
ENTRY_CHASE_CAP_PCT = _sp.ENTRY_CHASE_CAP_PCT  # buy stop-limit: limit = pivot * (1 + this/100); example value
ORDER_CLIENT_ID = 16                 # every order-writing session (IBKR: only the placing client can modify)
ORDER_READONLY_CLIENT_ID = 17        # dry runs, bars, reconcile reads
IBKR_ACCOUNT = ""                    # paper account (DU...), required for --submit
# ORDERS_DIR = os.path.expanduser("~/swing-trading-bot/Data/orders")   # default: Data/orders next to OUTPUT_DIR
ORDER_MAX_SCAN_AGE_MIN = 120         # submit: the decisions' Scan_Timestamp must be today and this fresh
ENTRY_SUBMIT_CUTOFF_ET = "09:29"     # v1 submits entries only before the open
CLOSE_WINDOW_ET = ("15:35", "15:48") # exit_manager --close-window: time-stop MOC placement (NYSE MOC
                                     # cutoff 15:50); shifted to 12:35-12:48 on early-close days
TRIM_STOP_MODE = "shared_oca"        # exit_manager: "shared_oca" (one OCA group; needs paper check 1 PASS:
                                     # a trim fill REDUCES the stop) or "split_stop" (the partial trim gets its own
                                     # stop in its own OCA group; correct whether TWS cancels or reduces)
