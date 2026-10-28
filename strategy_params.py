"""
Strategy parameters: EXAMPLE VALUES ONLY.

Every trading threshold the scanner, execution engine, risk manager and exit plan use lives here,
in one place. The numbers below are illustrative placeholders chosen so the code runs and the
tests exercise every branch. They are NOT the parameters of any live or paper strategy, they have
not been optimised, and nothing here is a recommendation to trade with them.

To run your own variant, edit this file (or override the scanner's tier settings in config.py,
which the scanner reads first).
"""

# ─── Position sizing (execution_engine.ExecutionPolicy, risk_manager.size_position) ───
RISK_PER_TRADE_PCT = 0.5            # % of equity lost if the initial stop is hit
MAX_POSITION_PCT = 20.0             # cap on any one position, % of equity
MAX_NEW_ENTRIES_PER_DAY = 3
MAX_SECTOR_CONCENTRATION_PCT = 35.0
MAX_TOTAL_OPEN_RISK_PCT = 3.0       # sum of capital at risk across open positions
MAX_GROUP_OPEN_RISK_PCT = 1.5       # per industry group; more goes to manual review
MAX_OPEN_POSITIONS = 8
MAX_PCT_OF_DAILY_DOLLAR_VOLUME = 1.0
EARNINGS_BLACKOUT_DAYS = 5
MIN_BREAKER_HISTORY_SESSIONS = 5    # equity-log sessions needed before the bot may submit entries

# ─── Stop bands by entry type ───
MIN_STOP_PCT = 1.0                  # tighter stops are noise and get skipped
MAX_STOP_PCT = 4.0                  # ORB, pullback and unknown entries
MAX_STOP_PCT_PIVOT = 7.0            # base-pattern pivot breakouts

# ─── Loss circuit breakers (risk_manager.CircuitBreakers) ───
DAILY_LOSS_LIMIT_PCT = 2.0          # no new entries today
WEEKLY_LOSS_LIMIT_PCT = 5.0         # reduced risk for the rest of the week
MONTHLY_DRAWDOWN_LIMIT_PCT = 10.0   # no new entries; review the system
WEEKLY_LOSS_RISK_MULT = 0.5

# ─── Exit plan (exit_rules) ───
TRIM_R = 1.5                        # sell a partial at +N R; the stop also moves to breakeven here
TRIM_FRACTION = 0.25
FAST_SESSIONS = 3                   # +N R within this many sessions after entry: skip the trim
RESPECT_LOOKBACK = 30               # sessions before entry used to pick the trailing MA
RESPECT_THRESHOLD = 0.70            # share of those closes above the MA for it to count
TIME_STOP_SESSIONS = 10             # exit if +TRIM_R hasn't come after this many sessions

# ─── Order entry (order_entry) ───
ENTRY_CHASE_CAP_PCT = 1.0           # buy stop-limit: limit = pivot * (1 + this/100)

# ─── Market regime ───
ELITE_CORRECTION_SIZE_MULT = 0.5    # elites still tradeable in a correction, at this size

# ─── Elite leader ("generals") rule (risk_manager.is_elite_leader) ───
ELITE_EPS_YOY_MIN = 80
ELITE_RS_MIN = 90
ELITE_COMPOSITE_MIN = 90
ELITE_RS_ONLY_MIN = 95
ELITE_RS_ONLY_YTD_MIN = 40

# ─── Scanner tiers (momentum_scanner; config.py may override any of these) ───
# Extension from the 50-day MA in ATR multiples
ATR_BUY_CAP = 4.0
ATR_EXTENDED_MODERATE_MAX = 7.0
ATR_EXTENDED_VERY_MAX = 10.0
# ADR (average daily range %) ceilings
TIER_BUY_MAX_ADR_BASE = 6.0
TIER_BUY_MAX_ADR_HIGH_EPS = 7.5
# Stop distance / ADR
STOP_ADR_RATIO_LUK_TIGHT = 0.5
STOP_ADR_RATIO_IDEAL = 1.0
STOP_ADR_RATIO_BUY_MAX = 1.5
STOP_ADR_RATIO_ELITE_MAX = 2.0
# Stop % caps
STOP_PCT_IDEAL = 4.0
STOP_PCT_BUY_MAX = 6.0
STOP_PCT_ELITE_MAX = MAX_STOP_PCT_PIVOT
STOP_PCT_MIN = MIN_STOP_PCT
# BUY tier
TIER_BUY_MIN_RS = 80
TIER_BUY_MIN_YTD = 0
TIER_BUY_MIN_EPS = 25
TIER_BUY_MIN_ADR = 2.5
TIER_BUY_HIGH_EPS_THRESHOLD = 60
TIER_BUY_MIN_EPS_RATING = 70
# WATCH / EXTENDED tiers
TIER_WATCH_MIN_RS = 75
TIER_EXTENDED_MIN_RS = 80
TIER_EXTENDED_MIN_YTD = 20
# ELITE tier
TIER_ELITE_MIN_YTD = 0.0
TIER_ELITE_MAX_ADR = 6.0
TIER_ELITE_REQUIRE_ABOVE_200MA = True
TIER_ELITE_MIN_EPS_RATING = 70
# RS leader tier: strong relative strength without the earnings check, at reduced size
RS_LEADER_MIN_RS = 90
RS_LEADER_MAX_PIVOT_DIST_PCT = 4.0  # price within ±N% of the pattern pivot
RS_LEADER_MAX_STOP_PCT = MAX_STOP_PCT_PIVOT
RS_LEADER_SIZE_MULT = 0.5
# Informational flags (not gates)
EXTENDED_NO_BASE_200MA_PCT = 50.0
LATE_RUN_DAYS_ABOVE_21EMA = 25
