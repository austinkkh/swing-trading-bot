#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
MOMENTUM STOCK SCANNER - COMPLETE SYSTEM (v2 with Regime + Risk Integration)
6-Tier + Parabolic System with IBD Filters & Priority Rating
Price Action Analysis: VCP Contraction + Swing Lows + Base Structure
Based on Mark Minervini / William O'Neil / Jeff Sun / Martin Luk

═══ NEW IN v2 ═══
- Market regime detection (gates ALL buy signals)
- Elite leader flagging (correction-mode exception for generals)
- Watch-only mode during corrections
- ELITE_ONLY.csv generated instead of BUY_LIST.csv in correction
═══════════════════════════════════════════════════════════════════════════════
"""

import logging
import logging.handlers
import os
import sys
import time
import pandas as pd
from datetime import datetime
from typing import Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from config import *

# Processes sharing one TWS need distinct client IDs (the watcher holds CLIENT_ID);
# scheduled jobs set TRADING_CLIENT_ID to connect alongside it.
WATCHER_CLIENT_ID = CLIENT_ID          # config CLIENT_ID: held all day by the launchd watcher
ADHOC_CLIENT_ID = 99                   # a CLI scan with no TRADING_CLIENT_ID (see main)
CLIENT_ID = int(os.environ.get('TRADING_CLIENT_ID', CLIENT_ID))

from ibkr_connector import IBKRConnector
from relative_strength import RelativeStrengthCalculator
from vcp_detector import VCPDetector
import pocket_pivot
import exit_rules
from backtest import add_indicators
from base_count import count_bases
from ibd_universe import load_ibd_universe
import symbol_hygiene

# ═══ NEW IMPORTS (v2) ═══
from market_regime import MarketRegimeDetector
from risk_manager import is_elite_leader

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# TIER CONFIG: example values from strategy_params.py; any of them set in config.py wins.
# ═══════════════════════════════════════════════════════════════════════════
import strategy_params as _sp

_TIER_SETTINGS = (
    'ATR_BUY_CAP', 'ATR_EXTENDED_MODERATE_MAX', 'ATR_EXTENDED_VERY_MAX',
    'TIER_BUY_MAX_ADR_BASE', 'TIER_BUY_MAX_ADR_HIGH_EPS',
    'STOP_ADR_RATIO_LUK_TIGHT', 'STOP_ADR_RATIO_IDEAL', 'STOP_ADR_RATIO_BUY_MAX', 'STOP_ADR_RATIO_ELITE_MAX',
    'STOP_PCT_IDEAL', 'STOP_PCT_BUY_MAX', 'STOP_PCT_ELITE_MAX', 'STOP_PCT_MIN',
    'TIER_BUY_MIN_RS', 'TIER_BUY_MIN_YTD', 'TIER_BUY_MIN_EPS', 'TIER_BUY_MIN_ADR',
    'TIER_BUY_HIGH_EPS_THRESHOLD', 'TIER_BUY_MIN_EPS_RATING',
    'TIER_WATCH_MIN_RS', 'TIER_EXTENDED_MIN_RS', 'TIER_EXTENDED_MIN_YTD',
    'TIER_ELITE_MIN_YTD', 'TIER_ELITE_MAX_ADR', 'TIER_ELITE_REQUIRE_ABOVE_200MA', 'TIER_ELITE_MIN_EPS_RATING',
    'RS_LEADER_MIN_RS', 'RS_LEADER_MAX_PIVOT_DIST_PCT', 'RS_LEADER_MAX_STOP_PCT', 'RS_LEADER_SIZE_MULT',
    'EXTENDED_NO_BASE_200MA_PCT', 'LATE_RUN_DAYS_ABOVE_21EMA', 'ELITE_CORRECTION_SIZE_MULT',
)
for _name in _TIER_SETTINGS:
    if _name not in globals():
        globals()[_name] = getattr(_sp, _name)

# Display caps
MAX_ELITE_DISPLAY = MAX_BUY_DISPLAY = MAX_WATCH_DISPLAY = MAX_EXTENDED_DISPLAY = 20

try:
    IBD_UNIVERSE_GLOB
except NameError:
    # Full IBD data-table export, named ibd_universe_YYYY-MM-DD.csv; newest file wins
    IBD_UNIVERSE_GLOB = str(Path(__file__).resolve().parent / "Data" / "ibd_universe_*.csv")

IBD_UNIVERSE_STALE_DAYS = 7   # weekly export; same limit as the dashboard (scan_display.py)


try:
    FARM_WAIT_SEC
except NameError:
    # How long (seconds of awake time) to wait for IBKR's historical data farm (ushmds) after
    # connecting, e.g. when TWS is still reconnecting after the Mac woke. Data reliability only.
    FARM_WAIT_SEC = 600
FARM_WAIT_SEC = int(os.environ.get('TRADING_FARM_WAIT_SEC', FARM_WAIT_SEC))

try:
    SCAN_MAX_FAILED_FRACTION
except NameError:
    # Whole-scan data-integrity gate (not a stock filter): if more than this share of the
    # universe (and at least SCAN_MAX_FAILED_MIN symbols) timed out or hit a session-level
    # data block, the data pull failed mid-scan. The scan then writes no file and exits 1, so
    # the pipeline retries instead of planning from a partial universe. Owner may tune these.
    SCAN_MAX_FAILED_FRACTION = 0.25
    SCAN_MAX_FAILED_MIN = 20




def days_above_ema(close: pd.Series, ema: pd.Series) -> int:
    """Closes since the last close below the EMA (0 if the latest close is below it)."""
    below = (close < ema).to_numpy()
    for k in range(len(below)):
        if below[-1 - k]:
            return k
    return len(below)


# In a CORRECTION the scanner is watch-only for BUY / RS leaders; elites may still trade at reduced size.


def effective_size_multiplier(stock: dict, tier: str, watch_only: bool) -> float:
    """Size multiplier exported to the execution engine (1.0 = full risk per trade)."""
    mult = float(stock.get('size_multiplier', 1.0))
    if watch_only and tier == 'TIER_ELITE_BUY':
        mult *= ELITE_CORRECTION_SIZE_MULT
    return mult


def entry_setup_label(stock) -> str:
    """
    The entry the execution engine should size for. Only a real VCP with price near its
    pivot is VCP_PIVOT (it gets the wide pivot stop band). A VCP-ish label without that
    (VCP score alone, VCP_LOOSE, far from the pivot) is BASE, which gets the tight band.
    """
    if stock.get('elite_entry_setup'):
        return stock['elite_entry_setup']
    setup = stock.get('setup_type') or ''
    if setup in ('VCP', 'VCP_LOOSE'):
        near_pivot = abs(stock.get('vcp_pivot_distance_pct') or 999) <= RS_LEADER_MAX_PIVOT_DIST_PCT
        return 'VCP_PIVOT' if stock.get('has_real_vcp') and near_pivot else 'BASE'
    return setup


# Industry group names (IBD's export has only the numeric group rank). industry.csv comes from
# a separate industry fetch (IBKR contract details); the theme dashboard from a separate
# theme study. Both are optional; are informational in the scan export: no tier, priority or size uses them.
DATA_DIR = Path(OUTPUT_DIR).expanduser().parent / "Data"
INDUSTRY_CSV = DATA_DIR / "fundamentals" / "industry.csv"
THEME_DASHBOARD_CSV = DATA_DIR / "themes" / "theme_dashboard_latest.csv"


def _file_key(path: Path):
    """(path, mtime) so a cache reloads when the file changes; (path, None) if missing, so a
    missing file is cached too (warned once, not once per exported row: review M4)."""
    try:
        return (str(path), path.stat().st_mtime)
    except OSError:
        return (str(path), None)


_industry_cache: dict = {'key': 'unset', 'map': {}}   # key (path, None) = file missing (warned once)
_theme_cache: dict = {'key': 'unset', 'map': {}}


def load_industry_map(path: Optional[Path] = None) -> Dict[str, str]:
    """symbol -> industry subcategory from industry.csv (same rule as theme_dashboard.load_industry).
    Missing or unreadable file: one warning and an empty map, never a failed scan."""
    path = Path(path or INDUSTRY_CSV)
    key = _file_key(path)
    if key == _industry_cache['key']:
        return _industry_cache['map']
    mapping: Dict[str, str] = {}
    if key[1] is None:
        logger.warning(f"⚠️ No {path}: Industry_Group will be blank "
                       "(optional industry map not present)")
    else:
        try:
            ind = pd.read_csv(path).dropna(subset=['subcategory']).drop_duplicates('symbol')
            mapping = dict(zip(ind['symbol'].astype(str).str.upper(), ind['subcategory'].astype(str).str.strip()))
        except Exception as e:                                   # noqa: BLE001
            logger.warning(f"⚠️ Couldn't read {path} ({e}); Industry_Group will be blank")
    _industry_cache.update(key=key, map=mapping)
    return mapping


def load_theme_map(path: Optional[Path] = None) -> Dict[str, dict]:
    """group -> {rank, emerging, as_of} from the theme dashboard; empty when the file is missing."""
    path = Path(path or THEME_DASHBOARD_CSV)
    key = _file_key(path)
    if key == _theme_cache['key']:
        return _theme_cache['map']
    mapping: Dict[str, dict] = {}
    if key[1] is not None:
        try:
            dash = pd.read_csv(path)
            for r in dash.itertuples(index=False):
                mapping[str(r.group).strip()] = {
                    'rank': int(r.rank) if pd.notna(r.rank) else None,
                    'emerging': str(r.emerging).strip().lower() == 'true',
                    'as_of': str(r.as_of_date) if pd.notna(r.as_of_date) else '',
                }
        except Exception as e:                                   # noqa: BLE001
            logger.warning(f"⚠️ Couldn't read {path} ({e}); theme columns will be blank")
    _theme_cache.update(key=key, map=mapping)
    return mapping


def theme_columns(group: str) -> dict:
    """Informational theme-dashboard columns for one industry group (blank when unranked).
    Research (abc81d8) found themes + RS + VCP lowered win rate: never feed these into tiers."""
    info = load_theme_map().get(str(group or '').strip()) if group else None
    if not info:
        return {'Theme_Group_Rank': None, 'Theme_Emerging': None, 'Theme_As_Of': ''}
    return {'Theme_Group_Rank': info['rank'], 'Theme_Emerging': info['emerging'], 'Theme_As_Of': info['as_of']}


# Why a historical fetch came back empty (ibkr_connector status -> scan-health text)
STATUS_REASONS = {
    'no_secdef': 'no_secdef: IBKR has no security definition (Error 200: delisted or renamed?)',
    'no_data': 'no_data: IBKR returned no bars (Error 162)',
    'pacing': 'pacing: IBKR pacing violation',
    'session_conflict': 'session_conflict: TWS trading session connected from a different IP (Error 162)',
}


SCAN_WORKERS_MAX = 10   # parallel base-filter threads (each holds one IBKR historical request)


def _first_not_none(*values):
    """First value that isn't None (0 is a real value, so `or` won't do)."""
    return next((v for v in values if v is not None), None)


def _eps_label(value):
    """Format EPS growth as a 6-char label. Missing data (None) shows n/a, not 0%."""
    if value is None:
        return "   n/a"
    return f"{value:>+5.0f}%"


def _num_or_na(value, spec):
    """Format a number with spec, or 'n/a' padded to the same width when missing."""
    if value is None:
        width = "".join(ch for ch in spec.split(".")[0] if ch.isdigit())
        return format("n/a", f"<{width}") if width else "n/a"
    return format(value, spec)


class MomentumScanner:
    """
    Complete 6-tier momentum scanner with price action analysis
    v2: Now with market regime gating and elite leader detection
    """
    
    def __init__(self):
        self.connector = None
        self.rs_calculator = None
        self.vcp_detector = None
        self.ibd_universe = {}           # symbol -> IBD ratings (ibd_universe.py)
        self.ibd_universe_date = None
        # ═══ v6: Universe builder integration ═══
        self.universe_obj = None  # Populated by _build_dynamic_universe()
        # ═══ v5: ELITE SPLIT INTO BUY vs HOLD ═══
        self.elite_buy_results = []      # ⚡ Elite with clean pullback entry
        self.elite_hold_results = []     # 📌 Elite tracked only (no entry today)
        self.buy_results = []            # 🎯 Active buy setups (non-elite)
        self.watch_results = []          # 📋 Building bases, high RS
        self.extended_results = []       # 🚀 Too extended (ATR > 4)
        self.rs_leader_results = []      # 🏃 Technical leaders, no earnings check (half size)
        self.scan_timestamp = None
        self.debug_info = {}
        # Per-scan data health: symbols that errored or returned no data, by stage
        self.scan_issues = self._empty_scan_issues()
        self.fetch_status = {}           # symbol -> ibkr fetch status this scan
        self.hygiene_dropped = []        # [(symbol, reason)] skipped by symbol_hygiene
        self.secdef_days = {}            # symbol -> consecutive Error-200 scan days
        self.scan_aborted = False        # True when run_full_scan gave up on data integrity
        # 'afterhours' CLI scans read the last daily bar as a completed session: main() sets this
        # so a run after 9:30 (pipeline retry, manual rescan) drops today's partial bar.
        # The watcher leaves it off: it watches intraday moves on purpose.
        self.complete_daily_bars_only = False
        
        # ═══ NEW (v2): Market regime state ═══
        self.regime_detector = None
        self.market_regime = None
        self.watch_only_mode = False
        self.active_universe_mode = None
        self.dynamic_universe_symbols = set()
        self.manual_watchlist_symbols = set()
        
    def initialize(self):
        """Initialize all components"""
        logger.info("="*70)
        logger.info("MOMENTUM SCANNER v2 - 6-TIER + REGIME GATING")
        logger.info("VCP Contraction + Swing Lows + Base Structure + Market Regime")
        if DEBUG_MODE:
            logger.info("🐛 DEBUG MODE ENABLED")
        logger.info("="*70)
        
        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
        
        logger.info("Connecting to TWS...")
        self.connector = IBKRConnector()
        self.connector.complete_daily_bars_only = self.complete_daily_bars_only
        
        if not self.connector.connect_and_run(TWS_HOST, TWS_PORT, CLIENT_ID):
            logger.error("❌ Failed to connect to TWS")
            return False
        # A connection is not the same as data: right after a wake TWS reconnects to IB's servers
        # and the HMDS farm is broken for a while (2110 / 2105). Wait for it instead of failing.
        self.connector.farm_wait_sec = FARM_WAIT_SEC
        # One budget for the whole run (initial wait + every SPY/QQQ retry): the pipeline sets
        # FARM_WAIT_SEC to the time left before its deadline, so an attempt can't run past it
        self.connector.farm_wait_budget = FARM_WAIT_SEC
        if not self.connector.wait_for_data_farms(max_wait=FARM_WAIT_SEC):
            logger.error(f"❌ SCAN ABORTED: IBKR data farm unavailable for {FARM_WAIT_SEC // 60} min "
                         f"({self.connector.farm_status_summary()}); no scan file written")
            return False
        
        self.rs_calculator = RelativeStrengthCalculator(self.connector)
        self.vcp_detector = VCPDetector(atr_period=ATR_PERIOD)
        
        # ═══ NEW (v2): Initialize regime detector ═══
        self.regime_detector = MarketRegimeDetector(self.connector)
        
        self.ibd_universe, self.ibd_universe_date, ibd_path = load_ibd_universe(IBD_UNIVERSE_GLOB)
        if self.ibd_universe:
            logger.info(f"✓ IBD universe: {len(self.ibd_universe)} symbols ({self.ibd_universe_date}) from {ibd_path}")
            self._warn_if_ibd_stale()
        else:
            logger.warning(f"⚠️ No IBD universe file at {IBD_UNIVERSE_GLOB}; no IBD ratings or EPS estimates")
        
        if not self.rs_calculator.fetch_spy_benchmark():
            logger.error(f"❌ SCAN ABORTED: SPY benchmark fetch failed after retries "
                         f"(data farm: {self.connector.farm_status_summary()}); no scan file written")
            return False
        
        logger.info("✓ Scanner initialized successfully")
        return True
    
    def _warn_if_ibd_stale(self, today=None) -> Optional[int]:
        """Log (not just draw on the dashboard) when the IBD export is old: its EPS Rating and A/D
        gate every BUY, so a stale file silently keeps passing old ratings (audit #12). Warning
        only; whether staleness should block BUYs is the owner's call. Returns the age if stale."""
        if not self.ibd_universe_date:
            return None
        age = ((today or datetime.now().date()) - self.ibd_universe_date).days
        if age > IBD_UNIVERSE_STALE_DAYS:
            logger.warning(f"⚠️ IBD universe export is {age} days old ({self.ibd_universe_date}); "
                           f"EPS Rating / A-D gates use those ratings. Export a fresh "
                           f"Data/ibd_universe_YYYY-MM-DD.csv")
            return age
        return None

    def _analyze_price_action(self, stock):
        """
        Comprehensive price action analysis
        Returns detailed base structure, volatility, and candle data
        """
        bars = stock.get('bars', [])
        if not bars:
            return None
        
        df = self._bars_to_dataframe(bars)
        
        # Calculate daily % moves
        df['daily_change_pct'] = ((df['close'] - df['open']) / df['open']) * 100
        df['daily_range_pct'] = ((df['high'] - df['low']) / df['low']) * 100
        df['true_range_pct'] = df['daily_range_pct']
        
        # Get recent 60 days for base analysis
        recent_60 = df.tail(60).copy()
        
        # Find base structure
        base_high = recent_60['high'].max()
        base_high_idx = recent_60['high'].idxmax()
        # Depth = drop from the base high to the lowest low AFTER it. The 60-day min low
        # alone would measure the rally into the high, not the base.
        base_low = recent_60.loc[base_high_idx:, 'low'].min()
        base_depth_pct = ((base_high - base_low) / base_high) * 100
        
        # Find base high date
        base_high_date = recent_60.loc[base_high_idx, 'date']
        
        # Analyze volatility contraction
        period_length = len(recent_60) // 3
        
        period1 = recent_60.iloc[:period_length]
        period2 = recent_60.iloc[period_length:period_length*2]
        period3 = recent_60.iloc[period_length*2:]
        
        vol1 = period1['daily_range_pct'].mean()
        vol2 = period2['daily_range_pct'].mean()
        vol3 = period3['daily_range_pct'].mean()
        
        # Calculate contraction %
        contraction_1_to_2 = ((vol1 - vol2) / vol1 * 100) if vol1 > 0 else 0
        contraction_2_to_3 = ((vol2 - vol3) / vol2 * 100) if vol2 > 0 else 0
        contraction_1_to_3 = ((vol1 - vol3) / vol1 * 100) if vol1 > 0 else 0
        
        # VCP quality based on contraction
        is_contracting = (contraction_1_to_2 > 15 and contraction_2_to_3 > 10)
        
        # Get last 10 daily candles
        last_10 = df.tail(10).copy()
        candle_data = []
        
        for idx, row in last_10.iterrows():
            candle_data.append({
                'date': row['date'].strftime('%Y-%m-%d'),
                'open': round(row['open'], 2),
                'high': round(row['high'], 2),
                'low': round(row['low'], 2),
                'close': round(row['close'], 2),
                'volume': int(row['volume']),
                'change_pct': round(row['daily_change_pct'], 2),
                'range_pct': round(row['daily_range_pct'], 2)
            })
        
        # Current price analysis
        current_price = df['close'].iloc[-1]
        yesterday_low = df['low'].iloc[-2] if len(df) >= 2 else df['low'].iloc[-1]
        yesterday_close = df['close'].iloc[-2] if len(df) >= 2 else df['close'].iloc[-1]
        
        # Distance from base high
        distance_from_base_high_pct = ((current_price - base_high) / base_high) * 100
        
        price_action = {
            'base_high': round(base_high, 2),
            'base_low': round(base_low, 2),
            'base_depth_pct': round(base_depth_pct, 1),
            'base_high_date': base_high_date.strftime('%Y-%m-%d'),
            'distance_from_base_high_pct': round(distance_from_base_high_pct, 1),
            'period1_volatility': round(vol1, 2),
            'period2_volatility': round(vol2, 2),
            'period3_volatility': round(vol3, 2),
            'contraction_1_to_2_pct': round(contraction_1_to_2, 1),
            'contraction_2_to_3_pct': round(contraction_2_to_3, 1),
            'contraction_1_to_3_pct': round(contraction_1_to_3, 1),
            'is_contracting': is_contracting,
            'yesterday_low': round(yesterday_low, 2),
            'yesterday_close': round(yesterday_close, 2),
            'current_price': round(current_price, 2),
            'last_10_candles': candle_data,
            'swing_low_3d': round(df['low'].iloc[-3:].min(), 2),
            'swing_low_5d': round(df['low'].iloc[-5:].min(), 2),
            'swing_low_7d': round(df['low'].iloc[-7:].min(), 2),
            'swing_low_10d': round(df['low'].iloc[-10:].min(), 2),
        }
        
        return price_action
    
    def _calculate_context_based_stop(self, stock):
        """Calculate stop using ACTUAL swing lows from price action"""
        price_action = stock.get('price_action')
        if not price_action:
            return self._mark_no_stop(stock, "No price action data")
        
        current_price = price_action['current_price']
        yesterday_low = price_action['yesterday_low']
        
        distance_10ema = stock.get('distance_from_10ema', 999)
        distance_21ema = stock.get('distance_from_21ema', 999)
        distance_50ma = stock.get('distance_from_50ma', 999)
        vcp_score = stock.get('vcp_score', 0)
        is_contracting = price_action['is_contracting']
        
        eps_yoy = stock.get('eps_yoy') or 0
        industry_rank = stock.get('industry_rank') or 197
        is_premium = (eps_yoy >= 100 and industry_rank <= 10)
        
        # DETERMINE STOP BASED ON CONTEXT
        if vcp_score >= 7.0 and is_contracting and -5 <= distance_50ma <= 15:
            stop_price = max(yesterday_low, price_action['swing_low_3d'])
            context = "VCP_BREAKOUT"
            description = f"Tight VCP breakout - use yesterday's low"
        elif vcp_score >= 5.0 and not is_contracting:
            stop_price = price_action['swing_low_7d']
            context = "LOOSE_BASE"
            description = f"Base needs more time - wider stop required"
        elif -2 <= distance_10ema <= 3 and is_contracting:
            stop_price = price_action['swing_low_3d']
            context = "10EMA_PULLBACK"
            description = f"10EMA pullback - 3-day low"
        elif -2 <= distance_21ema <= 5:
            stop_price = price_action['swing_low_5d']
            context = "21EMA_PULLBACK"
            description = f"21EMA pullback - 5-day low"
        elif -5 <= distance_50ma <= 5:
            stop_price = price_action['swing_low_7d']
            context = "50MA_BOUNCE"
            description = f"50MA bounce - 7-day swing low"
        elif 15 <= distance_50ma <= 30:
            stop_price = price_action['swing_low_10d']
            context = "EXTENDED"
            description = f"Extended - use 10-day low"
        else:
            # True 5-day swing low (max() with yesterday's low always returned yesterday's low,
            # a much tighter stop than intended)
            stop_price = price_action['swing_low_5d']
            context = "DEFAULT"
            description = f"Standard swing low (5-day)"
        
        stop_pct = ((current_price - stop_price) / current_price) * 100

        # A stop at or above price can't protect the position, so treat it as no stop
        if stop_pct <= 0:
            return self._mark_no_stop(stock, f"stop ${stop_price:.2f} at/above price ${current_price:.2f}")

        if stop_pct > 8.0:
            if is_premium:
                status = "MONITOR"
                warning = "⚠️ Wide stop - premium setup only"
            else:
                status = "WATCHLIST"
                warning = "🚨 Stop too wide - wait for better entry"
        elif stop_pct > 6.0:
            if is_premium:
                status = "ACCEPTABLE"
                warning = "✅ Acceptable for premium setup"
            else:
                status = "MONITOR"
                warning = "⚠️ Watch position size"
        else:
            status = "OPTIMAL"
            warning = "🎯 Tight risk - good entry"
        
        if not is_contracting and stop_pct > 5.0:
            warning += " | ⚠️ BASE NEEDS MORE TIME"
        
        stock['stop_loss'] = round(stop_price, 2)
        stock['stop_loss_pct'] = round(stop_pct, 2)  # 2dp: 0.96% mustn't round up past STOP_PCT_MIN
        stock['stop_context'] = context
        stock['stop_description'] = description
        stock['stop_status'] = status
        stock['stop_warning'] = warning
        
        return stop_price, stop_pct, status
    
    def _mark_no_stop(self, stock, reason):
        """Flag a stock with no valid stop. It stays listed but can never be BUY / ELITE_BUY."""
        stock['stop_loss'] = None
        stock['stop_loss_pct'] = None
        stock['stop_context'] = 'NO_STOP'
        stock['stop_description'] = reason
        stock['stop_status'] = 'NO_STOP'
        stock['stop_warning'] = f"🚫 No valid stop ({reason}) - not executable"
        return None, None, 'NO_STOP'

    def calculate_priority_rating(self, stock):
        """
        Priority rating, 0-110: EPS 20 + industry 15 + RS 15 + VCP 15 + 50MA distance 10
        + YTD 10 + volume 10 + stop quality 15. Compare within a scan, not as a percentage.
        """
        if not ENABLE_PRIORITY_RATING:
            return 0.0
        
        total_score = 0.0
        details = {}
        
        eps_yoy = stock.get('eps_yoy') or 0
        eps_currq = stock.get('eps_currq') or 0
        industry_rank = stock.get('industry_rank') or 197
        rs_rating = stock.get('rs_rating') or 0
        vcp_score = stock.get('vcp_score') or 0
        ytd_return = stock.get('ytd_return') or 0
        volume_ratio = stock.get('volume_ratio') or 1.0
        
        price_action = stock.get('price_action') or {}
        is_contracting = price_action.get('is_contracting', False)
        
        stock['eps_currq'] = eps_currq
        stock['industry_rank'] = industry_rank
        
        # 1. EPS Growth (20 pts)
        if eps_yoy >= 500:
            eps_score = 20.0
        elif eps_yoy >= 200:
            eps_score = 18.0
        elif eps_yoy >= 100:
            eps_score = 15.0
        elif eps_yoy >= 50:
            eps_score = 12.0
        elif eps_yoy >= 40:
            eps_score = 10.0
        elif eps_yoy >= 25:
            eps_score = 7.0
        else:
            eps_score = 0.0
        total_score += eps_score
        details['eps_score'] = eps_score
        
        # 2. Industry Rank (15 pts)
        if industry_rank <= 10:
            industry_score = 15.0
        elif industry_rank <= 20:
            industry_score = 13.0
        elif industry_rank <= 30:
            industry_score = 11.0
        elif industry_rank <= 50:
            industry_score = 9.0
        elif industry_rank <= 75:
            industry_score = 6.0
        elif industry_rank <= 100:
            industry_score = 3.0
        else:
            industry_score = 0.0
        total_score += industry_score
        details['industry_score'] = industry_score
        
        # 3. RS Rating (15 pts)
        rs_score = (rs_rating / 100.0) * 15.0
        total_score += rs_score
        details['rs_score'] = rs_score
        
        # 4. VCP Tightness WITH CONTRACTION BONUS (15 pts)
        base_vcp_points = min((vcp_score / 10.0) * 10.0, 10.0)
        
        if vcp_score >= 8.0 and is_contracting:
            tightness_bonus = 5.0
        elif vcp_score >= 7.0 and is_contracting:
            tightness_bonus = 4.0
        elif vcp_score >= 7.0:
            tightness_bonus = 2.0
        elif vcp_score >= 6.0 and is_contracting:
            tightness_bonus = 3.0
        elif vcp_score >= 6.0:
            tightness_bonus = 1.0
        else:
            tightness_bonus = 0.0
        
        vcp_total = min(base_vcp_points + tightness_bonus, 15.0)
        total_score += vcp_total
        details['vcp_points'] = vcp_total
        details['vcp_tightness_bonus'] = tightness_bonus
        details['is_contracting'] = is_contracting
        
        # 5. ATR Distance from 50MA (10 pts)
        current_price = stock.get('current_price', 0)
        ma_50 = stock.get('ma_50', 0)
        adr = stock.get('adr', 0)
        
        if current_price > 0 and ma_50 > 0 and adr > 0:
            atr = (current_price * adr) / 100.0
            price_distance = current_price - ma_50
            atr_distance = price_distance / atr if atr > 0 else 0
            
            if -1 <= atr_distance <= 2:
                pivot_score = 10.0
            elif 2 < atr_distance <= 3:
                pivot_score = 9.0
            elif 3 < atr_distance <= 5:
                pivot_score = 7.0
            elif 5 < atr_distance <= 7:
                pivot_score = 5.0
            elif 7 < atr_distance <= 10:
                pivot_score = 3.0
            elif -3 <= atr_distance < -1:
                pivot_score = 8.0
            elif atr_distance > 10:
                pivot_score = 0.0
            else:
                pivot_score = 0.0
        else:
            atr_distance = 0
            pivot_score = 0.0
        
        total_score += pivot_score
        details['pivot_score'] = pivot_score
        details['atr_distance'] = round(atr_distance, 1)
        
        # 6. YTD Performance (10 pts)
        if ytd_return >= 30:
            ytd_score = 10.0
        elif ytd_return >= 20:
            ytd_score = 8.0
        elif ytd_return >= 15:
            ytd_score = 6.0
        elif ytd_return >= 10:
            ytd_score = 4.0
        elif ytd_return >= 5:
            ytd_score = 2.0
        else:
            ytd_score = 0.0
        total_score += ytd_score
        details['ytd_score'] = ytd_score
        
        # 7. Volume (10 pts)
        if volume_ratio >= 1.5:
            volume_score = 10.0
        elif volume_ratio >= 1.2:
            volume_score = 8.0
        elif volume_ratio >= 1.0:
            volume_score = 6.0
        elif volume_ratio >= 0.8:
            volume_score = 4.0
        else:
            volume_score = 2.0
        total_score += volume_score
        details['volume_score'] = volume_score
        
        # 8. STOP QUALITY (15 pts)
        self._calculate_context_based_stop(stock)
        
        stop_pct = stock.get('stop_loss_pct')
        
        # No valid stop earns no stop-quality points (999 falls through to 0 below)
        # A stop tighter than STOP_PCT_MIN gets shaken out by noise, so it earns no points either
        if stop_pct is not None and stop_pct >= STOP_PCT_MIN and adr > 0:
            stop_adr_ratio = stop_pct / adr
        else:
            stop_adr_ratio = 999
        
        if stop_adr_ratio <= 0.5:
            base_score = 15.0
        elif stop_adr_ratio <= 0.6:
            base_score = 14.0
        elif stop_adr_ratio <= 0.75:
            base_score = 12.0
        elif stop_adr_ratio <= 1.0:
            base_score = 9.0
        elif stop_adr_ratio <= 1.5:
            base_score = 6.0
        elif stop_adr_ratio <= 2.0:
            base_score = 3.0
        else:
            base_score = 0.0
        
        if stop_pct is not None and stop_pct > 8.0:
            if eps_yoy >= 100 and industry_rank <= 10:
                adjustment = -2.0
            else:
                adjustment = -5.0
        else:
            adjustment = 0.0
        
        stop_quality_total = max(0, min(base_score + adjustment, 15.0))
        
        total_score += stop_quality_total
        details['stop_quality_score'] = stop_quality_total
        details['stop_adr_ratio'] = round(stop_adr_ratio, 2) if stop_pct is not None and adr > 0 else None
        details['stop_loss_pct'] = stop_pct
        
        is_power_stock = (eps_yoy >= 200 and rs_rating >= 95 and ytd_return >= 40)
        stock['is_power_stock'] = is_power_stock
        details['is_power_stock'] = is_power_stock
        
        stock['priority_rating'] = round(total_score, 1)
        stock['priority_details'] = details
        
        return round(total_score, 1)
    
    def _display_price_action_analysis(self, stock):
        """Display detailed price action breakdown"""
        symbol = stock['symbol']
        price_action = stock.get('price_action')
        
        if not price_action:
            return
        
        print("\n" + "="*100)
        print(f"📊 PRICE ACTION ANALYSIS: {symbol}")
        print("="*100 + "\n")
        
        print("🏗️  BASE STRUCTURE:")
        print(f"   • Base High: ${price_action['base_high']:.2f} ({price_action['base_high_date']})")
        print(f"   • Base Low:  ${price_action['base_low']:.2f}")
        print(f"   • Base Depth: {price_action['base_depth_pct']:.1f}%")
        print(f"   • Distance from High: {price_action['distance_from_base_high_pct']:+.1f}%")
        
        print("\n📉 VOLATILITY CONTRACTION (VCP Validation):")
        print(f"   • Period 1 (oldest):  {price_action['period1_volatility']:.2f}% avg daily range")
        print(f"   • Period 2 (middle):  {price_action['period2_volatility']:.2f}% avg daily range")
        print(f"   • Period 3 (recent):  {price_action['period3_volatility']:.2f}% avg daily range")
        print(f"   • Contraction 1→2: {price_action['contraction_1_to_2_pct']:+.1f}%")
        print(f"   • Contraction 2→3: {price_action['contraction_2_to_3_pct']:+.1f}%")
        print(f"   • Total Contraction: {price_action['contraction_1_to_3_pct']:+.1f}%")
        
        if price_action['is_contracting']:
            print("   ✅ STATUS: VOLATILITY CONTRACTING (True VCP!)")
        else:
            print("   ⚠️  STATUS: NOT CONTRACTING (Needs more time!)")
        
        print("\n🕯️  LAST 10 DAILY CANDLES:")
        print(f"   {'Date':<12} {'Open':<8} {'High':<8} {'Low':<8} {'Close':<8} {'Change%':<9} {'Range%':<8}")
        print("   " + "-"*90)
        
        for candle in price_action['last_10_candles']:
            print(f"   {candle['date']:<12} ${candle['open']:<7.2f} ${candle['high']:<7.2f} "
                  f"${candle['low']:<7.2f} ${candle['close']:<7.2f} {candle['change_pct']:>+7.2f}% "
                  f"{candle['range_pct']:>6.2f}%")
        
        print("\n🛑 STOP LOSS OPTIONS:")
        print(f"   • Yesterday's Low:  ${price_action['yesterday_low']:.2f} "
              f"({((price_action['current_price'] - price_action['yesterday_low']) / price_action['current_price'] * 100):.1f}% risk)")
        print(f"   • 3-Day Low:        ${price_action['swing_low_3d']:.2f} "
              f"({((price_action['current_price'] - price_action['swing_low_3d']) / price_action['current_price'] * 100):.1f}% risk)")
        print(f"   • 5-Day Low:        ${price_action['swing_low_5d']:.2f} "
              f"({((price_action['current_price'] - price_action['swing_low_5d']) / price_action['current_price'] * 100):.1f}% risk)")
        print(f"   • 7-Day Low:        ${price_action['swing_low_7d']:.2f} "
              f"({((price_action['current_price'] - price_action['swing_low_7d']) / price_action['current_price'] * 100):.1f}% risk)")
        print(f"   • 10-Day Low:       ${price_action['swing_low_10d']:.2f} "
              f"({((price_action['current_price'] - price_action['swing_low_10d']) / price_action['current_price'] * 100):.1f}% risk)")
        
        print("\n✅ RECOMMENDED STOP:")
        print(f"   • Price: ${stock['stop_loss']:.2f}")
        print(f"   • Risk: {stock['stop_loss_pct']:.1f}%")
        print(f"   • Context: {stock['stop_context']}")
        print(f"   • {stock['stop_description']}")
        print(f"   • Status: {stock['stop_warning']}")
        
        print("\n" + "="*100 + "\n")
    
    # ─────────────────────────────────────────────────────────────────
    # NEW: Normalize industry/sector from universe metadata
    # ─────────────────────────────────────────────────────────────────
    def _normalize_industry_from_meta(self, symbol: str) -> str:
        """Industry group name for symbol: universe metadata first (robust key lookup), then the
        industry.csv subcategory. The IBD export carries only the numeric group rank, so in practice
        the name comes from industry.csv. Manual-watchlist symbols (no metadata) get it too."""
        meta = self.universe_obj.metadata.get(symbol, {}) if self.universe_obj else {}
        # Try common keys used across sources
        for key in ('Industry_Group', 'industry_group', 'industry', 'Industry', 'Industry_Group_Name', 'Industry Group'):
            val = meta.get(key)
            if val and not (isinstance(val, float) and pd.isna(val)):
                return str(val).strip()
        return load_industry_map().get(str(symbol).upper(), '')
    
    def get_stock_universe(self):
        """Stocks to scan: the configured universe minus known-dead / renamed / quarantined
        tickers (symbol_hygiene; every mode goes through here). Dropped names are logged."""
        symbols = self._raw_stock_universe()
        if not symbols:
            return symbols
        # Weekly re-probe of quarantined names, full scans only (the watcher never writes state)
        if getattr(self, '_scan_type', None) == 'afterhours' and self.connector is not None:
            try:
                state = symbol_hygiene.load_health()
                lifted = symbol_hygiene.weekly_reprobe(state, self.connector, datetime.now().date())
                symbol_hygiene.save_health(state)
                if lifted:
                    logger.info(f"✓ Re-probe lifted quarantine: {', '.join(lifted)}")
            except Exception as e:                               # noqa: BLE001
                logger.warning(f"Symbol re-probe skipped: {e}")
        try:
            kept, dropped = symbol_hygiene.apply_symbol_hygiene(symbols)
        except Exception as e:                                   # noqa: BLE001
            logger.warning(f"⚠️ Symbol hygiene failed ({e}); scanning the full universe")
            return symbols
        self.hygiene_dropped = dropped
        if dropped:
            logger.info(f"✓ Skipped {len(dropped)} known dead/renamed/quarantined symbols: "
                        f"{', '.join(sym for sym, _ in dropped)}")
        return kept

    def _raw_stock_universe(self):
        """
        Get stocks to scan.
        
        v6: Now uses UniverseBuilder by default to pull from IBD xlsx + TradingView txt files.
        Falls back to STAGE2_WATCHLIST if no data files found or if SCAN_UNIVERSE is set
        to a legacy value.
        """
        # ═══ v6: Try UniverseBuilder first if SCAN_UNIVERSE is "DYNAMIC" or unset ═══
        if SCAN_UNIVERSE == "DYNAMIC" or SCAN_UNIVERSE == "IBD_UNIVERSE":
            self.active_universe_mode = SCAN_UNIVERSE
            return self._build_dynamic_universe()
        
        # Legacy modes
        logger.info(f"Building stock universe: {SCAN_UNIVERSE}")
        
        if SCAN_UNIVERSE == "CUSTOM":
            self.active_universe_mode = "CUSTOM"
            self.dynamic_universe_symbols = set()
            self.manual_watchlist_symbols = set()
            return CUSTOM_SYMBOLS
        elif SCAN_UNIVERSE == "STAGE2_WATCHLIST":
            self.active_universe_mode = "STAGE2_WATCHLIST"
            self.dynamic_universe_symbols = set()
            self.manual_watchlist_symbols = set(STAGE2_WATCHLIST)
            logger.info(f"✓ Stage 2 Watchlist: {len(STAGE2_WATCHLIST)} stocks")
            return STAGE2_WATCHLIST
        elif SCAN_UNIVERSE == "COMBINED":
            self.active_universe_mode = "COMBINED"
            us_all = self._fetch_all_us_stocks()
            combined = list(set(us_all + STAGE2_WATCHLIST))
            self.dynamic_universe_symbols = set(us_all)
            self.manual_watchlist_symbols = set(STAGE2_WATCHLIST)
            logger.info(f"✓ Combined: {len(combined)} stocks")
            return combined
        elif SCAN_UNIVERSE == "US_ALL":
            self.active_universe_mode = "US_ALL"
            symbols = self._fetch_all_us_stocks()
            self.dynamic_universe_symbols = set(symbols)
            self.manual_watchlist_symbols = set()
            logger.info(f"✓ US Universe: {len(symbols)} stocks")
            return symbols
        elif SCAN_UNIVERSE == "IBKR_PREFILTER":
            self.active_universe_mode = "IBKR_PREFILTER"
            symbols = self._fetch_all_us_stocks()
            self.dynamic_universe_symbols = set(symbols)
            self.manual_watchlist_symbols = set(STAGE2_WATCHLIST)
            merged = sorted(set(symbols) | self.manual_watchlist_symbols)
            logger.info(f"✓ IBKR prefilter seed: {len(symbols)} + manual Stage2: {len(self.manual_watchlist_symbols)}")
            logger.info(f"✓ Total to scan: {len(merged)} stocks")
            return merged
        else:
            # Default: try dynamic, fall back to STAGE2 if no data files
            self.active_universe_mode = "AUTO_DYNAMIC_FALLBACK"
            return self._build_dynamic_universe()
    
    def _build_dynamic_universe(self):
        """
        Build universe from IBD xlsx + TradingView text files.
        Falls back to STAGE2_WATCHLIST if no data files found.
        """
        try:
            from universe_builder import UniverseBuilder
        except ImportError:
            logger.warning("⚠️ universe_builder.py not found — falling back to STAGE2_WATCHLIST")
            return STAGE2_WATCHLIST
        
        logger.info("Building dynamic universe from IBD + TradingView sources...")
        
        try:
            builder = UniverseBuilder(data_dir='~/swing-trading-bot/Data')
            self.universe_obj = builder.build()
            self.dynamic_universe_symbols = set(self.universe_obj.symbols)
        except Exception as e:
            logger.error(f"❌ Universe builder error: {e}")
            logger.warning("⚠️ Falling back to STAGE2_WATCHLIST")
            self.dynamic_universe_symbols = set()
            self.manual_watchlist_symbols = set(STAGE2_WATCHLIST)
            return STAGE2_WATCHLIST
        
        if not self.universe_obj.symbols:
            logger.warning("⚠️ Universe builder returned 0 symbols — falling back to STAGE2_WATCHLIST")
            logger.warning("   (Check that ~/swing-trading-bot/Data/ contains IBD xlsx files)")
            self.dynamic_universe_symbols = set()
            self.manual_watchlist_symbols = set(STAGE2_WATCHLIST)
            return STAGE2_WATCHLIST
        
        # Merge with STAGE2_WATCHLIST as a safety net (your manual additions still respected)
        self.manual_watchlist_symbols = set(STAGE2_WATCHLIST)
        merged = sorted(self.dynamic_universe_symbols | self.manual_watchlist_symbols)
        
        logger.info(f"✓ Dynamic universe: {len(self.universe_obj.symbols)} from IBD/TV "
                    f"+ {len(set(STAGE2_WATCHLIST) - set(self.universe_obj.symbols))} extra from STAGE2_WATCHLIST")
        logger.info(f"✓ Total to scan: {len(merged)} stocks")
        
        return merged
    
    def _fetch_all_us_stocks(self):
        """Major US momentum stocks"""
        return [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO", "ORCL",
            "ADBE", "CRM", "CSCO", "INTC", "AMD", "QCOM", "TXN", "AMAT", "LRCX", "KLAC",
            "TSM", "ASML", "MU", "MRVL", "ON", "MPWR", "NXPI", "ADI", "MCHP", "ARM",
            "NOW", "PANW", "SNOW", "CRWD", "ZS", "DDOG", "NET", "PLTR", "COIN",
            "JPM", "BAC", "WFC", "GS", "MS", "V", "MA", "PYPL",
            "LLY", "JNJ", "UNH", "ABBV", "MRK", "TMO", "ISRG", "MRNA",
            "WMT", "HD", "COST", "NKE", "SBUX", "MCD", "LULU", "DECK",
            "CAT", "BA", "HON", "UPS", "RTX", "LMT", "GE", "DE", "ETN",
            "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO",
            "NFLX", "DIS", "CMCSA", "T", "VZ", "TMUS", "UBER", "ABNB",
        ]
    
    def run_full_scan(self, scan_type="afterhours"):
        """Execute complete scan"""
        self.scan_timestamp = datetime.now()
        self.scan_issues = self._empty_scan_issues()
        self.fetch_status = {}            # symbol -> ibkr fetch status (ok / no_secdef / timeout ...)
        self.hygiene_dropped = []
        self.scan_aborted = False
        self._scan_type = scan_type
        logger.info(f"\n{'='*70}")
        logger.info(f"STARTING {scan_type.upper()} SCAN: {self.scan_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"{'='*70}\n")
        
        # ════════════════════════════════════════════════════════
        # ═══ NEW (v2): MARKET REGIME CHECK (GATE ALL SIGNALS) ═══
        # ════════════════════════════════════════════════════════
        # The watcher rescans every 30s-5min: detect the regime once per day there, not each time
        # (SPY/QQQ refetches; the regime can't usefully change intraday). Review LT-02.
        pinned = (scan_type == "watch" and self.market_regime is not None
                  and self.market_regime.timestamp.date() == self.scan_timestamp.date())
        if not pinned:
            logger.info("📊 Detecting market regime...")
            self._set_regime_gating()
        
        start_time = time.time()
        
        symbols = self.get_stock_universe()
        if not symbols:
            logger.error("❌ No symbols")
            return False
        
        logger.info(f"Scanning {len(symbols)} stocks...\n")
        if scan_type != "watch":  # the watcher rescans its small watchlist all day
            self._save_universe_snapshot(symbols)
        
        filtered_stocks = self._parallel_filter_stocks(symbols)
        abort_reason = self._data_integrity_failure(len(symbols))
        if abort_reason:
            # A dropped socket or a farm outage mid-scan times out every remaining symbol; the
            # survivors would otherwise be written as a COMPLETE scan of the whole universe.
            logger.error(f"❌ SCAN ABORTED: {abort_reason}; no scan file written")
            self._log_scan_health(len(symbols), len(filtered_stocks))
            self.scan_aborted = True
            return False
        self._record_symbol_health(scan_type)
        
        if not filtered_stocks:
            logger.warning("⚠️ No stocks passed filters")
            self._log_scan_health(len(symbols), 0)
            return False
        
        logger.info(f"\n✓ {len(filtered_stocks)} stocks passed filters")
        
        logger.info("\nCalculating RS ratings...")
        stocks_with_rs = self._calculate_rs_for_stocks(filtered_stocks)
        self._apply_ibd_universe(stocks_with_rs)
        
        logger.info("\nDetecting VCP patterns...")
        stocks_with_vcp = self._detect_vcp_for_stocks(stocks_with_rs)
        
        logger.info("\nAnalyzing price action...")
        stocks_with_price_action = self._analyze_price_action_for_stocks(stocks_with_vcp)
        
        self._apply_tier_filters(stocks_with_price_action)
        
        if ENABLE_PRIORITY_RATING:
            logger.info("\nCalculating priority ratings...")
            self._calculate_priority_ratings_for_all_tiers()
        
        self._apply_stop_filter_to_tiers()
        # Base counts don't change intraday; skip the 2y fetches in the watcher's frequent rescans
        if scan_type != "watch":
            self._count_bases_for_finalists()
        
        if DEBUG_MODE:
            self._display_debug_info()
        
        elapsed = time.time() - start_time
        logger.info(f"\n{'='*70}")
        logger.info("✓ SCAN COMPLETE!")
        logger.info(f"✓ TIER_ELITE_BUY:  {len(self.elite_buy_results)} stocks ⚡ (actionable)")
        logger.info(f"✓ TIER_ELITE_HOLD: {len(self.elite_hold_results)} stocks 📌 (track only)")
        logger.info(f"✓ TIER_BUY:        {len(self.buy_results)} stocks 🎯")
        logger.info(f"✓ TIER_RS_LEADER:  {len(self.rs_leader_results)} stocks 🏃 (half size)")
        logger.info(f"✓ TIER_WATCH:      {len(self.watch_results)} stocks 📋")
        logger.info(f"✓ TIER_EXTENDED:   {len(self.extended_results)} stocks 🚀")
        logger.info(f"✓ Time: {elapsed:.1f}s")
        
        # ═══ NEW (v2): Show regime status in summary ═══
        if self.market_regime:
            logger.info(f"✓ Market Regime: {self.market_regime.status}")
            if self.watch_only_mode:
                total_actionable = len(self.elite_buy_results)
                logger.info(f"✓ Actionable (Elite buy only): {total_actionable}")
        
        logger.info(f"{'='*70}\n")
        
        # Watch rescans write no CSVs and draw no dashboard: they only feed the watcher's alerts
        # (they used to leave ~850 *_watch_* files and a 7.7 MB log). Review LT-02.
        self._log_scan_health(len(symbols), len(filtered_stocks))
        if scan_type == "watch":
            return True
        self._export_results(scan_type)
        self._render_results(scan_type)
        
        return True
    
    def _data_integrity_failure(self, total: int) -> Optional[str]:
        """Why this scan's data pull can't be trusted, or None. Checks the TWS socket, then the
        count of symbols that failed for transient/session reasons (timeouts, a session conflict)."""
        conn = self.connector
        if conn is not None and hasattr(conn, 'isConnected') and not conn.isConnected():
            return f"TWS connection lost during the scan ({total} symbols requested)"
        session = len(self.scan_issues.get('session', []))
        failed = len(self.scan_issues.get('timeout', [])) + session
        limit = max(SCAN_MAX_FAILED_MIN, SCAN_MAX_FAILED_FRACTION * total)
        if session and session >= min(limit, max(1, total // 2)):
            # Also catches the watcher's small watchlist, which the 20-symbol floor would miss
            return (f"{session} of {total} symbols blocked: TWS trading session is connected from a "
                    f"different IP (Error 162); log the other session out")
        if total and failed > limit:
            return (f"{failed} of {total} symbols timed out or were blocked "
                    f"(limit {limit:.0f}; data farm: {conn.farm_status_summary() if conn else 'n/a'})")
        return None

    def _record_symbol_health(self, scan_type):
        """Feed this scan's Error-200 results to the quarantine counter (full scans only)."""
        self.secdef_days = {}
        try:
            state = symbol_hygiene.load_health()
            if scan_type == 'afterhours':
                new_q, lifted = symbol_hygiene.record_scan_results(
                    state, dict(getattr(self, 'fetch_status', {})), self.scan_timestamp.date(),
                    ibd_symbols=self.ibd_universe.keys(), ibd_date=self.ibd_universe_date,
                    protected=symbol_hygiene.protected_symbols())
                symbol_hygiene.save_health(state)
                if new_q:
                    logger.warning(f"⚠️ Auto-quarantined (no security definition on "
                                   f"{symbol_hygiene.QUARANTINE_DAYS} scan days): {', '.join(new_q)}")
                if lifted:
                    logger.info(f"✓ Quarantine lifted: {', '.join(lifted)}")
            self.secdef_days = symbol_hygiene.secdef_days(state)
        except Exception as e:                                   # noqa: BLE001
            logger.warning(f"Symbol health not updated: {e}")

    # Stages that mean data or code failed (short history is a normal filter; timeouts are
    # transient and listed apart, so a busy watcher doesn't raise false "delisted" alarms)
    _ERROR_STAGES = ('no_data', 'session', 'filter_error', 'rs', 'vcp', 'price_action', 'base_count')
    _STAGE_LABELS = {
        'no_data': 'No data from IBKR',
        'timeout': 'IBKR timeouts (transient)',
        'session': 'Blocked: TWS session connected from a different IP (Error 162)',
        'short_history': 'Under 250 bars (new listing or partial data)',
        'filter_error': 'Errors in base filter',
        'rs': 'Errors in RS rating',
        'vcp': 'Errors in VCP detection',
        'price_action': 'Errors in price action',
        'base_count': 'Errors in base count',
        'pocket_pivot': 'Errors in pocket-pivot flag (stock kept, flag blank)',
    }

    @staticmethod
    def _empty_scan_issues():
        return {stage: [] for stage in MomentumScanner._STAGE_LABELS}

    def _record_issue(self, stage, symbol, reason):
        """Track a per-symbol problem for the scan summary (list.append is thread-safe)."""
        # setdefault: an unlisted stage must never raise here and drop the stock (review S1)
        self.scan_issues.setdefault(stage, []).append((symbol, reason))

    def _log_scan_health(self, scanned, passed):
        """Report at INFO how many symbols were analyzed and what failed, by stage."""
        logger.info(f"\n{'='*70}")
        self.scan_counts = {'scanned': scanned, 'passed': passed}
        logger.info("DATA HEALTH")
        logger.info(f"  Scanned: {scanned} | Passed base filters: {passed}")
        any_errors = False
        for stage, label in self._STAGE_LABELS.items():
            issues = self.scan_issues[stage]
            if not issues:
                continue
            examples = ", ".join(sym for sym, _ in issues[:5])
            more = f" (+{len(issues) - 5} more)" if len(issues) > 5 else ""
            if stage in self._ERROR_STAGES:
                any_errors = True
                logger.warning(f"  ⚠️ {label}: {len(issues)} - {examples}{more}")
                logger.warning(f"     first error: {issues[0][1][:100]}")
            else:
                logger.info(f"  {label}: {len(issues)} - {examples}{more}")
        dropped = getattr(self, 'hygiene_dropped', [])
        if dropped:
            logger.info(f"  Skipped by symbol hygiene (symbol_overrides.csv / quarantine): {len(dropped)} - "
                        f"{', '.join(sym for sym, _ in dropped)}")
        if not any_errors:
            logger.info("  ✓ No data errors")
        logger.info(f"{'='*70}\n")

    def _render_results(self, scan_type):
        """Show the end-of-scan view: rich panels if available, else the old text tables."""
        try:
            from scan_display import render_scan
        except ImportError as e:
            logger.warning(f"Rich display unavailable ({e}); using plain text view")
            self._display_results()
            self._display_top_price_action_analysis()
            return
        render_scan(self, scan_type=scan_type, export_prefix=getattr(self, 'export_prefix', None))

    def _set_regime_gating(self):
        """
        Detect the market regime and set watch-only mode. Fails closed: if the regime
        can't be determined, assume the worst (watch-only) rather than open every gate.
        """
        try:
            self.market_regime = self.regime_detector.detect_regime()
        except Exception as e:
            logger.error(f"❌ Regime detection error: {e}")
            self.market_regime = None

        if self.market_regime:
            self.regime_detector.display_regime(self.market_regime)
            self.watch_only_mode = not self.market_regime.should_take_new_positions
            if self.watch_only_mode:
                logger.warning("🔴 WATCH-ONLY MODE ACTIVE")
                logger.warning("   Only elite leaders will be flagged as actionable")
                logger.warning("   Regular buy list will NOT be generated")
                logger.warning("   Scanner will still analyze all stocks for tracking\n")
        else:
            logger.warning("⚠️ Regime detection failed — WATCH-ONLY as a precaution (only elite leaders actionable)")
            self.watch_only_mode = True

    def _save_universe_snapshot(self, symbols):
        """
        Point-in-time record of what this scan could see (Data/snapshots/universe_YYYY-MM-DD.csv).
        Backtests that only use today's lists over-weight stocks that kept working; dated
        snapshots let a future backtest trade only names that were really on the lists.
        """
        try:
            snap_dir = Path(OUTPUT_DIR).expanduser().parent / "Data" / "snapshots"
            snap_dir.mkdir(parents=True, exist_ok=True)
            uni = self.universe_obj
            rows = [{'symbol': sym,
                     'sources': ";".join(uni.get_sources(sym)) if uni else "",
                     'in_manual_watchlist': sym in self.manual_watchlist_symbols}
                    for sym in sorted(set(symbols))]
            path = snap_dir / f"universe_{self.scan_timestamp:%Y-%m-%d}.csv"
            pd.DataFrame(rows).to_csv(path, index=False)
            logger.info(f"✓ Universe snapshot: {path}")
        except Exception as e:
            logger.warning(f"⚠️ Universe snapshot not saved: {e}")

    def _rs_leader_check(self, stock, atr_multiple):
        """
        (qualifies, why not) for the RS Leader tier. why-not is None when the stock
        isn't an RS-leader candidate at all (RS under the bar, no IBD data, or no VCP).
        """
        ibd_rs = stock.get('ibd_rs')
        if ibd_rs is None or ibd_rs < RS_LEADER_MIN_RS or not stock.get('has_real_vcp'):
            return False, None
        acc_dist = stock.get('ibd_acc_dist') or ''
        pivot_dist = stock.get('vcp_pivot_distance_pct') or 0
        stop_pct = stock.get('stop_loss_pct')
        if acc_dist[:1] in ('D', 'E'):
            return False, f"A/D {acc_dist} (institutions selling)"
        if atr_multiple > ATR_BUY_CAP:
            return False, f"extended ({atr_multiple:.1f} ATR from 50MA)"
        if pivot_dist > RS_LEADER_MAX_PIVOT_DIST_PCT:
            return False, f"{pivot_dist:.0f}% past pivot"
        if pivot_dist < -RS_LEADER_MAX_PIVOT_DIST_PCT:
            return False, f"{-pivot_dist:.0f}% below pivot"
        if stop_pct is None:
            return False, "no valid stop"
        if stop_pct < STOP_PCT_MIN:
            return False, f"stop too tight ({stop_pct:.1f}%)"
        if stop_pct > RS_LEADER_MAX_STOP_PCT:
            return False, f"stop {stop_pct:.1f}% > {RS_LEADER_MAX_STOP_PCT:g}%"
        return True, None

    def _apply_ibd_universe(self, stocks):
        """
        Attach IBD ratings and use IBD's RS Rating in place of the homegrown one.
        The homegrown value disagrees with IBD's, usually lower (median 71 vs 94 on
        2026-10-22), so it's kept as rs_rating_calc for reference only.
        """
        if not self.ibd_universe:
            return
        covered = 0
        for stock in stocks:
            ratings = self.ibd_universe.get(stock['symbol'].upper())
            if not ratings:
                continue
            covered += 1
            stock.update(ratings)
            if ratings.get('ibd_rs') is not None:
                stock['rs_rating_calc'] = stock.get('rs_rating')
                stock['rs_rating'] = stock['ibd_rs_rating_real'] = ratings['ibd_rs']
        logger.info(f"✓ IBD universe covers {covered}/{len(stocks)} stocks")

    def _count_bases_for_finalists(self):
        """Base count needs ~2 years of bars, so fetch them only for names on the final lists."""
        finalists = (self.elite_buy_results + self.elite_hold_results + self.buy_results
                     + self.rs_leader_results + self.watch_results + self.extended_results)
        if not finalists:
            return
        logger.info(f"\nCounting bases for {len(finalists)} finalists (2y history)...")
        for stock in finalists:
            try:
                bars = self.connector.fetch_historical_data(stock['symbol'], duration="2 Y", bar_size="1 day")
                if not bars:
                    self._record_issue('base_count', stock['symbol'], 'no 2y bars returned')
                    continue
                stock.update(count_bases(self._bars_to_dataframe(bars)))
            except Exception as e:
                self._record_issue('base_count', stock['symbol'], str(e))
            time.sleep(REQUEST_DELAY)

    def _analyze_price_action_for_stocks(self, stocks):
        """Add price action analysis to all stocks"""
        for stock in stocks:
            try:
                price_action = self._analyze_price_action(stock)
                stock['price_action'] = price_action
            except Exception as e:
                self._record_issue('price_action', stock['symbol'], str(e))
                logger.debug(f"Price action error {stock['symbol']}: {e}")
                stock['price_action'] = None
        return stocks
    
    def _apply_stop_filter_to_tiers(self):
        """Move wide-stop BUY stocks to WATCH (elite stays put with warning)"""
        filtered = []
        moved = []
        for stock in self.buy_results:
            if stock.get('stop_status') == 'WATCHLIST':
                stock['moved_reason'] = f"Stop too wide: {stock['stop_loss_pct']:.1f}%"
                moved.append(stock)
            else:
                filtered.append(stock)
        
        self.buy_results = filtered
        self.watch_results.extend(moved)
        
        if moved:
            logger.info(f"⚠️ Moved {len(moved)} BUY → WATCH (stops too wide)")
    
    def _display_top_price_action_analysis(self):
        """Show detailed price action for top 3 actionable stocks (v5)"""
        all_buyable = self.elite_buy_results + self.buy_results
        
        if not all_buyable:
            return
        
        if self.watch_only_mode:
            if not self.elite_buy_results:
                return
            all_buyable = self.elite_buy_results
            print("\n" + "="*100)
            print("        ⚡ DETAILED PRICE ACTION - TOP 3 ELITE BUY ENTRIES ⚡        ")
            print("="*100)
        else:
            print("\n" + "="*100)
            print("        📊 DETAILED PRICE ACTION - TOP 3 BUYS 📊        ")
            print("="*100)
        
        all_buyable.sort(key=lambda x: x.get('priority_rating', 0), reverse=True)
        
        for stock in all_buyable[:3]:
            self._display_price_action_analysis(stock)
    
    def _parallel_filter_stocks(self, symbols):
        """Parallel base filtering"""
        filtered_stocks = []
        total = len(symbols)
        processed = 0
        
        logger.info("Applying base filters...")
        
        # At most SCAN_WORKERS_MAX requests in flight: IBKR paces historical-data requests, and 50
        # threads mostly queued behind each other (review S4)
        with ThreadPoolExecutor(max_workers=min(MAX_CONCURRENT_REQUESTS, SCAN_WORKERS_MAX)) as executor:
            future_to_symbol = {
                executor.submit(self._filter_single_stock, symbol): symbol 
                for symbol in symbols
            }
            
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                processed += 1
                
                try:
                    result = future.result()
                    if result:
                        filtered_stocks.append(result)
                    
                    if processed % 20 == 0:
                        logger.info(f"  Progress: {processed}/{total} ({len(filtered_stocks)} passed)")
                
                except Exception as e:
                    self._record_issue('filter_error', symbol, str(e))
                    logger.debug(f"Error {symbol}: {e}")

        return filtered_stocks

    def _filter_single_stock(self, symbol):
        """Apply base filters to single stock"""
        is_debug = symbol in DEBUG_SYMBOLS
        failures = []
        
        try:
            bars, status = self.connector.fetch_historical_data_ex(symbol, duration="1 Y", bar_size="1 day")
            self.fetch_status[symbol] = 'ok' if bars else status
            
            if not bars:
                # Keep IBKR's reason: a dead ticker (Error 200) and a slow reply need different fixes
                if status == 'timeout':
                    self._record_issue('timeout', symbol, 'IBKR timeout')
                elif status == 'session_conflict':
                    # Session-level block, not a fact about this ticker: counts toward the abort gate
                    self._record_issue('session', symbol, STATUS_REASONS[status])
                else:
                    self._record_issue('no_data', symbol, STATUS_REASONS.get(status, status))
                return None
            if len(bars) < 250:
                self._record_issue('short_history', symbol, f"{len(bars)} bars")
                if is_debug:
                    failures.append(f"Insufficient data: {len(bars)}")
                return None
            
            df = self._bars_to_dataframe(bars)
            current_price = df['close'].iloc[-1]
            
            if current_price < MIN_PRICE or current_price > MAX_PRICE:
                if is_debug:
                    failures.append(f"Price ${current_price:.2f} out of range")
                return None
            
            avg_volume = df['volume'].mean()
            if avg_volume < MIN_AVG_VOLUME:
                if is_debug:
                    failures.append(f"Volume {avg_volume:.0f} too low")
                return None
            
            monthly_turnover = current_price * avg_volume * 21
            if monthly_turnover < MIN_MONTHLY_TURNOVER:
                if is_debug:
                    failures.append(f"Turnover too low")
                return None
            
            ma_50 = df['close'].rolling(50).mean().iloc[-1]
            ma_200 = df['close'].rolling(200).mean().iloc[-1]
            ema_10 = df['close'].ewm(span=10, adjust=False).mean().iloc[-1]
            ema_21 = df['close'].ewm(span=21, adjust=False).mean().iloc[-1]
            
            distance_from_50ma = ((current_price - ma_50) / ma_50) * 100
            distance_from_200ma = ((current_price - ma_200) / ma_200) * 100
            distance_from_10ema = ((current_price - ema_10) / ema_10) * 100
            distance_from_21ema = ((current_price - ema_21) / ema_21) * 100
            days_above_21ema = days_above_ema(df['close'], df['close'].ewm(span=21, adjust=False).mean())
            # Pocket pivot on the latest bar (flag only, see pocket_pivot.py)
            try:
                pp_info = pocket_pivot.evaluate(add_indicators(df))
            except Exception as e:
                self._record_issue('pocket_pivot', symbol, str(e))
                pp_info = {}
            trail_ma = exit_rules.respected_ma(df)   # the MA this stock respects, for the exit plan
            
            if REQUIRE_ABOVE_50MA and current_price < ma_50 and distance_from_50ma < 20.0:
                if is_debug:
                    failures.append("Below 50MA")
                return None
            
            if REQUIRE_ABOVE_200MA and current_price < ma_200:
                if is_debug:
                    failures.append("Below 200MA")
                return None
            
            week_52_high = df['high'].iloc[-252:].max() if len(df) >= 252 else df['high'].max()
            week_52_low = df['low'].iloc[-252:].min() if len(df) >= 252 else df['low'].min()
            
            distance_from_52w_high = ((current_price - week_52_high) / week_52_high) * 100
            distance_from_52w_low = ((current_price - week_52_low) / week_52_low) * 100
            
            if distance_from_52w_low < MIN_DISTANCE_FROM_52W_LOW:
                if is_debug:
                    failures.append("Too close to 52W low")
                return None
            
            if distance_from_52w_high < -MAX_DISTANCE_FROM_52W_HIGH:
                if is_debug:
                    failures.append("Too far from 52W high")
                return None
            
            df['daily_range'] = ((df['high'] - df['low']) / df['close']) * 100
            adr = df['daily_range'].rolling(14).mean().iloc[-1]
            
            year_start = datetime(self.scan_timestamp.year, 1, 1)
            year_start_idx = df[df['date'] >= year_start].index[0] if any(df['date'] >= year_start) else 0
            ytd_return = ((df['close'].iloc[-1] / df['close'].iloc[year_start_idx]) - 1) * 100 if year_start_idx < len(df) else 0
            
            volume_50d_avg = df['volume'].iloc[-50:].mean()
            volume_ratio = avg_volume / volume_50d_avg if volume_50d_avg > 0 else 1.0
            
            price_change_1m = ((df['close'].iloc[-1] / df['close'].iloc[-21]) - 1) * 100 if len(df) >= 21 else 0
            price_change_3m = ((df['close'].iloc[-1] / df['close'].iloc[-63]) - 1) * 100 if len(df) >= 63 else 0
            
            if is_debug:
                self.debug_info[symbol] = {
                    'status': 'PASSED',
                    'price': current_price,
                    'ytd': ytd_return,
                    'distance_50ma': distance_from_50ma,
                    'distance_10ema': distance_from_10ema,
                    'adr': adr,
                }
            
            # ═══ v6: Inject IBD metadata from universe builder if available ═══
            ibd_data = {}
            universe_score = 0
            source_lists = []
            if self.universe_obj and symbol in self.universe_obj.metadata:
                meta = self.universe_obj.metadata[symbol]
                ibd_data = {
                    'ibd_rs_rating_real': meta.get('ibd_rs_rating'),
                    'ibd_eps_rating': meta.get('ibd_eps_rating'),
                    'ibd_composite_rating': meta.get('ibd_composite_rating'),
                    'ibd_acc_dist': meta.get('ibd_acc_dist'),
                    'ibd_smr_rating': meta.get('ibd_smr_rating'),
                    'ibd_industry_rank': meta.get('industry_rank'),
                    'ibd_eps_year_growth': meta.get('eps_year_growth'),
                    'ibd_eps_qtr_growth': meta.get('eps_qtr_growth'),
                }
                universe_score = meta.get('universe_score', 0)
                source_lists = self.universe_obj.source_lists.get(symbol, [])
            
            # NEW: Normalize industry group string for downstream reporting
            stock_industry = self._normalize_industry_from_meta(symbol)
            in_dynamic_universe = symbol in self.dynamic_universe_symbols
            in_manual_watchlist = symbol in self.manual_watchlist_symbols
            if in_dynamic_universe and in_manual_watchlist:
                source_origin = "dynamic+manual"
            elif in_dynamic_universe:
                source_origin = "dynamic"
            elif in_manual_watchlist:
                source_origin = "manual"
            else:
                source_origin = "other"
            
            return {
                'symbol': symbol,
                'bars': bars,
                'current_price': current_price,
                'ma_50': ma_50,
                'ma_200': ma_200,
                'ema_10': ema_10,
                'ema_21': ema_21,
                'distance_from_50ma': distance_from_50ma,
                'distance_from_200ma': distance_from_200ma,
                'distance_from_10ema': distance_from_10ema,
                'distance_from_21ema': distance_from_21ema,
                'days_above_21ema': days_above_21ema,
                'pp_info': pp_info,
                'trail_ma': trail_ma,
                'week_52_high': week_52_high,
                'week_52_low': week_52_low,
                'distance_from_52w_high': distance_from_52w_high,
                'distance_from_52w_low': distance_from_52w_low,
                'avg_volume': avg_volume,
                'volume_50d': volume_50d_avg,
                'volume_ratio': volume_ratio,
                'monthly_turnover': monthly_turnover,
                'adr': adr,
                'ytd_return': ytd_return,
                'price_change_1m': price_change_1m,
                'price_change_3m': price_change_3m,
                # v6: IBD metadata
                'universe_score': universe_score,
                'source_lists': source_lists,
                'source_origin': source_origin,
                'in_dynamic_universe': in_dynamic_universe,
                'in_manual_watchlist': in_manual_watchlist,
                'industry_group': stock_industry,   # <-- new field
                **ibd_data,
            }
            
        except Exception as e:
            self._record_issue('filter_error', symbol, str(e))
            if is_debug:
                self.debug_info[symbol] = {'status': 'ERROR', 'error': str(e)}
            return None
        
        finally:
            if is_debug and failures:
                self.debug_info[symbol] = {'status': 'FAILED', 'failures': failures}
    
    def _calculate_rs_for_stocks(self, stocks):
        """Calculate RS for all stocks"""
        for stock in stocks:
            try:
                rs_rating = self.rs_calculator.calculate_rs_rating(stock['bars'], periods=RS_LOOKBACK_PERIODS)
                stock['rs_rating'] = rs_rating if rs_rating else 0
            except Exception as e:
                self._record_issue('rs', stock['symbol'], str(e))
                stock['rs_rating'] = 0
        return stocks
    
    def _detect_vcp_for_stocks(self, stocks):
        """Detect VCP for all stocks (v2: captures pivot point and pullback data)"""
        for stock in stocks:
            try:
                vcp_result = self.vcp_detector.detect_vcp_pattern(
                    stock['bars'], 
                    min_contractions=VCP_CONTRACTION_REQUIRED
                )
                stock['vcp_score'] = vcp_result['score']
                stock['vcp_details'] = vcp_result
                # ═══ v6: Surface key VCP fields for tier filters and display ═══
                stock['vcp_pivot_price'] = vcp_result.get('pivot_price', 0)
                stock['vcp_pivot_distance_pct'] = vcp_result.get('pivot_distance_pct', 0)
                stock['vcp_num_contractions'] = vcp_result.get('num_contractions', 0)
                stock['vcp_final_pullback_pct'] = vcp_result.get('final_contraction_pct', 0)
                stock['vcp_days_in_pattern'] = vcp_result.get('days_in_pattern', 0)
                stock['vcp_volume_dryup'] = vcp_result.get('volume_dryup', False)
                stock['has_real_vcp'] = vcp_result.get('has_vcp', False)
            except Exception as e:
                self._record_issue('vcp', stock['symbol'], str(e))
                stock['vcp_score'] = 0
                stock['vcp_details'] = {'score': 0, 'details': f'Error: {e}'}
                stock['vcp_pivot_price'] = 0
                stock['vcp_pivot_distance_pct'] = 0
                stock['vcp_num_contractions'] = 0
                stock['vcp_final_pullback_pct'] = 0
                stock['vcp_days_in_pattern'] = 0
                stock['vcp_volume_dryup'] = False
                stock['has_real_vcp'] = False
        return stocks
    
    def _apply_tier_filters(self, stocks):
        """
        4-TIER CLASSIFICATION (v4) - ATR-normalized extension (Jeff Sun):
          TIER_ELITE:    Elite leaders, always actionable (bypass VCP gate)
          TIER_BUY:      Active buy setups (VCP/EMA/50MA), ATR <= 4 from 50MA
          TIER_WATCH:    High RS, building bases or stop too wide
          TIER_EXTENDED: ATR > 4 from 50MA (wait for pullback)
        """
        self.elite_buy_results = []
        self.elite_hold_results = []
        self.buy_results = []
        self.watch_results = []
        self.extended_results = []
        self.rs_leader_results = []
        
        for stock in stocks:
            
            # ─── EPS GROWTH (automatic; design decision) ───
            # Live: IBD's current-year EPS estimate, % change vs last year ("Curr Yr EPS Est. % Chg.")
            # from the weekly universe export; else the same column from IBD list files
            # (universe_builder). The hand-typed ibd_eps_*.csv lists are no longer read for EPS.
            # Backtests use SEC EDGAR actuals (Data/fundamentals/edgar_quarterly.csv) instead.
            eps_yoy = _first_not_none(stock.get('ibd_eps_est_yr_pct'), stock.get('ibd_eps_year_growth'))
            stock['eps_yoy'] = eps_yoy
            stock['eps_currq'] = _first_not_none(stock.get('ibd_eps_est_qtr_pct'), stock.get('ibd_eps_qtr_growth'))

            # IBD industry group rank (1 = strongest); 197 = unknown, ranks last
            ibd_industry_rank = stock.get('ibd_industry_rank')
            if ibd_industry_rank is not None and ibd_industry_rank > 0:
                stock['industry_rank'] = int(ibd_industry_rank)
            else:
                stock['industry_rank'] = stock.get('industry_rank') or 197
            
            # ─── ELITE LEADER FLAG ───
            # Use real IBD RS Rating if available (gold standard), fall back to homegrown
            rs_for_elite = stock.get('ibd_rs_rating_real') or stock.get('rs_rating')
            stock['is_elite_leader'] = is_elite_leader(
                eps_yoy=eps_yoy,
                rs_rating=rs_for_elite,
                ytd_return_pct=stock.get('ytd_return')
            )
            # When IBD covers the stock: no elite without real earnings, or while institutions sell.
            # Blocked names fall through to the normal tiers.
            # Each check runs on its own: a blank EPS Rating mustn't skip the A/D check.
            elite_eps = stock.get('ibd_eps_rating')
            acc_dist = stock.get('ibd_acc_dist') or ''
            if stock['is_elite_leader']:
                if elite_eps is not None and elite_eps < TIER_ELITE_MIN_EPS_RATING:
                    stock['is_elite_leader'] = False
                    stock['elite_blocked_reason'] = f"EPS Rtg {elite_eps:.0f} < {TIER_ELITE_MIN_EPS_RATING}"
                elif acc_dist[:1] in ('D', 'E'):
                    stock['is_elite_leader'] = False
                    stock['elite_blocked_reason'] = f"A/D {acc_dist} (institutions selling)"
            
            # ─── CALCULATE ATR MULTIPLE FROM 50MA (Jeff Sun metric) ───
            adr = stock.get('adr', 0)
            distance_50ma = stock.get('distance_from_50ma', 0)
            atr_multiple = distance_50ma / adr if adr > 0 else 0
            stock['atr_multiple_from_50ma'] = round(atr_multiple, 2)
            
            # ─── EXTENSION STATUS ───
            if atr_multiple > ATR_EXTENDED_VERY_MAX:
                stock['extension_status'] = "PARABOLIC"
                stock['extension_icon'] = "🔥"
            elif atr_multiple > ATR_EXTENDED_MODERATE_MAX:
                stock['extension_status'] = "VERY_EXTENDED"
                stock['extension_icon'] = "⚠️⚠️"
            elif atr_multiple > ATR_BUY_CAP:
                stock['extension_status'] = "MODERATELY_EXTENDED"
                stock['extension_icon'] = "⚠️"
            else:
                stock['extension_status'] = "NORMAL"
                stock['extension_icon'] = "✅"

            # ─── EXTENDED WITHOUT A BASE (flag only, never blocks a tier) ───
            # Far above the 200-day is fine with a proper base (e.g. SNDK, MU runs);
            # flag it only when it's that extended with no valid VCP underneath.
            dist_200 = stock.get('distance_from_200ma')
            stock['extended_no_base'] = (dist_200 is not None and dist_200 >= EXTENDED_NO_BASE_200MA_PCT
                                         and not stock.get('has_real_vcp', False))

            # ─── STOP LOSS (must exist before tier checks read stop_loss_pct) ───
            # Runs after eps_yoy / industry_rank are set above, since the stop uses them.
            self._calculate_context_based_stop(stock)

            # ════════════════════════════════════════════════════════
            # TIER ASSIGNMENT (priority order)
            # ════════════════════════════════════════════════════════
            
            # ─── TIER_ELITE: Split into BUY (clean entry) vs HOLD (no entry today) ───
            if stock['is_elite_leader']:
                stop_pct = stock.get('stop_loss_pct')
                has_stop = stop_pct is not None
                stop_adr = (stop_pct / adr) if has_stop and adr > 0 else 999
                
                # Base elite qualifications
                passes_elite_base = (
                    stock['ytd_return'] >= TIER_ELITE_MIN_YTD and
                    adr <= TIER_ELITE_MAX_ADR and
                    (not TIER_ELITE_REQUIRE_ABOVE_200MA or 
                     stock['current_price'] > stock.get('ma_200', 0))
                )
                
                if passes_elite_base:
                    # Check if there's a clean pullback entry today
                    dist_10ema = stock.get('distance_from_10ema', 999)
                    dist_21ema = stock.get('distance_from_21ema', 999)
                    
                    has_clean_entry = (
                        has_stop and  # No stop, no entry
                        stop_pct >= STOP_PCT_MIN and  # Tighter gets hit by noise
                        atr_multiple <= ATR_BUY_CAP and  # Not too extended
                        stop_pct <= STOP_PCT_ELITE_MAX and
                        stop_adr <= STOP_ADR_RATIO_ELITE_MAX and
                        (-2 <= dist_10ema <= 2 or       # At 10-EMA
                         -3 <= dist_21ema <= 3 or       # At 21-EMA
                         -5 <= distance_50ma <= 5)      # At 50MA
                    )
                    
                    if has_clean_entry:
                        # Label the entry setup
                        if -2 <= dist_10ema <= 2:
                            stock['elite_entry_setup'] = '10EMA'
                        elif -3 <= dist_21ema <= 3:
                            stock['elite_entry_setup'] = '21EMA'
                        else:
                            stock['elite_entry_setup'] = '50MA'
                        self.elite_buy_results.append(stock)
                    else:
                        # Track as HOLD with reason
                        if not has_stop:
                            stock['hold_reason'] = f"No valid stop ({stock.get('stop_description', 'unknown')}) - cannot execute"
                        elif stop_pct < STOP_PCT_MIN:
                            stock['hold_reason'] = f"Stop too tight ({stop_pct:.1f}% < {STOP_PCT_MIN:g}%) - noise will hit it"
                        elif atr_multiple > ATR_BUY_CAP:
                            stock['hold_reason'] = f"Extended ({atr_multiple:.1f} ATR from 50MA) - wait for pullback"
                        elif stop_pct > STOP_PCT_ELITE_MAX:
                            stock['hold_reason'] = f"Stop too wide ({stop_pct:.1f}%) - need better entry"
                        elif stop_adr > STOP_ADR_RATIO_ELITE_MAX:
                            stock['hold_reason'] = f"Stop/ADR too loose ({stop_adr:.1f}x)"
                        else:
                            stock['hold_reason'] = "No pullback entry today - monitor for EMA touch"
                        self.elite_hold_results.append(stock)
                    continue
                # If elite base fails (YTD, ADR cap, 200MA), fall through to normal tiers
            
            rs_rating = stock.get('rs_rating', 0)
            
            # ─── TIER_EXTENDED: ATR > 4 from 50MA (too far to chase) ───
            if atr_multiple > ATR_BUY_CAP:
                passes_extended = (
                    rs_rating >= TIER_EXTENDED_MIN_RS and
                    stock['ytd_return'] >= TIER_EXTENDED_MIN_YTD
                )
                if passes_extended:
                    self.extended_results.append(stock)
                continue
            
            # ─── TIER_BUY: Active setups (ATR <= 4) ───
            dist_10ema = stock.get('distance_from_10ema', 999)
            dist_21ema = stock.get('distance_from_21ema', 999)
            vcp_score = stock.get('vcp_score', 0)
            is_contracting = (stock.get('price_action') or {}).get('is_contracting', False)
            
            setup_type = None
            if vcp_score >= 4.0 and is_contracting:
                setup_type = "VCP"
            elif -2 <= dist_10ema <= 2:
                setup_type = "10EMA"
            elif -3 <= dist_21ema <= 3:
                setup_type = "21EMA"
            elif -5 <= distance_50ma <= 5:
                setup_type = "50MA"
            elif vcp_score >= 5.0 and atr_multiple <= 3.0:
                setup_type = "VCP_LOOSE"
            
            passes_buy_fundamentals = True
            eps_rating = stock.get('ibd_eps_rating')
            ad_selling = (stock.get('ibd_acc_dist') or '')[:1] in ('D', 'E')  # institutions net selling
            if eps_rating is not None:
                # IBD universe covers it: EPS Rating (earnings growth + stability)
                passes_buy_fundamentals = eps_rating >= TIER_BUY_MIN_EPS_RATING
            elif USE_IBD_EPS_FILTERS:
                passes_buy_fundamentals = (
                    eps_yoy is not None and eps_yoy >= TIER_BUY_MIN_EPS
                )
            
            if setup_type and rs_rating >= TIER_BUY_MIN_RS:
                # ADR cap: 7% base, 8% if EPS >= 50%
                eps_val = eps_yoy or 0
                adr_cap = (TIER_BUY_MAX_ADR_HIGH_EPS 
                          if eps_val >= TIER_BUY_HIGH_EPS_THRESHOLD 
                          else TIER_BUY_MAX_ADR_BASE)
                
                # Stop quality checks
                stop_pct = stock.get('stop_loss_pct')
                has_stop = stop_pct is not None
                stop_adr = (stop_pct / adr) if has_stop and adr > 0 else 999
                
                passes_buy = (
                    stock['ytd_return'] >= TIER_BUY_MIN_YTD and
                    stock['adr'] >= TIER_BUY_MIN_ADR and
                    adr <= adr_cap and
                    has_stop and  # No stop, no entry
                    stop_pct >= STOP_PCT_MIN and
                    stop_pct <= STOP_PCT_BUY_MAX and
                    stop_adr <= STOP_ADR_RATIO_BUY_MAX and
                    passes_buy_fundamentals and
                    not ad_selling
                )
                if passes_buy:
                    stock['setup_type'] = setup_type
                    # Flag tight-stop entries for priority boost
                    if stop_adr <= STOP_ADR_RATIO_LUK_TIGHT:
                        stock['stop_quality_flag'] = 'LUK_TIGHT'
                    elif stop_adr <= STOP_ADR_RATIO_IDEAL:
                        stock['stop_quality_flag'] = 'IDEAL'
                    else:
                        stock['stop_quality_flag'] = 'ACCEPTABLE'
                    self.buy_results.append(stock)
                    continue
                else:
                    # Failed BUY — track WHY it failed for watch reason
                    if not has_stop:
                        stock['watch_reason'] = f"No valid stop ({stock.get('stop_description', 'unknown')}) - cannot execute"
                    elif stop_pct < STOP_PCT_MIN:
                        stock['watch_reason'] = f"Stop too tight ({stop_pct:.1f}% < {STOP_PCT_MIN:g}%)"
                    elif adr > adr_cap:
                        stock['watch_reason'] = f"ADR too high ({adr:.1f}% > {adr_cap:.0f}% cap)"
                    elif stop_pct > STOP_PCT_BUY_MAX:
                        stock['watch_reason'] = f"Stop too wide ({stop_pct:.1f}% > {STOP_PCT_BUY_MAX}%)"
                    elif stop_adr > STOP_ADR_RATIO_BUY_MAX:
                        stock['watch_reason'] = f"Stop/ADR ratio too loose ({stop_adr:.1f}x)"
                    elif ad_selling:
                        stock['watch_reason'] = f"A/D {stock['ibd_acc_dist']} (institutions selling)"
                    elif not passes_buy_fundamentals and eps_rating is not None:
                        stock['watch_reason'] = f"EPS Rtg {eps_rating:.0f} < {TIER_BUY_MIN_EPS_RATING}"
                    elif not passes_buy_fundamentals:
                        stock['watch_reason'] = ("no IBD EPS" if eps_yoy is None
                                                 else f"EPS {eps_yoy:.0f}% < {TIER_BUY_MIN_EPS}% min")
                    else:
                        stock['watch_reason'] = "YTD or ADR below minimum"
            
            # ─── TIER_RS_LEADER: technical leader, no earnings check, reduced size ───
            is_rs_leader, rs_leader_miss = self._rs_leader_check(stock, atr_multiple)
            if is_rs_leader:
                stock['size_multiplier'] = RS_LEADER_SIZE_MULT
                stock['setup_type'] = 'VCP'  # the tier requires a VCP near its pivot: a pivot entry
                self.rs_leader_results.append(stock)
                continue
            
            # ─── TIER_WATCH: Catch-all for strong RS ───
            if rs_rating >= TIER_WATCH_MIN_RS:
                if stock.get('stop_status') == 'NO_STOP':
                    stock['setup_type'] = setup_type
                    stock['watch_reason'] = f"No valid stop ({stock.get('stop_description', 'unknown')}) - cannot execute"
                elif rs_leader_miss:
                    stock['setup_type'] = setup_type
                    stock['watch_reason'] = f"RS leader · {rs_leader_miss}"
                elif setup_type:
                    stock['setup_type'] = setup_type
                    buy_fail = stock.get('watch_reason')  # set above when the BUY check failed
                    stock['watch_reason'] = (f"{setup_type} setup · {buy_fail}" if buy_fail
                                             else f"{setup_type} setup · RS {rs_rating:.0f} < {TIER_BUY_MIN_RS} BUY min"
                                             if rs_rating < TIER_BUY_MIN_RS
                                             else f"{setup_type} setup - doesn't meet BUY thresholds")
                else:
                    stock['setup_type'] = None
                    stock['watch_reason'] = "High RS, no clean setup"
                self.watch_results.append(stock)
    
    def _calculate_priority_ratings_for_all_tiers(self):
        """Calculate and sort by priority (v5: split elite)"""
        for stock in self.elite_buy_results:
            self.calculate_priority_rating(stock)
        self.elite_buy_results.sort(key=lambda x: x.get('priority_rating', 0), reverse=True)
        
        for stock in self.elite_hold_results:
            self.calculate_priority_rating(stock)
        self.elite_hold_results.sort(key=lambda x: x.get('priority_rating', 0), reverse=True)
        
        for stock in self.buy_results:
            self.calculate_priority_rating(stock)
        self.buy_results.sort(key=lambda x: x.get('priority_rating', 0), reverse=True)
        
        for stock in self.watch_results:
            self.calculate_priority_rating(stock)
        self.watch_results.sort(key=lambda x: x.get('priority_rating', 0), reverse=True)
        
        for stock in self.rs_leader_results:
            self.calculate_priority_rating(stock)
        self.rs_leader_results.sort(key=lambda x: x.get('priority_rating', 0), reverse=True)
        
        for stock in self.extended_results:
            self.calculate_priority_rating(stock)
        self.extended_results.sort(key=lambda x: x.get('priority_rating', 0), reverse=True)
    
    def _display_debug_info(self):
        """Display debug info"""
        if not self.debug_info:
            return
        
        print("\n" + "="*110)
        print("                   🐛 DEBUG MODE 🐛                   ")
        print("="*110)
        
        for symbol in DEBUG_SYMBOLS:
            if symbol not in self.debug_info:
                print(f"\n❌ {symbol}: NOT FOUND")
                continue
            
            info = self.debug_info[symbol]
            print(f"\n{'='*110}")
            print(f"📊 {symbol}")
            print(f"{'='*110}")
            
            if info['status'] == 'ERROR':
                print(f"❌ ERROR: {info['error']}")
            elif info['status'] == 'FAILED':
                print("❌ FAILED:")
                for f in info['failures']:
                    print(f"   • {f}")
            else:
                print(f"✅ PASSED")
                for k, v in info.items():
                    if k != 'status':
                        print(f"   {k}: {v}")
        
        print("\n" + "="*110 + "\n")
    
    def _display_priority_breakdown(self, stocks, tier_name, max_display=3):
        """Display detailed priority score breakdown"""
        if not stocks or not ENABLE_PRIORITY_RATING:
            return
        
        print(f"\n{'─'*160}")
        print(f"📊 TOP {min(max_display, len(stocks))} {tier_name} - PRIORITY SCORE BREAKDOWN:")
        print(f"{'─'*160}\n")
        
        for idx, stock in enumerate(stocks[:max_display], 1):
            details = stock.get('priority_details', {})
            priority = stock.get('priority_rating', 0)
            
            stop_pct = stock.get('stop_loss_pct', 0)
            stop_adr_ratio = details.get('stop_adr_ratio', 0)
            adr = stock.get('adr', 0)
            vol_ratio = stock.get('volume_ratio', 0)
            atr_dist = details.get('atr_distance', 0)
            is_contracting = details.get('is_contracting', False)
            
            # ═══ NEW (v2): Show elite leader tag ═══
            elite_tag = " ⚡ ELITE" if stock.get('is_elite_leader', False) else ""
            
            print(f"  {idx}. {stock['symbol']}{elite_tag} - Total Priority: {priority:.1f}/100")
            print(f"     ├─ EPS Growth:       {details.get('eps_score', 0):>5.1f}/20 pts  (YoY: {_eps_label(stock.get('eps_yoy')).strip()})")
            print(f"     ├─ Industry Rank:    {details.get('industry_score', 0):>5.1f}/15 pts  (Rank: {stock.get('industry_rank', 197)})")
            print(f"     ├─ RS Rating:        {details.get('rs_score', 0):>5.1f}/15 pts  (RS: {stock['rs_rating']:.0f})")
            
            contraction_status = "CONTRACTING" if is_contracting else "NOT CONTRACTING"
            print(f"     ├─ VCP Tightness:    {details.get('vcp_points', 0):>5.1f}/15 pts  (Score: {stock['vcp_score']:.1f} | +{details.get('vcp_tightness_bonus', 0):.0f} bonus | {contraction_status})")
            
            print(f"     ├─ ATR Distance:     {details.get('pivot_score', 0):>5.1f}/10 pts  ({atr_dist:.1f} ATR from 50MA)")
            print(f"     ├─ YTD Performance:  {details.get('ytd_score', 0):>5.1f}/10 pts  (YTD: {stock['ytd_return']:>+6.1f}%)")
            print(f"     ├─ Volume Activity:  {details.get('volume_score', 0):>5.1f}/10 pts  ({vol_ratio:.2f}x)")
            print(f"     └─ STOP QUALITY:     {details.get('stop_quality_score', 0):>5.1f}/15 pts  ({stop_pct:.1f}% | {stop_adr_ratio:.2f}x ADR [{adr:.1f}%])")
            
            stop_warning = stock.get('stop_warning', '')
            print(f"     {stop_warning}")
            
            if details.get('is_power_stock', False):
                print(f"     💎 POWER STOCK FLAG")
            
            bars = int(priority / 5)
            bar_chart = '█' * bars + '░' * (20 - bars)
            print(f"     Rating: [{bar_chart}] {priority:.1f}%\n")
        
        print()
            
    def _display_results(self):
        """Display 4 tiers (v4)"""
        if not ENABLE_CONSOLE_OUTPUT:
            return
        
        timestamp = self.scan_timestamp.strftime('%Y-%m-%d %H:%M:%S')
        
        # ═══════════════════════════════════════════════════════
        # TIER_ELITE_BUY ⚡ (Actionable - has clean pullback entry)
        # ═══════════════════════════════════════════════════════
        print("\n" + "="*160)
        print("         ⚡ TIER_ELITE_BUY: GENERALS WITH CLEAN ENTRY (actionable now) ⚡         ")
        print(f"                                    {timestamp}                                    ")
        print("="*160)
        
        if not self.elite_buy_results:
            print("\n⚠️ No elite leaders have clean entries today (check ELITE_HOLD below)\n")
        else:
            print(f"\n{'#':<4} {'Symbol':<8} {'Priority':<9} {'Entry':<7} {'RS':<5} {'Price':<10} {'YTD%':<8} "
                  f"{'ATR×':<6} {'%50MA':<8} {'EPS%':<7} {'ADR':<5} {'Stop%':<6} {'Stop/ADR':<9}")
            print("-" * 160)
            
            for idx, stock in enumerate(self.elite_buy_results[:MAX_ELITE_DISPLAY], 1):
                pri = stock.get('priority_rating', 0)
                eps = stock.get('eps_yoy')
                stop = stock.get('stop_loss_pct', 0)
                adr = stock.get('adr', 0)
                stop_adr = stop/adr if adr > 0 else 0
                atr_x = stock.get('atr_multiple_from_50ma', 0)
                entry = stock.get('elite_entry_setup', '-')
                
                print(f"{idx:<4} {stock['symbol']:<8} {pri:<9.1f} {entry:<7} {stock['rs_rating']:<5.0f} "
                      f"${stock['current_price']:<9.2f} {stock['ytd_return']:>+7.1f}% "
                      f"{atr_x:<6.1f} {stock['distance_from_50ma']:>+7.1f}% {_eps_label(eps)} "
                      f"{stock['adr']:<5.1f} {stop:<6.1f} {stop_adr:<9.2f}")
            
            if len(self.elite_buy_results) > MAX_ELITE_DISPLAY:
                print(f"\n... +{len(self.elite_buy_results) - MAX_ELITE_DISPLAY} more")
            
            self._display_priority_breakdown(self.elite_buy_results, "ELITE_BUY", max_display=3)
        
        print("="*160)
        
        # ═══════════════════════════════════════════════════════
        # TIER_ELITE_HOLD 📌 (Track only - wait for pullback)
        # ═══════════════════════════════════════════════════════
        print("\n" + "="*160)
        print("         📌 TIER_ELITE_HOLD: GENERALS (tracking only, no entry today) 📌         ")
        print("="*160)
        
        if not self.elite_hold_results:
            print("\n⚠️ No elite leaders in hold-only status\n")
        else:
            print(f"\n{'#':<4} {'Symbol':<8} {'Priority':<9} {'RS':<5} {'Price':<10} {'YTD%':<8} "
                  f"{'ATR×':<6} {'EPS%':<7} {'ADR':<5} {'Stop%':<6} {'Why HOLD only'}")
            print("-" * 160)
            
            for idx, stock in enumerate(self.elite_hold_results[:MAX_ELITE_DISPLAY], 1):
                pri = stock.get('priority_rating', 0)
                eps = stock.get('eps_yoy')
                stop = stock.get('stop_loss_pct')
                atr_x = stock.get('atr_multiple_from_50ma', 0)
                reason = stock.get('hold_reason', '')
                
                print(f"{idx:<4} {stock['symbol']:<8} {pri:<9.1f} {stock['rs_rating']:<5.0f} "
                      f"${stock['current_price']:<9.2f} {stock['ytd_return']:>+7.1f}% "
                      f"{atr_x:<6.1f} {_eps_label(eps)} "
                      f"{stock['adr']:<5.1f} {_num_or_na(stop, '<6.1f')} {reason}")
            
            if len(self.elite_hold_results) > MAX_ELITE_DISPLAY:
                print(f"\n... +{len(self.elite_hold_results) - MAX_ELITE_DISPLAY} more")
            
            print(f"\n{'─'*160}")
            print("📌 These are strong leaders without clean entry setups today.")
            print("   Monitor for EMA pullbacks — they'll migrate to ELITE_BUY when entry becomes available.")
        
        print("="*160)
        
        # ═══════════════════════════════════════════════════════
        # TIER_BUY 🎯 (Active buy setups)
        # ═══════════════════════════════════════════════════════
        print("\n" + "="*160)
        print("         🎯 TIER_BUY: ACTIVE SETUPS (ATR ≤ 4 from 50MA, tight stops) 🎯         ")
        print("="*160)
        
        if not self.buy_results:
            print("\n⚠️ No active BUY setups today\n")
        else:
            print(f"\n{'#':<4} {'Symbol':<8} {'Priority':<9} {'Setup':<8} {'Quality':<10} "
                  f"{'RS':<5} {'Price':<10} {'YTD%':<8} {'ATR×':<6} {'EPS%':<7} {'ADR':<5} {'Stop%':<6} {'Stop/ADR':<9}")
            print("-" * 160)
            
            for idx, stock in enumerate(self.buy_results[:MAX_BUY_DISPLAY], 1):
                pri = stock.get('priority_rating', 0)
                eps = stock.get('eps_yoy')
                stop = stock.get('stop_loss_pct', 0)
                adr = stock.get('adr', 0)
                stop_adr = stop/adr if adr > 0 else 0
                atr_x = stock.get('atr_multiple_from_50ma', 0)
                setup = stock.get('setup_type', '-')
                quality = stock.get('stop_quality_flag', '-')
                quality_icon = '⭐' if quality == 'LUK_TIGHT' else ('✅' if quality == 'IDEAL' else '~')
                
                print(f"{idx:<4} {stock['symbol']:<8} {pri:<9.1f} {setup:<8} {quality_icon} {quality:<8} "
                      f"{stock['rs_rating']:<5.0f} "
                      f"${stock['current_price']:<9.2f} {stock['ytd_return']:>+7.1f}% "
                      f"{atr_x:<6.1f} {_eps_label(eps)} "
                      f"{stock['adr']:<5.1f} {stop:<6.1f} {stop_adr:<9.2f}")
            
            if len(self.buy_results) > MAX_BUY_DISPLAY:
                print(f"\n... +{len(self.buy_results) - MAX_BUY_DISPLAY} more")
            
            self._display_priority_breakdown(self.buy_results, "BUY", max_display=3)
            
            # ═══ v6: Show VCP pivot info for stocks with real VCPs ═══
            real_vcps = [s for s in self.buy_results if s.get('has_real_vcp', False)]
            if real_vcps:
                print(f"\n{'─'*160}")
                print(f"  📐 VCP PIVOT POINTS (set buy alerts above pivot):")
                print(f"{'─'*160}")
                print(f"  {'Symbol':<8} {'Pivot':<10} {'Distance':<10} {'Contractions':<14} "
                      f"{'Final PB':<10} {'Days':<6} {'Vol Dryup'}")
                for s in real_vcps[:10]:
                    print(f"  {s['symbol']:<8} ${s.get('vcp_pivot_price',0):<9.2f} "
                          f"{s.get('vcp_pivot_distance_pct',0):>+7.1f}%  "
                          f"{s.get('vcp_num_contractions',0):<14} "
                          f"-{s.get('vcp_final_pullback_pct',0):<8.1f}% "
                          f"{s.get('vcp_days_in_pattern',0):<6} "
                          f"{'YES' if s.get('vcp_volume_dryup',False) else 'no'}")
        
        print("="*160)
        
        # ═══════════════════════════════════════════════════════
        # TIER_WATCH 📋 (Building bases / stops too wide)
        # ═══════════════════════════════════════════════════════
        print("\n" + "="*160)
        print("         📋 TIER_WATCH: BUILDING BASES / Stop issues 📋         ")
        print("="*160)
        
        if not self.watch_results:
            print("\n⚠️ No watch candidates today\n")
        else:
            print(f"\n{'#':<4} {'Symbol':<8} {'Priority':<9} {'RS':<5} {'Price':<10} {'YTD%':<8} "
                  f"{'ATR×':<6} {'ADR':<5} {'Stop%':<6} {'Why watching'}")
            print("-" * 160)
            
            for idx, stock in enumerate(self.watch_results[:MAX_WATCH_DISPLAY], 1):
                pri = stock.get('priority_rating', 0)
                stop = stock.get('stop_loss_pct')
                atr_x = stock.get('atr_multiple_from_50ma', 0)
                reason = stock.get('watch_reason', stock.get('moved_reason', ''))
                
                print(f"{idx:<4} {stock['symbol']:<8} {pri:<9.1f} {stock['rs_rating']:<5.0f} "
                      f"${stock['current_price']:<9.2f} {stock['ytd_return']:>+7.1f}% "
                      f"{atr_x:<6.1f} {stock['adr']:<5.1f} {_num_or_na(stop, '<6.1f')} {reason}")
            
            if len(self.watch_results) > MAX_WATCH_DISPLAY:
                print(f"\n... +{len(self.watch_results) - MAX_WATCH_DISPLAY} more")
        
        print("="*160)
        
        # ═══════════════════════════════════════════════════════
        # TIER_EXTENDED 🚀 (Too far from 50MA — wait for pullback)
        # ═══════════════════════════════════════════════════════
        print("\n" + "="*160)
        print("         🚀 TIER_EXTENDED: ATR > 4 FROM 50MA (wait for pullback) 🚀         ")
        print("="*160)
        
        if not self.extended_results:
            print("\n⚠️ No extended stocks today\n")
        else:
            print(f"\n{'#':<4} {'Symbol':<8} {'Status':<22} {'Priority':<9} {'RS':<5} {'Price':<10} "
                  f"{'YTD%':<8} {'ATR×':<6} {'%50MA':<8} {'ADR':<5}")
            print("-" * 160)
            
            for idx, stock in enumerate(self.extended_results[:MAX_EXTENDED_DISPLAY], 1):
                pri = stock.get('priority_rating', 0)
                atr_x = stock.get('atr_multiple_from_50ma', 0)
                ext_status = stock.get('extension_status', '')
                ext_icon = stock.get('extension_icon', '')
                status_label = f"{ext_icon} {ext_status.replace('_', ' ').title()}"
                
                print(f"{idx:<4} {stock['symbol']:<8} {status_label:<22} {pri:<9.1f} {stock['rs_rating']:<5.0f} "
                      f"${stock['current_price']:<9.2f} {stock['ytd_return']:>+7.1f}% "
                      f"{atr_x:<6.1f} {stock['distance_from_50ma']:>+7.1f}% {stock['adr']:<5.1f}")
            
            if len(self.extended_results) > MAX_EXTENDED_DISPLAY:
                print(f"\n... +{len(self.extended_results) - MAX_EXTENDED_DISPLAY} more")
            
            print(f"\n{'─'*160}")
            print("⚠️ WARNING: Wait for pullback to 10-EMA/21-EMA/50-MA before entry")
            print("   Parabolic (10+ ATR) needs 20-30% correction before considering")
        
        print("="*160 + "\n")
    
    def _export_results(self, scan_type):
        """Export to CSV (v4: 4-tier)"""
        if not ENABLE_CSV_EXPORT:
            return
        
        date_str = self.scan_timestamp.strftime('%Y-%m-%d')
        time_str = self.scan_timestamp.strftime('%H%M')
        self.export_prefix = str(Path(OUTPUT_DIR) / f"scan_{date_str}_{time_str}_{scan_type}")
        regime_status = self.market_regime.status if self.market_regime else 'UNKNOWN'
        
        all_results = []
        
        def create_row(stock, tier, tier_pri):
            eps_yoy = stock.get('eps_yoy')
            ind_rank = stock.get('industry_rank') or 197
            atr_dist = stock.get('priority_details', {}).get('atr_distance', 0)
            stop_pct = stock.get('stop_loss_pct')
            adr = stock.get('adr', 0)
            stop_adr_ratio = (stop_pct / adr) if stop_pct is not None and adr > 0 else None
            stop_price = stock.get('stop_loss')
            # price_action is None when its analysis failed; .get on None would crash the export
            price_action = stock.get('price_action') or {}
            
            return {
                'Scan_Date': date_str,
                'Scan_Time': time_str,
                'Scan_Timestamp': self.scan_timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'Scan_Type': scan_type,
                'Universe_Mode': self.active_universe_mode or SCAN_UNIVERSE,
                'Tier': tier,
                'Tier_Priority': tier_pri,
                'Symbol': stock['symbol'],
                'Priority_Rating': round(stock.get('priority_rating', 0), 1),
                'Is_Elite_Leader': stock.get('is_elite_leader', False),
                'Regime': regime_status,
                'Setup_Type': stock.get('setup_type', ''),
                # How the entry is taken; the execution engine picks the stop band from it.
                'Entry_Setup': entry_setup_label(stock),
                'Stop_Quality_Flag': stock.get('stop_quality_flag', ''),
                'RS_Rating': round(stock['rs_rating'], 1),
                'Price': round(stock['current_price'], 2),
                'YTD_%': round(stock['ytd_return'], 1),
                'Distance_50MA_%': round(stock['distance_from_50ma'], 1),
                'Distance_200MA_%': round(stock['distance_from_200ma'], 1) if stock.get('distance_from_200ma') is not None else None,
                'Extended_No_Base': stock.get('extended_no_base', False),
                'Days_Above_21EMA': stock.get('days_above_21ema'),
                # Forward-test flags (no tier or BUY rule uses them): pocket pivot today, and whether
                # it sits inside a VCP base (detector VCP with price still below its pivot)
                'Pocket_Pivot': (stock.get('pp_info') or {}).get('pocket_pivot', False),
                'Pocket_Pivot_In_VCP': bool((stock.get('pp_info') or {}).get('pocket_pivot')
                                            and stock.get('has_real_vcp')
                                            and (stock.get('vcp_pivot_price') or 0) > stock['current_price']),
                'PP_Stop': (stock.get('pp_info') or {}).get('pp_stop'),
                'PP_Stop_Pct': (stock.get('pp_info') or {}).get('pp_stop_pct'),
                'ADR_Pct': (stock.get('pp_info') or {}).get('adr_pct'),
                'Dollar_Vol_Month_M': (stock.get('pp_info') or {}).get('dollar_vol_month_m'),
                # Exit plan: after +TRIM_R, trail this MA (the one the stock respected; exit_rules.py)
                'Trail_MA': stock.get('trail_ma', 'ema21'),
                'Late_Run_No_Reset': (stock.get('days_above_21ema') or 0) > LATE_RUN_DAYS_ABOVE_21EMA,
                'Base_Count': stock.get('base_number'),
                'Size_Multiplier': effective_size_multiplier(stock, tier, self.watch_only_mode),
                # IBD industry group rank (unique per group), blank when unknown; the execution
                # engine caps combined risk per group with it. Industry_Rank uses 197 for unknown.
                'IBD_Group_Rank': int(stock['ibd_industry_rank']) if stock.get('ibd_industry_rank') is not None else None,
                'Regime_Blocked': bool(self.watch_only_mode and tier in ('TIER_BUY', 'TIER_RS_LEADER')),
                'IBD_SMR': stock.get('ibd_smr_rating') or '',
                'RS_Calc': stock.get('rs_rating_calc'),
                'Elite_Blocked': stock.get('elite_blocked_reason') or '',
                'Base_Forming': stock.get('base_forming', False),
                'ATR_Multiple_50MA': stock.get('atr_multiple_from_50ma', 0),
                'Extension_Status': stock.get('extension_status', ''),
                'EPS_YoY_%': round(eps_yoy, 0) if eps_yoy is not None else None,
                'Industry_Rank': ind_rank,
                # Informational (reports, engine's 'Sector' column); no rule reads it
                'Industry_Group': (stock.get('industry_group')
                                   or self._normalize_industry_from_meta(stock['symbol'])),
                'VCP_Score': round(stock['vcp_score'], 1),
                'IBD_RS_Real': stock.get('ibd_rs_rating_real') or '',
                'IBD_EPS_Rating': stock.get('ibd_eps_rating') or '',
                'IBD_Composite': stock.get('ibd_composite_rating') or '',
                'IBD_Acc_Dist': stock.get('ibd_acc_dist') or '',
                'Universe_Score': stock.get('universe_score', 0),
                'Source_Lists': ','.join(stock.get('source_lists', [])),
                'Source_Origin': stock.get('source_origin', 'unknown'),
                'In_Dynamic_Universe': stock.get('in_dynamic_universe', False),
                'In_Manual_Watchlist': stock.get('in_manual_watchlist', False),
                'VCP_Pivot': round(stock.get('vcp_pivot_price', 0), 2),
                'VCP_Dist_From_Pivot_%': round(stock.get('vcp_pivot_distance_pct', 0), 2),
                'VCP_Contractions': stock.get('vcp_num_contractions', 0),
                'VCP_Final_PB_%': round(stock.get('vcp_final_pullback_pct', 0), 1),
                'VCP_Days': stock.get('vcp_days_in_pattern', 0),
                'VCP_Vol_Dryup': stock.get('vcp_volume_dryup', False),
                'Has_Real_VCP': stock.get('has_real_vcp', False),
                'Is_Contracting': price_action.get('is_contracting', False),
                'Contraction_%': round(price_action.get('contraction_1_to_3_pct', 0), 1),
                'Base_Depth_%': round(price_action.get('base_depth_pct', 0), 1),
                'ADR_%': round(stock['adr'], 1),
                'Stop_Loss_Price': round(stop_price, 2) if stop_price is not None else None,
                'Stop_Loss_%': round(stop_pct, 2) if stop_pct is not None else None,
                'Stop_Context': stock.get('stop_context', ''),
                'Stop_Status': stock.get('stop_status', ''),
                'Stop_ADR_Ratio': round(stop_adr_ratio, 2) if stop_adr_ratio is not None else None,
                'Yesterday_Low': round(price_action.get('yesterday_low', 0), 2),
                'Swing_Low_3D': round(price_action.get('swing_low_3d', 0), 2),
                'Swing_Low_5D': round(price_action.get('swing_low_5d', 0), 2),
                'Swing_Low_7D': round(price_action.get('swing_low_7d', 0), 2),
                'Volume_Ratio': round(stock['volume_ratio'], 2),
                'Is_Power_Stock': stock.get('is_power_stock', False),
                'Watch_Reason': stock.get('watch_reason', ''),
                # Theme dashboard (informational only, appended last); Theme_As_Of shows staleness
                **theme_columns(stock.get('industry_group')
                                or self._normalize_industry_from_meta(stock['symbol'])),
            }
        
        for stock in self.elite_buy_results:
            all_results.append(create_row(stock, 'TIER_ELITE_BUY', 0))
        for stock in self.elite_hold_results:
            all_results.append(create_row(stock, 'TIER_ELITE_HOLD', 1))
        for stock in self.buy_results:
            all_results.append(create_row(stock, 'TIER_BUY', 2))
        for stock in self.watch_results:
            all_results.append(create_row(stock, 'TIER_WATCH', 3))
        for stock in self.rs_leader_results:
            all_results.append(create_row(stock, 'TIER_RS_LEADER', 2.5))
        for stock in self.extended_results:
            all_results.append(create_row(stock, 'TIER_EXTENDED', 4))
        
        if not all_results:
            logger.warning("⚠️ No stocks to export")
            return
        
        df = pd.DataFrame(all_results)
        df = df.sort_values(['Tier_Priority', 'Priority_Rating'], ascending=[True, False])
        
        filepath = Path(OUTPUT_DIR) / f"scan_{date_str}_{time_str}_{scan_type}_COMPLETE.csv"
        df.to_csv(filepath, index=False)
        logger.info(f"\n✅ COMPLETE SCAN: {filepath}")
        logger.info(f"   Total: {len(df)} stocks")
        
        # ═══ BUY LIST: Elite_Buy + Buy (in correction: Elite_Buy only) ═══
        buyable = ['TIER_ELITE_BUY', 'TIER_BUY', 'TIER_RS_LEADER']
        df_buy = df[df['Tier'].isin(buyable)].copy()
        
        if self.watch_only_mode:
            df_buy = df_buy[df_buy['Tier'] == 'TIER_ELITE_BUY'].copy()
            
            if df_buy.empty:
                logger.warning("\n🔴 No Elite BUY setups found")
                logger.warning("   No actionable trades today — watch ELITE_HOLD list for pullbacks")
            else:
                df_buy = df_buy.sort_values('Priority_Rating', ascending=False)
                df_buy.insert(0, 'Buy_Priority', range(1, len(df_buy) + 1))
                
                filepath_buy = Path(OUTPUT_DIR) / f"scan_{date_str}_{time_str}_{scan_type}_ELITE_ONLY.csv"
                df_buy.to_csv(filepath_buy, index=False)
                logger.info(f"\n⚡ ELITE LEADERS ONLY (correction mode): {filepath_buy}")
                logger.info(f"   Actionable: {len(df_buy)} stocks | Regime: {regime_status}")
                
                logger.info(f"\n   ELITE LEADERS (correction-approved, use 0.125% risk):")
                for idx, row in df_buy.iterrows():
                    logger.info(f"     {row['Buy_Priority']}. {row['Symbol']} - "
                              f"Priority: {row['Priority_Rating']:.1f}, "
                              f"EPS: {_eps_label(row['EPS_YoY_%']).strip()}, "
                              f"RS: {row['RS_Rating']:.0f}, "
                              f"ATR×: {row['ATR_Multiple_50MA']:.1f}, "
                              f"Stop: {row['Stop_Loss_%']:.1f}%")
                
                logger.info(f"\n   ⚠️  Check your 25% total invested cap before adding")
        
        elif not df_buy.empty:
            df_buy = df_buy.sort_values('Priority_Rating', ascending=False)
            df_buy.insert(0, 'Buy_Priority', range(1, len(df_buy) + 1))
            
            filepath_buy = Path(OUTPUT_DIR) / f"scan_{date_str}_{time_str}_{scan_type}_BUY_LIST.csv"
            df_buy.to_csv(filepath_buy, index=False)
            logger.info(f"\n💰 BUY LIST: {filepath_buy}")
            logger.info(f"   Actionable: {len(df_buy)} stocks | Regime: {regime_status}")
            
            if len(df_buy) >= 3:
                logger.info(f"\n   TOP 3 BUY PRIORITY:")
                for idx, row in df_buy.head(3).iterrows():
                    elite_flag = " ⚡ ELITE" if row['Tier'] == 'TIER_ELITE_BUY' else ""
                    quality_flag = ""
                    if row['Stop_Quality_Flag'] == 'LUK_TIGHT':
                        quality_flag = " ⭐ LUK-TIGHT"
                    elif row['Stop_Quality_Flag'] == 'IDEAL':
                        quality_flag = " ✅ IDEAL"
                    contract_flag = " ✅ CONTRACTING" if row['Is_Contracting'] else ""
                    
                    logger.info(f"     {row['Buy_Priority']}. {row['Symbol']} ({row['Tier']}) - "
                              f"Priority: {row['Priority_Rating']:.1f}, "
                              f"Setup: {row['Setup_Type']}, "
                              f"Stop: {row['Stop_Loss_%']:.1f}% ({row['Stop_ADR_Ratio']:.1f}x ADR)"
                              f"{elite_flag}{quality_flag}{contract_flag}")
        
        # ═══ ELITE_HOLD separate CSV (track-only leaders) ═══
        df_hold = df[df['Tier'] == 'TIER_ELITE_HOLD'].copy()
        if not df_hold.empty:
            df_hold = df_hold.sort_values('Priority_Rating', ascending=False)
            filepath_hold = Path(OUTPUT_DIR) / f"scan_{date_str}_{time_str}_{scan_type}_ELITE_HOLD.csv"
            df_hold.to_csv(filepath_hold, index=False)
            logger.info(f"\n📌 ELITE_HOLD (tracking only): {filepath_hold}")
            logger.info(f"   Leaders tracked for future pullback: {len(df_hold)} stocks")
        
        # ═══ WATCHLIST CSV ═══
        watch = ['TIER_WATCH', 'TIER_EXTENDED']
        df_watch = df[df['Tier'].isin(watch)].copy()
        
        if not df_watch.empty:
            filepath_watch = Path(OUTPUT_DIR) / f"scan_{date_str}_{time_str}_{scan_type}_WATCHLIST.csv"
            df_watch.to_csv(filepath_watch, index=False)
            logger.info(f"\n👁️ WATCHLIST: {filepath_watch}")
            logger.info(f"   Watch + extended: {len(df_watch)} stocks")
        
        # ─────────────────────────────────────────────────────────────────
        # NEW: Sector summary export (counts + average priority)
        # ─────────────────────────────────────────────────────────────────
        try:
            # groupby drops NaN/blank keys silently: name them so no symbol vanishes from the summary
            groups = df['Industry_Group'].fillna('').replace('', 'Unknown')
            sector_df = df.groupby(groups.rename('Industry_Group')).agg(
                symbol_count=('Symbol', 'count'),
                avg_priority=('Priority_Rating', 'mean'),
                avg_tier_priority=('Tier_Priority', 'mean')
            ).reset_index().sort_values(['symbol_count', 'avg_priority'], ascending=[False, False])
            sector_file = Path(OUTPUT_DIR) / f"scan_{date_str}_{time_str}_{scan_type}_SECTOR_SUMMARY.csv"
            sector_df.to_csv(sector_file, index=False)
            logger.info(f"\n📊 SECTOR SUMMARY: {sector_file}  (groups: {len(sector_df)})")
        except Exception as e:
            logger.warning(f"Failed to write sector summary: {e}")
    
    def _bars_to_dataframe(self, bars):
        """Convert bars to DataFrame"""
        data = {
            'date': [bar.date for bar in bars],
            'open': [bar.open for bar in bars],
            'high': [bar.high for bar in bars],
            'low': [bar.low for bar in bars],
            'close': [bar.close for bar in bars],
            'volume': [bar.volume for bar in bars]
        }
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        return df
    
    def cleanup(self):
        """Cleanup"""
        if self.connector:
            self.connector.disconnect_and_stop()
        logger.info("Scanner shutdown complete")


def scan_log_handler(log_dir=None, day: Optional[datetime] = None) -> logging.Handler:
    """The day's scan log, size-capped: at LOG_MAX_MB it rolls to .1 ... .LOG_BACKUP_COUNT.

    One file per day as before; a runaway day (DEBUG_MODE, ibapi wire spam) now stops at
    LOG_MAX_MB x (LOG_BACKUP_COUNT + 1) instead of growing without limit. Only this handler's own
    roll-overs drop the oldest same-day copy; earlier days' files are never touched.
    """
    log_dir = Path(log_dir or LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f'momentum_scan_{(day or datetime.now()).strftime("%Y%m%d")}.log'
    max_mb = float(globals().get('LOG_MAX_MB', 50))
    backups = int(globals().get('LOG_BACKUP_COUNT', 5))
    return logging.handlers.RotatingFileHandler(path, maxBytes=int(max_mb * 1024 * 1024),
                                                backupCount=backups, encoding='utf-8')


def main():
    """Main entry point"""
    logging.basicConfig(
        level=logging.DEBUG if DEBUG_MODE else logging.INFO,
        format='[%(asctime)s] %(levelname)s - %(message)s',
        datefmt='%H:%M:%S',
        handlers=[
            scan_log_handler(),
            logging.StreamHandler()
        ]
    )
    # ibapi logs every wire message at DEBUG (~1M lines per scan); keep it to warnings
    logging.getLogger('ibapi').setLevel(logging.WARNING)
    
    # A CLI scan on the watcher's id either can't connect or knocks the watcher off (and
    # launchd's relaunch then knocks the scan off mid-run): default to the ad-hoc id, and
    # refuse an explicit TRADING_CLIENT_ID equal to the watcher's (audit #10)
    global CLIENT_ID
    if 'TRADING_CLIENT_ID' not in os.environ:
        CLIENT_ID = ADHOC_CLIENT_ID
    if CLIENT_ID == WATCHER_CLIENT_ID:
        logger.error(f"❌ Client id {CLIENT_ID} is the watcher's; set TRADING_CLIENT_ID to another id "
                     f"(scheduled scan 11, ad-hoc {ADHOC_CLIENT_ID})")
        return 3
    logger.info(f"TWS client id {CLIENT_ID}")
    scanner = MomentumScanner()
    scanner.complete_daily_bars_only = True
    # Non-zero on failure, so the scheduled pipeline stops instead of planning from an older file
    status = 1
    try:
        if not scanner.initialize():
            logger.error("❌ Initialization failed")
            return 1
        
        # No universe or no survivors writes no CSV: exit 2 so the pipeline sees it (review S12)
        # A data-integrity abort (connection lost / mass timeouts) is a TWS failure: exit 1, retryable
        if scanner.run_full_scan(scan_type="afterhours"):
            status = 0
        else:
            status = 1 if scanner.scan_aborted else 2
        
    except KeyboardInterrupt:
        logger.info("\n⚠️ Scan interrupted by user")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}", exc_info=True)
    finally:
        scanner.cleanup()
        
        print("\n" + "="*70)
        print("SCAN COMPLETE" if status == 0 else "SCAN FAILED")
        print("="*70)
    return status


if __name__ == "__main__":
    sys.exit(main())
