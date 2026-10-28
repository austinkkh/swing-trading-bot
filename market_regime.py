"""
═══════════════════════════════════════════════════════════════════════════════
MARKET REGIME DETECTION MODULE
Based on William O'Neil's Distribution Day Methodology (IBD)
Determines market health to gate ALL trading signals

This module answers: "Should I be taking new positions RIGHT NOW?"
─────────────────────────────────────────────────────────────────────────────
Usage:
    from market_regime import MarketRegimeDetector

    detector = MarketRegimeDetector(ibkr_connector)
    regime = detector.detect_regime()

    if regime.should_take_new_positions:
        size = base_size * regime.position_size_multiplier
        # proceed with signals...
    else:
        # watchlist only — manage existing positions
═══════════════════════════════════════════════════════════════════════════════
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# REGIME CLASSIFICATIONS
# ═══════════════════════════════════════════════════════════════════════════
REGIME_CONFIRMED_UPTREND = "CONFIRMED_UPTREND"
REGIME_UPTREND_UNDER_PRESSURE = "UPTREND_UNDER_PRESSURE"
REGIME_RALLY_ATTEMPT = "RALLY_ATTEMPT"
REGIME_CORRECTION = "CORRECTION"

# ═══════════════════════════════════════════════════════════════════════════
# THRESHOLDS (tune these based on your risk tolerance)
# ═══════════════════════════════════════════════════════════════════════════

# Distribution Day detection
# A distribution day = index closes DOWN by threshold% on HIGHER volume
DIST_DAY_DECLINE_THRESHOLD = -0.20   # Index falls 0.20%+ to qualify
DIST_DAY_LOOKBACK = 25               # Trading sessions to look back
DIST_DAY_EXPIRY_GAIN = 5.0           # D-day expires if index rallies 5%+ above it

# Distribution Day regime thresholds
DIST_HEALTHY_MAX = 3                 # 0-3 D-days = healthy market
DIST_PRESSURE_MIN = 4                # 4-5 D-days = under pressure
DIST_SERIOUS_MIN = 6                 # 6+ D-days = correction warning

# Drawdown thresholds (from 52-week high)
DRAWDOWN_PRESSURE = -3.0             # 3%+ drawdown = pressure zone
DRAWDOWN_CORRECTION = -8.0           # 8%+ drawdown = correction

# Follow-through day (confirms new uptrend after correction)
FTD_MIN_GAIN_PCT = 1.50              # Index must gain 1.50%+ 
FTD_MIN_DAY = 4                      # Must be on day 4+ of rally attempt

# Moving average slope detection
MA_SLOPE_LOOKBACK = 10               # Days to measure MA slope

# Data reliability (not a regime rule): extra attempts for the SPY/QQQ fetch after a timeout
INDEX_FETCH_RETRIES = 3

# Position size multipliers by regime
REGIME_POSITION_MULTIPLIERS = {
    REGIME_CONFIRMED_UPTREND: 1.00,      # Full size
    REGIME_UPTREND_UNDER_PRESSURE: 0.50, # Half size
    REGIME_RALLY_ATTEMPT: 0.25,          # Quarter size, top quality only
    REGIME_CORRECTION: 0.00,             # NO new positions
}


# ═══════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class DistributionDay:
    """A single distribution day event"""
    date: str
    decline_pct: float
    volume_change_pct: float
    index: str  # "SPY" or "QQQ"


@dataclass
class IndexHealth:
    """Health metrics for a single index"""
    symbol: str
    current_price: float
    ma_50: float
    ma_200: float
    above_50ma: bool
    above_200ma: bool
    ma_50_rising: bool
    ma_200_rising: bool
    distance_from_50ma_pct: float
    drawdown_from_high_pct: float
    high_52w: float


@dataclass
class MarketRegime:
    """Complete market regime analysis output"""
    # Primary output
    status: str
    timestamp: datetime
    position_size_multiplier: float
    should_take_new_positions: bool
    reasoning: str
    
    # Index health
    spy_health: IndexHealth
    qqq_health: IndexHealth
    
    # Distribution days
    distribution_day_count: int
    distribution_days: List[DistributionDay] = field(default_factory=list)
    
    # Risk flags
    warnings: List[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════
# MARKET REGIME DETECTOR
# ═══════════════════════════════════════════════════════════════════════════
class MarketRegimeDetector:
    """
    Detects market regime using IBD/O'Neil methodology.
    
    Regime determines position sizing and whether to take new signals at all.
    """
    
    def __init__(self, ibkr_connector):
        self.connector = ibkr_connector
        self.spy_df: Optional[pd.DataFrame] = None
        self.qqq_df: Optional[pd.DataFrame] = None
    
    # ─────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────────────────────────────
    def detect_regime(self) -> Optional[MarketRegime]:
        """
        Main entry point. Returns MarketRegime object or None on failure.
        """
        logger.info("Detecting market regime...")
        
        # Fetch index data
        if not self._fetch_index_data():
            logger.error("❌ Failed to fetch index data for regime detection")
            return None
        
        # Analyze each index
        spy_health = self._analyze_index_health(self.spy_df, "SPY")
        qqq_health = self._analyze_index_health(self.qqq_df, "QQQ")
        
        # Count distribution days (SPY is primary)
        spy_dist_days = self._count_distribution_days(self.spy_df, "SPY")
        qqq_dist_days = self._count_distribution_days(self.qqq_df, "QQQ")
        
        # Combine distribution days (take higher count as market signal)
        all_dist_days = spy_dist_days + qqq_dist_days
        all_dist_days.sort(key=lambda d: d.date, reverse=True)
        dist_count = max(len(spy_dist_days), len(qqq_dist_days))
        
        # Classify regime
        status, reasoning, warnings = self._classify_regime(
            spy_health, qqq_health, dist_count, spy_dist_days
        )
        
        multiplier = REGIME_POSITION_MULTIPLIERS[status]
        
        regime = MarketRegime(
            status=status,
            timestamp=datetime.now(),
            position_size_multiplier=multiplier,
            should_take_new_positions=(multiplier > 0),
            reasoning=reasoning,
            spy_health=spy_health,
            qqq_health=qqq_health,
            distribution_day_count=dist_count,
            distribution_days=all_dist_days[:10],  # Most recent 10
            warnings=warnings
        )
        
        return regime
    
    def display_regime(self, regime: MarketRegime) -> None:
        """Pretty-print regime analysis to console"""
        if not regime:
            print("⚠️ Regime detection failed")
            return
        
        # Emoji by regime
        regime_icons = {
            REGIME_CONFIRMED_UPTREND: "🟢",
            REGIME_UPTREND_UNDER_PRESSURE: "🟡",
            REGIME_RALLY_ATTEMPT: "🟠",
            REGIME_CORRECTION: "🔴"
        }
        icon = regime_icons.get(regime.status, "⚪")
        
        print("\n" + "="*75)
        print(f"  {icon}  MARKET REGIME: {regime.status}  {icon}")
        print("="*75)
        print(f"\n  Timestamp: {regime.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Position size multiplier: {regime.position_size_multiplier:.2f}x")
        print(f"  Take new positions: {'YES' if regime.should_take_new_positions else 'NO'}")
        print(f"\n  Reasoning: {regime.reasoning}")
        
        print(f"\n  ── INDEX HEALTH ──")
        for h in [regime.spy_health, regime.qqq_health]:
            print(f"\n  {h.symbol}:")
            print(f"    Price: ${h.current_price:.2f}  |  "
                  f"50MA: ${h.ma_50:.2f}  |  200MA: ${h.ma_200:.2f}")
            print(f"    Above 50MA: {'✓' if h.above_50ma else '✗'}  |  "
                  f"Above 200MA: {'✓' if h.above_200ma else '✗'}  |  "
                  f"50MA rising: {'✓' if h.ma_50_rising else '✗'}")
            print(f"    Distance from 50MA: {h.distance_from_50ma_pct:+.2f}%  |  "
                  f"Drawdown from 52W high: {h.drawdown_from_high_pct:+.2f}%")
        
        print(f"\n  ── DISTRIBUTION DAYS (last {DIST_DAY_LOOKBACK} sessions) ──")
        print(f"    Count: {regime.distribution_day_count}")
        if regime.distribution_day_count == 0:
            print(f"    Status: Clean tape, institutions holding")
        elif regime.distribution_day_count <= DIST_HEALTHY_MAX:
            print(f"    Status: Healthy (≤{DIST_HEALTHY_MAX} is normal)")
        elif regime.distribution_day_count < DIST_SERIOUS_MIN:
            print(f"    Status: ⚠️ Under pressure")
        else:
            print(f"    Status: 🚨 Heavy distribution")
        
        if regime.distribution_days:
            print(f"\n    Most recent:")
            for dd in regime.distribution_days[:5]:
                print(f"      {dd.date}  {dd.index}  "
                      f"{dd.decline_pct:+.2f}%  "
                      f"vol {dd.volume_change_pct:+.1f}%")
        
        if regime.warnings:
            print(f"\n  ── WARNINGS ──")
            for w in regime.warnings:
                print(f"    ⚠️ {w}")
        
        print("\n" + "="*75 + "\n")
    
    # ─────────────────────────────────────────────────────────────────
    # DATA FETCHING
    # ─────────────────────────────────────────────────────────────────
    def _fetch_index_data(self) -> bool:
        """Fetch SPY and QQQ historical data"""
        try:
            logger.info("Fetching SPY data...")
            # retries: a transient timeout must not drop the scan to watch-only
            spy_bars = self.connector.fetch_historical_data(
                "SPY", duration="1 Y", bar_size="1 day", retries=INDEX_FETCH_RETRIES
            )
            if not spy_bars or len(spy_bars) < 200:
                logger.error(f"❌ Insufficient SPY data")
                return False
            self.spy_df = self._bars_to_dataframe(spy_bars)
            
            logger.info("Fetching QQQ data...")
            qqq_bars = self.connector.fetch_historical_data(
                "QQQ", duration="1 Y", bar_size="1 day", retries=INDEX_FETCH_RETRIES
            )
            if not qqq_bars or len(qqq_bars) < 200:
                logger.error(f"❌ Insufficient QQQ data")
                return False
            self.qqq_df = self._bars_to_dataframe(qqq_bars)
            
            logger.info(f"✓ Loaded SPY ({len(self.spy_df)} days) "
                       f"and QQQ ({len(self.qqq_df)} days)")
            return True
            
        except Exception as e:
            logger.error(f"Error fetching index data: {e}")
            return False
    
    # ─────────────────────────────────────────────────────────────────
    # INDEX HEALTH ANALYSIS
    # ─────────────────────────────────────────────────────────────────
    def _analyze_index_health(self, df: pd.DataFrame, symbol: str) -> IndexHealth:
        """Analyze MA positioning and drawdown for an index"""
        current_price = df['close'].iloc[-1]
        
        ma_50 = df['close'].rolling(50).mean().iloc[-1]
        ma_200 = df['close'].rolling(200).mean().iloc[-1]
        
        # MA slope (rising if current MA > MA from N days ago)
        ma_50_series = df['close'].rolling(50).mean()
        ma_200_series = df['close'].rolling(200).mean()
        
        ma_50_rising = ma_50_series.iloc[-1] > ma_50_series.iloc[-MA_SLOPE_LOOKBACK]
        ma_200_rising = ma_200_series.iloc[-1] > ma_200_series.iloc[-MA_SLOPE_LOOKBACK]
        
        # 52-week high and drawdown
        high_52w = df['high'].iloc[-252:].max() if len(df) >= 252 else df['high'].max()
        drawdown_pct = ((current_price - high_52w) / high_52w) * 100
        
        return IndexHealth(
            symbol=symbol,
            current_price=current_price,
            ma_50=ma_50,
            ma_200=ma_200,
            above_50ma=(current_price > ma_50),
            above_200ma=(current_price > ma_200),
            ma_50_rising=ma_50_rising,
            ma_200_rising=ma_200_rising,
            distance_from_50ma_pct=((current_price - ma_50) / ma_50) * 100,
            drawdown_from_high_pct=drawdown_pct,
            high_52w=high_52w
        )
    
    # ─────────────────────────────────────────────────────────────────
    # DISTRIBUTION DAY DETECTION
    # ─────────────────────────────────────────────────────────────────
    def _count_distribution_days(
        self, df: pd.DataFrame, symbol: str
    ) -> List[DistributionDay]:
        """
        Count distribution days in the lookback period.
        
        Distribution Day = index closes DOWN >= threshold% on HIGHER volume
        than the previous trading day.
        
        Distribution days expire if the index rallies DIST_DAY_EXPIRY_GAIN%
        above the distribution day's close.
        """
        if len(df) < DIST_DAY_LOOKBACK + 1:
            return []
        
        # Work with recent window
        window = df.tail(DIST_DAY_LOOKBACK + 1).copy().reset_index(drop=True)
        window['daily_change_pct'] = window['close'].pct_change() * 100
        window['volume_change_pct'] = window['volume'].pct_change() * 100
        window['higher_volume'] = window['volume'] > window['volume'].shift(1)
        
        # Identify candidate distribution days
        candidates = window[
            (window['daily_change_pct'] <= DIST_DAY_DECLINE_THRESHOLD) &
            (window['higher_volume'] == True)
        ].copy()
        
        # Check expiry: has the index rallied 5%+ above the D-day close since then?
        current_price = df['close'].iloc[-1]
        dist_days = []
        
        for _, row in candidates.iterrows():
            dd_close = row['close']
            gain_since = ((current_price - dd_close) / dd_close) * 100
            
            if gain_since < DIST_DAY_EXPIRY_GAIN:
                # Still valid
                dist_days.append(DistributionDay(
                    date=row['date'].strftime('%Y-%m-%d'),
                    decline_pct=row['daily_change_pct'],
                    volume_change_pct=row['volume_change_pct'],
                    index=symbol
                ))
        
        return dist_days
    
    # ─────────────────────────────────────────────────────────────────
    # REGIME CLASSIFICATION LOGIC
    # ─────────────────────────────────────────────────────────────────
    def _classify_regime(
        self,
        spy_health: IndexHealth,
        qqq_health: IndexHealth,
        dist_count: int,
        spy_dist_days: List[DistributionDay]
    ) -> Tuple[str, str, List[str]]:
        """
        Apply O'Neil regime rules to classify market.
        Returns (regime_status, reasoning, warnings)
        """
        warnings = []
        reasons = []
        
        # ─── CORRECTION CHECKS (highest priority) ─────────────────────
        # Rule 1: Either index below 50MA AND in drawdown > pressure threshold
        if (not spy_health.above_50ma or not qqq_health.above_50ma):
            if (spy_health.drawdown_from_high_pct <= DRAWDOWN_PRESSURE or
                qqq_health.drawdown_from_high_pct <= DRAWDOWN_PRESSURE):
                reasons.append(
                    f"Index below 50MA "
                    f"(SPY: {'above' if spy_health.above_50ma else 'BELOW'}, "
                    f"QQQ: {'above' if qqq_health.above_50ma else 'BELOW'})"
                )
                # Severe correction?
                if (spy_health.drawdown_from_high_pct <= DRAWDOWN_CORRECTION or
                    qqq_health.drawdown_from_high_pct <= DRAWDOWN_CORRECTION):
                    reasons.append(
                        f"Drawdown severe: SPY {spy_health.drawdown_from_high_pct:.1f}%, "
                        f"QQQ {qqq_health.drawdown_from_high_pct:.1f}%"
                    )
                    return (
                        REGIME_CORRECTION,
                        " | ".join(reasons),
                        warnings + ["Severe drawdown — preserve capital"]
                    )
        
        # Rule 2: Heavy distribution
        if dist_count >= DIST_SERIOUS_MIN:
            reasons.append(f"{dist_count} distribution days in last {DIST_DAY_LOOKBACK} sessions")
            return (
                REGIME_CORRECTION,
                " | ".join(reasons),
                warnings + [
                    f"Institutional selling — {dist_count} D-days is a major warning"
                ]
            )
        
        # Rule 3: 200MA break is serious
        if not spy_health.above_200ma or not qqq_health.above_200ma:
            reasons.append("Index below 200MA — long-term trend broken")
            return (
                REGIME_CORRECTION,
                " | ".join(reasons),
                warnings + ["200MA break signals major trend change"]
            )
        
        # ─── RALLY ATTEMPT CHECKS ─────────────────────────────────────
        # Recent correction that's trying to rally back
        if (spy_health.drawdown_from_high_pct <= DRAWDOWN_CORRECTION * 0.75):
            # Still deep in drawdown but holding 50MA — rally attempt
            if spy_health.above_50ma or qqq_health.above_50ma:
                reasons.append(
                    f"Attempting rally from drawdown "
                    f"(SPY {spy_health.drawdown_from_high_pct:.1f}% from high)"
                )
                warnings.append("Unconfirmed rally — wait for follow-through day")
                return (
                    REGIME_RALLY_ATTEMPT,
                    " | ".join(reasons),
                    warnings
                )
        
        # ─── UPTREND UNDER PRESSURE CHECKS ─────────────────────────────
        pressure_conditions = 0
        pressure_reasons = []
        
        if dist_count >= DIST_PRESSURE_MIN:
            pressure_conditions += 1
            pressure_reasons.append(f"{dist_count} D-days")
        
        if not spy_health.ma_50_rising or not qqq_health.ma_50_rising:
            pressure_conditions += 1
            pressure_reasons.append("50MA flattening")
        
        if (spy_health.drawdown_from_high_pct <= DRAWDOWN_PRESSURE or
            qqq_health.drawdown_from_high_pct <= DRAWDOWN_PRESSURE):
            pressure_conditions += 1
            pressure_reasons.append(
                f"Drawdown (SPY: {spy_health.drawdown_from_high_pct:.1f}%)"
            )
        
        if pressure_conditions >= 1:
            reasons.extend(pressure_reasons)
            return (
                REGIME_UPTREND_UNDER_PRESSURE,
                " | ".join(reasons),
                warnings + [
                    "Trade smaller size, highest-quality setups only",
                    "Tighten stops on existing positions"
                ]
            )
        
        # ─── CONFIRMED UPTREND (default if nothing else triggers) ──────
        reasons.append("Both indices above 50MA and 200MA")
        reasons.append(f"50MA rising, {dist_count} D-days (healthy)")
        if spy_health.drawdown_from_high_pct > -2.0:
            reasons.append("Near 52W highs")
        
        return (REGIME_CONFIRMED_UPTREND, " | ".join(reasons), warnings)
    
    # ─────────────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────────────
    def _bars_to_dataframe(self, bars) -> pd.DataFrame:
        """Convert IBKR bars to DataFrame"""
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


# ═══════════════════════════════════════════════════════════════════════════
# STANDALONE TEST
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    """
    Run this module directly to test regime detection.
    Requires IBKR TWS running on port 7497.
    """
    import sys
    from pathlib import Path
    
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Import connector from your project
    try:
        from ibkr_connector import IBKRConnector
        from config import TWS_HOST, TWS_PORT, CLIENT_ID
    except ImportError:
        print("❌ Run from your project directory (needs ibkr_connector.py and config.py)")
        sys.exit(1)
    
    print("\n" + "="*70)
    print("  MARKET REGIME DETECTOR - STANDALONE TEST")
    print("="*70)
    
    connector = IBKRConnector()
    if not connector.connect_and_run(TWS_HOST, TWS_PORT, CLIENT_ID):
        print("❌ Failed to connect to TWS")
        sys.exit(1)
    
    try:
        detector = MarketRegimeDetector(connector)
        regime = detector.detect_regime()
        detector.display_regime(regime)
        
        # Show how to use the output
        if regime:
            print("\n  ── USAGE EXAMPLE ──")
            print(f"  regime.should_take_new_positions = {regime.should_take_new_positions}")
            print(f"  regime.position_size_multiplier = {regime.position_size_multiplier}")
            print(f"\n  If you normally risk 0.5% per trade:")
            adjusted = 0.5 * regime.position_size_multiplier
            print(f"  → Adjusted risk: {adjusted:.3f}% per trade")
            print()
    finally:
        connector.disconnect_and_stop()
