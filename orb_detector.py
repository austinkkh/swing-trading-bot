"""
═══════════════════════════════════════════════════════════════════════════════
OPENING RANGE BREAKOUT (ORB) DETECTOR
Based on Qullamaggie / Kullamägi / Minervini methodology
─────────────────────────────────────────────────────────────────────────────
Configurable ORB rules (default 10-minute ORB):
  1. First ORB window (default 9:30-9:40) defines the opening range
  2. ORH = high of opening bar
  3. ORL = low of opening bar (acts as stop)
  4. Entry = price breaks ORH after ORB formation on volume confirmation
  5. Risk = (ORH - ORL) / current_price * 100

Usage:
  detector = ORBDetector(ibkr_connector)
  signal = detector.check_orb_breakout("SNDK")
  if signal['has_breakout']:
      enter_trade(signal)
═══════════════════════════════════════════════════════════════════════════════
"""

import logging
import pandas as pd

from market_time import bar_lag_minutes, now_et, parse_ibkr_times
import strategy_params as _sp
from datetime import datetime, timedelta, time as dt_time
from dataclasses import dataclass, field
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# ORB CONFIG
# ═══════════════════════════════════════════════════════════════════════════
ORB_BAR_SIZE = "5 mins"              # Intraday fetch granularity
ORB_WINDOW_MINUTES = 10              # Default ORB window (set 5 for experiment)
ORB_OPENING_TIME = dt_time(9, 30)    # Market open
# Opening range is complete once the window has elapsed after the open (9:40 for 10 min)
if ORB_WINDOW_MINUTES % 5 != 0:
    raise ValueError(f"ORB_WINDOW_MINUTES must be a multiple of 5 (5-min bars), got {ORB_WINDOW_MINUTES}")
ORB_FORMATION_END = (datetime.combine(datetime.today(), ORB_OPENING_TIME)
                     + timedelta(minutes=ORB_WINDOW_MINUTES)).time()
ORB_HUNT_END = dt_time(10, 30)       # Stop hunting after this (stale signals)

# Breakout criteria
ORB_BREAKOUT_BUFFER_PCT = 0.1        # Price must be 0.1%+ above ORH (not just touch)
ORB_VOLUME_CONFIRM_MULT = 1.5        # Current minute volume > 1.5x avg
ORB_MAX_RISK_PCT = _sp.MAX_STOP_PCT  # Entry -> ORL stop must be within the tight stop band
ORB_MIN_STOP_PCT = 1.0               # Tighter stops get hit by noise (matches STOP_PCT_MIN)
ORB_MIN_GAP_PCT = -1.0               # Reject if gap-down more than 1%
ORB_MAX_EXTENSION_PCT = 3.0          # Reject if already 3%+ above ORH (chased it)

ORB_BAR_MINUTES = 5                  # matches ORB_BAR_SIZE
DELAYED_DATA_MINUTES = 15            # IBKR delayed data (TWS 2188) trails the clock by ~15 min


@dataclass
class ORBSignal:
    """Output of ORB check for a single stock"""
    symbol: str
    has_breakout: bool
    
    # Opening range data
    orh: float = 0.0
    orl: float = 0.0
    or_range_pct: float = 0.0    # (ORH-ORL)/ORL * 100
    
    # Current state
    current_price: float = 0.0
    distance_above_orh_pct: float = 0.0
    volume_ratio: float = 0.0
    
    # Risk metrics
    stop_price: float = 0.0      # ORL is the stop
    stop_pct: float = 0.0
    
    # Status
    status: str = "NO_DATA"      # NO_DATA / BEFORE_FORMATION / FORMING / NO_BREAK / BROKE_OUT / EXTENDED / RISK_TOO_HIGH
    reason: str = ""
    # Minutes the newest bar's end trails the clock (None = no bars). ~15 = delayed data:
    # the price above is that stale. Informational; nothing gates on it yet.
    data_lag_min: Optional[float] = None
    
    timestamp: Optional[datetime] = None


class ORBDetector:
    """
    Detect Opening Range Breakouts using IBKR intraday bars.
    """
    
    def __init__(self, ibkr_connector):
        self.connector = ibkr_connector
        self._cache = {}  # symbol → (date, opening_bar) — avoid re-fetching
    
    def is_orb_window_active(self) -> bool:
        """ORB hunt is active after ORB formation window until 10:30am"""
        now = now_et().time()
        return ORB_FORMATION_END <= now <= ORB_HUNT_END
    
    def is_orb_forming(self) -> bool:
        """Opening ORB window still forming"""
        now = now_et().time()
        return ORB_OPENING_TIME <= now < ORB_FORMATION_END
    
    def check_orb_breakout(self, symbol: str) -> ORBSignal:
        """
        Main entry: check if symbol has a valid ORB breakout right now.
        """
        signal = ORBSignal(symbol=symbol, has_breakout=False, timestamp=now_et())
        
        # ─── TIME CHECKS ───
        if self.is_orb_forming():
            signal.status = "FORMING"
            signal.reason = f"Opening ORB window still forming (before {ORB_FORMATION_END.strftime('%H:%M')})"
            return signal
        
        if not self.is_orb_window_active():
            signal.status = "OUT_OF_WINDOW"
            signal.reason = f"Outside ORB hunt window ({ORB_FORMATION_END.strftime('%H:%M')}-10:30)"
            return signal
        
        # ─── FETCH INTRADAY BARS ───
        bars = self._fetch_intraday_bars(symbol)
        if not bars or len(bars) < 1:
            signal.status = "NO_DATA"
            signal.reason = "Could not fetch intraday bars"
            return signal
        
        df = self._bars_to_dataframe(bars)
        if not df.empty:
            signal.data_lag_min = bar_lag_minutes(df['date'].iloc[-1].to_pydatetime(), ORB_BAR_MINUTES,
                                                  signal.timestamp)
        
        # ─── IDENTIFY OPENING RANGE ───
        opening_range = self._get_opening_range(df)
        if opening_range is None:
            signal.status = "NO_OPENING_BAR"
            signal.reason = "No usable opening-range bars found"
            return signal
        
        signal.orh = round(opening_range['high'], 2)
        signal.orl = round(opening_range['low'], 2)
        signal.or_range_pct = round(((signal.orh - signal.orl) / signal.orl) * 100, 2)
        signal.stop_price = signal.orl
        
        # ─── GET CURRENT PRICE (latest bar close) ───
        current_bar = df.iloc[-1]
        signal.current_price = round(current_bar['close'], 2)
        signal.stop_pct = round(((signal.current_price - signal.orl) / signal.current_price) * 100, 2)
        
        # ─── BREAKOUT CHECK: Price above ORH? ───
        breakout_threshold = signal.orh * (1 + ORB_BREAKOUT_BUFFER_PCT / 100)
        if signal.current_price < breakout_threshold:
            signal.status = "NO_BREAK"
            signal.reason = f"Price ${signal.current_price:.2f} below ORH ${signal.orh:.2f} + buffer"
            return signal
        
        signal.distance_above_orh_pct = round(
            ((signal.current_price - signal.orh) / signal.orh) * 100, 2
        )
        
        # ─── EXTENSION CHECK: Already chased? ───
        if signal.distance_above_orh_pct > ORB_MAX_EXTENSION_PCT:
            signal.status = "EXTENDED"
            signal.reason = f"Already {signal.distance_above_orh_pct:.1f}% above ORH (chased)"
            return signal
        
        # ─── RISK CHECK: entry -> ORL stop inside the tight stop band ───
        # The range width alone isn't the risk: entries up to ORB_MAX_EXTENSION_PCT above ORH
        # make the real stop distance wider than the range.
        if signal.stop_pct > ORB_MAX_RISK_PCT:
            signal.status = "RISK_TOO_HIGH"
            signal.reason = f"Stop {signal.stop_pct:.1f}% (entry to ORL) exceeds {ORB_MAX_RISK_PCT}% max"
            return signal
        if signal.stop_pct < ORB_MIN_STOP_PCT:
            signal.status = "STOP_TOO_TIGHT"
            signal.reason = f"Stop {signal.stop_pct:.2f}% (entry to ORL) under {ORB_MIN_STOP_PCT}% minimum"
            return signal
        
        # ─── VOLUME CONFIRMATION ───
        signal.volume_ratio = self._calc_volume_ratio(df, delay_min=self._data_delay_min())
        if signal.volume_ratio < ORB_VOLUME_CONFIRM_MULT:
            signal.status = "NO_VOLUME"
            signal.reason = f"Volume {signal.volume_ratio:.1f}x avg, need {ORB_VOLUME_CONFIRM_MULT}x"
            return signal
        
        # ─── ALL CHECKS PASSED ───
        signal.has_breakout = True
        signal.status = "BROKE_OUT"
        signal.reason = "Valid ORB breakout with volume confirmation"
        
        return signal
    
    def _data_delay_min(self) -> float:
        """Minutes the feed trails the clock: DELAYED_DATA_MINUTES once TWS sent 2188, else 0."""
        return DELAYED_DATA_MINUTES if getattr(self.connector, 'data_delayed', False) else 0

    def _fetch_intraday_bars(self, symbol: str) -> Optional[list]:
        """Fetch today's 5-min bars from IBKR"""
        try:
            bars = self.connector.fetch_historical_data(
                symbol=symbol,
                duration="1 D",
                bar_size=ORB_BAR_SIZE
            )
            return bars
        except Exception as e:
            logger.debug(f"Intraday fetch error {symbol}: {e}")
            return None
    
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
        # IBKR intraday dates look like '20260724 09:30:00 US/Eastern'; pd.to_datetime can't
        # parse the zone, which made every ORB check fail silently. Parse to naive Eastern time.
        df['date'] = parse_ibkr_times(df['date'])
        return df
    
    def _get_opening_range(self, df: pd.DataFrame):
        """Find ORB opening range using first N five-minute bars."""
        if df.empty:
            return None
        
        today = now_et().date()
        today_bars = df[df['date'].dt.date == today].sort_values('date').reset_index(drop=True)
        
        if today_bars.empty:
            return None

        # Only bars that start inside the opening window. Taking the first N rows would
        # pull in later bars (even the one still forming) when the 9:30 bar is missing.
        bar_times = today_bars['date'].dt.time
        window = today_bars[(bar_times >= ORB_OPENING_TIME) & (bar_times < ORB_FORMATION_END)]
        bars_needed = ORB_WINDOW_MINUTES // 5
        if len(window) < bars_needed or window['date'].iloc[0].time() != ORB_OPENING_TIME:
            return None
        # The window's last bar is closed only once a later bar exists. With delayed data the
        # 9:35 bar is still forming at ~9:50 real time: a partial bar must not set ORH/ORL.
        if not (bar_times >= ORB_FORMATION_END).any():
            return None
        return {
            'high': float(window['high'].max()),
            'low': float(window['low'].min()),
        }
    
    def _calc_volume_ratio(self, df: pd.DataFrame, now=None, delay_min: float = 0) -> float:
        """
        Latest bar's volume vs the average of today's earlier bars. The latest 5-minute bar is
        usually still forming, so its volume is scaled to a full-bar pace (comparing a 1-minute-old
        bar with full bars made real breakouts look like ~0.2x volume).
        delay_min: how far the feed trails the wall clock (15 on delayed data). "Still forming"
        is judged on the data clock, else a delayed partial bar looks complete and its volume
        is understated (breakouts rejected as NO_VOLUME).
        """
        if len(df) < 2:
            return 1.0
        now = (now or now_et()) - timedelta(minutes=delay_min)   # data clock
        today = df[df['date'].dt.date == now.date()]
        if len(today) < 2:
            today = df
        last = today.iloc[-1]
        prior_avg = today.iloc[:-1]['volume'].mean()
        if prior_avg == 0:
            return 1.0
        last_vol = float(last['volume'])
        elapsed = (now - last['date']).total_seconds()
        if 0 <= elapsed < 300:  # still forming
            last_vol *= 300 / max(elapsed, 30)
        return round(last_vol / prior_avg, 2)

    def display_signal(self, signal: ORBSignal):
        """Pretty-print an ORB signal"""
        if signal.has_breakout:
            print("\n" + "🚨"*40)
            print(f"\n  ⚡⚡⚡  ORB BREAKOUT: {signal.symbol}  ⚡⚡⚡")
            print("\n" + "🚨"*40)
            print(f"\n  Opening Range:    ${signal.orl:.2f} - ${signal.orh:.2f}")
            print(f"  Range Width:      {signal.or_range_pct:.2f}%")
            print(f"  Current Price:    ${signal.current_price:.2f}")
            print(f"  Above ORH:        +{signal.distance_above_orh_pct:.2f}%")
            print(f"  Volume:           {signal.volume_ratio:.1f}x average")
            print(f"  Stop (ORL):       ${signal.stop_price:.2f} (-{signal.stop_pct:.2f}% risk)")
            print("\n" + "🚨"*40 + "\n")
        else:
            print(f"  {signal.symbol}: {signal.status} — {signal.reason}")


# ═══════════════════════════════════════════════════════════════════════════
# STANDALONE TEST
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    """
    Test ORB detection on a few symbols.
    Run during market hours (after ORB formation - 10:30am) for live data.
    """
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    try:
        from ibkr_connector import IBKRConnector
        from config import TWS_HOST, TWS_PORT, CLIENT_ID
    except ImportError:
        print("❌ Run from the repo directory")
        sys.exit(1)
    
    print("\n" + "="*70)
    print("  ORB DETECTOR - STANDALONE TEST")
    print("="*70)
    print(f"  Current time (ET): {now_et().strftime('%H:%M:%S')}")
    
    now = now_et().time()
    if now < ORB_OPENING_TIME:
        print(f"  ⚠️  Market not open yet (opens 9:30)")
    elif now < ORB_FORMATION_END:
        print(f"  ⏳  Opening ORB window still forming (until {ORB_FORMATION_END.strftime('%H:%M')})")
    elif now > ORB_HUNT_END:
        print(f"  ⏰  Past ORB hunt window (ended 10:30)")
    else:
        print(f"  ✅  ORB hunt window ACTIVE")
    
    print("="*70 + "\n")
    
    connector = IBKRConnector()
    if not connector.connect_and_run(TWS_HOST, TWS_PORT, CLIENT_ID):
        print("❌ Failed to connect to TWS")
        sys.exit(1)
    
    try:
        detector = ORBDetector(connector)
        
        test_symbols = ["SNDK", "LITE", "CIEN", "VIAV", "AMD", "NVDA"]
        
        print(f"Testing {len(test_symbols)} symbols...\n")
        
        for symbol in test_symbols:
            signal = detector.check_orb_breakout(symbol)
            detector.display_signal(signal)
        
    finally:
        connector.disconnect_and_stop()
