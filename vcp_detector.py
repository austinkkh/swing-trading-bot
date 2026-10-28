#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
VCP DETECTOR v2 — Real Volatility Contraction Pattern Detection
Based on Mark Minervini's SEPA / Master Trader Program methodology

Key differences from v1:
  - Uses pivot detection (swing high/low algorithm) instead of "tightness scoring"
  - Identifies actual pullbacks, not just average ATR contraction
  - Minervini's reset rule: if a pullback is DEEPER than the previous one,
    the contraction count resets (pattern restart)
  - Outputs structured data with pivot point, pullback details, and pivot distance
  - Distinguishes Stage 2 trending pullbacks from real base-building VCPs

VCP Definition:
  1. Stock has formed a base (sideways or slightly downward consolidation)
  2. Within the base: 3+ pullbacks, each shallower than the previous
  3. Each pullback off a swing high to a swing low
  4. Final pullback typically <10%, ideally <5%
  5. Volume dries up into the final contraction
  6. Pivot point = the most recent swing high (the breakout level)
═══════════════════════════════════════════════════════════════════════════════
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════
PIVOT_LOOKBACK = 5                  # N bars on each side for pivot detection
VCP_LOOKBACK_BARS = 90              # Look back N days for pattern
MIN_CONTRACTIONS_REQUIRED = 3       # Minimum to qualify as VCP
MIN_PATTERN_DAYS = 14               # Pattern must span at least N days
MAX_PATTERN_DAYS = 65               # Don't reach back further than N days

# Tightness thresholds
EXCELLENT_FINAL_PULLBACK = 5.0      # Final pullback ≤ 5% = excellent
GOOD_FINAL_PULLBACK = 8.0           # Final pullback ≤ 8% = good  
MAX_FINAL_PULLBACK = 12.0           # Final pullback ≤ 12% = acceptable
MAX_INITIAL_PULLBACK = 35.0         # Initial pullback ≤ 35%

# Volume validation
VOLUME_DRYUP_THRESHOLD = 0.85       # Recent volume must be ≤ 85% of earlier

# Contraction sequence
NEAR_EQUAL_TOLERANCE = 0.10         # Older pullback may be up to 10% shallower (relative) and still count
LIVE_PULLBACK_MIN_PCT = 2.0         # In-progress dips smaller than this are noise, not a contraction


# ═══════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class Pivot:
    """A swing high or swing low"""
    index: int
    date: pd.Timestamp
    price: float
    pivot_type: str  # 'HIGH' or 'LOW'
    volume: float = 0.0


@dataclass
class Pullback:
    """A swing-high → swing-low pullback"""
    swing_high: Pivot
    swing_low: Pivot
    depth_pct: float
    duration_bars: int
    
    def __repr__(self):
        return f"Pullback(-{self.depth_pct:.1f}%, {self.duration_bars}d)"


@dataclass
class VCPResult:
    """Output of VCP detection"""
    has_vcp: bool = False
    score: float = 0.0
    
    # Pattern structure
    num_contractions: int = 0
    pullbacks: List[Pullback] = field(default_factory=list)
    
    # Pivot data
    pivot_price: float = 0.0
    pivot_distance_pct: float = 0.0
    final_contraction_pct: float = 0.0
    initial_contraction_pct: float = 0.0
    
    # Quality metrics
    tightness_progression: str = "UNKNOWN"  # DECREASING / RESET / INSUFFICIENT
    volume_dryup: bool = False
    days_in_pattern: int = 0
    
    # For backwards compat with v1
    details: str = ""
    
    def to_dict(self):
        return {
            'has_vcp': self.has_vcp,
            'score': self.score,
            'num_contractions': self.num_contractions,
            'pivot_price': round(self.pivot_price, 2),
            'pivot_distance_pct': round(self.pivot_distance_pct, 2),
            'final_contraction_pct': round(self.final_contraction_pct, 2),
            'initial_contraction_pct': round(self.initial_contraction_pct, 2),
            'tightness_progression': self.tightness_progression,
            'volume_dryup': self.volume_dryup,
            'days_in_pattern': self.days_in_pattern,
            'pullbacks': [
                {
                    'high_date': pb.swing_high.date.strftime('%Y-%m-%d'),
                    'high_price': round(pb.swing_high.price, 2),
                    'low_date': pb.swing_low.date.strftime('%Y-%m-%d'),
                    'low_price': round(pb.swing_low.price, 2),
                    'depth_pct': round(pb.depth_pct, 2),
                    'duration_bars': pb.duration_bars
                }
                for pb in self.pullbacks
            ],
            'details': self.details
        }


# ═══════════════════════════════════════════════════════════════════════════
# VCP DETECTOR
# ═══════════════════════════════════════════════════════════════════════════
class VCPDetector:
    """
    Real VCP detection using pivot algorithm + Minervini reset rule.
    
    Backwards compatible with v1 interface — has detect_vcp_pattern() method
    that returns dict-like result for the existing scanner.
    """
    
    def __init__(self, atr_period: int = 14, pivot_lookback: int = PIVOT_LOOKBACK):
        self.atr_period = atr_period
        self.pivot_lookback = pivot_lookback
    
    # ─────────────────────────────────────────────────────────────────
    # PUBLIC INTERFACE (v1-compatible)
    # ─────────────────────────────────────────────────────────────────
    def detect_vcp_pattern(self, bars, min_contractions: int = MIN_CONTRACTIONS_REQUIRED) -> dict:
        """
        v1-compatible entry point. Returns dict with same keys as v1 + new fields.
        """
        if not bars or len(bars) < 30:
            return self._empty_result("Insufficient data")
        
        df = self._bars_to_dataframe(bars)
        result = self.detect(df, min_contractions=min_contractions)
        
        # Convert to v1-compatible dict
        return {
            'score': result.score,
            'has_vcp': result.has_vcp,
            'num_contractions': result.num_contractions,
            'pivot_price': round(result.pivot_price, 2),
            'pivot_distance_pct': round(result.pivot_distance_pct, 2),
            'final_contraction_pct': round(result.final_contraction_pct, 2),
            'initial_contraction_pct': round(result.initial_contraction_pct, 2),
            'tightness_progression': result.tightness_progression,
            'volume_dryup': result.volume_dryup,
            'days_in_pattern': result.days_in_pattern,
            'pullbacks': [
                {
                    'high_price': round(pb.swing_high.price, 2),
                    'low_price': round(pb.swing_low.price, 2),
                    'depth_pct': round(pb.depth_pct, 2),
                    'duration_bars': pb.duration_bars
                }
                for pb in result.pullbacks
            ],
            'details': result.details,
            # Legacy compat fields (v1 returned these)
            'atr_pct': 0,
            'range_pct': 0,
            'volume_trend': 'declining' if result.volume_dryup else 'mixed',
            'distance_from_pivot': round(result.pivot_distance_pct, 2)
        }
    
    def detect(self, df: pd.DataFrame, min_contractions: int = MIN_CONTRACTIONS_REQUIRED) -> VCPResult:
        """
        Main detection logic. Operates on a DataFrame with OHLCV columns.
        """
        result = VCPResult()
        
        if len(df) < 30:
            result.details = "Insufficient data"
            return result
        
        # Use only recent N bars for pattern detection
        window = df.tail(VCP_LOOKBACK_BARS).reset_index(drop=True)
        
        # ─── 1. Find all pivot highs and lows ───
        pivots = self._find_pivots(window)
        
        if len(pivots) < 4:  # Need at least 2 highs + 2 lows
            result.details = f"Too few pivots ({len(pivots)})"
            return result
        
        # ─── 2. Build pullback sequence (including the one still forming) ───
        all_pullbacks = self._build_pullbacks(pivots, window)
        
        if not all_pullbacks:
            result.details = "No valid pullbacks identified"
            return result
        
        # ─── 3. Apply Minervini's reset rule ───
        contracting_sequence = self._apply_reset_rule(all_pullbacks)
        
        result.num_contractions = len(contracting_sequence)
        result.pullbacks = contracting_sequence
        
        if len(contracting_sequence) < min_contractions:
            result.details = (
                f"Only {len(contracting_sequence)} contracting pullbacks "
                f"(need {min_contractions}). Last reset interrupted progression."
            )
            return result
        
        # ─── 4. Final contraction must be tight enough ───
        final_pb = contracting_sequence[-1]
        result.final_contraction_pct = final_pb.depth_pct
        result.initial_contraction_pct = contracting_sequence[0].depth_pct
        
        if final_pb.depth_pct > MAX_FINAL_PULLBACK:
            result.details = f"Final contraction too loose ({final_pb.depth_pct:.1f}%)"
            return result
        
        if contracting_sequence[0].depth_pct > MAX_INITIAL_PULLBACK:
            result.details = f"Initial pullback too deep ({contracting_sequence[0].depth_pct:.1f}%)"
            return result
        
        # ─── 5. Pattern duration check ───
        first_high = contracting_sequence[0].swing_high
        last_low = contracting_sequence[-1].swing_low
        days_span = last_low.index - first_high.index
        result.days_in_pattern = days_span
        
        if days_span < MIN_PATTERN_DAYS:
            result.details = f"Pattern too short ({days_span} bars)"
            return result
        
        if days_span > MAX_PATTERN_DAYS:
            result.details = f"Pattern too old ({days_span} bars)"
            return result
        
        # ─── 6. Pivot point = last swing high in sequence ───
        result.pivot_price = contracting_sequence[-1].swing_high.price
        current_price = window['close'].iloc[-1]
        result.pivot_distance_pct = ((current_price - result.pivot_price) / result.pivot_price) * 100
        
        # ─── 7. Volume dryup check ───
        result.volume_dryup = self._check_volume_dryup(window, contracting_sequence)
        
        # ─── 8. Tightness progression label ───
        result.tightness_progression = "DECREASING"  # We know this since reset rule passed
        
        # ─── 9. SCORE THE VCP ───
        result.score = self._score_vcp(contracting_sequence, result)
        result.has_vcp = result.score >= 5.0
        
        result.details = (
            f"{result.num_contractions} contractions, "
            f"{result.initial_contraction_pct:.0f}% → {result.final_contraction_pct:.0f}%, "
            f"pivot ${result.pivot_price:.2f}, {result.days_in_pattern}d in pattern"
        )
        
        return result
    
    # ─────────────────────────────────────────────────────────────────
    # PIVOT DETECTION
    # ─────────────────────────────────────────────────────────────────
    def _find_pivots(self, df: pd.DataFrame) -> List[Pivot]:
        """
        Find swing highs and swing lows using N-bar lookback algorithm.
        A bar at index i is a swing high if its high > all highs in [i-N, i-1] and [i+1, i+N].
        """
        pivots = []
        n = self.pivot_lookback
        
        if len(df) < (2 * n + 1):
            return pivots
        
        for i in range(n, len(df) - n):
            current_high = df['high'].iloc[i]
            current_low = df['low'].iloc[i]
            
            # Check if swing high
            left_highs = df['high'].iloc[i-n:i].values
            right_highs = df['high'].iloc[i+1:i+n+1].values
            
            if current_high > left_highs.max() and current_high > right_highs.max():
                pivots.append(Pivot(
                    index=i,
                    date=df['date'].iloc[i] if 'date' in df.columns else pd.Timestamp.now(),
                    price=current_high,
                    pivot_type='HIGH',
                    volume=df['volume'].iloc[i]
                ))
            
            # Check if swing low
            left_lows = df['low'].iloc[i-n:i].values
            right_lows = df['low'].iloc[i+1:i+n+1].values
            
            if current_low < left_lows.min() and current_low < right_lows.min():
                pivots.append(Pivot(
                    index=i,
                    date=df['date'].iloc[i] if 'date' in df.columns else pd.Timestamp.now(),
                    price=current_low,
                    pivot_type='LOW',
                    volume=df['volume'].iloc[i]
                ))
        
        # Sort by index
        pivots.sort(key=lambda p: p.index)
        return pivots
    
    # ─────────────────────────────────────────────────────────────────
    # PULLBACK BUILDING
    # ─────────────────────────────────────────────────────────────────
    def _build_pullbacks(self, pivots: List[Pivot], df: Optional[pd.DataFrame] = None) -> List[Pullback]:
        """
        Walk pivots in chronological order. Each HIGH→LOW pair = one pullback.
        We need alternating HIGH/LOW structure.
        With df, also appends the in-progress pullback that confirmed pivots can't see yet.
        """
        if len(pivots) < 2:
            return []
        
        # Filter to alternating sequence (avoid two highs in a row)
        cleaned = self._clean_pivot_sequence(pivots)
        
        pullbacks = []
        
        for i in range(len(cleaned) - 1):
            current = cleaned[i]
            next_p = cleaned[i + 1]
            
            # Only count HIGH → LOW transitions as pullbacks
            if current.pivot_type == 'HIGH' and next_p.pivot_type == 'LOW':
                depth_pct = ((current.price - next_p.price) / current.price) * 100
                duration = next_p.index - current.index
                
                if depth_pct > 0:  # Sanity check
                    pullbacks.append(Pullback(
                        swing_high=current,
                        swing_low=next_p,
                        depth_pct=depth_pct,
                        duration_bars=duration
                    ))

        if df is not None:
            live, replaces_last = self._live_pullback(cleaned, df, pullbacks)
            if live and replaces_last:
                pullbacks[-1] = live
            elif live:
                pullbacks.append(live)

        return pullbacks

    def _live_pullback(self, cleaned: List[Pivot], df: pd.DataFrame,
                       pullbacks: List[Pullback]) -> Tuple[Optional[Pullback], bool]:
        """
        The pullback still forming at the right edge of the chart, and whether it
        replaces the last confirmed pullback.

        A pivot needs N bars on each side to confirm, so the last N bars can never
        hold one. That hides the most recent decline, which in a VCP is usually the
        final, tightest contraction (and in a breaking stock, the deepest one).
        Take the highest high since the last confirmed swing low, and the lowest
        low after it. If that low undercuts the last confirmed low, the bounce in
        between was not a new swing: the last pullback simply kept going, so it's
        re-measured from its own swing high instead of the bounce high.
        """
        lows = [p for p in cleaned if p.pivot_type == 'LOW']
        segment = df.iloc[lows[-1].index + 1:] if lows else df
        if len(segment) < 2:
            return None, False

        hi_i = int(segment['high'].idxmax())
        lo_i = int(df['low'].iloc[hi_i:].idxmin())
        hi, lo = df['high'].iloc[hi_i], df['low'].iloc[lo_i]

        def pivot_at(i: int, price: float, kind: str) -> Pivot:
            return Pivot(index=i, date=df['date'].iloc[i] if 'date' in df.columns else pd.Timestamp.now(),
                         price=price, pivot_type=kind, volume=df['volume'].iloc[i])

        last = pullbacks[-1] if pullbacks else None
        if last and lows and lo < lows[-1].price and last.swing_low.index == lows[-1].index:
            depth_pct = (last.swing_high.price - lo) / last.swing_high.price * 100
            return Pullback(swing_high=last.swing_high, swing_low=pivot_at(lo_i, lo, 'LOW'),
                            depth_pct=depth_pct, duration_bars=lo_i - last.swing_high.index), True

        depth_pct = (hi - lo) / hi * 100
        if depth_pct < LIVE_PULLBACK_MIN_PCT or lo_i <= hi_i:
            return None, False
        if last and hi_i <= last.swing_low.index:
            return None, False  # Already counted as a confirmed pullback

        return Pullback(swing_high=pivot_at(hi_i, hi, 'HIGH'), swing_low=pivot_at(lo_i, lo, 'LOW'),
                        depth_pct=depth_pct, duration_bars=lo_i - hi_i), False
    
    def _clean_pivot_sequence(self, pivots: List[Pivot]) -> List[Pivot]:
        """
        Ensure pivots alternate HIGH/LOW. If two HIGHs in a row, keep the higher.
        If two LOWs in a row, keep the lower.
        """
        if not pivots:
            return []
        
        cleaned = [pivots[0]]
        
        for p in pivots[1:]:
            last = cleaned[-1]
            
            if p.pivot_type == last.pivot_type:
                # Same type — keep the more extreme one
                if p.pivot_type == 'HIGH' and p.price > last.price:
                    cleaned[-1] = p
                elif p.pivot_type == 'LOW' and p.price < last.price:
                    cleaned[-1] = p
                # else: keep current
            else:
                cleaned.append(p)
        
        return cleaned
    
    # ─────────────────────────────────────────────────────────────────
    # MINERVINI'S RESET RULE
    # ─────────────────────────────────────────────────────────────────
    def _apply_reset_rule(self, pullbacks: List[Pullback]) -> List[Pullback]:
        """
        Walk pullbacks in order. Build the LONGEST decreasing sequence ending at the most recent pullback.
        
        Minervini's rule: if a pullback is deeper than the previous one, RESET.
        We want the contraction sequence that LEADS UP TO the most recent pullback.
        
        Algorithm:
          - Start from most recent pullback, walk backward
          - Add pullbacks while each is shallower than the next (chronologically deeper backward)
          - Stop when we hit a pullback that's deeper than the previous (older) one
        """
        if not pullbacks:
            return []
        
        # We want a chronological forward sequence: pb[0] deepest, pb[-1] tightest
        # Walk backward from the most recent pullback
        sequence = [pullbacks[-1]]
        
        for i in range(len(pullbacks) - 2, -1, -1):
            candidate = pullbacks[i]
            deepest_newer = max(pb.depth_pct for pb in sequence)

            # Going BACKWARD in time, depths should INCREASE (older = deeper).
            # Near-equal still counts (e.g. 10% then 10%), within NEAR_EQUAL_TOLERANCE.
            # Compare to the deepest pullback already in the sequence, not just the
            # oldest: otherwise the tolerance compounds and 5 -> 5.5 -> 6 -> 6.6 -> 7.3%
            # (volatility expanding) would pass as a contraction.
            if candidate.depth_pct >= deepest_newer * (1 - NEAR_EQUAL_TOLERANCE):
                sequence.insert(0, candidate)
            else:
                # Reset — older pullback is shallower than newer one, breaks the contraction story
                break
        
        return sequence
    
    # ─────────────────────────────────────────────────────────────────
    # VOLUME DRYUP
    # ─────────────────────────────────────────────────────────────────
    def _check_volume_dryup(self, df: pd.DataFrame, sequence: List[Pullback]) -> bool:
        """
        Compare average volume during the most recent contraction vs the first contraction.
        Volume should dry up over time.
        """
        if len(sequence) < 2:
            return False
        
        first_pb = sequence[0]
        last_pb = sequence[-1]
        
        first_vol = df['volume'].iloc[first_pb.swing_high.index:first_pb.swing_low.index + 1].mean()
        last_vol = df['volume'].iloc[last_pb.swing_high.index:last_pb.swing_low.index + 1].mean()
        
        if first_vol == 0:
            return False
        
        return last_vol < first_vol * VOLUME_DRYUP_THRESHOLD
    
    # ─────────────────────────────────────────────────────────────────
    # SCORING
    # ─────────────────────────────────────────────────────────────────
    def _score_vcp(self, sequence: List[Pullback], result: VCPResult) -> float:
        """
        Score 0-10 based on:
          - Number of contractions (3 = baseline, 4+ = bonus)
          - Tightness of final contraction (lower = better)
          - Tightness progression ratio (steeper drop = better)
          - Volume dryup
          - Days in pattern (longer base = stronger)
        """
        score = 0.0
        
        # Number of contractions: 3 = 3pts, 4 = 4pts, 5+ = 5pts
        n = len(sequence)
        score += min(n, 5)  # Caps at 5
        
        # Final contraction tightness
        final = result.final_contraction_pct
        if final <= EXCELLENT_FINAL_PULLBACK:
            score += 3.0
        elif final <= GOOD_FINAL_PULLBACK:
            score += 2.0
        elif final <= MAX_FINAL_PULLBACK:
            score += 1.0
        
        # Progression ratio: how dramatically did pullbacks tighten?
        initial = result.initial_contraction_pct
        if initial > 0:
            ratio = final / initial
            if ratio <= 0.25:        # Final is ≤25% of initial (huge tightening)
                score += 1.5
            elif ratio <= 0.5:       # Final is ≤50% of initial
                score += 1.0
            elif ratio <= 0.75:
                score += 0.5
        
        # Volume dryup
        if result.volume_dryup:
            score += 1.0
        
        # Days in pattern (sweet spot is 25-50 days)
        days = result.days_in_pattern
        if 25 <= days <= 50:
            score += 0.5
        
        return min(score, 10.0)
    
    # ─────────────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────────────
    def _bars_to_dataframe(self, bars) -> pd.DataFrame:
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
    
    def _empty_result(self, reason: str) -> dict:
        """Return v1-compatible empty result"""
        return {
            'score': 0,
            'has_vcp': False,
            'num_contractions': 0,
            'pivot_price': 0,
            'pivot_distance_pct': 0,
            'final_contraction_pct': 0,
            'initial_contraction_pct': 0,
            'tightness_progression': 'INSUFFICIENT',
            'volume_dryup': False,
            'days_in_pattern': 0,
            'pullbacks': [],
            'details': reason,
            'atr_pct': 0,
            'range_pct': 0,
            'volume_trend': 'unknown',
            'distance_from_pivot': 0
        }


# ═══════════════════════════════════════════════════════════════════════════
# STANDALONE TEST
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    """
    Test VCP detection on synthetic data + a few real symbols if connected.
    """
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "="*70)
    print("  VCP DETECTOR v2 - TEST")
    print("="*70)
    
    # ─── Synthetic VCP test ───
    print("\n📊 Test 1: Synthetic perfect VCP")
    
    # Build a fake price series with 3 progressively tighter pullbacks
    np.random.seed(42)
    base_price = 100.0
    prices = []
    dates = pd.date_range(end='2026-04-05', periods=80, freq='D')
    
    # Bar 0-15: rise to 120 (initial run)
    for i in range(15):
        prices.append(base_price + i * 1.3 + np.random.uniform(-0.5, 0.5))
    
    # Bar 15-30: pullback 1 to 100 (~17% drop)
    high1 = prices[-1]
    for i in range(15):
        prices.append(high1 - (i+1) * 1.3 + np.random.uniform(-0.3, 0.3))
    
    # Bar 30-40: rise to 117
    low1 = prices[-1]
    for i in range(10):
        prices.append(low1 + (i+1) * 1.7 + np.random.uniform(-0.3, 0.3))
    
    # Bar 40-50: pullback 2 to 108 (~8% drop)
    high2 = prices[-1]
    for i in range(10):
        prices.append(high2 - (i+1) * 0.9 + np.random.uniform(-0.2, 0.2))
    
    # Bar 50-58: rise to 116
    low2 = prices[-1]
    for i in range(8):
        prices.append(low2 + (i+1) * 1.0 + np.random.uniform(-0.2, 0.2))
    
    # Bar 58-65: pullback 3 to 112 (~3% drop, very tight)
    high3 = prices[-1]
    for i in range(7):
        prices.append(high3 - (i+1) * 0.55 + np.random.uniform(-0.1, 0.1))
    
    # Bar 65-80: tight consolidation around 113
    low3 = prices[-1]
    for i in range(15):
        prices.append(low3 + 0.1 * i + np.random.uniform(-0.2, 0.2))
    
    df = pd.DataFrame({
        'date': dates,
        'open': prices,
        'high': [p + np.random.uniform(0.1, 0.5) for p in prices],
        'low': [p - np.random.uniform(0.1, 0.5) for p in prices],
        'close': prices,
        'volume': [1_000_000 * (1.5 - 0.7 * (i / 80)) for i in range(80)]  # Volume drying up
    })
    
    detector = VCPDetector()
    result = detector.detect(df, min_contractions=3)
    
    print(f"\n  has_vcp:               {result.has_vcp}")
    print(f"  score:                 {result.score:.1f}/10")
    print(f"  num_contractions:      {result.num_contractions}")
    print(f"  pivot_price:           ${result.pivot_price:.2f}")
    print(f"  pivot_distance:        {result.pivot_distance_pct:+.2f}%")
    print(f"  initial → final:       {result.initial_contraction_pct:.1f}% → {result.final_contraction_pct:.1f}%")
    print(f"  days_in_pattern:       {result.days_in_pattern}")
    print(f"  volume_dryup:          {result.volume_dryup}")
    print(f"  details:               {result.details}")
    
    if result.pullbacks:
        print(f"\n  Pullback sequence:")
        for i, pb in enumerate(result.pullbacks, 1):
            print(f"    {i}. -{pb.depth_pct:.1f}%  ({pb.duration_bars} bars)  "
                  f"${pb.swing_high.price:.2f} → ${pb.swing_low.price:.2f}")
    
    # ─── Test 2: Reset rule ───
    print("\n📊 Test 2: Reset rule (deeper pullback should reset count)")
    
    # Sequence: -20%, -12%, -18% (RESET HERE), -7%, -4%
    # Should detect only 2 contractions (-7%, -4%) not 5
    # Build with pivots at clear positions
    np.random.seed(1)
    test2_prices = []
    
    # Run-up
    for i in range(10):
        test2_prices.append(100 + i * 2)
    h1 = test2_prices[-1]  # 118
    
    # Pullback 1: -20% to ~94
    for i in range(8):
        test2_prices.append(h1 - (i+1) * 3)
    l1 = test2_prices[-1]
    
    # Rise
    for i in range(8):
        test2_prices.append(l1 + (i+1) * 2.5)
    h2 = test2_prices[-1]
    
    # Pullback 2: -12% to ~98
    for i in range(6):
        test2_prices.append(h2 - (i+1) * 2.3)
    l2 = test2_prices[-1]
    
    # Rise
    for i in range(6):
        test2_prices.append(l2 + (i+1) * 2.5)
    h3 = test2_prices[-1]
    
    # Pullback 3: -18% (RESET — deeper than -12%)
    for i in range(7):
        test2_prices.append(h3 - (i+1) * 2.8)
    l3 = test2_prices[-1]
    
    # Rise
    for i in range(6):
        test2_prices.append(l3 + (i+1) * 2.6)
    h4 = test2_prices[-1]
    
    # Pullback 4: -7%
    for i in range(5):
        test2_prices.append(h4 - (i+1) * 1.6)
    l4 = test2_prices[-1]
    
    # Rise
    for i in range(5):
        test2_prices.append(l4 + (i+1) * 1.8)
    h5 = test2_prices[-1]
    
    # Pullback 5: -4%
    for i in range(4):
        test2_prices.append(h5 - (i+1) * 1.0)
    
    test2_dates = pd.date_range(end='2026-04-05', periods=len(test2_prices), freq='D')
    df2 = pd.DataFrame({
        'date': test2_dates,
        'open': test2_prices,
        'high': [p + 0.5 for p in test2_prices],
        'low': [p - 0.5 for p in test2_prices],
        'close': test2_prices,
        'volume': [1_000_000] * len(test2_prices)
    })
    
    result2 = detector.detect(df2, min_contractions=2)
    print(f"\n  num_contractions:      {result2.num_contractions} (expected: 2 — sequence after reset)")
    print(f"  details:               {result2.details}")
    if result2.pullbacks:
        print(f"  Final sequence:")
        for i, pb in enumerate(result2.pullbacks, 1):
            print(f"    {i}. -{pb.depth_pct:.1f}%")
    
    print("\n" + "="*70)
    print("  Tests complete")
    print("="*70 + "\n")
