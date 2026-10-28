"""
Base Count — which base a stock is on since its stage low (O'Neil / IBD style)

1st and 2nd bases tend to work better than 3rd and later ones, which come
late in a move. Rules:
  - Count from the stage low: the lowest low in the history given.
  - A base starts when price falls MIN_BASE_DEPTH_PCT below its running high
    (the base high) and ends on a close above that high. Shorter than
    MIN_BASE_BARS is a pullback, not a base.
  - A new base counts +1 only if its high is BASE_ADVANCE_PCT above the
    previous base's high. Otherwise it's base-on-base: same count. That's how
    a cluster of small shelves (e.g. DINO) counts as one base.
  - Reset to 1 when a base undercuts the previous base's low, or corrects
    more than MAX_BASE_DEPTH_PCT.
Needs about 2 years of daily bars: with 1 year, earlier bases are missed.
"""

from typing import List, Optional

import pandas as pd

MIN_BASE_DEPTH_PCT = 8.0     # Smaller dips are noise
MIN_BASE_BARS = 15           # IBD prefers 25 (5 weeks); 15 keeps tight VCPs
BASE_ADVANCE_PCT = 20.0      # Run needed between bases for a new count
MAX_BASE_DEPTH_PCT = 50.0    # Deeper than this breaks the stage


def find_bases(df: pd.DataFrame) -> List[dict]:
    """Bases since the stage low, oldest first. The last one may still be forming (end_i None)."""
    df = df.reset_index(drop=True)
    start = int(df['low'].idxmin())
    bases = []
    run_hi, run_hi_i = df['high'].iloc[start], start
    current = None

    for i in range(start + 1, len(df)):
        high, low, close = df['high'].iloc[i], df['low'].iloc[i], df['close'].iloc[i]
        if current is None:
            if high > run_hi:
                run_hi, run_hi_i = high, i
            elif (run_hi - low) / run_hi * 100 >= MIN_BASE_DEPTH_PCT:
                current = {'hi_i': run_hi_i, 'hi': run_hi, 'lo': low, 'lo_i': i}
        else:
            if low < current['lo']:
                current['lo'], current['lo_i'] = low, i
            if close > current['hi']:  # breakout ends the base
                current['end_i'] = i
                if i - current['hi_i'] >= MIN_BASE_BARS:
                    bases.append(current)
                current, run_hi, run_hi_i = None, high, i

    if current is not None and len(df) - 1 - current['hi_i'] >= MIN_BASE_BARS:
        current['end_i'] = None
        bases.append(current)
    return bases


def count_bases(df: pd.DataFrame) -> dict:
    """
    Base number of the latest base, and whether that base is still forming.
    base_number is None when no base has formed since the stage low.
    """
    bases = find_bases(df)
    n, prev = 0, None
    for b in bases:
        b['depth_pct'] = (b['hi'] - b['lo']) / b['hi'] * 100
        if prev is None:
            n, b['reason'] = 1, 'first base'
        elif b['depth_pct'] > MAX_BASE_DEPTH_PCT:
            n, b['reason'] = 1, f"reset: {b['depth_pct']:.0f}% correction"
        elif b['lo'] < prev['lo']:
            n, b['reason'] = 1, 'reset: undercut prior base low'
        elif b['hi'] >= prev['hi'] * (1 + BASE_ADVANCE_PCT / 100):
            n, b['reason'] = n + 1, 'new base'
        else:
            b['reason'] = 'base-on-base'
        b['number'] = n
        prev = b

    last: Optional[dict] = bases[-1] if bases else None
    return {
        'base_number': last['number'] if last else None,
        'base_forming': bool(last and last['end_i'] is None),
        'base_count_reason': last['reason'] if last else 'no base since stage low',
        'last_base_depth_pct': round(last['depth_pct'], 1) if last else None,
    }
