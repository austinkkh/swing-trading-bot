"""
Pocket pivot detection — one definition shared by the scanner and the backtests.

A pocket pivot day d (Morales/Kacher style):
  - Minervini trend template holds at d (backtest.add_indicators 'trend_ok')
  - up day: close > open and close > prior close
  - volume > the highest down-day volume of the 10 sessions before d
  - close above the 10-day EMA by at most 6% (not extended)
Trade: buy near d's close, stop at d's low (within the pivot stop band).
"In VCP": the VCP detector sees a valid VCP at d's close and the close is still below its pivot,
i.e. an early entry inside the base. Flag only; no trading rule uses it.
"""

from typing import List, Optional

import numpy as np
import pandas as pd

import strategy_params as sp

MAX_EXT_ABOVE_EMA10 = 1.06
DOWN_VOLUME_LOOKBACK = 10
MIN_STOP_PCT, MAX_STOP_PCT = sp.MIN_STOP_PCT, sp.MAX_STOP_PCT_PIVOT


def is_pocket_pivot(df: pd.DataFrame, d: int) -> bool:
    """df needs open/high/low/close/volume plus 'ema10' and 'trend_ok' (backtest.add_indicators)."""
    if d < DOWN_VOLUME_LOOKBACK + 1:
        return False
    o, c, v = df['open'].iat[d], df['close'].iat[d], df['volume'].iat[d]
    if not bool(df['trend_ok'].iat[d]) or not (c > o and c > df['close'].iat[d - 1]):
        return False
    closes = df['close'].iloc[d - DOWN_VOLUME_LOOKBACK - 1:d].to_numpy()
    vols = df['volume'].iloc[d - DOWN_VOLUME_LOOKBACK:d].to_numpy()
    down_vol = vols[closes[1:] < closes[:-1]]
    max_down = down_vol.max() if len(down_vol) else 0.0
    e10 = df['ema10'].iat[d]
    return bool(v > max_down and e10 < c <= e10 * MAX_EXT_ABOVE_EMA10)


def pocket_pivot_days(df: pd.DataFrame, start: int = 201) -> List[int]:
    return [d for d in range(max(start, DOWN_VOLUME_LOOKBACK + 1), len(df)) if is_pocket_pivot(df, d)]


def adr_pct(df: pd.DataFrame, d: int) -> float:
    """20-day average (high - low) / close, % ."""
    w = df.iloc[max(0, d - 19):d + 1]
    return float(((w['high'] - w['low']) / w['close']).mean() * 100)


def dollar_volume_month_m(df: pd.DataFrame, d: int) -> float:
    """21-day average dollar volume x 21, in $ millions."""
    w = df.iloc[max(0, d - 20):d + 1]
    return float((w['close'] * w['volume']).mean() * 21 / 1e6)


def evaluate(df: pd.DataFrame, d: Optional[int] = None, detector=None) -> dict:
    """Pocket-pivot status at bar d (default: the last bar) with the trade's numbers."""
    d = len(df) - 1 if d is None else d
    pp = is_pocket_pivot(df, d)
    entry, stop = float(df['close'].iat[d]), float(df['low'].iat[d])
    stop_pct = (entry - stop) / entry * 100 if entry > 0 else np.nan
    in_vcp = False
    pivot = None
    if pp and detector is not None:
        res = detector.detect(df.iloc[:d + 1][['date', 'open', 'high', 'low', 'close', 'volume']], 3)
        pivot = res.pivot_price if res.has_vcp else None
        in_vcp = bool(res.has_vcp and res.pivot_price > entry)
    return {
        'pocket_pivot': pp,
        'pocket_pivot_in_vcp': in_vcp,
        'pp_entry': round(entry, 2), 'pp_stop': round(stop, 2), 'pp_stop_pct': round(stop_pct, 2),
        'pp_stop_ok': bool(MIN_STOP_PCT <= stop_pct <= MAX_STOP_PCT),
        'vcp_pivot': round(pivot, 2) if pivot else None,
        'adr_pct': round(adr_pct(df, d), 2),
        'dollar_vol_month_m': round(dollar_volume_month_m(df, d), 1),
    }
