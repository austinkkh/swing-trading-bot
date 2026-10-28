"""
Exit rules: the exit plan, parameterised by strategy_params.py (example values).

  1. Initial stop: the setup's structural stop.
  2. Sell TRIM_FRACTION at +TRIM_R, UNLESS the stock reaches +TRIM_R within FAST_SESSIONS sessions
     after the entry day: then it's behaving like a monster and the full position is held.
  3. Move the stop to breakeven once the high reaches +TRIM_R.
  4. After +TRIM_R, a daily close below the trail line exits the rest at the next open. The trail line
     is the moving average the stock respected before entry: the fastest of 5-EMA, 10-SMA, 21-EMA it
     closed above on >= RESPECT_THRESHOLD of the RESPECT_LOOKBACK sessions before entry, else the 21-EMA.
  5. Time stop: if +TRIM_R hasn't been reached after TIME_STOP_SESSIONS sessions, exit at the close.

`manage()` turns the plan into today's actions for one open position, from daily bars. It decides;
it never sends orders.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd

from strategy_params import (TRIM_R, TRIM_FRACTION, FAST_SESSIONS,   # example values
                             RESPECT_LOOKBACK, RESPECT_THRESHOLD, TIME_STOP_SESSIONS)
TRAIL_LINES = ('ema5', 'sma10', 'ema21')     # fastest first
TRAIL_LABELS = {'ema5': '5-day EMA', 'sma10': '10-day SMA', 'ema21': '21-day EMA'}


def add_trail_lines(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(ema5=df['close'].ewm(span=5, adjust=False).mean(),
                     sma10=df['close'].rolling(10).mean(),
                     ema21=df['close'].ewm(span=21, adjust=False).mean())


def respected_ma(df: pd.DataFrame, entry_index: Optional[int] = None) -> str:
    """
    Trail line for a trade entered at bar entry_index (default: the bar after the last one, i.e. a
    scan run after the close for tomorrow's entry). Uses only the RESPECT_LOOKBACK bars before the entry.
    """
    d = add_trail_lines(df) if not set(TRAIL_LINES) <= set(df.columns) else df
    t = len(d) if entry_index is None else entry_index
    w = d.iloc[max(0, t - RESPECT_LOOKBACK):t]
    if len(w) < RESPECT_LOOKBACK:
        return 'ema21'
    for line in TRAIL_LINES:
        if (w['close'] > w[line]).mean() >= RESPECT_THRESHOLD:
            return line
    return 'ema21'


@dataclass
class ExitState:
    entry_price: float
    initial_stop: float
    entry_date: pd.Timestamp
    trail_line: str = 'ema21'
    trimmed: bool = False
    hold_full: bool = False          # +TRIM_R came within FAST_SESSIONS: never trim
    reached_2r: bool = False


@dataclass
class ExitActions:
    stop_price: float
    actions: List[str] = field(default_factory=list)    # e.g. 'SELL_TRIM_AT_TARGET', 'EXIT_NEXT_OPEN', 'TIME_STOP_AT_CLOSE'
    notes: List[str] = field(default_factory=list)
    take_profit_order: Optional[float] = None            # price for a resting trim sell, when one should exist


def manage(state: ExitState, df: pd.DataFrame) -> ExitActions:
    """
    Today's exit actions for one position, evaluated after the latest daily close in df (bars on or
    after the entry date). Updates `state` in place (trimmed / hold_full / reached_2r).
    """
    d = add_trail_lines(df)
    r = state.entry_price - state.initial_stop
    since = d[d['date'] >= state.entry_date].reset_index(drop=True)
    if r <= 0 or since.empty:
        return ExitActions(stop_price=state.initial_stop, notes=['no bars since entry or invalid stop'])
    target = state.entry_price + TRIM_R * r
    hit = since.index[since['high'] >= target]
    if len(hit) and not state.reached_2r:
        state.reached_2r = True
        if hit[0] <= FAST_SESSIONS:            # index 0 = entry day
            state.hold_full = True
    out = ExitActions(stop_price=state.initial_stop)
    sessions = len(since)
    if state.reached_2r:
        out.stop_price = max(state.initial_stop, state.entry_price)
        if state.hold_full:
            out.notes.append(f"+{TRIM_R:g}R within {FAST_SESSIONS} sessions: holding full size (no trim sale)")
        elif not state.trimmed:
            out.actions.append('SELL_TRIM_AT_TARGET')
            state.trimmed = True
        line_value = float(d[state.trail_line].iat[-1])
        if float(d['close'].iat[-1]) < line_value:
            out.actions.append('EXIT_NEXT_OPEN')
            out.notes.append(f"close below the {TRAIL_LABELS[state.trail_line]} ({line_value:.2f})")
    else:
        if sessions > FAST_SESSIONS and not state.hold_full:
            out.take_profit_order = round(target, 2)   # after the fast window, a resting trim sell at +TRIM_R
        if sessions > TIME_STOP_SESSIONS:
            out.actions.append('TIME_STOP_AT_CLOSE')
            out.notes.append(f"+{TRIM_R:g}R not reached in {TIME_STOP_SESSIONS} sessions")
    return out


def describe(trail_line: str = 'ema21') -> dict:
    """Text for decision files and alerts."""
    return {
        'Exit_Trim_Rule': f"Sell {TRIM_FRACTION:.0%} at +{TRIM_R:g}R unless +{TRIM_R:g}R comes within {FAST_SESSIONS} sessions (then hold full size)",
        'Exit_Trail_Rule': f"After +{TRIM_R:g}R: stop to breakeven; exit on a close below the {TRAIL_LABELS.get(trail_line, trail_line)}",
        'Invalidation_Rule': f"Structural stop; time stop after {TIME_STOP_SESSIONS} sessions without +{TRIM_R:g}R",
        'Trail_MA': trail_line,
    }


def simulate_trade(df: pd.DataFrame, t: int, entry: float, stop: float, trail_line: str = 'ema21'):
    """
    Daily-bar walk of the adopted exit, driven by manage() itself, so the forward tracker and the
    backtests score exactly the plan the bot follows. Entry fills at bar t (any stop hit on the
    entry bar counts at the stop; later gaps fill at the open). Conservative: the stop is checked
    before the target on every bar.
    Returns (gross R, exit bar index, trimmed). A trade still open is marked to the last close and
    its exit index is the last bar.
    """
    state = ExitState(entry_price=entry, initial_stop=stop, entry_date=df['date'].iat[t], trail_line=trail_line)
    r = entry - stop
    pnl, frac, cur_stop, pending_exit, tp = 0.0, 1.0, stop, False, None
    for i in range(t, len(df)):
        o, h, l, c = (float(df[k].iat[i]) for k in ('open', 'high', 'low', 'close'))
        if pending_exit:
            return pnl + frac * (o - entry) / r, i, state.trimmed
        if l <= cur_stop:
            return pnl + frac * ((min(o, cur_stop) if i > t else cur_stop) - entry) / r, i, state.trimmed
        if tp is not None and h >= tp and not state.trimmed and not state.hold_full:
            pnl += TRIM_FRACTION * (max(o, tp) - entry) / r      # resting trim sell at +TRIM_R filled
            frac -= TRIM_FRACTION
            state.trimmed = True
        acts = manage(state, df.iloc[:i + 1])
        if 'SELL_TRIM_AT_TARGET' in acts.actions and frac > 1 - 1e-9:
            # +TRIM_R printed after the fast window and no resting order filled it yet: sell at the target
            pnl += TRIM_FRACTION * (max(o, entry + TRIM_R * r) - entry) / r
            frac -= TRIM_FRACTION
        cur_stop = acts.stop_price
        tp = acts.take_profit_order
        if 'TIME_STOP_AT_CLOSE' in acts.actions:
            return pnl + frac * (c - entry) / r, i, state.trimmed
        if 'EXIT_NEXT_OPEN' in acts.actions:
            pending_exit = True
    return pnl + frac * (float(df['close'].iat[-1]) - entry) / r, len(df) - 1, state.trimmed
