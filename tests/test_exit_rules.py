"""Adopted exit plan: trim at +TRIM_R unless it comes fast, breakeven, trail the respected MA."""

import numpy as np
import pandas as pd
import pytest

import exit_rules as er


def bars(closes, start="2026-01-02", spread=0.01):
    c = np.array(closes, dtype=float)
    return pd.DataFrame({'date': pd.bdate_range(start, periods=len(c)), 'open': c, 'high': c * (1 + spread),
                         'low': c * (1 - spread), 'close': c, 'volume': 1e6})


def test_fast_trend_respects_the_5ema():
    assert er.respected_ma(bars(np.linspace(50, 100, 80))) == 'ema5'


def test_choppy_stock_falls_back_to_21ema():
    rng = np.random.default_rng(0)
    assert er.respected_ma(bars(100 + rng.normal(0, 3, 80))) == 'ema21'


def history_then_trade(trade_closes, entry=100.0):
    pre = np.linspace(80, entry, 60)
    df = bars(np.r_[pre, trade_closes])
    return df, df['date'].iat[60]


ENTRY, STOP = 100.0, 97.0
TARGET = ENTRY + er.TRIM_R * (ENTRY - STOP)
FAST = er.FAST_SESSIONS


def below_target(n):
    """n sessions drifting up from the entry, highs well under the target."""
    return [ENTRY + 0.1 * k for k in range(n)]


def test_fast_target_means_hold_full_size():
    df, d0 = history_then_trade(np.linspace(ENTRY, TARGET + 2, FAST + 1))   # target by the last fast session
    st = er.ExitState(entry_price=ENTRY, initial_stop=STOP, entry_date=d0)
    a = er.manage(st, df)
    assert st.hold_full and 'SELL_TRIM_AT_TARGET' not in a.actions
    assert a.stop_price == ENTRY                                  # breakeven


def test_slow_target_sells_the_trim():
    df, d0 = history_then_trade(below_target(FAST + 2) + [TARGET + 0.5])
    st = er.ExitState(entry_price=ENTRY, initial_stop=STOP, entry_date=d0)
    a = er.manage(st, df)
    assert not st.hold_full and a.actions == ['SELL_TRIM_AT_TARGET'] and a.stop_price == ENTRY
    assert er.manage(st, df).actions == []                        # only once


def test_take_profit_order_appears_after_the_fast_window():
    df, d0 = history_then_trade(below_target(FAST + 1))           # entry day + FAST, below the target
    st = er.ExitState(entry_price=ENTRY, initial_stop=STOP, entry_date=d0)
    assert er.manage(st, df).take_profit_order == pytest.approx(round(TARGET, 2))
    df_in, _ = history_then_trade(below_target(FAST))
    assert er.manage(er.ExitState(ENTRY, STOP, d0), df_in).take_profit_order is None


def test_target_on_the_last_fast_session_still_counts_as_fast():
    df, d0 = history_then_trade(below_target(FAST) + [TARGET + 0.5])   # index FAST
    st = er.ExitState(entry_price=ENTRY, initial_stop=STOP, entry_date=d0)
    er.manage(st, df)
    assert st.hold_full


def test_close_below_trail_line_after_target_exits_next_open():
    df, d0 = history_then_trade([100, 102, 104, 107, 110, 112, 100.5])
    st = er.ExitState(entry_price=ENTRY, initial_stop=STOP, entry_date=d0, trail_line='ema5')
    assert 'EXIT_NEXT_OPEN' in er.manage(st, df).actions


def test_time_stop_without_target():
    closes = [ENTRY + 0.1 * (k % 5) for k in range(er.TIME_STOP_SESSIONS + 1)]
    df, d0 = history_then_trade(closes)
    assert 'TIME_STOP_AT_CLOSE' in er.manage(er.ExitState(ENTRY, STOP, d0), df).actions
    df_in, _ = history_then_trade(closes[:-1])
    assert 'TIME_STOP_AT_CLOSE' not in er.manage(er.ExitState(ENTRY, STOP, d0), df_in).actions


def test_describe_names_the_line():
    assert "10-day SMA" in er.describe('sma10')['Exit_Trail_Rule']
