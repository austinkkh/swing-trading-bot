"""Pocket pivot: the shared definition used by the scanner, tracker and backtests."""

import numpy as np
import pandas as pd

import backtest as bt
import pocket_pivot as pp


def frame(n=260, last_volume=3_000_000, last_up=True, ext=1.02):
    """Steady uptrend (trend template holds); the last bar is the candidate day."""
    close = np.linspace(50, 100, n)
    df = pd.DataFrame({'date': pd.bdate_range("2025-01-02", periods=n), 'open': close * 0.995,
                       'high': close * 1.01, 'low': close * 0.99, 'close': close,
                       'volume': np.full(n, 1_000_000.0)})
    # a few down days with heavy volume in the prior 10 sessions
    for k in (3, 6):
        df.loc[n - 1 - k, 'close'] = df.loc[n - 2 - k, 'close'] * 0.99
        df.loc[n - 1 - k, 'volume'] = 2_000_000
    df = bt.add_indicators(df)
    e10 = df['ema10'].iat[-2]
    last = n - 1
    df.loc[last, 'close'] = e10 * ext
    df.loc[last, 'open'] = df.loc[last, 'close'] * (0.98 if last_up else 1.02)
    df.loc[last, 'high'] = df.loc[last, 'close'] * 1.005
    df.loc[last, 'low'] = df.loc[last, 'close'] * 0.97
    df.loc[last, 'volume'] = last_volume
    return bt.add_indicators(df[['date', 'open', 'high', 'low', 'close', 'volume']])


def test_up_day_beating_down_volume_is_a_pocket_pivot():
    df = frame()
    assert pp.is_pocket_pivot(df, len(df) - 1)


def test_volume_must_beat_every_recent_down_day():
    df = frame(last_volume=1_500_000)    # below the 2M down days
    assert not pp.is_pocket_pivot(df, len(df) - 1)


def test_extended_above_10ema_is_not_a_pocket_pivot():
    df = frame(ext=1.10)
    assert not pp.is_pocket_pivot(df, len(df) - 1)


def test_down_close_is_not_a_pocket_pivot():
    df = frame(last_up=False)
    assert not pp.is_pocket_pivot(df, len(df) - 1)


def test_evaluate_reports_trade_numbers():
    info = pp.evaluate(frame())
    assert info['pocket_pivot'] and info['pp_stop'] < info['pp_entry']
    assert 2.5 < info['pp_stop_pct'] < 3.5 and info['pp_stop_ok']
    assert info['adr_pct'] > 0 and info['dollar_vol_month_m'] > 0
