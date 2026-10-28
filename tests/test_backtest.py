"""Backtest trade simulation and signal timing (no lookahead)."""

import numpy as np
import pandas as pd
import pytest

import backtest as bt
from helpers import price_path

F = bt.TRIM_FRACTION


def bars(rows, ema21=None):
    """rows: [(open, high, low, close), ...] starting 2025-01-02. ema21 defaults far below (never trails out)."""
    df = pd.DataFrame(rows, columns=['open', 'high', 'low', 'close'])
    df['date'] = pd.bdate_range("2025-01-02", periods=len(df))
    df['ema21'] = ema21 if ema21 is not None else 1.0
    return df


def run(rows, entry=100.0, stop=98.0, ema21=None):
    sig = bt.Signal('TEST', fill_index=0, entry=entry, stop=stop)
    return bt.simulate(bars(rows, ema21), sig, "TEST")


def test_stop_hit_is_minus_one_r():
    t = run([(100, 100.5, 99.5, 100), (99.5, 99.8, 97.5, 98)])
    assert t.exit_reason == "stop"
    assert t.r_gross == pytest.approx(-1.0, abs=1e-3)


def test_gap_below_stop_fills_at_open():
    t = run([(100, 100.5, 99.5, 100), (96, 96.5, 95, 95.5)])  # opens 2 points under the stop
    assert t.r_gross == pytest.approx(-2.0, abs=1e-3)


def test_trim_then_breakeven_stop_keeps_the_trimmed_r():
    # +1R (102) hit on day 1, then price falls back through entry
    t = run([(100, 100.5, 99.5, 100), (101, 102.5, 100.5, 102), (101, 101, 99.0, 99.5)])
    assert t.trimmed and t.exit_reason == "breakeven stop"
    assert t.r_gross == pytest.approx(F * 1.0, abs=1e-3)


def test_trail_exits_next_open_after_close_below_21ema():
    ema = [90, 90, 90, 105, 105]
    t = run([(100, 100.5, 99.5, 100), (101, 103, 100.5, 102.5), (103, 104.5, 102.5, 104),
             (104, 104.2, 103, 103.5), (103.8, 104, 103, 103.5)], ema21=ema)
    assert t.exit_reason == "close < 21EMA"
    # the trim at +1R, the rest out at day 4's open (103.8 -> +1.9R)
    assert t.r_gross == pytest.approx(F * 1.0 + (1 - F) * 1.9, abs=1e-3)


def test_time_stop_without_reaching_one_r():
    rows = [(100, 101, 99.5, 100.5)] * (bt.TIME_STOP_DAYS + 4)
    t = run(rows)
    assert t.exit_reason == "time stop"
    assert t.days_held == bt.TIME_STOP_DAYS
    assert t.r_gross == pytest.approx(0.25, abs=1e-3)


def test_costs_reduce_r():
    t = run([(100, 100.5, 99.5, 100), (99.5, 99.8, 97.5, 98)])
    assert t.r_net < t.r_gross


@pytest.mark.parametrize("stop", [100 * (1 - bt.MIN_STOP_PCT / 2 / 100),    # too tight
                                  100 * (1 - (bt.MAX_STOP_PCT + 2) / 100)])  # too wide
def test_stops_outside_band_are_not_traded(stop):
    assert run([(100, 101, 99, 100)] * 3, stop=stop) is None


def test_pullback_fills_next_open_with_signal_day_low_as_stop():
    df = bt.add_indicators(price_path([(220, 150)], start=50.0))
    t = len(df) - 3
    df.loc[t, ['trend_ok', 'open', 'close', 'low']] = [True, df.loc[t, 'ema10'] * 0.995,
                                                       df.loc[t, 'ema10'] * 1.01, df.loc[t, 'ema10'] * 0.998]
    sigs = [s for s in bt.pullback_signals(df) if s.fill_index == t + 1]
    assert sigs, "expected a pullback signal on the tagged day"
    assert sigs[0].entry == pytest.approx(df.loc[t + 1, 'open'])
    assert sigs[0].stop == pytest.approx(df.loc[t, 'low'])


def test_vcp_detection_never_sees_the_fill_day():
    df = bt.add_indicators(price_path([(220, 150), (40, 160)], start=50.0))
    df['trend_ok'] = True
    seen = []

    class Spy:
        def detect(self, window, _min):
            seen.append(window['date'].iloc[-1])
            from vcp_detector import VCPResult
            return VCPResult()  # never a VCP; we only check what data it was given

    bt.vcp_signals(df, detector=Spy())
    # each call for fill day t receives bars through t-1 only
    assert all(last < df['date'].iloc[-1] for last in seen)
    assert max(seen) == df['date'].iloc[-2]


def run_rules(rows, rules, ema21=None):
    sig = bt.Signal('TEST', fill_index=0, entry=100.0, stop=98.0)
    return bt.simulate(bars(rows, ema21), sig, "TEST", rules)


def test_trim_bar_closing_below_entry_exits_at_breakeven_same_bar():
    # +1R (102) printed, but the bar closes at 99.5: the trip back through entry is certain
    t = run_rules([(100, 100.5, 99.5, 100), (101, 102.5, 99.0, 99.5)], bt.LIVE)
    assert t.exit_reason == "breakeven stop (same bar)"
    assert t.r_gross == pytest.approx(F * 1.0, abs=1e-3)


def test_pessimistic_intrabar_assumes_the_dip_came_after_the_high():
    # Trim bar dips to 99.8 but closes at 102: unknowable order; pessimistic exits, realistic holds
    rows = [(100, 100.5, 99.5, 100), (100.5, 102.5, 99.8, 102), (102, 104, 101.5, 103.5)] + [(103.5, 104, 103, 103.5)] * 3
    pess = run_rules(rows, bt.Rules(name="p", intrabar="pessimistic"))
    real = run_rules(rows, bt.LIVE)
    assert pess.exit_reason == "breakeven stop (same bar)"
    assert real.exit_reason != "breakeven stop (same bar)"


def test_live_rule_exits_below_21ema_even_before_trim():
    ema = [99, 99, 101, 101, 101]  # day 2 closes at 100.5 < 101 before +1R was reached
    rows = [(100, 100.5, 99.5, 100), (100, 101, 99.5, 100.8), (100.8, 101, 100.2, 100.5),
            (100.4, 100.6, 99.8, 100.2), (100.2, 100.5, 99.9, 100.1)]
    live = run_rules(rows, bt.LIVE, ema21=ema)
    after_trim_only = run_rules(rows, bt.Rules(name="a", exit_below_21ema_before_trim=False), ema21=ema)
    assert live.exit_reason == "close < 21EMA" and live.days_held == 3
    assert after_trim_only.exit_reason != "close < 21EMA"


def test_max_drawdown_counts_from_zero():
    assert bt.max_drawdown(pd.Series([-1.0, -1.0, -1.0, 2.0])) == pytest.approx(3.0)
    assert bt.max_drawdown(pd.Series([1.0, -0.5, 2.0, -3.0])) == pytest.approx(3.0)


def test_regime_is_unknown_during_warmup():
    spy = price_path([(260, 150)], start=100.0)
    labels = bt.market_regime(spy)
    assert set(labels.iloc[:199]) == {'UNKNOWN'}
    assert labels.iloc[-1] == 'UP'


# ─── Research knobs (defaults must equal live rules) ───
def run_knobs(rows, rules, entry=100.0, stop=98.0):
    df = bars(rows)
    df['ema10'] = 1.0
    return bt.simulate(df, bt.Signal('TEST', 0, entry, stop), "TEST", rules)


def test_trim_at_two_r_with_half():
    rules = bt.Rules(trim_r=2.0, trim_fraction=0.5)
    t = run_knobs([(100, 100.5, 99.5, 100), (101, 104.5, 100.5, 104), (101, 101, 99.0, 99.5)], rules)
    assert t.trimmed and t.r_gross == pytest.approx(0.5 * 2.0, abs=1e-3)


def test_stop_cap_tightens_a_wide_stop():
    rules = bt.Rules(stop_cap_pct=5.0)
    t = run_knobs([(100, 100.5, 99.5, 100)] * 3, rules, stop=92.0)   # 8% structural stop -> 5%
    assert t.stop == pytest.approx(95.0) and t.stop_pct == pytest.approx(5.0)


def test_shorter_time_stop():
    t = run_knobs([(100, 101, 99.5, 100.5)] * 12, bt.Rules(time_stop_days=5))
    assert t.exit_reason == "time stop" and t.days_held == 5


def test_default_rules_are_the_live_rules():
    assert bt.Rules() == bt.LIVE
    assert (bt.LIVE.trim_r, bt.LIVE.trim_fraction, bt.LIVE.time_stop_days, bt.LIVE.trail_ema,
            bt.LIVE.stop_cap_pct) == (1.0, bt.TRIM_FRACTION, bt.TIME_STOP_DAYS, 'ema21', None)


def test_a_pullback_trade_does_not_hide_a_vcp_breakout():
    df = bars([(100, 100.5, 99.5, 100)] * 15)
    sigs = [bt.Signal('EMA_PULLBACK', 0, 100.0, 98.0), bt.Signal('VCP_BREAKOUT', 2, 100.0, 96.0)]
    setups = [t.setup for t in bt.trades_for(df, sigs, "TEST")]
    assert setups == ['EMA_PULLBACK', 'VCP_BREAKOUT']
