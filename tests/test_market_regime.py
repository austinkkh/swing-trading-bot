"""Regime classifier (the gate that can zero the BUY list): one synthetic case per state, D-days."""

import numpy as np
import pandas as pd
import pytest

import market_regime as mr


def health(symbol="SPY", above_50=True, above_200=True, rising_50=True, drawdown=-1.0):
    return mr.IndexHealth(symbol=symbol, current_price=100, ma_50=98, ma_200=90, above_50ma=above_50,
                          above_200ma=above_200, ma_50_rising=rising_50, ma_200_rising=True,
                          distance_from_50ma_pct=2.0, drawdown_from_high_pct=drawdown, high_52w=101)


def classify(spy, qqq=None, d_days=0):
    return mr.MarketRegimeDetector(None)._classify_regime(spy, qqq or health("QQQ"), d_days, [])[0]


def test_clean_uptrend():
    assert classify(health()) == mr.REGIME_CONFIRMED_UPTREND


def test_four_distribution_days_is_under_pressure():
    assert classify(health(), d_days=4) == mr.REGIME_UPTREND_UNDER_PRESSURE


def test_six_distribution_days_is_correction():
    assert classify(health(), d_days=6) == mr.REGIME_CORRECTION


def test_below_50ma_with_severe_drawdown_is_correction():
    assert classify(health(above_50=False, drawdown=-9.0)) == mr.REGIME_CORRECTION


def test_below_50ma_with_moderate_drawdown_is_only_pressure():
    """Documents today's rule: below the 50-day and -4% isn't a correction (Rule 1 needs -8%)."""
    assert classify(health(above_50=False, drawdown=-4.0)) == mr.REGIME_UPTREND_UNDER_PRESSURE


def test_below_200ma_is_correction():
    assert classify(health(), qqq=health("QQQ", above_200=False)) == mr.REGIME_CORRECTION


def test_deep_drawdown_holding_50ma_is_rally_attempt():
    assert classify(health(drawdown=-6.5)) == mr.REGIME_RALLY_ATTEMPT


def index_frame(closes, volumes):
    n = len(closes)
    return pd.DataFrame({"date": pd.bdate_range("2026-01-02", periods=n), "close": closes,
                         "high": closes, "low": closes, "open": closes, "volume": volumes})


def test_distribution_day_counted_and_expires_after_5pct_rally():
    closes = np.full(40, 100.0)
    vols = np.full(40, 1_000_000.0)
    closes[30:] = 99.5           # day 30: -0.5% on higher volume = a D-day
    vols[30] = 1_500_000
    det = mr.MarketRegimeDetector(None)
    assert len(det._count_distribution_days(index_frame(closes, vols), "SPY")) == 1
    closes[-1] = 99.5 * 1.06     # index now 6% above the D-day close: it expires
    assert len(det._count_distribution_days(index_frame(closes, vols), "SPY")) == 0


def test_down_day_on_lower_volume_is_not_distribution():
    closes = np.full(40, 100.0)
    closes[30:] = 99.0
    vols = np.full(40, 1_000_000.0)
    vols[30] = 900_000
    assert mr.MarketRegimeDetector(None)._count_distribution_days(index_frame(closes, vols), "SPY") == []
