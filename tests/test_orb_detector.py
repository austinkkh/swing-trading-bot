"""Opening range from the first N five-minute bars (default 10-minute ORB)."""

from datetime import datetime, timedelta

import pandas as pd

import orb_detector as orb
from market_time import now_et


def bars(day, highs, lows):
    start = datetime.combine(day, orb.ORB_OPENING_TIME)
    return pd.DataFrame({
        'date': [start + timedelta(minutes=5 * i) for i in range(len(highs))],
        'high': highs, 'low': lows,
    })


def detector():
    return orb.ORBDetector(ibkr_connector=None)


def test_formation_end_follows_window():
    opened = datetime.combine(now_et().date(), orb.ORB_OPENING_TIME)
    assert orb.ORB_FORMATION_END == (opened + timedelta(minutes=orb.ORB_WINDOW_MINUTES)).time()


def test_opening_range_spans_the_whole_window():
    # 10-minute ORB = first two 5-minute bars: high of both, low of both, third bar ignored
    df = bars(now_et().date(), highs=[101, 103, 110], lows=[99, 100, 95])
    assert detector()._get_opening_range(df) == {'high': 103.0, 'low': 99.0}


def test_opening_range_waits_for_enough_bars():
    df = bars(now_et().date(), highs=[101], lows=[99])
    assert detector()._get_opening_range(df) is None


def test_no_bars_today_means_no_range():
    """The old code fell back to the first bar in the data, i.e. yesterday's open."""
    df = bars(now_et().date() - timedelta(days=1), highs=[101, 103], lows=[99, 100])
    assert detector()._get_opening_range(df) is None


def test_missing_opening_bar_means_no_range():
    """Without the 9:30 bar, later bars (even the one still forming) must not become the range."""
    day = now_et().date()
    start = datetime.combine(day, orb.ORB_OPENING_TIME) + timedelta(minutes=5)  # first bar 9:35
    df = pd.DataFrame({'date': [start + timedelta(minutes=5 * i) for i in range(3)],
                       'high': [101, 103, 104], 'low': [99, 100, 101]})
    assert detector()._get_opening_range(df) is None


def breakout_signal(monkeypatch, highs, lows, closes, volumes):
    """Run check_orb_breakout on synthetic 5-min bars, inside the hunt window."""
    day = now_et().date()
    start = datetime.combine(day, orb.ORB_OPENING_TIME)
    df = pd.DataFrame({'date': [start + timedelta(minutes=5 * i) for i in range(len(highs))],
                       'open': closes, 'high': highs, 'low': lows, 'close': closes, 'volume': volumes})
    d = detector()
    monkeypatch.setattr(d, "is_orb_forming", lambda: False)
    monkeypatch.setattr(d, "is_orb_window_active", lambda: True)
    monkeypatch.setattr(d, "_fetch_intraday_bars", lambda symbol: ["bars"])
    monkeypatch.setattr(d, "_bars_to_dataframe", lambda bars: df)
    return d.check_orb_breakout("TEST")


def test_stop_measured_from_entry_not_range_width(monkeypatch):
    # Range 100-102.9 (2.9% wide) passes a width check, but entry at 105.9 puts the stop 5.6% away
    sig = breakout_signal(monkeypatch, highs=[102.9, 102.5, 106], lows=[100, 100.5, 103],
                          closes=[102, 102, 105.9], volumes=[1000, 1000, 3000])
    assert not sig.has_breakout
    assert sig.status == "RISK_TOO_HIGH"


def test_stop_too_tight_is_rejected(monkeypatch):
    # Range 100.0-100.3 with entry 100.45: stop 0.45% away, under the 1% minimum
    sig = breakout_signal(monkeypatch, highs=[100.3, 100.25, 100.5], lows=[100.0, 100.05, 100.3],
                          closes=[100.1, 100.2, 100.45], volumes=[1000, 1000, 3000])
    assert not sig.has_breakout
    assert sig.status == "STOP_TOO_TIGHT"


def test_valid_breakout(monkeypatch):
    # Range 100-101.5, entry 102 (0.5% above ORH): stop 2% away, 3x volume
    sig = breakout_signal(monkeypatch, highs=[101.5, 101.2, 102.2], lows=[100, 100.4, 101.4],
                          closes=[101, 101, 102], volumes=[1000, 1000, 3000])
    assert sig.has_breakout, sig.reason


def test_ibkr_intraday_date_strings_parse():
    # The real IBKR format; plain pd.to_datetime raised on it, so ORB never fired
    from types import SimpleNamespace
    raw = [SimpleNamespace(date=f"20260724 09:{m:02d}:00 US/Eastern", open=1, high=1, low=1, close=1, volume=1)
           for m in (30, 35, 40)]
    df = detector()._bars_to_dataframe(raw)
    assert df['date'].dt.strftime("%H:%M").tolist() == ["09:30", "09:35", "09:40"]


def test_other_zone_is_converted_to_eastern():
    import market_time as mt
    assert mt.parse_ibkr_times(["20260724 06:30:00 US/Pacific"]).iloc[0].strftime("%H:%M") == "09:30"


def test_forming_bar_volume_is_scaled_to_full_bar_pace():
    day = datetime(2026, 7, 24)
    times = [datetime.combine(day, orb.ORB_OPENING_TIME) + timedelta(minutes=5 * i) for i in range(3)]
    df = pd.DataFrame({'date': times, 'volume': [1000, 1000, 400]})
    # 1 minute into the 9:40 bar: 400 shares -> 2000-share pace = 2x the 1000 average
    assert detector()._calc_volume_ratio(df, now=times[-1] + timedelta(minutes=1)) == 2.0
    # completed bar: no scaling
    assert detector()._calc_volume_ratio(df, now=times[-1] + timedelta(minutes=6)) == 0.4
