"""TWS 2188 (no real-time API market data -> ~15-min delayed bars): logged once, measured,
shown on alerts; ORB never uses a partial opening bar. No signal gate added (owner decision)."""

import logging
import threading
from datetime import datetime, timedelta
from types import SimpleNamespace

import pandas as pd

import ibkr_connector as ic
import orb_detector as orb
import watch_mode as wm
from market_time import bar_lag_minutes, now_et

MSG = "Requested market data is not subscribed. Delayed market data is available."


def test_2188_warns_once_then_debug(caplog):
    conn = ic.IBKRConnector()
    with caplog.at_level(logging.DEBUG, logger="ibkr_connector"):
        conn.error(1, 1792715947541, 2188, MSG, "")
        conn.error(2, 1792715947541, 2188, MSG, "")
        conn.error(3, 2188, MSG)                                  # old signature too
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and "delayed" in warnings[0].getMessage()
    assert conn.data_delayed
    assert conn.errors == {}                                      # not a request error


def test_request_with_2188_still_returns_its_bars():
    conn = ic.IBKRConnector()
    conn.serverVersion = lambda: 219

    def emit(req_id):
        conn.error(req_id, 1792715947541, 2188, MSG, "")
        for i in range(3):
            conn.historicalData(req_id, SimpleNamespace(close=float(i)))
        conn.historicalDataEnd(req_id, "", "")
    conn.reqHistoricalData = lambda req_id, *a, **k: threading.Thread(target=emit, args=(req_id,)).start()
    assert len(conn.fetch_historical_data("SPY", timeout=5)) == 3


def test_bar_lag_minutes():
    assert bar_lag_minutes(datetime(2026, 10, 26, 9, 55), 5, datetime(2026, 10, 26, 10, 15)) == 15
    assert bar_lag_minutes(datetime(2026, 10, 26, 10, 10), 5, datetime(2026, 10, 26, 10, 12)) <= 0


def five_min(day, n, start=orb.ORB_OPENING_TIME):
    t0 = datetime.combine(day, start)
    return pd.DataFrame({'date': [t0 + timedelta(minutes=5 * i) for i in range(n)],
                         'open': 100.0, 'high': [101.0 + i for i in range(n)], 'low': 99.0,
                         'close': 100.5, 'volume': 1000.0})


def test_opening_range_waits_for_the_window_to_close():
    day = now_et().date()
    det = orb.ORBDetector(None)
    assert det._get_opening_range(five_min(day, 2)) is None       # newest bar is the 9:35 (forming)
    assert det._get_opening_range(five_min(day, 3)) == {'high': 102.0, 'low': 99.0}


def test_volume_ratio_uses_the_data_clock_on_delayed_data():
    day = datetime(2026, 7, 24)
    times = [datetime.combine(day, orb.ORB_OPENING_TIME) + timedelta(minutes=5 * i) for i in range(3)]
    df = pd.DataFrame({'date': times, 'volume': [1000, 1000, 400]})
    wall = times[-1] + timedelta(minutes=16)                      # wall clock ~15 min ahead of the data
    det = orb.ORBDetector(None)
    assert det._calc_volume_ratio(df, now=wall) == 0.4            # wall clock: bar looks complete
    assert det._calc_volume_ratio(df, now=wall, delay_min=15) == 2.0   # data clock: 1 min in, scaled


def test_signal_carries_the_data_lag(monkeypatch):
    now = now_et().replace(hour=10, minute=15, second=0, microsecond=0)
    df = five_min(now.date(), 6)                                  # newest bar starts 9:55
    det = orb.ORBDetector(SimpleNamespace(data_delayed=True))
    monkeypatch.setattr(orb, "now_et", lambda: now)
    monkeypatch.setattr(det, "is_orb_forming", lambda: False)
    monkeypatch.setattr(det, "is_orb_window_active", lambda: True)
    monkeypatch.setattr(det, "_fetch_intraday_bars", lambda s: ["bars"])
    monkeypatch.setattr(det, "_bars_to_dataframe", lambda b: df)
    assert det.check_orb_breakout("SPY").data_lag_min == 15


def test_stale_data_banner():
    assert "DATA DELAYED ~15m" in wm.WatchModeMonitor.stale_data_banner(15.0)
    assert wm.WatchModeMonitor.stale_data_banner(0.0) is None
    assert wm.WatchModeMonitor.stale_data_banner(-3.0, data_delayed=True) is None   # measured fresh wins
    assert "2188" in wm.WatchModeMonitor.stale_data_banner(None, data_delayed=True)
    assert wm.WatchModeMonitor.stale_data_banner(None) is None


def test_orb_scan_records_the_worst_lag():
    mon = wm.WatchModeMonitor()
    mon.watchlist_symbols = ["A", "B"]
    lags = {"A": 14.0, "B": 16.0}
    det = SimpleNamespace(check_orb_breakout=lambda s: orb.ORBSignal(symbol=s, has_breakout=False,
                                                                      data_lag_min=lags[s]))
    mon.run_orb_scan(det)
    assert mon.last_data_lag == 16.0
