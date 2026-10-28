"""Audit #5: an 'afterhours' scan run after 9:30 ET must not read today's partial daily bar as
yesterday's completed session. The watcher keeps intraday bars. No TWS needed."""

from datetime import datetime
from types import SimpleNamespace

import ibkr_connector as ic
import market_time as mt
import momentum_scanner as ms


def bars(*dates):
    return [SimpleNamespace(date=d, close=1.0) for d in dates]


B = bars("20261022", "20261023", "20261026")
drop = ic.IBKRConnector._drop_forming_bar


def test_partial_bar_dropped_during_the_session():
    assert [b.date for b in drop(B, now=datetime(2026, 10, 26, 10, 10))] == ["20261022", "20261023"]


def test_premarket_and_after_close_keep_every_bar():
    assert drop(B, now=datetime(2026, 10, 26, 9, 20)) == B
    assert drop(B, now=datetime(2026, 10, 26, 16, 5)) == B


def test_no_bar_for_today_yet_is_left_alone():
    older = bars("20261022", "20261023")
    assert drop(older, now=datetime(2026, 10, 26, 10, 0)) == older


def test_early_close_day_bar_is_complete_after_13():
    b = bars("20261125", "20261127")
    assert mt.market_close_time(datetime(2026, 11, 27).date()) == mt.EARLY_CLOSE
    assert drop(b, now=datetime(2026, 11, 27, 12, 0)) == b[:1]
    assert drop(b, now=datetime(2026, 11, 27, 13, 30)) == b


def test_only_the_afterhours_cli_turns_it_on(monkeypatch):
    assert ms.MomentumScanner().complete_daily_bars_only is False      # watcher default
    seen = {}
    monkeypatch.setattr(ms.MomentumScanner, "initialize",
                        lambda self: seen.setdefault('flag', self.complete_daily_bars_only) and False)
    monkeypatch.setattr(ms.MomentumScanner, "cleanup", lambda self: None)
    ms.main()
    assert seen['flag'] is True


def test_connector_applies_it_to_daily_bars_only(monkeypatch):
    conn = ic.IBKRConnector()
    conn.serverVersion = lambda: 219
    conn.complete_daily_bars_only = True
    monkeypatch.setattr(mt, "now_et", lambda: datetime(2026, 10, 26, 11, 0))

    def req(req_id, contract, end, duration, bar_size, *a):
        for b in (B if bar_size == "1 day" else bars("20261026 10:55:00")):
            conn.historicalData(req_id, b)
        conn.historicalDataEnd(req_id, "", "")
    monkeypatch.setattr(conn, "reqHistoricalData", req)
    daily, status = conn.fetch_historical_data_ex("MU", duration="1 Y", bar_size="1 day")
    assert status == 'ok' and len(daily) == 2
    intraday, _ = conn.fetch_historical_data_ex("MU", duration="1 D", bar_size="5 mins")
    assert len(intraday) == 1
