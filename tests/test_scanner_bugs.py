"""Code-review batch 4: scanner bugs S1, S4, S12, RS date alignment."""

import threading
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import momentum_scanner as ms
from ibkr_connector import IBKRConnector
from relative_strength import RelativeStrengthCalculator


def test_unlisted_issue_stage_never_raises():
    """S1: a pocket-pivot error used to raise KeyError and drop a stock that passed every filter."""
    scanner = ms.MomentumScanner()
    scanner._record_issue('pocket_pivot', 'MU', 'boom')
    scanner._record_issue('some_new_stage', 'MU', 'boom')
    assert scanner.scan_issues['pocket_pivot'] == [('MU', 'boom')]


def test_request_ids_unique_across_threads():
    """S4: 50 threads x 200 ids must never share one."""
    conn = IBKRConnector()
    ids, barrier = [], threading.Barrier(50)

    def grab():
        barrier.wait()
        local = [conn.get_next_req_id() for _ in range(200)]
        ids.extend(local)

    threads = [threading.Thread(target=grab) for _ in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(ids) == len(set(ids)) == 10_000


@pytest.mark.parametrize("scan_ok, code", [(True, 0), (False, 2)])
def test_main_exit_code_reflects_scan_result(monkeypatch, tmp_path, scan_ok, code):
    """S12: an empty scan writes no CSV, so main() must not report success."""
    monkeypatch.setattr(ms, "LOG_DIR", str(tmp_path))
    monkeypatch.setattr(ms.MomentumScanner, "initialize", lambda self: True)
    monkeypatch.setattr(ms.MomentumScanner, "run_full_scan", lambda self, scan_type: scan_ok)
    monkeypatch.setattr(ms.MomentumScanner, "cleanup", lambda self: None)
    assert ms.main() == code


def bars(dates, closes):
    return [SimpleNamespace(date=d.strftime("%Y%m%d"), open=c, high=c, low=c, close=c, volume=1e6)
            for d, c in zip(dates, closes)]


def test_rs_compares_spy_over_the_stocks_own_dates():
    """A stock missing SPY's last 5 sessions must be measured against SPY on its own dates."""
    days = pd.bdate_range("2025-01-02", periods=300)
    spy = RelativeStrengthCalculator(connector=None)
    spy_close = np.linspace(100, 130, 300)
    spy_close[-5:] = 200.0                       # a SPY jump the stock's data doesn't cover
    spy.spy_data = spy._bars_to_dataframe(bars(days, spy_close))
    stock = bars(days[:-5], spy_close[:-5])      # tracks SPY exactly, last bar 5 sessions old
    assert spy.calculate_rs_rating(stock) == pytest.approx(50.0)   # same return as SPY = RS 50
