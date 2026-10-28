"""
Whole-scan data-integrity gate (review H1, audit #8): a TWS disconnect, a farm outage or a
session conflict mid-scan must not turn the survivors of a partial data pull into a COMPLETE scan.
No TWS needed: the connector is scripted.
"""

import logging

import pytest

import momentum_scanner as ms
from test_data_reliability import ScriptedConnector, daily_bars


def _scanner(conn, symbols, monkeypatch, tmp_path):
    monkeypatch.setattr(ms, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(ms, "LOG_DIR", str(tmp_path))
    scanner = ms.MomentumScanner()
    scanner.connector = conn
    monkeypatch.setattr(scanner, "_set_regime_gating", lambda: None)
    monkeypatch.setattr(scanner, "get_stock_universe", lambda: list(symbols))
    monkeypatch.setattr(scanner, "_save_universe_snapshot", lambda s: None)
    exported = []
    monkeypatch.setattr(scanner, "_export_results", lambda t: exported.append(t))
    monkeypatch.setattr(scanner, "_render_results", lambda t: None)
    # Downstream stages are not under test here
    for name in ("_calculate_rs_for_stocks", "_detect_vcp_for_stocks", "_analyze_price_action_for_stocks"):
        monkeypatch.setattr(scanner, name, lambda stocks: stocks)
    for name in ("_apply_ibd_universe", "_apply_tier_filters"):
        monkeypatch.setattr(scanner, name, lambda stocks: None)
    for name in ("_calculate_priority_ratings_for_all_tiers", "_apply_stop_filter_to_tiers",
                 "_count_bases_for_finalists"):
        monkeypatch.setattr(scanner, name, lambda: None)
    monkeypatch.setattr(scanner, "_record_symbol_health", lambda t: None)
    return scanner, exported


class TimeoutConnector(ScriptedConnector):
    """Symbols in `bad` reply with `status`; everything else gets a full year of bars."""

    def __init__(self, bad, status='timeout', connected=True):
        super().__init__()
        self.bad, self.status, self._connected = set(bad), status, connected

    def isConnected(self):
        return self._connected

    def _fetch_historical_once(self, symbol, duration, bar_size, what_to_show, timeout):
        self.requests.append(symbol)
        return (None, self.status) if symbol in self.bad else (daily_bars(), 'ok')


SYMBOLS = [f"S{i:03d}" for i in range(100)]


def test_mass_timeouts_abort_the_scan_without_a_file(monkeypatch, tmp_path, caplog):
    conn = TimeoutConnector(bad=SYMBOLS[30:])          # 70% timed out: the farm died mid-scan
    scanner, exported = _scanner(conn, SYMBOLS, monkeypatch, tmp_path)
    with caplog.at_level(logging.ERROR):
        assert scanner.run_full_scan("afterhours") is False
    assert scanner.scan_aborted is True and exported == []
    assert any("SCAN ABORTED" in r.getMessage() and "70 of 100" in r.getMessage() for r in caplog.records)


def test_a_few_timeouts_do_not_abort(monkeypatch, tmp_path):
    conn = TimeoutConnector(bad=SYMBOLS[:15])          # under the 20-symbol floor
    scanner, exported = _scanner(conn, SYMBOLS, monkeypatch, tmp_path)
    scanner.run_full_scan("afterhours")
    assert scanner.scan_aborted is False


def test_lost_connection_aborts_even_with_survivors(monkeypatch, tmp_path):
    conn = TimeoutConnector(bad=[], connected=False)
    scanner, exported = _scanner(conn, SYMBOLS, monkeypatch, tmp_path)
    assert scanner.run_full_scan("afterhours") is False
    assert scanner.scan_aborted is True and exported == []


def test_hard_no_data_errors_are_not_an_integrity_failure(monkeypatch, tmp_path):
    """Error 200/162 per symbol are real answers (dead tickers), not a failed data pull."""
    conn = TimeoutConnector(bad=SYMBOLS[:60], status='no_secdef')
    scanner, _ = _scanner(conn, SYMBOLS, monkeypatch, tmp_path)
    scanner.run_full_scan("afterhours")
    assert scanner.scan_aborted is False


@pytest.mark.parametrize("aborted, code", [(True, 1), (False, 2)])
def test_main_exit_code_retryable_only_on_integrity_abort(monkeypatch, tmp_path, aborted, code):
    monkeypatch.setattr(ms, "LOG_DIR", str(tmp_path))
    monkeypatch.setattr(ms.MomentumScanner, "initialize", lambda self: True)
    monkeypatch.setattr(ms.MomentumScanner, "cleanup", lambda self: None)

    def scan(self, scan_type):
        self.scan_aborted = aborted
        return False
    monkeypatch.setattr(ms.MomentumScanner, "run_full_scan", scan)
    assert ms.main() == code


# ─── Error 162 "different IP" (audit #8) ───
def test_error_162_different_ip_is_a_session_conflict_not_no_data():
    import ibkr_connector as ic
    msg = "Historical Market Data Service error message:Trading TWS session is connected from a different IP address"
    assert ic.IBKRConnector._error_status({'code': 162, 'msg': msg}) == 'session_conflict'
    assert ic.IBKRConnector._error_status({'code': 162, 'msg': "HMDS query returned no data: X"}) == 'no_data'
    assert ic.IBKRConnector._error_status({'code': 162, 'msg': "API historical data query cancelled / pacing violation"}) == 'pacing'


def test_session_conflict_aborts_a_full_scan(monkeypatch, tmp_path):
    conn = TimeoutConnector(bad=SYMBOLS, status='session_conflict')
    scanner, exported = _scanner(conn, SYMBOLS, monkeypatch, tmp_path)
    assert scanner.run_full_scan("afterhours") is False
    assert scanner.scan_aborted is True and exported == []
    assert len(scanner.scan_issues['session']) == 100 and scanner.scan_issues['no_data'] == []


def test_session_conflict_aborts_a_small_watch_scan_too(monkeypatch, tmp_path, caplog):
    """The watcher's 10-name list is under the 20-symbol floor; a session block must still show."""
    watch = SYMBOLS[:10]
    conn = TimeoutConnector(bad=watch, status='session_conflict')
    scanner, _ = _scanner(conn, watch, monkeypatch, tmp_path)
    with caplog.at_level(logging.ERROR):
        assert scanner.run_full_scan("watch") is False
    assert any("different IP" in r.getMessage() for r in caplog.records)


# ─── Audit #10: CLI scans never take the watcher's client id ───
def _main_client(monkeypatch, tmp_path, env):
    monkeypatch.setattr(ms, "LOG_DIR", str(tmp_path))
    if env is None:
        monkeypatch.delenv("TRADING_CLIENT_ID", raising=False)
    else:
        monkeypatch.setenv("TRADING_CLIENT_ID", env)
        monkeypatch.setattr(ms, "CLIENT_ID", int(env))
    seen = {}
    monkeypatch.setattr(ms.MomentumScanner, "initialize", lambda self: seen.setdefault('id', ms.CLIENT_ID) and False)
    monkeypatch.setattr(ms.MomentumScanner, "cleanup", lambda self: None)
    return ms.main(), seen.get('id')


def test_cli_scan_without_env_uses_the_adhoc_id(monkeypatch, tmp_path):
    monkeypatch.setattr(ms, "CLIENT_ID", ms.WATCHER_CLIENT_ID)
    rc, used = _main_client(monkeypatch, tmp_path, None)
    assert used == ms.ADHOC_CLIENT_ID != ms.WATCHER_CLIENT_ID


def test_scheduled_id_is_kept(monkeypatch, tmp_path):
    rc, used = _main_client(monkeypatch, tmp_path, "11")
    assert used == 11


def test_explicit_watcher_id_is_refused(monkeypatch, tmp_path):
    rc, used = _main_client(monkeypatch, tmp_path, str(ms.WATCHER_CLIENT_ID))
    assert rc == 3 and used is None


# ─── Audit #12: a stale IBD export is logged, not only drawn on the dashboard ───
def test_stale_ibd_export_logs_a_warning(caplog):
    from datetime import date
    sc = ms.MomentumScanner()
    sc.ibd_universe_date = date(2026, 10, 1)
    with caplog.at_level(logging.WARNING):
        assert sc._warn_if_ibd_stale(today=date(2026, 10, 26)) == 25
    assert any("25 days old" in r.getMessage() for r in caplog.records)
    sc.ibd_universe_date = date(2026, 10, 22)
    assert sc._warn_if_ibd_stale(today=date(2026, 10, 26)) is None
