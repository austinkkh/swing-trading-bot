"""Dead / renamed tickers: IBKR's reason is kept, Error 200 quarantines after N scan days,
timeouts never do, nothing is dropped silently and nothing is ever added. No TWS needed."""

import json
from datetime import date, datetime
from io import StringIO

import pytest
from rich.console import Console

import momentum_scanner as ms
import scan_display
import symbol_hygiene as sh

D1, D2, D3 = date(2026, 10, 26), date(2026, 10, 27), date(2026, 10, 28)


@pytest.fixture
def files(tmp_path, monkeypatch):
    monkeypatch.setattr(sh, "OVERRIDES_CSV", tmp_path / "symbol_overrides.csv")
    monkeypatch.setattr(sh, "HEALTH_JSON", tmp_path / "symbol_health.json")
    monkeypatch.setattr(sh, "SNAPSHOT_DIR", tmp_path / "account")
    monkeypatch.setattr(sh, "PIPELINE_SNAPSHOT_DIR", tmp_path / "paper_runs")
    return tmp_path


def write_overrides(path, rows):
    path.write_text("symbol,action,new_symbol,reason,evidence,added,review_after\n" + "\n".join(rows) + "\n")


def snapshot(dirpath, held=(), pending=(), ts=None):
    dirpath.mkdir(parents=True, exist_ok=True)
    ts = ts or datetime.now()
    (dirpath / f"account_snapshot_{ts:%Y-%m-%d_%H%M}.json").write_text(json.dumps(
        {'timestamp': ts.isoformat(timespec='seconds'), 'held_symbols': list(held),
         'pending_entry_orders': list(pending)}))


# ─── Scanner keeps IBKR's reason ───
class StatusConn:
    def __init__(self, statuses):
        self.statuses = statuses

    def fetch_historical_data_ex(self, symbol, **kw):
        return None, self.statuses[symbol]


def test_error_200_and_timeouts_are_recorded_apart():
    scanner = ms.MomentumScanner()
    scanner.connector = StatusConn({'DEAD': 'no_secdef', 'SLOW': 'timeout', 'EMPTY': 'no_data'})
    for sym in ('DEAD', 'SLOW', 'EMPTY'):
        assert scanner._filter_single_stock(sym) is None
    no_data = dict(scanner.scan_issues['no_data'])
    assert no_data['DEAD'].startswith('no_secdef') and no_data['EMPTY'].startswith('no_data')
    assert [s for s, _ in scanner.scan_issues['timeout']] == ['SLOW']
    assert 'timeout' not in ms.MomentumScanner._ERROR_STAGES
    assert scanner.fetch_status == {'DEAD': 'no_secdef', 'SLOW': 'timeout', 'EMPTY': 'no_data'}


# ─── Quarantine counter ───
def test_quarantine_counts_distinct_scan_days():
    state = {'symbols': {}}
    for _ in range(3):
        assert sh.record_scan_results(state, {'X': 'no_secdef'}, D1, protected=set()) == ([], [])
    assert state['symbols']['X']['consecutive_secdef_days'] == 1     # same day thrice = 1
    sh.record_scan_results(state, {'X': 'no_secdef'}, D2, protected=set())
    new_q, _ = sh.record_scan_results(state, {'X': 'no_secdef'}, D3, protected=set())
    assert new_q == ['X'] and state['symbols']['X']['quarantined_on'] == D3.isoformat()


def test_a_success_in_between_resets_the_count():
    state = {'symbols': {}}
    sh.record_scan_results(state, {'X': 'no_secdef'}, D1, protected=set())
    sh.record_scan_results(state, {'X': 'ok'}, D2, protected=set())
    new_q, _ = sh.record_scan_results(state, {'X': 'no_secdef'}, D3, protected=set())
    assert new_q == [] and state['symbols']['X']['consecutive_secdef_days'] == 1


@pytest.mark.parametrize("status", ['timeout', 'no_data', 'pacing', 'error:504'])
def test_timeouts_and_162_never_count(status):
    """Valid names (MPC, VLO, P) time out in a busy watcher: they must never be quarantined."""
    state = {'symbols': {}}
    for d in (D1, D2, D3, date(2026, 10, 29)):
        sh.record_scan_results(state, {'MPC': status}, d, protected=set())
    assert state['symbols'] == {}


def test_newer_ibd_export_lifts_the_quarantine():
    state = {'symbols': {'ACA': {'consecutive_secdef_days': 3, 'first_fail': '2026-10-23',
                                 'last_fail': '2026-10-27', 'quarantined_on': '2026-10-27'}}}
    # An export dated on/before the first failure proves nothing
    assert sh.record_scan_results(state, {}, D3, ibd_symbols={'ACA'}, ibd_date=date(2026, 10, 22),
                                  protected=set()) == ([], [])
    _, lifted = sh.record_scan_results(state, {}, D3, ibd_symbols={'ACA'}, ibd_date=date(2026, 10, 28),
                                       protected=set())
    assert lifted == ['ACA'] and state['symbols'] == {}


def test_weekly_reprobe_lifts_names_that_exist_again():
    class Conn:
        calls = []

        def fetch_contract_details(self, sym, timeout=10):
            self.calls.append(sym)
            return object() if sym == 'BACK' else None
    state = {'symbols': {s: {'quarantined_on': '2026-10-20', 'consecutive_secdef_days': 3}
                         for s in ('BACK', 'GONE')}, 'last_reprobe': '2026-10-20'}
    conn = Conn()
    assert sh.weekly_reprobe(state, conn, D1) == ['BACK']           # new ISO week -> probe
    assert set(state['symbols']) == {'GONE'} and state['last_reprobe'] == D1.isoformat()
    assert sh.weekly_reprobe(state, conn, D2) == [] and len(conn.calls) == 2   # once a week


def test_held_or_ordered_symbols_are_never_quarantined_or_skipped(files):
    state = {'symbols': {}}
    for d in (D1, D2, D3):
        sh.record_scan_results(state, {'HELD': 'no_secdef'}, d, protected={'HELD'})
    assert not state['symbols']['HELD']['quarantined_on']
    # Unknown positions (no snapshot) -> no new quarantine either
    for d in (D1, D2, D3):
        sh.record_scan_results(state, {'UNK': 'no_secdef'}, d, protected=None)
    assert not state['symbols']['UNK']['quarantined_on']
    write_overrides(sh.OVERRIDES_CSV, ["HELD,skip,,acquired,,2026-10-26,2027-01-26"])
    q_state = {'symbols': {'ORD': {'quarantined_on': '2026-10-27', 'consecutive_secdef_days': 3}}}
    snapshot(sh.SNAPSHOT_DIR, held=['HELD'], pending=['ORD'])
    kept, dropped = sh.apply_symbol_hygiene(['HELD', 'ORD', 'MU'], state=q_state)
    assert kept == ['HELD', 'ORD', 'MU'] and dropped == []


def test_quarantine_needs_a_recent_snapshot(files):
    q_state = {'symbols': {'DEAD': {'quarantined_on': '2026-10-27', 'consecutive_secdef_days': 3}}}
    assert sh.apply_symbol_hygiene(['DEAD'], overrides={}, state=q_state)[0] == ['DEAD']   # no snapshot
    snapshot(sh.SNAPSHOT_DIR)
    kept, dropped = sh.apply_symbol_hygiene(['DEAD'], overrides={}, state=q_state)
    assert kept == [] and dropped[0][1].startswith('auto-quarantined')


def test_pipeline_snapshot_counts_for_quarantine(files):
    """Review M2: the 9:20 pipeline writes Scans/paper_runs/account_snapshot_*.json, never
    Data/account; with only old manual snapshots there, quarantine silently stopped working."""
    from datetime import timedelta
    q_state = {'symbols': {'DEAD': {'quarantined_on': '2026-10-27', 'consecutive_secdef_days': 3}}}
    snapshot(sh.SNAPSHOT_DIR, held=['OLD'], ts=datetime.now() - timedelta(days=10))   # stale manual run
    assert sh.protected_symbols() is None
    snapshot(sh.PIPELINE_SNAPSHOT_DIR, held=['DEAD'])                               # today's pipeline run
    assert sh.protected_symbols() == {'DEAD'}
    snapshot(sh.PIPELINE_SNAPSHOT_DIR, held=[], ts=datetime.now() + timedelta(minutes=1))
    assert sh.protected_symbols() == set()                                        # newest by timestamp
    kept, dropped = sh.apply_symbol_hygiene(['DEAD', 'MU'], overrides={}, state=q_state)
    assert kept == ['MU'] and dropped[0][0] == 'DEAD'


def test_newest_snapshot_wins_across_folders(files):
    from datetime import timedelta
    snapshot(sh.PIPELINE_SNAPSHOT_DIR, held=['A'], ts=datetime.now() - timedelta(hours=5))
    snapshot(sh.SNAPSHOT_DIR, held=['B'])
    assert sh.protected_symbols() == {'B'}


# ─── Overrides ───
def test_rename_dedupes_and_never_adds(files):
    write_overrides(sh.OVERRIDES_CSV, ["PSTG,rename,P,renamed,,2026-10-26,2027-01-26",
                                       "VSCO,rename,VSXY,renamed,,2026-10-26,2027-01-26"])
    kept, dropped = sh.apply_symbol_hygiene(['PSTG', 'P', 'VSCO', 'MU'], state={'symbols': {}}, protected=set())
    assert kept == ['P', 'MU']                                       # VSXY not added
    assert dict(dropped) == {'PSTG': 'renamed to P (already scanned)',
                             'VSCO': 'renamed to VSXY (not added: owner to approve)'}


@pytest.mark.parametrize("content", ["symbol,action\nX,delete\n", "\x00\x01garbage"])
def test_malformed_files_fail_open(files, content):
    sh.OVERRIDES_CSV.write_text(content)
    sh.HEALTH_JSON.write_text("{not json")
    snapshot(sh.SNAPSHOT_DIR)
    assert sh.apply_symbol_hygiene(['X', 'MU']) == (['X', 'MU'], [])


def test_seeded_override_file_is_valid():
    rows = sh.load_overrides(sh.REPO_DIR / "symbol_overrides.csv")
    assert rows and all(o.action in ('skip', 'rename') for o in rows.values())
    assert rows['OLDT'].new_symbol == 'NEWT'


# ─── One choke point, every mode ───
@pytest.mark.parametrize("mode", ["DYNAMIC", "STAGE2_WATCHLIST", "CUSTOM", "SOMETHING_ELSE"])
def test_hygiene_runs_in_every_universe_mode(files, monkeypatch, mode):
    write_overrides(sh.OVERRIDES_CSV, ["CFLT,skip,,acquired,,2026-10-26,2027-01-26"])
    monkeypatch.setattr(ms, "SCAN_UNIVERSE", mode)
    monkeypatch.setattr(ms, "CUSTOM_SYMBOLS", ["MU", "CFLT"])
    monkeypatch.setattr(ms, "STAGE2_WATCHLIST", ["MU", "CFLT"])
    monkeypatch.setattr(ms.MomentumScanner, "_build_dynamic_universe", lambda self: ["MU", "CFLT"])
    scanner = ms.MomentumScanner()
    assert scanner.get_stock_universe() == ["MU"]
    assert scanner.hygiene_dropped == [("CFLT", "skip: acquired")]


def test_watcher_watchlist_is_filtered(files, monkeypatch, tmp_path):
    import pandas as pd
    import watch_mode
    write_overrides(sh.OVERRIDES_CSV, ["CFLT,skip,,acquired,,2026-10-26,2027-01-26"])
    pd.DataFrame({"Symbol": ["MU", "CFLT"], "Tier": ["TIER_BUY", "TIER_BUY"]}).to_csv(
        tmp_path / "scan_2026-10-26_0920_afterhours_COMPLETE.csv", index=False)
    monkeypatch.setattr(watch_mode, "OUTPUT_DIR", str(tmp_path))
    mon = watch_mode.WatchModeMonitor()
    assert mon.load_latest_elite_hold() and mon.watchlist_symbols == ["MU"]


# ─── Visible, never silent ───
def render_warnings(scanner) -> str:
    buf = StringIO()
    Console(file=buf, width=200).print(scan_display._warnings(scanner))
    return buf.getvalue()


def test_display_shows_skips_new_secdefs_and_due_reviews(files, monkeypatch):
    write_overrides(sh.OVERRIDES_CSV, ["CFLT,skip,,acquired,,2026-07-01,2026-10-01"])   # review overdue
    scanner = ms.MomentumScanner()
    scanner.hygiene_dropped = [("CFLT", "skip: acquired"), ("GONE", "auto-quarantined 2026-10-20 (...)")]
    scanner.scan_issues['no_data'] = [("ACA", ms.STATUS_REASONS['no_secdef'])]
    scanner.scan_issues['timeout'] = [("MPC", "IBKR timeout")]
    scanner.secdef_days = {"ACA": 2}
    scanner.ibd_universe_date = date.today()
    text = render_warnings(scanner)
    assert "Skipped 1 known dead/renamed symbols (symbol_overrides.csv) + 1 auto-quarantined: CFLT, GONE" in text
    assert "No security definition (1): ACA (day 2 of 3)" in text
    assert "Override review due: CFLT" in text
    assert "transient, not delisted" in text and "likely delisted" not in text


def test_timeouts_alone_are_not_a_warning(files):
    scanner = ms.MomentumScanner()
    scanner.scan_issues['timeout'] = [("MPC", "IBKR timeout")]
    scanner.ibd_universe_date = date.today()
    text = render_warnings(scanner)
    assert "No data problems" in text and "MPC" in text


def test_scan_health_log_lists_skipped_symbols(caplog):
    import logging
    scanner = ms.MomentumScanner()
    scanner.hygiene_dropped = [("CFLT", "skip: acquired")]
    with caplog.at_level(logging.INFO, logger="momentum_scanner"):   # the watcher test sets it to WARNING
        scanner._log_scan_health(10, 5)
    assert any("Skipped by symbol hygiene" in r.getMessage() and "CFLT" in r.getMessage() for r in caplog.records)
