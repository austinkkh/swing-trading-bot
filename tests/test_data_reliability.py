"""
Data-farm reliability (2026-10-26: the 9:20 scan ran during a 45 s wake while TWS was still
reconnecting, timed out once on SPY and gave up). No TWS needed: connectors are scripted.
"""

import logging
import os
import stat
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import ibkr_connector as ic
import momentum_scanner as ms

ROOT = Path(__file__).resolve().parents[1]


def daily_bars(n=260, start=100.0):
    days = pd.bdate_range("2025-09-01", periods=n)
    return [SimpleNamespace(date=d.strftime("%Y%m%d"), open=start + i, high=start + i + 1, low=start + i - 1,
                            close=start + i, volume=1e6) for i, d in enumerate(days)]


class ScriptedConnector(ic.IBKRConnector):
    """Real connector logic (farm state, retries) with the TWS calls replaced."""

    def __init__(self, replies=None):
        super().__init__()
        self.replies = list(replies or [])   # per _fetch_historical_once call: (bars, status)
        self.requests = []

    def connect_and_run(self, host, port, client_id):
        return True

    def disconnect_and_stop(self):
        pass

    def _fetch_historical_once(self, symbol, duration, bar_size, what_to_show, timeout):
        self.requests.append(symbol)
        return self.replies.pop(0) if self.replies else (daily_bars(), 'ok')


@pytest.fixture
def no_sleep(monkeypatch):
    """Fake monotonic clock: _sleep advances it instantly."""
    clock = {'t': 0.0, 'hooks': []}

    def sleep(s):
        clock['t'] += s
        for hook in list(clock['hooks']):
            hook(clock['t'])
    monkeypatch.setattr(ic, "_monotonic", lambda: clock['t'])
    monkeypatch.setattr(ic, "_sleep", sleep)
    return clock


@pytest.fixture
def scanner_env(monkeypatch, tmp_path):
    monkeypatch.setattr(ms, "load_ibd_universe", lambda glob: ({}, None, None))
    monkeypatch.setattr(ms, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(ms, "LOG_DIR", str(tmp_path))

    def install(conn):
        monkeypatch.setattr(ms, "IBKRConnector", lambda: conn)
        return conn
    return install


def test_scan_aborts_when_the_farm_stays_broken(scanner_env, no_sleep, caplog, monkeypatch):
    conn = scanner_env(ScriptedConnector())
    conn.error(-1, 2105, "HMDS data farm connection is broken:ushmds")
    with caplog.at_level(logging.ERROR):
        assert ms.MomentumScanner().initialize() is False
    assert no_sleep['t'] >= ms.FARM_WAIT_SEC                  # it waited first
    assert conn.requests == []                                 # and never asked for SPY
    assert any("data farm unavailable" in r.getMessage() for r in caplog.records)
    monkeypatch.setattr(ms.MomentumScanner, "run_full_scan", lambda self, scan_type: pytest.fail("no scan"))
    assert ms.main() == 1                                      # pipeline stops before the engine


def test_scan_proceeds_once_the_farm_recovers(scanner_env, no_sleep):
    conn = scanner_env(ScriptedConnector())
    conn.error(-1, 2110, "Connectivity between Trader Workstation and server is broken.")
    conn.error(-1, 2105, "HMDS data farm connection is broken:ushmds")
    no_sleep['hooks'].append(lambda t: t >= 5 and conn.error(-1, 2106, "HMDS data farm connection is OK:ushmds"))
    scanner = ms.MomentumScanner()
    assert scanner.initialize() is True
    assert 5 <= no_sleep['t'] < 10
    assert scanner.rs_calculator.spy_data is not None and conn.requests == ['SPY']


def test_spy_timeout_once_then_success_still_initializes(scanner_env, no_sleep):
    conn = scanner_env(ScriptedConnector(replies=[(None, 'timeout')]))
    scanner = ms.MomentumScanner()
    assert scanner.initialize() is True
    assert conn.requests == ['SPY', 'SPY']


def test_spy_hard_error_is_not_retried(scanner_env, no_sleep):
    conn = scanner_env(ScriptedConnector(replies=[(None, 'no_data')]))
    assert ms.MomentumScanner().initialize() is False
    assert conn.requests == ['SPY']


def test_regime_retries_a_transient_index_timeout(no_sleep):
    import market_regime as mr
    conn = ScriptedConnector(replies=[(None, 'timeout')])
    det = mr.MarketRegimeDetector(conn)
    assert det._fetch_index_data() is True                     # not dropped to watch-only
    assert conn.requests == ['SPY', 'SPY', 'QQQ']


# ─── Watcher ───
class FakeScanner:
    def __init__(self, init_ok=True, farms_ok=True):
        self.init_ok = init_ok
        self.connector = ScriptedConnector()
        if not farms_ok:
            self.connector.error(-1, 2105, "HMDS data farm connection is broken:ushmds")
        self.connector.isConnected = lambda: True
        self.connector.serverVersion = lambda: 219

    def initialize(self):
        return self.init_ok

    def cleanup(self):
        pass


def _watcher(monkeypatch, scanner):
    import watch_mode as wm
    monkeypatch.setattr(wm, "MomentumScanner", lambda: scanner)
    monkeypatch.setattr(wm, "ORBDetector", lambda conn: None)
    mon = wm.WatchModeMonitor()
    monkeypatch.setattr(mon, "load_latest_elite_hold", lambda: True)
    monkeypatch.setattr(mon, "reload_if_new_scan", lambda: False)
    monkeypatch.setattr(mon, "get_current_phase", lambda: ("ORB_HUNT_FAST", 30, True))
    return wm, mon


def test_watcher_init_failure_exits_nonzero(monkeypatch):
    _, mon = _watcher(monkeypatch, FakeScanner(init_ok=False))
    with pytest.raises(SystemExit) as e:
        mon.run_forever()
    assert e.value.code == 1


def test_watcher_skips_scans_while_the_farm_is_down(monkeypatch, caplog):
    wm, mon = _watcher(monkeypatch, FakeScanner(farms_ok=False))
    monkeypatch.setattr(mon, "run_orb_scan", lambda det: pytest.fail("must not scan"))
    monkeypatch.setattr(mon, "run_migration_scan", lambda sc: pytest.fail("must not scan"))

    def stop(_):
        raise KeyboardInterrupt
    monkeypatch.setattr(wm.time, "sleep", stop)
    with caplog.at_level(logging.WARNING):
        mon.run_forever()
    assert mon.scan_count == 0
    assert any("skipping scan" in r.getMessage() for r in caplog.records)


# ─── Pipeline shell retry (stub PYTHON, fake `date`) ───
def _pipeline(tmp_path, scan_codes, now_hhmm="0921", attempts=3, notify="0"):
    work = tmp_path / "repo"
    work.mkdir()
    script = work / "run_daily_pipeline.sh"
    script.write_text((ROOT / "run_daily_pipeline.sh").read_text())
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "calls.log"
    codes = tmp_path / "codes"
    codes.write_text("\n".join(str(c) for c in scan_codes) + "\n")
    stub = bin_dir / "py"
    stub.write_text(f"""#!/bin/bash
echo "$1" >> {calls}
case "$1" in
  -) echo {tmp_path / 'out'} ;;
  momentum_scanner.py)
     echo "farm=$TRADING_FARM_WAIT_SEC" >> {calls}
     code=$(head -n1 {codes}); tail -n +2 {codes} > {codes}.t; mv {codes}.t {codes}
     exit "${{code:-1}}" ;;
  notify.py) echo "$*" >> {tmp_path / 'notify.log'} ;;
  *) exit 0 ;;
esac
""")
    fake_date = bin_dir / "date"
    fake_date.write_text(f"""#!/bin/bash
case "$1" in
  +%u) echo 3 ;;
  +%H%M) echo {now_hhmm} ;;
  *) /bin/date "$@" ;;
esac
""")
    for f in (stub, fake_date):
        f.chmod(f.stat().st_mode | stat.S_IEXEC)
    env = dict(os.environ, PYTHON=str(stub), PATH=f"{bin_dir}:{os.environ['PATH']}",
               TRADING_SCAN_RETRY_SLEEP="0", TRADING_SCAN_ATTEMPTS=str(attempts), TRADING_NOTIFY=notify)
    proc = subprocess.run(["bash", str(script)], cwd=work, env=env, capture_output=True, text=True, timeout=60)
    lines = calls.read_text().split() if calls.exists() else []
    return proc, lines


def test_pipeline_retries_a_failed_scan_up_to_the_limit(tmp_path):
    proc, calls = _pipeline(tmp_path, [1, 1, 1])
    assert proc.returncode == 1
    assert calls.count("momentum_scanner.py") == 3
    assert "execution_engine.py" not in calls and "account_snapshot.py" not in calls
    assert "scan attempt 3/3" in proc.stdout


def test_pipeline_no_retry_after_the_deadline(tmp_path):
    proc, calls = _pipeline(tmp_path, [1, 1, 1], now_hhmm="0950")
    assert proc.returncode == 1 and calls.count("momentum_scanner.py") == 1
    assert "deadline" in proc.stdout


def test_pipeline_does_not_retry_an_empty_scan(tmp_path):
    proc, calls = _pipeline(tmp_path, [2])
    assert proc.returncode == 2 and calls.count("momentum_scanner.py") == 1


def test_pipeline_recovers_on_a_retry(tmp_path):
    # Second attempt succeeds; the stub writes no COMPLETE file, so the pipeline still stops
    # (fail safe), but only after the scanner was retried and before the engine.
    proc, calls = _pipeline(tmp_path, [1, 0])
    assert calls.count("momentum_scanner.py") == 2
    assert "wrote no COMPLETE file" in proc.stdout and proc.returncode == 1
    assert "execution_engine.py" not in calls


def test_pipeline_caps_each_attempts_farm_wait_at_the_deadline(tmp_path):
    """M1: at 09:40 with a 09:45 deadline an attempt may wait for the farm 300 s, not 600 s."""
    proc, calls = _pipeline(tmp_path, [1, 1], now_hhmm="0940", attempts=2)
    assert calls.count("farm=300") == 2
    (tmp_path / "early").mkdir()
    proc, calls = _pipeline(tmp_path / "early", [2], now_hhmm="0921")
    assert "farm=600" in calls                                  # 24 min left: the normal cap


def _notifications(tmp_path):
    log = tmp_path / "notify.log"
    return log.read_text().splitlines() if log.exists() else []


def test_pipeline_notifies_a_final_scan_failure_once(tmp_path):
    proc, calls = _pipeline(tmp_path, [2], notify="1")
    sent = _notifications(tmp_path)
    assert proc.returncode == 2 and len(sent) == 1
    assert "Premarket scan FAILED" in sent[0] and "--level error" in sent[0] and "exited 2" in sent[0]


def test_pipeline_notifies_any_other_failure_on_exit(tmp_path):
    proc, calls = _pipeline(tmp_path, [0], notify="1")       # scan ok, but no COMPLETE file
    sent = _notifications(tmp_path)
    assert proc.returncode == 1 and len(sent) == 1
    assert "Premarket pipeline FAILED" in sent[0] and "exited 1" in sent[0]


def test_pipeline_notify_off_sends_nothing(tmp_path):
    proc, calls = _pipeline(tmp_path, [2])
    assert _notifications(tmp_path) == [] and "notify.py" not in calls


def test_pipeline_run_after_the_deadline_keeps_the_full_wait(tmp_path):
    proc, calls = _pipeline(tmp_path, [2], now_hhmm="1010")
    assert "farm=600" in calls


def test_farm_waits_share_one_budget_across_retries(no_sleep):
    """M1: SPY retries no longer wait 600 s each; together they get the run's one budget."""
    conn = ScriptedConnector(replies=[(None, 'timeout')] * 4)
    conn.error(-1, 2105, "HMDS data farm connection is broken:ushmds")
    conn.farm_wait_budget = 100
    bars, status = conn.fetch_historical_data_ex("SPY", retries=3)
    assert bars is None and status == 'timeout'
    assert conn.farm_wait_budget == 0
    assert no_sleep['t'] <= 15 + 100 + 1                       # one backoff + the budget, not 3 x 600
    assert conn.requests == ['SPY']                            # farm still down: no pointless retries


def test_watcher_uses_short_farm_waits(monkeypatch):
    scanner = FakeScanner()
    scanner.connector.farm_wait_budget = 600
    wm, mon = _watcher(monkeypatch, scanner)
    monkeypatch.setattr(mon, "run_orb_scan", lambda det: 0)
    monkeypatch.setattr(mon, "run_migration_scan", lambda sc: 0)

    def stop(_):
        raise KeyboardInterrupt
    monkeypatch.setattr(wm.time, "sleep", stop)
    mon.run_forever()
    assert scanner.connector.farm_wait_sec == wm.WATCH_FARM_WAIT_SEC
    assert scanner.connector.farm_wait_budget is None


def test_watcher_probes_a_stale_link_flag_and_clears_it(monkeypatch):
    """L5: link marked down (2110) and TWS never sends 1102 / farm-OK: after 5 min one cheap
    probe that gets an answer clears the flag; probes are rate-limited."""
    import watch_mode as wm
    conn = ScriptedConnector()
    conn.error(-1, 2110, "Connectivity between Trader Workstation and server is broken.")
    probes = []
    conn.fetch_contract_details = lambda sym, timeout=10: probes.append(sym) or None
    mon = wm.WatchModeMonitor()
    mon.data_down_since = 0.0
    assert mon.probe_stale_link(conn, 60) is False and probes == []          # too early
    assert mon.probe_stale_link(conn, 301) is False and probes == ['SPY']    # no answer
    assert mon.probe_stale_link(conn, 400) is False and probes == ['SPY']    # rate-limited
    conn.fetch_contract_details = lambda sym, timeout=10: probes.append(sym) or "DETAILS"
    assert mon.probe_stale_link(conn, 700) is True and conn.server_link_ok is True
    assert wm.WatchModeMonitor.data_ready(conn) is True


def test_watcher_probe_never_clears_an_explicitly_broken_farm():
    import watch_mode as wm
    conn = ScriptedConnector()
    conn.error(-1, 2110, "Connectivity between Trader Workstation and server is broken.")
    conn.error(-1, 2105, "HMDS data farm connection is broken:ushmds")
    conn.fetch_contract_details = lambda sym, timeout=10: "DETAILS"
    mon = wm.WatchModeMonitor()
    mon.data_down_since = 0.0
    mon.probe_stale_link(conn, 400)
    assert wm.WatchModeMonitor.data_ready(conn) is False                     # ushmds still broken


