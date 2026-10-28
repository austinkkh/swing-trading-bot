"""morning_report.py: read-only health rows, offline (fake connector, fake launchctl, temp files)."""

import json
import os
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

import morning_report as mr
import strategy_params as sp

NOW = datetime(2026, 11, 4, 11, 0)          # Wednesday, after the 9:20 pipeline


@pytest.fixture
def paths(tmp_path):
    for d in ("Scans", "Logs/scheduler", "launchd", "Data/orders", "Data/account"):
        (tmp_path / d).mkdir(parents=True)
    return mr.Paths(root=tmp_path, scans=tmp_path / "Scans", logs=tmp_path / "Logs", launchd=tmp_path / "launchd",
                    ibd_glob=str(tmp_path / "Data" / "ibd_universe_*.csv"), orders=tmp_path / "Data" / "orders",
                    equity_log=tmp_path / "Data" / "account" / "equity_log.csv",
                    adjustments=tmp_path / "Data" / "account" / "capital_adjustments.csv")


def write_log(paths, name, *lines):
    p = paths.logs / "scheduler" / name
    p.write_text("\n".join(lines) + "\n")
    return p


# ─── pipeline log ───

def test_pipeline_row_states(paths):
    assert mr.pipeline_row(paths, NOW).status == "WARN"                          # no log at all
    write_log(paths, "premarket_pipeline_2026-11-04_0920.log", "==> Running momentum scanner", "==> Done")
    assert mr.pipeline_row(paths, NOW).status == "OK"
    write_log(paths, "premarket_pipeline_2026-11-04_0930.log", "==> Done", "WARN: order plan failed (exit 1)")
    assert mr.pipeline_row(paths, NOW).status == "WARN"
    write_log(paths, "premarket_pipeline_2026-11-04_0940.log", "SCAN FAILED",
              "ERROR: momentum scanner failed 3 time(s)")
    row = mr.pipeline_row(paths, NOW)
    assert row.status == "FAIL" and "0940" in row.detail and "scanner failed" in row.detail
    write_log(paths, "premarket_pipeline_2026-11-04_0950.log", "==> Running momentum scanner")
    assert mr.pipeline_row(paths, NOW).status == "WARN"                          # no Done line


def test_pipeline_row_flags_no_run_today(paths):
    write_log(paths, "premarket_pipeline_2026-11-03_0920.log", "==> Done")
    row = mr.pipeline_row(paths, NOW)
    assert row.status == "WARN" and "no run today" in row.detail


# ─── scan / ibd ───

def test_scan_row_fresh_and_stale(paths):
    assert mr.scan_row(paths, NOW).status == "FAIL"
    (paths.scans / "scan_2026-11-03_0920_afterhours_COMPLETE.csv").write_text("x")
    assert mr.scan_row(paths, NOW).status == "WARN"                              # today's is due by 10:00
    assert mr.scan_row(paths, datetime(2026, 11, 4, 9, 0)).status == "OK"        # before 10: yesterday's is fine
    (paths.scans / "scan_2026-11-04_0920_afterhours_COMPLETE.csv").write_text("x")
    (paths.scans / "scan_2026-11-04_1030_watch_COMPLETE.csv").write_text("x")
    row = mr.scan_row(paths, NOW)
    assert row.status == "OK" and "watcher scan" in row.detail
    # Monday morning: Friday's scan is the newest one expected
    (paths.scans / "scan_2026-11-04_0920_afterhours_COMPLETE.csv").unlink()
    (paths.scans / "scan_2026-11-03_0920_afterhours_COMPLETE.csv").unlink()
    (paths.scans / "scan_2026-11-06_0920_afterhours_COMPLETE.csv").write_text("x")
    assert mr.scan_row(paths, datetime(2026, 11, 9, 8, 0)).status == "OK"


def test_ibd_row_age(paths):
    assert mr.ibd_row(paths, NOW).status == "FAIL"
    (paths.root / "Data" / "ibd_universe_2026-10-30.csv").write_text("x")
    assert mr.ibd_row(paths, NOW).status == "OK"
    (paths.root / "Data" / "ibd_universe_2026-10-30.csv").unlink()
    (paths.root / "Data" / "ibd_universe_2026-10-20.csv").write_text("x")
    row = mr.ibd_row(paths, NOW)
    assert row.status == "WARN" and "15 day" in row.detail


# ─── ledger ───

def test_ledger_row(paths):
    assert mr.ledger_row(paths).status == "OK"
    ev = {"trade_date": "2026-11-04", "symbol": "NVDA", "setup": "PIVOT_BREAKOUT", "role": "entry",
          "state": "intended", "order_ref": "BOT-ENTRY-20261104-NVDA", "order_id": 100}
    (paths.orders / "ledger.jsonl").write_text(json.dumps(ev) + "\n")
    row = mr.ledger_row(paths)
    assert row.status == "WARN" and "1 live bot order" in row.detail
    (paths.orders / "ledger.jsonl").write_text("{not json\n" + json.dumps(ev) + "\n")
    assert mr.ledger_row(paths).status == "FAIL"


# ─── launchd ───

PLIST = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict><key>Label</key><string>{label}</string>
<key>StandardErrorPath</key><string>{err}</string></dict></plist>"""


def add_plist(paths, label, err="/nonexistent.log"):
    (paths.launchd / f"{label}.plist").write_text(PLIST.format(label=label, err=err))


def runner(listing):
    calls = []

    def run(cmd, **kw):
        calls.append(cmd)
        return SimpleNamespace(stdout=listing, returncode=0)
    run.calls = calls
    return run


def test_launchd_row(paths):
    add_plist(paths, "com.example.trading-watcher")
    add_plist(paths, "com.example.trading.premarket")
    ok = runner("PID\tStatus\tLabel\n123\t-15\tcom.example.trading-watcher\n-\t0\tcom.example.trading.premarket\n")
    row = mr.launchd_row(paths, ok)
    assert row.status == "OK" and "trading-watcher running" in row.detail
    assert ok.calls == [["launchctl", "list"]]                       # list only: read-only
    failed = runner("123\t0\tcom.example.trading-watcher\n-\t1\tcom.example.trading.premarket\n")
    assert "last exit 1" in mr.launchd_row(paths, failed).detail
    missing = runner("123\t0\tcom.example.trading-watcher\n")
    row = mr.launchd_row(paths, missing)
    assert row.status == "WARN" and "NOT loaded: trading.premarket" in row.detail


# ─── disk / logs (item 3: report pipeline log size, delete nothing) ───

def test_disk_and_log_rows(paths):
    write_log(paths, "premarket_pipeline_2026-11-03_0920.log", "x" * 1000)
    write_log(paths, "premarket_pipeline_2026-11-04_0920.log", "x" * 1000)
    big = paths.logs / "watcher-launchd.err.log"
    with big.open("wb") as f:
        f.truncate(int((mr.LAUNCHD_LOG_MAX_MB + 1) * 1024 * 1024))   # sparse file
    add_plist(paths, "com.example.trading-watcher", err=str(big))
    gb = 1024 ** 3
    rows = {r.name: r for r in mr.disk_rows(paths, lambda p: SimpleNamespace(free=50 * gb))}
    assert rows["disk"].status == "OK"
    assert rows["logs"].status == "WARN" and "2 pipeline run log(s)" in rows["logs"].detail
    assert "watcher-launchd.err.log" in rows["logs"].detail
    assert len(list((paths.logs / "scheduler").glob("*.log"))) == 2 and big.exists()     # nothing deleted
    assert mr.disk_rows(paths, lambda p: SimpleNamespace(free=1 * gb))[0].status == "FAIL"
    assert mr.disk_rows(paths, lambda p: SimpleNamespace(free=5 * gb))[0].status == "WARN"


# ─── breakers from the equity log ───

def equity(paths, rows):
    lines = ["timestamp,account,net_liquidation"] + [f"{t},DUX1,{v}" for t, v in rows]
    paths.equity_log.write_text("\n".join(lines) + "\n")


def test_breakers_from_equity_log(paths):
    assert mr.breakers_row(None, "DUX1", NOW, paths).status == "WARN"            # no history
    days = ["2026-10-28", "2026-10-29", "2026-10-30", "2026-11-02", "2026-11-03"]
    equity(paths, [(f"{d}T09:20:00", 100000) for d in days[:3]])
    row = mr.breakers_row(None, "DUX1", NOW, paths)
    assert row.status == "WARN" and f"3 of {sp.MIN_BREAKER_HISTORY_SESSIONS} sessions" in row.detail
    equity(paths, [(f"{d}T09:20:00", 100000) for d in days] + [("2026-11-04T09:20:00", 100000)])
    row = mr.breakers_row(None, "DUX1", NOW, paths)
    assert row.status == "OK" and "entries allowed" in row.detail
    equity(paths, [(f"{d}T09:20:00", 100000) for d in days] + [("2026-11-04T09:20:00", 100000 * (1 - (sp.DAILY_LOSS_LIMIT_PCT + 0.5) / 100))])
    row = mr.breakers_row(None, "DUX1", NOW, paths)
    assert row.status == "FAIL" and "Daily loss" in row.detail


def test_breakers_from_snapshot():
    weekly = -sp.WEEKLY_LOSS_LIMIT_PCT - 0.5
    snap = {"breaker_history_sessions": sp.MIN_BREAKER_HISTORY_SESSIONS + 1, "breaker_history_start": "2026-10-27",
            "daily_pnl_pct": -0.2, "weekly_pnl_pct": weekly, "month_drawdown_pct": weekly}
    row = mr.breakers_row(snap, "DUX1", NOW, mr.Paths())
    assert row.status == "OK" and f"risk x{sp.WEEKLY_LOSS_RISK_MULT:g}" in row.detail


# ─── TWS (fake connector, read-only) ───

class FakeConn:
    instances = []

    def __init__(self, read_only=True, connect_ok=True, accounts=("DUX1",), farms=None, positions=(), orders=()):
        self.read_only = read_only
        self.connect_ok, self.accounts = connect_ok, accounts
        self.farm_status = farms if farms is not None else {"usfarm": True, "ushmds": True}
        self.server_link_ok = True
        self.positions, self.orders = list(positions), list(orders)
        self.connected_to = None
        self.disconnected = False
        FakeConn.instances.append(self)

    def connect_and_run(self, host, port, client_id):
        self.connected_to = (port, client_id)
        return self.connect_ok

    def fetch_managed_accounts(self, timeout=10):
        return list(self.accounts)

    def fetch_account_state(self, timeout=10):
        return {"positions": self.positions, "open_orders": self.orders,
                "account_values": {a: {"NetLiquidation": "100000"} for a in self.accounts}}

    def placeOrder(self, *a):
        raise AssertionError("the report must never place an order")

    def disconnect_and_stop(self):
        self.disconnected = True


def factory(**kw):
    made = []

    def f(read_only=True):
        c = FakeConn(read_only=read_only, **kw)
        made.append(c)
        return c
    f.made = made
    return f


@pytest.mark.parametrize("port", [7496, 4001])
def test_live_port_refused_before_connecting(port, paths):
    f = factory()
    rows, snap = mr.check_tws(port, "DUX1", f, sleep=lambda s: None, now=NOW, paths=paths)
    assert rows[0].status == "FAIL" and "LIVE" in rows[0].detail and f.made == [] and snap is None


def test_tws_down(paths):
    rows, snap = mr.check_tws(7497, "DUX1", factory(connect_ok=False), sleep=lambda s: None, now=NOW, paths=paths)
    assert rows[0].status == "FAIL" and snap is None


def test_tws_account_unprotected_and_no_writes(paths):
    equity(paths, [("2026-11-03T09:20:00", 100000)])
    before = paths.equity_log.read_text()
    pos = [{"account": "DUX1", "symbol": "NVDA", "sec_type": "STK", "shares": 10.0, "avg_cost": 100.0}]
    f = factory(positions=pos, farms={"usfarm": True, "ushmds": False})
    rows, snap = mr.check_tws(7497, "DUX1", f, sleep=lambda s: None, now=NOW, paths=paths)
    by = {r.name: r for r in rows}
    assert f.made[0].read_only is True and f.made[0].connected_to == (7497, 18) and f.made[0].disconnected
    assert by["tws"].status == "FAIL" and "ushmds" in by["tws"].detail
    assert by["account"].status == "WARN" and "NVDA 10 sh" in by["account"].detail
    assert snap["breaker_history_sessions"] == 1
    assert paths.equity_log.read_text() == before                # report never appends the equity log


def test_live_account_in_session_fails(paths):
    rows, snap = mr.check_tws(7497, None, factory(accounts=("DUX1", "U99")), sleep=lambda s: None, now=NOW,
                              paths=paths)
    assert rows[-1].name == "account" and rows[-1].status == "FAIL" and snap is None


# ─── report + one banner ───

def test_build_report_survives_a_broken_check(paths, monkeypatch):
    monkeypatch.setattr(mr, "ibd_row", lambda p, n: 1 / 0)
    rows = mr.build_report(7497, "DUX1", use_tws=False, paths=paths, now=NOW, launchctl_runner=runner(""),
                           disk_usage=lambda p: SimpleNamespace(free=50 * 1024 ** 3))
    names = [r.name for r in rows]
    assert names[:3] == ["tws", "breakers", "pipeline"] and "ibd" in names and "launchd" in names
    assert {r.name: r for r in rows}["ibd"].status == "FAIL"
    assert "FAIL" in mr.format_table(rows)


def test_summary_message():
    ok = [mr.Row("disk", "OK", "")]
    assert mr.summary_message(ok) is None
    level, msg = mr.summary_message(ok + [mr.Row("scan", "WARN", ""), mr.Row("tws", "FAIL", "")])
    assert level == "error" and msg.startswith("1 FAIL, 1 WARN") and "scan WARN" in msg


def test_main_posts_one_banner(monkeypatch, capsys):
    rows = [mr.Row("tws", "FAIL", "down"), mr.Row("scan", "WARN", "old"), mr.Row("disk", "OK", "fine")]
    monkeypatch.setattr(mr, "build_report", lambda *a, **k: rows)
    posted = []
    monkeypatch.setattr(mr.notifier, "notify", lambda *a, **k: posted.append(a))
    assert mr.main(["--no-tws"]) == 1
    assert len(posted) == 1 and posted[0][0] == "Morning report" and posted[0][2] == "error"
    assert "FAIL  tws" in capsys.readouterr().out
    posted.clear()
    assert mr.main(["--no-tws", "--no-notify"]) == 1 and posted == []


def test_ledger_row_ignores_trade_status_events(paths):
    """A closed trade's 'trade' event (state closed, not in TERMINAL) is not a live bot order."""
    base = {"trade_date": "2026-11-04", "symbol": "NVDA", "setup": "PIVOT_BREAKOUT"}
    events = [{**base, "role": "entry", "state": "filled", "order_ref": "BOT-ENTRY-20261104-NVDA", "order_id": 100,
               "action": "BUY", "quantity": 10, "filled_qty": 10},
              {**base, "role": "stop", "state": "filled", "order_ref": "BOT-STOP-20261104-NVDA", "order_id": 101,
               "action": "SELL", "quantity": 10, "filled_qty": 10},
              {**base, "role": "trade", "state": "closed", "order_ref": "BOT-TRADE-20261104-NVDA"}]
    (paths.orders / "ledger.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    row = mr.ledger_row(paths)
    assert row.status == "OK" and row.detail.startswith("0 live bot order(s), 0 open trade(s)"), row.detail
