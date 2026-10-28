"""ibkr_alert_setter safety: preview by default, paper-only when staging orders."""

import sys

import pytest

import ibkr_alert_setter as alerts


class FakeConnector:
    def __init__(self, accounts=("DUXXXXXXX",)):
        self.accounts = list(accounts)
        self.connected = False

    def fetch_managed_accounts(self, timeout=8):
        return self.accounts

    def connect_and_run(self, host, port, client_id):
        self.connected = True
        return True

    def disconnect_and_stop(self):
        pass


def test_live_port_is_refused():
    with pytest.raises(SystemExit):
        alerts.confirm_paper_session(FakeConnector(), port=7496)


def test_live_account_is_refused():
    with pytest.raises(SystemExit):
        alerts.confirm_paper_session(FakeConnector(accounts=["DUXXXXXXX", "U7654321"]), port=7497)


def test_no_accounts_is_refused():
    with pytest.raises(SystemExit):
        alerts.confirm_paper_session(FakeConnector(accounts=[]), port=7497)


def test_paper_session_passes():
    alerts.confirm_paper_session(FakeConnector(), port=7497)  # no exit


def test_preview_is_the_default_and_never_connects(monkeypatch):
    fake = FakeConnector()
    monkeypatch.setattr(alerts, "IBKRConnector", lambda **kw: fake)
    monkeypatch.setattr(alerts.AlertManager, "set_alerts_from_scan", lambda self, tiers, dry_run: None)
    monkeypatch.setattr(sys, "argv", ["ibkr_alert_setter.py"])
    alerts.main()
    assert fake.connected is False


# ─── Batch 2: alerts come only from engine decisions on a fresh scan ───
from datetime import datetime, timedelta

import pandas as pd

import execution_engine as ee

NOW = datetime.now()


def decisions_file(tmp_path, scan_age_h=1.0, **row):
    stamp = (NOW - timedelta(hours=scan_age_h)).isoformat()   # the engine ages it on the real clock
    base = {"Symbol": "MU", "Tier": "TIER_BUY", "Stop_Loss_%": 3.0, "Price": 98.0, "Scan_Timestamp": stamp,
            "Has_Real_VCP": True, "VCP_Pivot": 100.0, "VCP_Dist_From_Pivot_%": -2.0, "Entry_Setup": "VCP_PIVOT"}
    base.update(row)
    out = ee.build_execution_decisions(pd.DataFrame([base]), mode="afterhours", orb_window=10)
    path = tmp_path / "scan_2026-10-26_0920_execution_decisions.csv"
    out.to_csv(path, index=False)
    return path




def test_engine_buy_row_becomes_an_alert_candidate(tmp_path):
    path = decisions_file(tmp_path)
    df = alerts.AlertManager(None).load_alert_candidates(path, alerts.DEFAULT_TIERS, now=NOW)
    assert list(df["Symbol"]) == ["MU"]


def test_engine_skip_never_gets_an_alert(tmp_path):
    """Regime-blocked (or any SKIP) rows used to be staged straight from the scan CSV."""
    path = decisions_file(tmp_path, Regime_Blocked=True)
    assert alerts.AlertManager(None).load_alert_candidates(path, alerts.DEFAULT_TIERS, now=NOW).empty


def test_stale_scan_gets_no_alerts(tmp_path):
    path = decisions_file(tmp_path, scan_age_h=30)
    assert alerts.AlertManager(None).load_alert_candidates(path, alerts.DEFAULT_TIERS, now=NOW).empty


def test_raw_scan_csv_is_refused(tmp_path):
    path = tmp_path / "scan_x_COMPLETE.csv"
    pd.DataFrame([{"Symbol": "MU", "Tier": "TIER_BUY", "VCP_Pivot": 100.0}]).to_csv(path, index=False)
    assert alerts.AlertManager(None).load_alert_candidates(path, alerts.DEFAULT_TIERS, now=NOW).empty


def test_setter_looks_only_for_decisions_files(tmp_path, monkeypatch):
    monkeypatch.setattr(alerts, "OUTPUT_DIR", str(tmp_path))
    (tmp_path / "scan_2026-10-26_watch_COMPLETE.csv").write_text("Symbol\nMU\n")
    with pytest.raises(FileNotFoundError):
        alerts.AlertManager(None).find_latest_decisions_csv()


def test_one_undated_row_refuses_the_file(tmp_path):
    """Audit: .max() skipped NaN, so an undated row rode along with a dated one."""
    path = decisions_file(tmp_path)
    df = pd.read_csv(path)
    df = pd.concat([df, df.assign(Symbol="WDC", Scan_Timestamp=None)])
    df.to_csv(path, index=False)
    assert alerts.AlertManager(None).load_alert_candidates(path, alerts.DEFAULT_TIERS, now=NOW).empty


def test_future_dated_scan_is_refused(tmp_path):
    path = decisions_file(tmp_path, scan_age_h=-48)
    assert alerts.AlertManager(None).load_alert_candidates(path, alerts.DEFAULT_TIERS, now=NOW).empty


def test_missing_vcp_columns_refuse_the_file(tmp_path):
    path = decisions_file(tmp_path)
    pd.read_csv(path).drop(columns=["Has_Real_VCP"]).to_csv(path, index=False)
    assert alerts.AlertManager(None).load_alert_candidates(path, alerts.DEFAULT_TIERS, now=NOW).empty


def test_engine_skips_undated_and_future_rows():
    rows = [{"Symbol": "A", "Tier": "TIER_BUY", "Stop_Loss_%": 3.0, "Scan_Timestamp": None},
            {"Symbol": "B", "Tier": "TIER_BUY", "Stop_Loss_%": 3.0,
             "Scan_Timestamp": (datetime.now() + timedelta(hours=48)).isoformat()},
            {"Symbol": "C", "Tier": "TIER_BUY", "Stop_Loss_%": 3.0, "Scan_Timestamp": datetime.now().isoformat()}]
    out = ee.build_execution_decisions(pd.DataFrame(rows), mode="afterhours", orb_window=10)
    assert list(out["Decision"]) == [ee.SKIP, ee.SKIP, ee.BUY]
