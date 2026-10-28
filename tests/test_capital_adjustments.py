"""Circuit-breaker inputs net of recorded cash moves. Thresholds and risk_manager are unchanged:
a deposit must never mask a loss and an unrecorded drop must never be excused."""

import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest

import account_snapshot as acct
import risk_manager as rm
import strategy_params as sp
from account_snapshot import CapitalAdjustment as Adj

ROOT = Path(__file__).resolve().parents[1]


def t(day, hour=9):
    return datetime(2026, 10, day, hour, 20)


def breaker(ch):
    return rm.circuit_breaker(ch["daily_pnl_pct"], ch["weekly_pnl_pct"], ch["month_drawdown_pct"])


def test_recorded_withdrawal_is_not_a_loss():
    history = [(t(22), 100_000), (t(23), 100_000)]
    ch = acct.equity_changes(history, 80_000, t(27), [Adj(t(24, 12), 'withdrawal', -20_000)])
    assert ch["daily_pnl_pct"] == pytest.approx(0) and ch["weekly_pnl_pct"] == pytest.approx(0)
    assert ch["month_drawdown_pct"] == pytest.approx(0)
    assert breaker(ch)[0] is None


def test_unrecorded_withdrawal_still_halts():
    ch = acct.equity_changes([(t(22), 100_000), (t(23), 100_000)], 80_000, t(27))
    assert ch["month_drawdown_pct"] == pytest.approx(-20.0)
    assert breaker(ch)[0]


def test_recorded_deposit_cannot_mask_a_losing_day():
    """+10k deposit today on 100k: the day is measured on 110k, so a loss past the daily limit still fires
    although the account ends above yesterday's 100k."""
    loss = sp.DAILY_LOSS_LIMIT_PCT + 0.3
    ch = acct.equity_changes([(t(26), 100_000)], 110_000 * (1 - loss / 100), t(27, 16),
                             [Adj(t(27, 10), 'deposit', 10_000)])
    assert ch["daily_pnl_pct"] == pytest.approx(-loss, abs=1e-3)
    assert breaker(ch)[0]


def test_reset_drops_every_earlier_point():
    ch = acct.equity_changes([(t(22), 30_600), (t(23), 30_500)], 25_000, t(27), [Adj(t(26), 'reset')])
    assert ch == {"daily_pnl_pct": None, "weekly_pnl_pct": None, "month_drawdown_pct": None}
    ch = acct.equity_changes([(t(22), 30_600), (t(26, 10), 25_000)], 24_500, t(27), [Adj(t(26), 'reset')])
    assert ch["month_drawdown_pct"] == pytest.approx(-2.0)           # measured from the reset only


def test_future_adjustments_are_ignored():
    ch = acct.equity_changes([(t(26), 100_000)], 80_000, t(27), [Adj(t(28), 'withdrawal', -20_000)])
    assert ch["daily_pnl_pct"] == pytest.approx(-20.0)


def test_flow_moves_only_points_before_it():
    points, used = acct.apply_adjustments([(t(22), 100_000), (t(26), 90_000)], t(27),
                                          [Adj(t(23), 'withdrawal', -10_000)])
    assert points == [(t(22), 90_000), (t(26), 90_000)] and len(used) == 1


def test_upward_fallback_still_restarts_without_a_record():
    up = acct.equity_changes([(t(23), 30_600), (t(26), 100_000)], 99_500, t(27))
    assert up["daily_pnl_pct"] == pytest.approx(-0.5)


def test_adjustments_csv_roundtrip_and_validation(tmp_path):
    path = tmp_path / "capital_adjustments.csv"
    now = t(27, 12)
    acct.record_adjustment("dux1", "withdrawal", -5000, at=t(27, 10), note="test", path=path, now=now)
    acct.record_adjustment("DUX2", "reset", at=t(27, 11), path=path, now=now)
    assert acct.read_adjustments("DUX1", path) == [Adj(t(27, 10), 'withdrawal', -5000.0, 'test')]
    assert acct.read_adjustments("DUX2", path)[0].amount is None
    with pytest.raises(ValueError, match="paper"):
        acct.record_adjustment("U123", "deposit", 100, path=path, now=now)
    with pytest.raises(ValueError, match="future"):
        acct.record_adjustment("DUX1", "deposit", 100, at=t(28), path=path, now=now)
    with pytest.raises(ValueError, match="negative"):
        acct.record_adjustment("DUX1", "withdrawal", 100, path=path, now=now)


def test_adjustments_for_another_account_are_not_read(tmp_path):
    path = tmp_path / "a.csv"
    acct.record_adjustment("DUX2", "withdrawal", -20_000, at=t(24), path=path, now=t(27))
    ch = acct.equity_changes([(t(23), 100_000)], 80_000, t(27), acct.read_adjustments("DUX1", path))
    assert ch["month_drawdown_pct"] == pytest.approx(-20.0)


# ─── Implied-flow detection (warning only: the engine gate is the owner's call) ───
def prev(net_liq, daily, ts=None, positions=0):
    return {'timestamp': ts or t(27, 9), 'net_liquidation': net_liq, 'daily_pnl': daily, 'positions': positions}


def test_net_liq_move_not_explained_by_pnl_is_flagged():
    flow = acct.implied_capital_flow(prev(100_000, 0.0), 105_000, 0.0, t(27, 15))
    assert flow['checked'] and flow['flagged'] and flow['amount'] == 5000


def test_move_explained_by_pnl_is_not_flagged():
    flow = acct.implied_capital_flow(prev(100_000, 0.0), 105_000, 5000.0, t(27, 15))
    assert flow['checked'] and not flow['flagged']


def test_recorded_adjustment_covers_the_move():
    flow = acct.implied_capital_flow(prev(100_000, 0.0), 105_000, 0.0, t(27, 15),
                                     [Adj(t(27, 12), 'deposit', 5000)])
    assert not flow['flagged']


@pytest.mark.parametrize("p, daily", [(None, 0.0), (prev(100_000, None), 0.0), (prev(100_000, 0.0), None),
                                      (prev(100_000, 0.0, ts=t(26, 15)), 0.0)])
def test_missing_pnl_or_another_day_is_not_checked(p, daily):
    flow = acct.implied_capital_flow(p, 105_000, daily, t(27, 15))
    assert flow['checked'] is False and flow['reason']


def test_snapshot_carries_the_flag_and_applied_adjustments():
    state = {"positions": [], "open_orders": [], "account_values": {"DUX1": {"NetLiquidation": "105000",
                                                                            "TotalCashValue": "105000"}},
             "pnl": {"DUX1": {"daily": 0.0, "unrealized": 0.0, "realized": 0.0}}}
    snap = acct.build_snapshot(state, now=t(27, 15), equity_history=[(t(26), 100_000)],
                               adjustments=[Adj(t(26, 12), 'deposit', 1000)], prev_pnl=prev(100_000, 0.0))
    assert snap['unexplained_capital_change']['amount'] == 5000
    assert snap['adjustments_applied'][0]['kind'] == 'deposit' and snap['total_cash'] == 105_000


def test_positions_wiped_with_a_jump_is_a_probable_reset():
    state = {"positions": [], "open_orders": [], "account_values": {"DUX1": {"NetLiquidation": "100000"}}}
    snap = acct.build_snapshot(state, now=t(27, 9), equity_history=[],
                               prev_pnl=prev(30_500, None, ts=t(26, 15), positions=8))
    assert "probable paper reset" in snap['unexplained_capital_change']['reason']


def test_pnl_callback_ignores_unset_values():
    from ibkr_connector import IBKRConnector
    conn = IBKRConnector()
    conn._pnl_req[5] = "DUX1"
    conn.pnl(5, 12.5, 1.7976931348623157e308, 0.0)
    assert conn.pnl_values["DUX1"] == {'daily': 12.5, 'unrealized': None, 'realized': 0.0}


def test_record_cli_writes_only_the_csv_and_never_connects(tmp_path, monkeypatch):
    import ibkr_connector
    monkeypatch.setattr(acct, "ADJUSTMENTS_CSV", tmp_path / "adj.csv")
    monkeypatch.setattr(acct, "EQUITY_LOG", tmp_path / "equity_log.csv")
    monkeypatch.setattr(ibkr_connector.IBKRConnector, "connect_and_run",
                        lambda *a, **k: pytest.fail("must not connect"))
    acct.append_equity_log({"timestamp": t(23).isoformat(), "account": "DUX1", "net_liquidation": 100_000},
                           tmp_path / "equity_log.csv")
    acct.append_equity_log({"timestamp": t(26).isoformat(), "account": "DUX1", "net_liquidation": 80_000},
                           tmp_path / "equity_log.csv")
    monkeypatch.setattr(sys, "argv", ["account_snapshot.py", "--record-adjustment", "-20000",
                                      "--account", "DUX1", "--at", t(24).isoformat()])
    # read_equity_log's default path is bound at import: point it at the temp log
    real_read = acct.read_equity_log
    monkeypatch.setattr(acct, "read_equity_log", lambda a: real_read(a, tmp_path / "equity_log.csv"))
    assert acct.main() == 0
    assert (tmp_path / "adj.csv").read_text().count("withdrawal") == 1
    assert not (tmp_path / "snapshots").exists()


# ─── Review H2: an adjustment recorded late must not double-count the move ───
def _cli(monkeypatch, tmp_path, argv, capsys=None):
    monkeypatch.setattr(acct, "ADJUSTMENTS_CSV", tmp_path / "adj.csv")
    real_read = acct.read_equity_log
    monkeypatch.setattr(acct, "read_equity_log",
                        lambda a, path=None: real_read(a, path or tmp_path / "equity_log.csv"))
    monkeypatch.setattr(sys, "argv", ["account_snapshot.py", *argv])
    return acct.main()


def _log(tmp_path, *points):
    for ts, eq in points:
        acct.append_equity_log({"timestamp": ts.isoformat(), "account": "DUX1", "net_liquidation": eq},
                               tmp_path / "equity_log.csv")


def test_record_cli_refuses_without_at(tmp_path, monkeypatch, capsys):
    _log(tmp_path, (t(19, 16), 100_000), (t(20, 16), 100_000))
    assert _cli(monkeypatch, tmp_path, ["--record-adjustment", "-20000", "--account", "DUX1"]) == 1
    out = capsys.readouterr().out
    assert "--at is required" in out and "equity log: 2026-10-20 16:20" in out
    assert not (tmp_path / "adj.csv").exists()


def test_late_recorded_withdrawal_with_true_time_keeps_a_losing_day_a_loss(tmp_path, monkeypatch):
    """Mon/Tue 100k, withdraw 20k Wed 10:00, Wed 16:00 logs 80k; recorded Thu with the real time.
    Thu 72k is a real -10% day and must read as -10%, not +20%."""
    _log(tmp_path, (t(19, 16), 100_000), (t(20, 16), 100_000), (t(21, 16), 80_000))
    assert _cli(monkeypatch, tmp_path, ["--record-adjustment", "-20000", "--account", "DUX1",
                                        "--at", datetime(2026, 10, 21, 10, 0).isoformat()]) == 0
    history = acct.read_equity_log("DUX1", tmp_path / "equity_log.csv")
    adjustments = acct.read_adjustments("DUX1", tmp_path / "adj.csv")
    ch = acct.equity_changes(history, 72_000, t(22, 16), adjustments)
    assert ch["daily_pnl_pct"] == pytest.approx(-10)
    assert ch["month_drawdown_pct"] == pytest.approx(-10)


def test_record_cli_warns_when_snapshots_already_contain_the_move(tmp_path, monkeypatch, capsys):
    _log(tmp_path, (t(19, 16), 100_000), (t(21, 16), 80_000))
    assert _cli(monkeypatch, tmp_path, ["--record-adjustment", "-20000", "--account", "DUX1",
                                        "--at", "now"]) == 0
    assert "already including" not in capsys.readouterr().out      # 'now' is after every entry
    assert _cli(monkeypatch, tmp_path, ["--record-adjustment", "-5000", "--account", "DUX1",
                                        "--at", t(20, 12).isoformat()]) == 0
    out = capsys.readouterr().out
    assert "1 equity-log entry at or after --at" in out and "2026-10-21 16:20" in out
