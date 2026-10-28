"""The one sizing rule and the loss circuit breakers (risk_manager), and how the engine applies them."""

from datetime import datetime

import pandas as pd
import pytest

import account_snapshot as acct
import execution_engine as ee
import risk_manager as rm
import strategy_params as sp

RISK, CAP = sp.RISK_PER_TRADE_PCT, sp.MAX_POSITION_PCT


# ─── size_position ───
def test_position_is_risk_over_stop():
    stop = RISK / CAP * 100 * 2                      # half the cap
    pos, risk = rm.size_position(RISK, stop, CAP)
    assert pos == pytest.approx(CAP / 2) and risk == pytest.approx(RISK)


def test_cap_shrinks_position_and_actual_risk():
    stop = RISK / CAP * 100 / 2                      # uncapped would be twice the cap
    pos, risk = rm.size_position(RISK, stop, CAP)
    assert pos == pytest.approx(CAP) and risk == pytest.approx(RISK / 2)


def test_wider_stop_means_smaller_position_same_risk():
    stop = RISK / CAP * 100 * 2
    tight, wide = rm.size_position(RISK, stop, CAP), rm.size_position(RISK, stop * 2, CAP)
    assert wide[0] == pytest.approx(tight[0] / 2) and wide[1] == tight[1]


@pytest.mark.parametrize("stop", [None, 0.0, -1.0])
def test_no_stop_no_size(stop):
    assert rm.size_position(RISK, stop, CAP) == (0.0, 0.0)


# ─── circuit_breaker ───
def test_daily_loss_halts():
    halt, mult, _ = rm.circuit_breaker(-sp.DAILY_LOSS_LIMIT_PCT - 0.1, 0.0, 0.0)
    assert halt and "Daily loss" in halt and mult == 0
    assert rm.circuit_breaker(-sp.DAILY_LOSS_LIMIT_PCT + 0.1, 0.0, 0.0)[0] is None


def test_monthly_drawdown_halts():
    halt, _, _ = rm.circuit_breaker(0.0, 0.0, -sp.MONTHLY_DRAWDOWN_LIMIT_PCT - 0.5)
    assert halt and "Monthly drawdown" in halt


def test_weekly_loss_cuts_risk():
    weekly = -sp.WEEKLY_LOSS_LIMIT_PCT - 0.2
    halt, mult, note = rm.circuit_breaker(-0.5, weekly, weekly - 0.8)
    assert halt is None and mult == sp.WEEKLY_LOSS_RISK_MULT and "Weekly loss" in note


def test_unknown_equity_change_never_trips():
    assert rm.circuit_breaker(None, None, None) == (None, 1.0, "")


# ─── equity history (account_snapshot.equity_changes) ───
def t(day, hour=9):
    return datetime(2026, 10, day, hour, 20)


def test_equity_changes_daily_weekly_month():
    # Thu 1 Oct 100k ... Fri 23 Oct 104k (month peak) ... Mon 26 Oct 101k; now Tue 27 Oct 99k
    history = [(t(1), 100_000), (t(23), 104_000), (t(26), 101_000)]
    ch = acct.equity_changes(history, 99_000, t(27))
    assert ch["daily_pnl_pct"] == pytest.approx(-1.9802, abs=1e-3)    # vs Mon 26
    assert ch["weekly_pnl_pct"] == pytest.approx(-4.8077, abs=1e-3)   # vs Fri 23 (before Monday)
    assert ch["month_drawdown_pct"] == pytest.approx(-4.8077, abs=1e-3)


def test_no_history_means_unknown():
    assert acct.equity_changes([], 100_000, t(27)) == {
        "daily_pnl_pct": None, "weekly_pnl_pct": None, "month_drawdown_pct": None}


def test_account_reset_up_restarts_history():
    """Paper reset 30.6k -> 100k is a capital change, not a +227% day."""
    up = acct.equity_changes([(t(23), 30_600), (t(26), 100_000)], 99_500, t(27))
    assert up["daily_pnl_pct"] == pytest.approx(-0.5) and up["month_drawdown_pct"] == pytest.approx(-0.5)


def test_big_drop_is_never_excused_as_a_withdrawal():
    """Safety audit: a 26% real loss must halt, not reset the history and switch the breakers off."""
    ch = acct.equity_changes([(t(22), 100_000), (t(23), 96_000), (t(26), 95_000)], 70_000, t(27))
    assert ch["daily_pnl_pct"] == pytest.approx(-26.3158, abs=1e-3)
    assert ch["month_drawdown_pct"] == pytest.approx(-30.0)
    assert rm.circuit_breaker(ch["daily_pnl_pct"], ch["weekly_pnl_pct"], ch["month_drawdown_pct"])[0]


def test_equity_log_roundtrip(tmp_path):
    log = tmp_path / "equity_log.csv"
    acct.append_equity_log({"timestamp": "2026-10-26T09:20:00", "account": "DUX1", "net_liquidation": 100000.0}, log)
    acct.append_equity_log({"timestamp": "2026-10-26T09:21:00", "account": "DUX2", "net_liquidation": 5.0}, log)
    assert acct.read_equity_log("DUX1", log) == [(datetime(2026, 10, 26, 9, 20), 100000.0)]


# ─── Engine: one sizing rule, breakers, max positions ───
def decide(rows, snapshot=None, policy=None):
    df = pd.DataFrame(rows)
    if snapshot is not None:
        df = ee.apply_account_snapshot(df, snapshot)
    return ee.build_execution_decisions(df, mode="afterhours", orb_window=10, policy=policy)


def row(symbol="A", **kw):
    return {"Symbol": symbol, "Tier": "TIER_BUY", "Stop_Loss_%": 2.0, "Price": 100.0, **kw}


def snap(**kw):
    base = {"account": "DUX1", "net_liquidation": 100_000, "total_open_risk_pct": 0.0,
            "group_open_risk_pct": {}, "held_symbols": [], "unprotected_symbols": []}
    return {**base, **kw}


def test_engine_position_and_risk_come_from_size_position():
    out = decide([row()]).iloc[0]
    pos, risk = rm.size_position(ee.ExecutionPolicy().risk_per_trade_pct, 2.0, ee.ExecutionPolicy().max_position_pct)
    assert out["Position_Pct"] == pytest.approx(pos) and out["Risk_Pct_Target"] == pytest.approx(risk)


def test_engine_carries_exit_plan_and_take_profit():
    out = decide([row(Trail_MA="sma10")]).iloc[0]
    assert out["Take_Profit_Price"] == pytest.approx(100.0 * (1 + sp.TRIM_R * 2.0 / 100))   # +TRIM_R R on a 2% stop
    assert out["Trail_MA"] == "sma10" and f"{sp.TRIM_FRACTION:.0%}" in out["Exit_Trim_Rule"]


def test_daily_loss_breaker_skips_every_entry():
    loss = -sp.DAILY_LOSS_LIMIT_PCT - 0.5
    out = decide([row()], snap(daily_pnl_pct=loss, weekly_pnl_pct=loss, month_drawdown_pct=loss)).iloc[0]
    assert out["Decision"] == ee.SKIP and "Daily loss" in out["Reason"]


def test_weekly_loss_breaker_cuts_risk():
    weekly = -sp.WEEKLY_LOSS_LIMIT_PCT - 0.5
    stop = RISK / CAP * 100                          # full risk exactly fills the cap: no capping
    out = decide([row(**{"Stop_Loss_%": stop})],
                 snap(daily_pnl_pct=0.0, weekly_pnl_pct=weekly, month_drawdown_pct=weekly)).iloc[0]
    assert out["Decision"] == ee.BUY
    assert out["Risk_Pct_Target"] == pytest.approx(ee.ExecutionPolicy().risk_per_trade_pct * sp.WEEKLY_LOSS_RISK_MULT)


def test_missing_equity_history_is_reported_not_checked():
    out = decide([row()], snap()).iloc[0]
    assert out["Decision"] == ee.BUY and "loss circuit breakers" in out["Reason"]


def test_max_open_positions_limits_todays_entries():
    held = [f"H{i}" for i in range(sp.MAX_OPEN_POSITIONS - 1)]
    out = decide([row("A"), row("B")], snap(held_symbols=held))
    assert list(out["Decision"]) == [ee.BUY, ee.SKIP]
    assert "Max open positions" in out.iloc[1]["Reason"]


def test_executor_sizes_with_the_same_rule():
    import ibkr_paper_executor as ex
    decisions = decide([row(**{"Stop_Loss_%": 2.0})])
    intents, _ = ex._build_buy_intents(decisions)
    assert intents.iloc[0]["Position_Pct_Of_Equity"] == pytest.approx(decisions.iloc[0]["Position_Pct"])


def test_max_open_positions_counts_manual_review_rows():
    """Reviewer: approving a MANUAL_REVIEW row opens a position too, so it needs a free slot."""
    held = [f"H{i}" for i in range(sp.MAX_OPEN_POSITIONS)]
    out = decide([row("A", **{"Stop_Loss_%": 5.0}), row("B")], snap(held_symbols=held))
    assert list(out["Decision"]) == [ee.SKIP, ee.SKIP]
    assert all("Max open positions" in r for r in out["Reason"])


# ─── breaker history (owner decision 2026-10-26 (b)) ───
def test_breaker_history_counts_distinct_dates_excluding_today():
    history = [(t(20), 100_000), (t(21), 100_100), (t(21, 15), 100_200), (t(22), 99_900), (t(27, 8), 99_800)]
    h = acct.breaker_history(history, 99_800, t(27))
    assert h == {"breaker_history_sessions": 3, "breaker_history_start": "2026-10-20"}


def test_breaker_history_restarts_after_a_recorded_reset():
    from account_snapshot import CapitalAdjustment as Adj
    history = [(t(20), 30_000), (t(21), 30_100), (t(26, 10), 100_000), (t(27), 100_100)]
    h = acct.breaker_history(history, 100_000, t(28), [Adj(t(26), 'reset')])
    assert h == {"breaker_history_sessions": 2, "breaker_history_start": "2026-10-26"}


def test_breaker_history_restarts_after_an_unrecorded_jump():
    history = [(t(20), 30_000), (t(21), 30_100), (t(26), 100_000), (t(27), 100_100)]
    assert acct.breaker_history(history, 100_000, t(28))["breaker_history_sessions"] == 2
    # the jump is today itself: no usable history yet
    assert acct.breaker_history([(t(26), 30_000)], 100_000, t(27))["breaker_history_sessions"] == 0


def test_snapshot_carries_breaker_history():
    st = {"positions": [], "open_orders": [], "account_values": {"DUX1": {"NetLiquidation": "100000"}}}
    s = acct.build_snapshot(st, now=t(27), equity_history=[(t(26), 100_000)])
    assert s["breaker_history_sessions"] == 1
    assert acct.build_snapshot(st, now=t(27))["breaker_history_sessions"] is None
