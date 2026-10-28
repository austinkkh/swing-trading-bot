"""Read-only account snapshot: capital at risk from stop orders, and how the engine uses it."""

from datetime import datetime, timedelta

import pandas as pd
import pytest

import account_snapshot as acct
import execution_engine as ee
import strategy_params as sp

FALLBACK = acct.UNPROTECTED_FALLBACK_RISK_PCT / 100
P = ee.ExecutionPolicy()
RISK = P.risk_per_trade_pct


def pos(symbol, shares, cost, account="DUX1"):
    return {"account": account, "symbol": symbol, "sec_type": "STK", "shares": shares, "avg_cost": cost}


def stop(symbol, qty, px, action="SELL", kind="STP", account="DUX1", perm=None):
    return {"perm_id": perm, "account": account, "symbol": symbol, "action": action, "order_type": kind,
            "quantity": qty, "aux_price": px, "lmt_price": 0.0, "trail_stop_price": 1.7976931348623157e308}


def state(positions, orders, net_liq="100000", accounts=("DUX1",)):
    return {"positions": positions, "open_orders": orders,
            "account_values": {a: {"NetLiquidation": net_liq} for a in accounts}}


# ─── Risk per position ───
def test_risk_is_entry_minus_stop_times_shares():
    r = acct.position_risks([pos("MU", 100, 50.0)], [stop("MU", 100, 47.5)])[0]
    assert r["risk_dollars"] == pytest.approx(250)
    assert r["unprotected_shares"] == 0


def test_stop_at_or_above_entry_frees_the_risk():
    r = acct.position_risks([pos("MU", 100, 50.0)], [stop("MU", 100, 52.0)])[0]
    assert r["risk_dollars"] == 0


def test_missing_or_partial_stop_is_unprotected():
    none = acct.position_risks([pos("MU", 100, 50.0)], [])[0]
    part = acct.position_risks([pos("MU", 100, 50.0)], [stop("MU", 60, 48.0)])[0]
    assert none["unprotected_shares"] == 100
    assert part["unprotected_shares"] == 40
    # stop risk 60 x 2 = 120, plus the fallback % of cost on the 40 uncovered shares
    assert part["risk_dollars"] == pytest.approx(120 + 40 * 50 * FALLBACK)


def test_unprotected_shares_charged_fallback_risk():
    """Owner decision 2026-10-25: no stop never means zero risk; charge a fallback % of the position's cost."""
    none = acct.position_risks([pos("MU", 100, 50.0)], [])[0]
    assert none["fallback_risk_dollars"] == pytest.approx(5_000 * FALLBACK) and FALLBACK > 0
    assert none["risk_dollars"] == pytest.approx(5_000 * FALLBACK)
    short = acct.position_risks([pos("SPY", -10, 500.0)], [])[0]
    assert short["risk_dollars"] == pytest.approx(5_000 * FALLBACK)
    covered = acct.position_risks([pos("MU", 100, 50.0)], [stop("MU", 100, 47.5)])[0]
    assert covered["fallback_risk_dollars"] == 0


def test_unprotected_positions_count_toward_total_open_risk():
    snap = acct.build_snapshot(state([pos("MU", 100, 50.0), pos("AAPL", 10, 200.0)], []))
    # (5,000 + 2,000) x fallback % on a 100k account
    assert snap["total_open_risk_pct"] == pytest.approx(7_000 * FALLBACK / 100_000 * 100)
    assert snap["unprotected_symbols"] == ["AAPL", "MU"]


def test_worst_stops_count_when_over_covered():
    r = acct.position_risks([pos("MU", 100, 50.0)], [stop("MU", 100, 49.0, perm=1), stop("MU", 100, 45.0, perm=2)])[0]
    assert r["risk_dollars"] == pytest.approx(500)


def test_repeated_callbacks_for_one_order_count_once():
    r = acct.position_risks([pos("MU", 100, 50.0)], [stop("MU", 50, 48.0, perm=7)] * 2)[0]
    assert r["protected_shares"] == 50


def test_buy_side_or_other_symbol_orders_do_not_protect_a_long():
    orders = [stop("MU", 100, 48.0, action="BUY"), stop("WDC", 100, 48.0), {**stop("MU", 100, 48.0), "order_type": "LMT"}]
    assert acct.position_risks([pos("MU", 100, 50.0)], orders)[0]["unprotected_shares"] == 100


def test_trailing_stop_uses_trail_stop_price():
    trail = {**stop("MU", 100, 0.0, kind="TRAIL"), "trail_stop_price": 48.0}
    assert acct.position_risks([pos("MU", 100, 50.0)], [trail])[0]["risk_dollars"] == pytest.approx(200)


def test_short_uses_buy_stop_above():
    r = acct.position_risks([pos("XYZ", -100, 20.0)], [stop("XYZ", 100, 21.0, action="BUY")])[0]
    assert r["risk_dollars"] == pytest.approx(100) and r["unprotected_shares"] == 0


# ─── Snapshot ───
def test_snapshot_totals_and_groups():
    groups = {"MU": 3, "WDC": 3, "RNG": 40}
    s = acct.build_snapshot(state([pos("MU", 100, 50.0), pos("WDC", 100, 30.0), pos("RNG", 10, 40.0)],
                                  [stop("MU", 100, 48.5), stop("WDC", 100, 29.0), stop("RNG", 10, 41.0)]),
                            group_of=groups.get)
    assert s["total_open_risk_pct"] == pytest.approx(0.25)   # $150 + $100 + $0 on $100k
    assert s["group_open_risk_pct"] == {"3": pytest.approx(0.25), "40": 0.0}
    assert s["held_symbols"] == ["MU", "RNG", "WDC"]


def test_live_account_in_session_is_refused():
    with pytest.raises(ValueError, match="Non-paper"):
        acct.build_snapshot(state([], [], accounts=("DUX1", "U999")))


def test_several_paper_accounts_need_a_choice():
    with pytest.raises(ValueError, match="--account"):
        acct.build_snapshot(state([], [], accounts=("DUX1", "DUX2")))
    assert acct.build_snapshot(state([], [], accounts=("DUX1", "DUX2")), account="DUX2")["account"] == "DUX2"


def test_staged_entry_orders_are_listed_not_counted():
    s = acct.build_snapshot(state([], [stop("HNGE", 50, 30.0, action="BUY")]))
    assert s["pending_entry_orders"] == ["HNGE"] and s["total_open_risk_pct"] == 0


def test_alert_setter_orders_are_listed_apart_from_entries():
    """Audit #11: the alert setter's 1-share staged alerts (orderRef ALERT) are not entries."""
    alert = dict(stop("MU", 1, 100.0, action="BUY", kind="LMT"), order_ref="ALERT")
    s = acct.build_snapshot(state([], [alert, stop("HNGE", 50, 30.0, action="BUY")]))
    assert s["pending_entry_orders"] == ["HNGE"] and s["alert_orders"] == ["MU"]


def test_alert_orders_are_tagged_when_staged():
    import ibkr_alert_setter as al
    import ibkr_connector as ic
    placed = []

    class Conn:
        next_order_id = 7

        def fetch_contract_details(self, symbol):
            from types import SimpleNamespace
            return SimpleNamespace(contract=SimpleNamespace(conId=1))

        def placeOrder(self, oid, contract, order):
            placed.append(order)
    setter = al.AlertManager.__new__(al.AlertManager)
    setter.connector = Conn()
    result = setter.create_alert("MU", 100.0)
    assert result["status"] == "PLACED", result
    assert placed and placed[0].orderRef == ic.ALERT_ORDER_REF and placed[0].transmit is False
    cond = placed[0].conditions[0]
    assert (cond.conId, cond.exchange, cond.isMore, cond.price) == (1, "SMART", True, 100.0)
    assert cond.isConjunctionConnection is True
    cond.make_fields()                                   # serializes with the installed ibapi


def test_stale_snapshot_is_ignored(tmp_path):
    s = acct.build_snapshot(state([], []), now=datetime.now() - timedelta(hours=3))
    path = acct.save_snapshot(s, tmp_path)
    assert acct.load_snapshot(str(path)) is None
    fresh = acct.save_snapshot(acct.build_snapshot(state([], [])), tmp_path / "fresh")
    assert acct.load_snapshot(str(fresh))["account"] == "DUX1"


# ─── Engine ───
def scan(*rows):
    base = {"Tier": "TIER_BUY", "Stop_Loss_%": RISK / P.max_position_pct * 100}   # full risk exactly fills the cap
    return pd.DataFrame([{**base, **r} for r in rows])


def snapshot(total=0.0, groups=None, held=(), unprotected=()):
    return {"account": "DUX1", "total_open_risk_pct": total, "group_open_risk_pct": groups or {},
            "held_symbols": list(held), "unprotected_symbols": list(unprotected)}


def decide(df, snap):
    return ee.build_execution_decisions(ee.apply_account_snapshot(df, snap), mode="afterhours", orb_window=10)


def test_existing_open_risk_is_counted_cumulatively():
    out = decide(scan({"Symbol": "A"}, {"Symbol": "B"}), snapshot(total=P.max_total_open_risk_pct - 1.5 * RISK))
    assert list(out["Decision"]) == [ee.BUY, ee.SKIP]   # room for one more full risk, not two
    assert "Open risk cap" in out.iloc[1]["Reason"]


def test_existing_group_risk_from_the_account():
    out = decide(scan({"Symbol": "STX", "IBD_Group_Rank": 3}),
                 snapshot(groups={"3": P.max_group_open_risk_pct - RISK / 2}))
    assert out.iloc[0]["Decision"] == ee.MANUAL_REVIEW


def test_already_held_symbol_is_skipped():
    out = decide(scan({"Symbol": "MU"}), snapshot(held=["MU"]))
    assert out.iloc[0]["Decision"] == ee.SKIP and "Already holding" in out.iloc[0]["Reason"]


def test_unprotected_position_sends_buys_to_review():
    out = decide(scan({"Symbol": "A"}), snapshot(held=["MU"], unprotected=["MU"]))
    assert out.iloc[0]["Decision"] == ee.MANUAL_REVIEW and "MU" in out.iloc[0]["Reason"]


def test_missing_snapshot_blocks_entries():
    out = decide(scan({"Symbol": "A"}), None)
    assert out.iloc[0]["Decision"] == ee.SKIP and "unknown" in out.iloc[0]["Reason"]


def test_without_snapshot_columns_engine_behaves_as_before():
    out = ee.build_execution_decisions(scan({"Symbol": "A"}, {"Symbol": "B"}), mode="afterhours", orb_window=10)
    assert list(out["Decision"]) == [ee.BUY, ee.BUY]


# ─── Audit fixes ───
def test_inactive_or_cancelled_stop_does_not_protect():
    for status in ("Inactive", "PendingCancel", "Cancelled"):
        r = acct.position_risks([pos("MU", 100, 50.0)], [{**stop("MU", 100, 48.0), "status": status}])[0]
        assert r["unprotected_shares"] == 100, status
    ok = acct.position_risks([pos("MU", 100, 50.0)], [{**stop("MU", 100, 48.0), "status": "Submitted"}])[0]
    assert ok["unprotected_shares"] == 0


def test_option_stop_does_not_protect_stock():
    opt = {**stop("MU", 100, 48.0), "sec_type": "OPT"}
    assert acct.position_risks([pos("MU", 100, 50.0)], [opt])[0]["unprotected_shares"] == 100
    by_conid = acct.position_risks([{**pos("MU", 100, 50.0), "con_id": 1}], [{**stop("MU", 100, 48.0), "con_id": 2}])[0]
    assert by_conid["unprotected_shares"] == 100


def test_repeats_without_perm_id_count_once():
    o = {**stop("MU", 60, 48.0), "perm_id": 0, "client_id": 0, "order_id": 0}
    r = acct.position_risks([pos("MU", 100, 50.0)], [o, dict(o)])[0]
    assert r["protected_shares"] == 60 and r["unprotected_shares"] == 40


def test_snapshot_for_another_account_is_ignored(tmp_path):
    path = acct.save_snapshot(acct.build_snapshot(state([], [])), tmp_path)
    assert acct.load_snapshot(str(path), account="DUX2") is None
    assert acct.load_snapshot(str(path), account="DUX1") is not None


def test_manual_review_rows_respect_the_open_risk_cap():
    # Group cap sends STX to review; approving it would breach the total open-risk cap -> SKIP instead
    out = decide(scan({"Symbol": "STX", "IBD_Group_Rank": 3}),
                 snapshot(total=P.max_total_open_risk_pct - RISK / 2, groups={"3": P.max_group_open_risk_pct - RISK / 2}))
    assert out.iloc[0]["Decision"] == ee.SKIP
    assert "Group risk cap" in out.iloc[0]["Reason"] and "Open risk cap" in out.iloc[0]["Reason"]


def test_thin_stock_position_capped_by_daily_dollar_volume():
    # $42M/month = $2M/day; the liquidity % of that, on a $1M account (well under the position cap)
    cap = 2_000_000 * sp.MAX_PCT_OF_DAILY_DOLLAR_VOLUME / 100 / 1_000_000 * 100
    assert cap < P.max_position_pct
    snap = {**snapshot(), "net_liquidation": 1_000_000}
    out = decide(scan({"Symbol": "THIN", "Dollar_Vol_Month_M": 42.0, "Stop_Loss_%": 2.0}), snap)
    row = out.iloc[0]
    assert row["Position_Cap_Pct"] == pytest.approx(cap)
    assert row["Risk_Pct_Target"] == pytest.approx(cap * 2.0 / 100)   # capped position x 2% stop
    assert "Liquidity caps position" in row["Reason"]


def test_liquid_stock_is_not_capped():
    snap = {**snapshot(), "net_liquidation": 100_000}
    row = decide(scan({"Symbol": "BIG", "Dollar_Vol_Month_M": 2000.0}), snap).iloc[0]
    assert row["Position_Cap_Pct"] == pytest.approx(P.max_position_pct)
    assert "Liquidity" not in row["Reason"]
