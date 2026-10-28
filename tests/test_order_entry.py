"""order_entry: worst-fill sizing, plan checks, bracket fields, submit sequence (offline only)."""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

import config
import order_broker as ob
import order_entry as oe
import order_ledger as ol
import strategy_params as sp
from execution_engine import ExecutionPolicy
from helpers import FakeBroker
from order_ledger import LedgerKey, OrderLedger, make_order_ref

POLICY = ExecutionPolicy()
NOW = datetime(2026, 11, 2, 9, 24)          # Monday, before the 09:29 cutoff (ET)
TODAY = NOW.date()
LO, HI = POLICY.min_stop_pct, POLICY.max_stop_pct_pivot
AUX, LMT = oe.entry_prices(150.0)
STOP = round(LMT * (1 - (LO + HI) / 2 / 100), 2)    # mid-band at the worst fill


def row(symbol="NVDA", pivot=150.0, stop=STOP, close=148.0, **kw):
    base = {"Symbol": symbol, "Decision": "BUY", "Entry_Type": "PIVOT_BREAKOUT", "Tier": "TIER_BUY",
            "VCP_Pivot": pivot, "Stop_Price": stop, "Entry_Price": close, "Risk_Pct_Budget": sp.RISK_PER_TRADE_PCT,
            "Position_Cap_Pct": sp.MAX_POSITION_PCT, "IBD_Group_Rank": None, "Scan_Timestamp": "2026-11-02 09:21:00"}
    base.update(kw)
    return base


def rows(*rs):
    return pd.DataFrame(list(rs) or [row()])


def snap(net_liq=100_000.0, sessions=sp.MIN_BREAKER_HISTORY_SESSIONS, total=0.0, held=(), unprotected=(), groups=None, pending=()):
    return {"account": "DUX1", "net_liquidation": net_liq, "total_open_risk_pct": total,
            "group_open_risk_pct": groups or {}, "held_symbols": list(held),
            "unprotected_symbols": list(unprotected), "pending_entry_orders": list(pending),
            "alert_orders": [], "breaker_history_sessions": sessions}


@pytest.fixture
def ledger(tmp_path):
    return OrderLedger(tmp_path / "ledger.jsonl", run_id="t", clock=lambda: NOW)


def plan(ledger, df=None, snapshot=None, open_orders=(), now=NOW, submit=True):
    return oe.build_plan(df if df is not None else rows(), snapshot if snapshot is not None else snap(),
                         ledger, list(open_orders), POLICY, now, submit=submit)


# ─── prices and sizing ───

def test_entry_prices_cap_never_exceeded():
    mult = 1 + config.ENTRY_CHASE_CAP_PCT / 100
    assert oe.entry_prices(150) == (150.00, round(150 * mult, 2))
    for pivot in (10.0, 33.33, 49.99, 77.77, 123.45, 999.99, 12.3456):
        aux, lmt = oe.entry_prices(pivot)
        assert lmt <= aux * mult + 1e-9 and lmt >= aux * mult - 0.01


def test_sizing_uses_the_limit():
    p = oe.size_at_worst_fill(pd.Series(row()), 100_000, POLICY)
    stop_pct = (LMT - STOP) / LMT * 100
    assert p.ok and p.stop_pct_at_limit == pytest.approx(stop_pct, abs=1e-3)
    assert p.stop_pct_at_pivot == pytest.approx((AUX - STOP) / AUX * 100, abs=1e-3)
    # risk budget / stop % at the limit = position %, floored to whole shares at the limit
    pos_pct = min(sp.RISK_PER_TRADE_PCT / stop_pct * 100, sp.MAX_POSITION_PCT)
    shares = int(100_000 * pos_pct / 100 / LMT)
    assert p.shares == shares and p.position_pct == pytest.approx(shares * LMT / 100_000 * 100, abs=1e-3)
    assert p.risk_pct_at_limit == pytest.approx(shares * (LMT - STOP) / 100_000 * 100, abs=1e-3)


def test_stop_inside_band_at_pivot_but_outside_at_limit_is_skipped():
    stop = round(LMT * (1 - HI / 100) - 0.05, 2)          # just past the band's max at the limit
    p = oe.size_at_worst_fill(pd.Series(row(stop=stop)), 100_000, POLICY)
    assert p.stop_pct_at_pivot < HI and p.stop_pct_at_limit > HI
    assert p.status == oe.SKIP_STOP_BAND_AT_LIMIT and "worst-case fill" in p.reason


def test_too_tight_stop_and_bad_stop():
    tight = round(AUX * (1 - LO / 2 / 100), 2)
    assert oe.size_at_worst_fill(pd.Series(row(stop=tight)), 100_000, POLICY).status == oe.SKIP_STOP_BAND
    assert oe.size_at_worst_fill(pd.Series(row(stop=151.0)), 100_000, POLICY).status == oe.SKIP_BAD_STOP
    assert oe.size_at_worst_fill(pd.Series(row(stop=None, Stop_Pct=None)), 100_000, POLICY).status == oe.SKIP_BAD_STOP


def test_zero_shares_is_a_skip():
    p = oe.size_at_worst_fill(pd.Series(row(pivot=1500, stop=STOP * 10, close=1480)), 3_000, POLICY)
    assert p.shares == 0 and p.status == oe.SKIP_ZERO_SHARES


def test_old_decisions_without_budget_are_skipped():
    p = oe.size_at_worst_fill(pd.Series(row(Risk_Pct_Budget=None)), 100_000, POLICY)
    assert p.status == oe.SKIP_MISSING_FIELDS


# ─── plan checks ───

def test_only_buy_rows_are_loaded(tmp_path):
    f = tmp_path / "d.csv"
    pd.DataFrame([row(), row("AMD", Decision="MANUAL_REVIEW"), row("TSLA", Decision="SKIP")]).to_csv(f, index=False)
    out = oe.load_buy_rows(f, now=datetime(2026, 11, 2, 9, 25))
    assert list(out["Symbol"]) == ["NVDA"]


@pytest.mark.parametrize("stamp,now", [("2026-11-01 16:30:00", datetime(2026, 11, 2, 9, 25)),
                                       ("2026-11-02 06:00:00", datetime(2026, 11, 2, 9, 25)),
                                       (None, datetime(2026, 11, 2, 9, 25))])
def test_stale_or_other_day_scan_refuses_the_file(tmp_path, stamp, now):
    f = tmp_path / "d.csv"
    pd.DataFrame([row(Scan_Timestamp=stamp)]).to_csv(f, index=False)
    with pytest.raises(ValueError):
        oe.load_buy_rows(f, now=now)


def test_non_pivot_buy_is_not_auto(ledger):
    p = plan(ledger, rows(row(Entry_Type="ORB_10M_BREAKOUT"), row("AMD", VCP_Pivot=0)))
    assert [x.status for x in p] == [oe.NOT_AUTO_NO_PIVOT, oe.NOT_AUTO_NO_PIVOT]


def test_close_above_limit_is_skipped_and_between_is_noted(ledger):
    assert plan(ledger, rows(row(close=LMT + 0.5)))[0].status == oe.SKIP_PIVOT_EXCEEDED
    p = plan(ledger, rows(row(close=round((AUX + LMT) / 2, 2))))[0]
    assert p.ok and oe.NOTE_ABOVE_PIVOT in p.notes


def test_held_ledger_and_manual_buy_order_are_skipped(ledger):
    assert plan(ledger, snapshot=snap(held=["NVDA"]))[0].status == oe.SKIP_HELD
    manual = {"symbol": "NVDA", "action": "BUY", "order_ref": ""}
    assert plan(ledger, open_orders=[manual])[0].status == oe.SKIP_OPEN_BUY_ORDER
    alert = {"symbol": "NVDA", "action": "BUY", "order_ref": "ALERT"}
    p = plan(ledger, open_orders=[alert])[0]
    assert p.ok and any("ALERT" in n for n in p.notes)
    ledger.record_intent(LedgerKey(TODAY, "NVDA", "PIVOT_BREAKOUT"), "entry",
                         make_order_ref("entry", TODAY, "NVDA"), 5, qty=1)
    assert plan(ledger)[0].status == oe.SKIP_IN_LEDGER


def test_daily_auto_entry_cap(ledger):
    p = plan(ledger, rows(row(), row("AMD")))
    assert [x.status for x in p] == [oe.PLANNED, oe.SKIP_DAILY_ENTRY_CAP]     # MAX_AUTO_ENTRIES_PER_DAY = 1
    ledger.record_intent(LedgerKey(TODAY, "TSLA", "PIVOT_BREAKOUT"), "entry",
                         make_order_ref("entry", TODAY, "TSLA"), 5, qty=1)
    ledger.record({"trade_date": TODAY, "symbol": "TSLA", "setup": "PIVOT_BREAKOUT", "role": "entry",
                   "order_ref": make_order_ref("entry", TODAY, "TSLA"), "order_id": 5, "state": "cancelled",
                   "reason": "never_reached_tws"})
    assert plan(ledger)[0].status == oe.SKIP_DAILY_ENTRY_CAP                  # a never-sent entry used the slot


def test_caps_are_cumulative_and_count_pending_bot_entries(ledger, monkeypatch):
    monkeypatch.setattr(config, "MAX_AUTO_ENTRIES_PER_DAY", 5, raising=False)
    # each row risks r at the limit; room for 1.5 r under each cap: one row fits, a second doesn't
    r = oe.size_at_worst_fill(pd.Series(row()), 100_000, POLICY).risk_pct_at_limit
    total = POLICY.max_total_open_risk_pct - 1.5 * r
    p = plan(ledger, rows(row(), row("AMD")), snapshot=snap(total=total))
    assert [x.status for x in p] == [oe.PLANNED, oe.SKIP_OPEN_RISK_CAP]
    groups = {"3": POLICY.max_group_open_risk_pct - 1.5 * r}
    p = plan(ledger, rows(row(IBD_Group_Rank=3), row("AMD", IBD_Group_Rank=3)), snapshot=snap(groups=groups))
    assert [x.status for x in p] == [oe.PLANNED, oe.SKIP_GROUP_RISK_CAP]
    held = [f"S{i}" for i in range(POLICY.max_open_positions - 1)]
    p = plan(ledger, rows(row(), row("AMD")), snapshot=snap(held=held))
    assert [x.status for x in p] == [oe.PLANNED, oe.SKIP_MAX_POSITIONS]
    # a working bot entry from earlier counts its planned risk and its slot
    ledger.record_intent(LedgerKey(TODAY, "TSLA", "PIVOT_BREAKOUT"), "entry",
                         make_order_ref("entry", TODAY, "TSLA"), 5, qty=10, plan={"risk_pct": r})
    assert plan(ledger, rows(row()), snapshot=snap(total=total))[0].status == oe.SKIP_OPEN_RISK_CAP
    assert plan(ledger, rows(row()), snapshot=snap(held=held))[0].status == oe.SKIP_MAX_POSITIONS


@pytest.mark.parametrize("sessions", [sp.MIN_BREAKER_HISTORY_SESSIONS - 1, None])
def test_breaker_history_halts_submit(ledger, sessions):
    p = plan(ledger, snapshot=snap(sessions=sessions))[0]
    assert p.status == oe.HALTED_NO_BREAKER_HISTORY
    assert f"of {sp.MIN_BREAKER_HISTORY_SESSIONS} sessions since the last reset" in p.reason


def test_breaker_history_dry_run_plans_with_a_note(ledger):
    p = plan(ledger, snapshot=snap(sessions=sp.MIN_BREAKER_HISTORY_SESSIONS - 1), submit=False)[0]
    assert p.ok and any("submit would halt" in n for n in p.notes)


def test_unprotected_position_halts(ledger):
    assert plan(ledger, snapshot=snap(unprotected=["MU"]))[0].status == oe.HALTED_UNPROTECTED
    assert plan(ledger, snapshot=snap(unprotected=["MU"]), submit=False)[0].status == oe.HALTED_UNPROTECTED


def test_after_cutoff_or_holiday_refuses(ledger):
    assert plan(ledger, now=datetime(2026, 11, 2, 9, 29))[0].status == oe.HALTED_AFTER_CUTOFF
    assert plan(ledger, now=datetime(2026, 11, 2, 9, 45), submit=False)[0].ok      # dry run may plan late
    assert plan(ledger, now=datetime(2026, 11, 26, 9, 0))[0].status == oe.HALTED_NOT_TRADING_DAY


def test_missing_snapshot_halts(ledger):
    p = oe.build_plan(rows(), None, ledger, [], POLICY, NOW, submit=False)[0]
    assert p.status == oe.HALTED_NO_SNAPSHOT


# ─── bracket and submit ───

def test_bracket_fields(ledger):
    p = plan(ledger)[0]
    parent, child = oe.build_bracket(p, 1000, "DUX1")
    assert (parent.action, parent.orderType, parent.tif, parent.transmit, parent.outsideRth) == \
        ("BUY", "STP LMT", "DAY", False, False)
    assert (parent.auxPrice, parent.lmtPrice, parent.totalQuantity) == (AUX, LMT, p.shares) and p.shares > 0
    assert (child.action, child.orderType, child.tif, child.transmit, child.parentId) == \
        ("SELL", "STP", "GTC", True, 1000)
    assert child.auxPrice == STOP and child.totalQuantity == p.shares and child.outsideRth is False
    assert child.ocaGroup == "BOT-OCA-20261102-NVDA" and child.ocaType == 3
    assert parent.orderRef == "BOT-ENTRY-20261102-NVDA" and child.orderRef == "BOT-STOP-20261102-NVDA"
    assert parent.account == child.account == "DUX1"


def test_submit_writes_intents_then_places_and_records_working(ledger):
    broker = FakeBroker()
    plans = oe.submit_plans(plan(ledger), broker, ledger, TODAY)
    assert plans[0].status == oe.WORKING
    assert [o.orderRef for _, _, o in broker.placed] == ["BOT-ENTRY-20261102-NVDA", "BOT-STOP-20261102-NVDA"]
    assert [o.transmit for _, _, o in broker.placed] == [False, True]
    cur = ledger.current()
    assert set(cur["state"]) == {"working"} and set(cur["role"]) == {"entry", "stop"}
    states = [e["state"] for e in ledger.events() if e["role"] == "entry"]
    assert states == ["intended", "submitted", "working"]           # intent fsynced before placing


def test_order_ids_stay_above_the_ledger(ledger):
    ledger.record_intent(LedgerKey(TODAY, "TSLA", "PIVOT_BREAKOUT"), "stop",
                         make_order_ref("stop", TODAY, "TSLA"), 5000, qty=1)
    plans = oe.submit_plans(plan(ledger), FakeBroker(first_id=10), ledger, TODAY)
    assert plans[0].parent_id == 5001 and plans[0].child_id == 5002


def test_rejected_ack_is_recorded(ledger, monkeypatch):
    monkeypatch.setattr(oe, "notify", lambda msg: None)
    plans = oe.submit_plans(plan(ledger), FakeBroker(ack=("rejected", "TWS 201: margin")), ledger, TODAY)
    assert plans[0].status == oe.REJECTED and "201" in plans[0].reason
    assert set(ledger.current()["state"]) == {"rejected"}


def test_submit_refuses_a_read_only_broker(ledger):
    broker = FakeBroker(submit_enabled=False)
    with pytest.raises(PermissionError):
        oe.submit_plans(plan(ledger), broker, ledger, TODAY)
    assert broker.placed == [] and ledger.events() == []


def test_drop_between_parent_and_child(ledger):
    broker = FakeBroker(fail_on_place=2)
    plans = oe.submit_plans(plan(ledger), broker, ledger, TODAY)
    assert plans[0].status == oe.FAILED_CHILD and len(broker.placed) == 1     # parent only, staged
    report = ledger.reconcile(broker.state(), TODAY)
    assert [b["order_ref"] for b in report.incomplete_brackets] == ["BOT-ENTRY-20261102-NVDA"]
    assert report.incomplete_brackets[0]["staged"] is True
    child = ledger.current().query("role == 'stop'").iloc[-1]
    assert child["state"] == "cancelled" and child["reason"] == "never_reached_tws"
    # the parent went to the socket: 'submitted' (TWS may hide an untransmitted order), never
    # never_reached_tws, and it was cancelled at once
    parent = ledger.current().query("role == 'entry'").iloc[-1]
    assert parent["state"] in ("submitted", "working") and parent["reason"] != "never_reached_tws"
    assert broker.cancelled == [plans[0].parent_id]
    # the next submit run cancels it again if TWS still shows it; a rerun never resends the entry
    assert oe.cancel_incomplete_brackets(report, broker, 16)[0]["order_id"] == plans[0].parent_id
    assert broker.cancelled == [plans[0].parent_id] * 2
    assert plan(ledger)[0].status == oe.SKIP_IN_LEDGER


def test_parent_hidden_at_tws_is_not_marked_never_reached(ledger):
    """TWS may not list an untransmitted parent in reqAllOpenOrders: the ledger must not call it
    never_reached_tws (cancelled) while it still sits in the TWS window."""
    broker = FakeBroker(fail_on_place=2)
    plans = oe.submit_plans(plan(ledger), broker, ledger, TODAY)
    broker.state().open_orders.clear()
    ledger.reconcile(broker.state(), TODAY)
    parent = ledger.current().query("role == 'entry'").iloc[-1]
    assert plans[0].status == oe.FAILED_CHILD
    assert parent["state"] == "submitted" and parent["reason"] != "never_reached_tws"


def test_child_less_parent_cancelled_even_if_transmit_reads_true(ledger):
    """ibapi never decodes Order.transmit, so a staged parent reads transmit=True at reconcile."""
    broker = FakeBroker(fail_on_place=2)
    oe.submit_plans(plan(ledger), broker, ledger, TODAY)
    for o in broker.state().open_orders:
        o["transmit"] = True
    report = ledger.reconcile(broker.state(), TODAY)
    assert len(oe.cancel_incomplete_brackets(report, broker, 16)) == 1
    # another client's parent is never cancelled (IBKR would refuse); the read-only broker never cancels
    assert oe.cancel_incomplete_brackets(report, broker, 99) == []
    assert oe.cancel_incomplete_brackets(report, FakeBroker(submit_enabled=False), 16) == []


def test_filled_parent_is_not_an_incomplete_bracket(ledger):
    broker = FakeBroker(fail_on_place=2)
    oe.submit_plans(plan(ledger), broker, ledger, TODAY)
    broker.state().open_orders[0]["status"] = "Filled"
    assert ledger.reconcile(broker.state(), TODAY).incomplete_brackets == []


class AckBroker(FakeBroker):
    """FakeBroker with a per-order ack (parent vs child)."""

    def __init__(self, acks, **kw):
        super().__init__(**kw)
        self.acks = acks          # order index (0 parent, 1 child) -> ack

    def wait_ack(self, order_id, timeout=10.0):
        idx = [p[0] for p in self.placed].index(order_id)
        return self.acks.get(idx, ("working", "PreSubmitted"))


@pytest.mark.parametrize("child_ack", [("rejected", "TWS 110: price does not conform to min tick"),
                                       ("unknown", "no order status within 10s")])
def test_child_stop_not_acked_cancels_parent_and_stops(ledger, monkeypatch, child_ack):
    monkeypatch.setattr(config, "MAX_AUTO_ENTRIES_PER_DAY", 2, raising=False)
    broker = AckBroker({1: child_ack})
    plans = oe.submit_plans(plan(ledger, rows(row(), row("AMD"))), broker, ledger, TODAY)
    assert [p.status for p in plans] == [oe.FAILED_CHILD_ACK, oe.NOT_SENT]
    assert broker.cancelled == [plans[0].parent_id]          # parent cancelled in the same run
    assert len(broker.placed) == 2                            # AMD never sent
    cur = ledger.current().set_index("role")
    assert cur.loc["entry", "state"] != "working" or bool(cur.loc["entry", "cancel_requested"])
    assert cur.loc["stop", "state"] == ("rejected" if child_ack[0] == "rejected" else "submitted")
    assert child_ack[1] in plans[0].reason


def test_failed_child_ack_exits_2(monkeypatch):
    assert oe.FAILED_CHILD_ACK in oe.FAILURE_STATUSES and oe.FAILED_CHILD in oe.FAILURE_STATUSES


def test_failed_parent_stops_the_run_and_leaves_the_intent(ledger, monkeypatch):
    monkeypatch.setattr(config, "MAX_AUTO_ENTRIES_PER_DAY", 2, raising=False)
    broker = FakeBroker(fail_on_place=1)
    plans = oe.submit_plans(plan(ledger, rows(row(), row("AMD"))), broker, ledger, TODAY)
    assert [p.status for p in plans] == [oe.FAILED_PARENT, oe.NOT_SENT]
    assert set(ledger.current()["state"]) == {"intended"}           # reconcile decides later
    assert not ledger.has_entry(TODAY, "AMD")


def test_oversell_guard_cancels_the_intents(ledger):
    stray = {"symbol": "NVDA", "action": "SELL", "quantity": 50, "status": "Submitted", "perm_id": 9}
    broker = FakeBroker(open_orders=[stray])
    plans = oe.submit_plans(plan(ledger), broker, ledger, TODAY)
    assert plans[0].status == oe.SKIP_OVERSELL_GUARD and broker.placed == []
    assert set(ledger.current()["state"]) == {"cancelled"}


# ─── output and CLI ───

def test_plan_csv_columns(tmp_path, ledger):
    path = oe.write_plan(plan(ledger, rows(row(), row("AMD", Entry_Type="UNKNOWN"))), tmp_path / "plan.csv",
                         run_id="r1", mode="plan-only", now=NOW)
    df = pd.read_csv(path)
    assert list(df.columns) == oe.PLAN_COLUMNS and len(df) == 2
    oe.write_plan(plan(ledger), path, run_id="r2", now=NOW)                    # appends a block
    assert list(pd.read_csv(path)["run_id"]) == ["r1", "r1", "r2"]


def _decisions_file(tmp_path, **kw):
    f = tmp_path / "scan_execution_decisions.csv"
    pd.DataFrame([row(Scan_Timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), **kw)]).to_csv(f, index=False)
    return f


@pytest.fixture
def cli_env(tmp_path, monkeypatch):
    monkeypatch.setattr(ol, "ORDERS_DIR", tmp_path / "orders")
    monkeypatch.setattr(oe, "now_et", lambda: NOW)
    monkeypatch.setattr(oe, "notify", lambda msg: None)
    return tmp_path


def test_plan_only_never_builds_a_connector(cli_env, monkeypatch):
    import ibkr_connector

    def boom(*a, **k):
        raise AssertionError("plan-only must not construct a connector")
    monkeypatch.setattr(ibkr_connector, "IBKRConnector", boom)
    monkeypatch.setattr(ob, "IBKRConnector", boom)
    monkeypatch.setattr(ob, "open_session", boom)
    snap_path = cli_env / "snap.json"
    snap_path.write_text(json.dumps({**snap(), "timestamp": datetime.now().isoformat(timespec="seconds")}))
    rc = oe.main(["--decisions", str(_decisions_file(cli_env)), "--account-snapshot", str(snap_path), "--plan-only"])
    assert rc == 0
    out = pd.read_csv(cli_env / "orders" / "plan_2026-11-02.csv")
    assert list(out["status"]) == [oe.PLANNED] and list(out["mode"]) == ["plan-only"]
    assert not (cli_env / "orders" / "ledger.jsonl").exists()                 # nothing recorded, nothing sent


def test_plan_only_without_snapshot_halts_with_exit_2(cli_env):
    assert oe.main(["--decisions", str(_decisions_file(cli_env)), "--plan-only"]) == 2


def test_submit_without_config_switch_is_refused(cli_env, monkeypatch):
    monkeypatch.setattr(config, "AUTO_SUBMIT_ENABLED", False, raising=False)
    broker = FakeBroker(submit_enabled=False)
    opened = []
    monkeypatch.setattr(ob, "open_session", lambda port, account, cli, cid: opened.append(cid) or broker)
    monkeypatch.setattr(oe, "_snapshot_from_state", lambda state, account, now: snap())
    rc = oe.main(["--decisions", str(_decisions_file(cli_env)), "--account", "DUX1", "--submit"])
    assert rc == 2 and broker.placed == [] and opened == [17]                # read-only client
    assert list(pd.read_csv(cli_env / "orders" / "plan_2026-11-02.csv")["mode"]) == ["submit-refused"]


def test_submit_with_both_gates_places_one_bracket(cli_env, monkeypatch):
    monkeypatch.setattr(config, "AUTO_SUBMIT_ENABLED", True, raising=False)
    broker = FakeBroker(submit_enabled=True)
    opened = []
    monkeypatch.setattr(ob, "open_session", lambda port, account, cli, cid: opened.append(cid) or broker)
    monkeypatch.setattr(oe, "_snapshot_from_state", lambda state, account, now: snap())
    rc = oe.main(["--decisions", str(_decisions_file(cli_env)), "--account", "DUX1", "--submit"])
    assert rc == 0 and opened == [16] and len(broker.placed) == 2
    # rerun: the ledger key blocks a second entry, nothing more is placed
    oe.main(["--decisions", str(_decisions_file(cli_env)), "--account", "DUX1", "--submit"])
    assert len(broker.placed) == 2


def test_submit_halts_without_breaker_history(cli_env, monkeypatch):
    monkeypatch.setattr(config, "AUTO_SUBMIT_ENABLED", True, raising=False)
    broker = FakeBroker(submit_enabled=True)
    monkeypatch.setattr(ob, "open_session", lambda port, account, cli, cid: broker)
    monkeypatch.setattr(oe, "_snapshot_from_state", lambda state, account, now: snap(sessions=2))
    assert oe.main(["--decisions", str(_decisions_file(cli_env)), "--account", "DUX1", "--submit"]) == 2
    assert broker.placed == []


def test_second_run_holding_the_lock_exits_3(cli_env, monkeypatch):
    monkeypatch.setattr(ob, "open_session", lambda *a: FakeBroker(submit_enabled=False))
    with ol.ledger_lock(cli_env / "orders"):
        assert oe.main(["--decisions", str(_decisions_file(cli_env)), "--account", "DUX1"]) == 3


def test_holiday_exits_0_before_anything(cli_env, monkeypatch):
    monkeypatch.setattr(oe, "now_et", lambda: datetime(2026, 11, 26, 9, 0))
    assert oe.main(["--decisions", "/nonexistent.csv", "--plan-only"]) == 0
