"""fill_monitor: stop coverage per position, bot-only fixes, submit gates (offline only)."""

from datetime import date, datetime

import pandas as pd
import pytest

import config
import fill_monitor as fm
import order_ledger as ol
from helpers import FakeBroker
from order_ledger import LedgerKey, OrderLedger, make_oca_group, make_order_ref

D1 = date(2026, 11, 2)
ACCT = "DUX1"
RTH = datetime(2026, 11, 4, 12, 30)        # Wednesday midday ET
AFTER = datetime(2026, 11, 4, 17, 0)
ENTRY_REF = make_order_ref("entry", D1, "NVDA")
STOP_REF = make_order_ref("stop", D1, "NVDA")
OCA = make_oca_group(D1, "NVDA")
KEY = f"{D1.isoformat()}:NVDA:PIVOT_BREAKOUT"


@pytest.fixture
def ledger(tmp_path):
    return OrderLedger(tmp_path / "ledger.jsonl", run_id="t", clock=lambda: RTH)


def filled_trade(ledger, filled=120, entry_state="filled"):
    """A bot bracket whose entry filled `filled` shares; the child stop is working at 141."""
    key = LedgerKey(D1, "NVDA", "PIVOT_BREAKOUT")
    common = dict(client_id=16, account=ACCT, qty=120)
    ledger.record_intent(key, "entry", ENTRY_REF, 1043, action="BUY", order_type="STP LMT",
                         aux_price=150.0, lmt_price=153.0, tif="DAY", **common)
    ledger.record_intent(key, "stop", STOP_REF, 1044, action="SELL", order_type="STP", aux_price=141.0,
                         tif="GTC", parent_order_id=1043, oca_group=OCA, **common)
    base = {**key.as_fields(), "client_id": 16}
    for role, ref, oid in (("entry", ENTRY_REF, 1043), ("stop", STOP_REF, 1044)):
        ledger.record({**base, "role": role, "order_ref": ref, "order_id": oid, "state": "submitted"})
        ledger.record({**base, "role": role, "order_ref": ref, "order_id": oid, "state": "working"})
    ledger.record({**base, "role": "entry", "order_ref": ENTRY_REF, "order_id": 1043, "state": entry_state,
                   "filled_qty": filled, "avg_fill_price": 151.0})
    return key


def stop_order(qty=120, oid=1044, ref=STOP_REF, client_id=16, aux=141.0, parent=1043, symbol="NVDA",
               order_type="STP", status="PreSubmitted"):
    return {"order_id": oid, "perm_id": 0, "client_id": client_id, "account": ACCT, "symbol": symbol,
            "sec_type": "STK", "action": "SELL", "order_type": order_type, "quantity": float(qty),
            "aux_price": aux, "lmt_price": 0.0, "status": status, "order_ref": ref, "parent_id": parent,
            "tif": "GTC", "oca_group": OCA if ref.startswith("BOT-") else "", "transmit": True}


def pos(symbol="NVDA", shares=120.0, cost=151.0):
    return {"account": ACCT, "symbol": symbol, "sec_type": "STK", "shares": shares, "avg_cost": cost}


def broker(positions=(), orders=(), submit=True, quotes=None, client_id=16):
    return FakeBroker(account=ACCT, submit_enabled=submit, positions=positions, open_orders=orders,
                      quotes=quotes, client_id=client_id, first_id=2000)


def check(ledger, b, exit_state=None, now=RTH):
    return fm.check_protection(b.state(), ledger, exit_state or {}, quote=b.last_price, now=now, account=ACCT)


def one(rows, symbol="NVDA"):
    found = [r for r in rows if r.symbol == symbol]
    assert len(found) == 1
    return found[0]


def no_orders_sent(b):
    return not (b.placed or b.modified or b.cancelled)


# ─── decisions ───

def test_fully_covered_is_ok_and_needs_no_quote(ledger):
    filled_trade(ledger)
    b = broker([pos()], [stop_order()])
    r = one(check(ledger, b))
    assert (r.action, r.protected, r.unprotected, r.bot_trade) == (fm.OK, 120, 0, KEY)
    assert b.quote_calls == []


def test_partial_fill_resizes_the_bot_stop_to_the_position(ledger):
    filled_trade(ledger)
    b = broker([pos(shares=120)], [stop_order(qty=60)], quotes={"NVDA": (148.0, False)})
    r = one(check(ledger, b))
    assert r.action == fm.RESIZE_STOP and r.unprotected == 60
    fixes = fm.plan_fixes([r], submit=True)
    assert fixes[0]["action"] == fm.RESIZE_STOP and fixes[0]["new_qty"] == 120
    fm.apply_fixes(fixes, b, ledger)
    oid, _, order = b.modified[0]
    assert (oid, order.totalQuantity, order.auxPrice, order.parentId) == (1044, 120, 141.0, 1043)
    assert order.ocaGroup == OCA and order.tif == "GTC" and order.orderRef == STOP_REF
    stop_row = ledger.current().query("order_ref == @STOP_REF").iloc[-1]
    assert stop_row["qty"] == 120 and "resize_stop_intent 60->120" in stop_row["reason"]


def test_working_entry_rest_is_covered_by_its_child(ledger):
    """Parent still working with 60 of 120 filled: a 120-share child is expected, not oversized."""
    filled_trade(ledger, filled=60, entry_state="partially_filled")
    parent = {**stop_order(qty=120, oid=1043, ref=ENTRY_REF, parent=0, order_type="STP LMT"),
              "action": "BUY", "filled": 60.0, "status": "Submitted"}
    b = broker([pos(shares=60)], [parent, stop_order(qty=120)])
    r = one(check(ledger, b))
    assert r.action == fm.OK and r.pending_entry_qty == 60 and r.oversized == 0


def test_no_stop_places_one_at_the_exit_state_stop(ledger):
    filled_trade(ledger)
    b = broker([pos()], [], quotes={"NVDA": (155.0, False)})
    exit_state = {"trades": {KEY: {"current_stop": 151.37, "status": "open"}}}
    r = one(check(ledger, b, exit_state))
    assert r.action == fm.PLACE_STOP and r.stop == 151.37 and r.ledger_stop == 141.0
    fm.apply_fixes(fm.plan_fixes([r], submit=True), b, ledger)
    oid, contract, order = b.placed[0]
    assert (contract.symbol, order.action, order.orderType, order.totalQuantity) == ("NVDA", "SELL", "STP", 120)
    assert (order.auxPrice, order.tif, order.outsideRth, order.transmit) == (151.37, "GTC", False, True)
    assert order.orderRef == make_order_ref("stop_fix", D1, "NVDA") and order.ocaGroup == OCA
    assert order.ocaType == 3 and order.account == ACCT and oid >= 2000
    fix_row = ledger.current().query("role == 'stop_fix'").iloc[-1]
    assert fix_row["state"] == "working" and fix_row["qty"] == 120 and fix_row["client_id"] == 16


def test_no_stop_falls_back_to_the_ledger_stop(ledger):
    filled_trade(ledger)
    r = one(check(ledger, broker([pos()], [], quotes={"NVDA": (150.0, False)})))
    assert r.action == fm.PLACE_STOP and r.stop == 141.0


def test_fix_intent_is_written_before_place(ledger):
    filled_trade(ledger)
    b = FakeBroker(account=ACCT, positions=[pos()], quotes={"NVDA": (150.0, False)}, fail_on_place=1)
    fixes = fm.apply_fixes(fm.plan_fixes(check(ledger, b), submit=True), b, ledger)
    assert fixes[0]["result"].startswith("failed")
    assert ledger.current().query("role == 'stop_fix'").iloc[-1]["state"] == "intended"


def test_price_below_stop_flattens_only_with_both_gates(ledger):
    filled_trade(ledger)
    b = broker([pos()], [], quotes={"NVDA": (140.0, False)})
    r = one(check(ledger, b))
    assert r.action == fm.FLATTEN
    # Read-only: downgraded to an alert, nothing sent
    assert fm.plan_fixes([r], submit=False)[0]["action"] == "ALERT_FLATTEN"
    fm.apply_fixes(fm.plan_fixes([r], submit=True), b, ledger)
    _, _, order = b.placed[0]
    assert (order.orderType, order.tif, order.totalQuantity) == ("MKT", "DAY", 120)
    assert order.orderRef == make_order_ref("flatten", D1, "NVDA")


@pytest.mark.parametrize("quote,now", [((140.0, True), RTH), ((None, True), RTH), ((140.0, False), AFTER)])
def test_delayed_quote_or_closed_market_only_alerts(ledger, quote, now):
    filled_trade(ledger)
    b = broker([pos()], [], quotes={"NVDA": quote})
    r = one(check(ledger, b, now=now))
    if quote[0] is None:
        assert r.action == fm.PLACE_STOP and "no quote" in r.note   # protect it anyway
    else:
        assert r.action == fm.ALERT_FLATTEN and fm.plan_fixes([r], submit=True) == []


def test_oversized_bot_stop_shrinks(ledger):
    filled_trade(ledger)
    b = broker([pos(shares=120)], [stop_order(qty=150)])
    r = one(check(ledger, b))
    assert r.action == fm.SHRINK_STOP and r.oversized == 30
    fm.apply_fixes(fm.plan_fixes([r], submit=True), b, ledger)
    assert b.modified[0][2].totalQuantity == 120


def test_oversized_manual_stop_is_only_reported(ledger):
    b = broker([pos("AAPL", 20, 200.0)], [stop_order(qty=30, oid=7, ref="", client_id=0, symbol="AAPL",
                                                     parent=0, aux=190.0)])
    r = one(check(ledger, b), "AAPL")
    assert r.action == fm.REPORT_OVERSIZED and r.bot_trade is None
    assert fm.plan_fixes([r], submit=True) == []


def test_orphan_bot_sell_is_cancelled(ledger):
    filled_trade(ledger)
    b = broker([], [stop_order(qty=120)])            # position gone (stopped out by hand / reset)
    rows = check(ledger, b)
    r = one(rows)
    assert r.action == fm.CANCEL_ORPHAN_SELL
    fm.apply_fixes(fm.plan_fixes(rows, submit=True), b, ledger)
    assert b.cancelled == [1044]
    assert ledger.current().query("order_ref == @STOP_REF").iloc[-1]["cancel_requested"] is True


def test_child_of_a_working_parent_is_not_an_orphan(ledger):
    parent = {**stop_order(oid=1043, ref=ENTRY_REF, parent=0, order_type="STP LMT"), "action": "BUY"}
    assert check(ledger, broker([], [parent, stop_order()])) == []


def test_another_clients_order_is_never_modified(ledger):
    filled_trade(ledger)
    b = broker([pos()], [stop_order(qty=60, client_id=17)], quotes={"NVDA": (150.0, False)})
    fixes = fm.plan_fixes(check(ledger, b), submit=True)
    assert fixes[0]["action"] == "ALERT_RESIZE_STOP"
    fm.apply_fixes(fixes, b, ledger)
    assert no_orders_sent(b)


def test_new_stop_capped_by_working_trim(ledger):
    """A working 40-share trim LMT leaves room for an 80-share stop only (no oversell)."""
    filled_trade(ledger)
    trim = {**stop_order(qty=40, oid=1050, ref=make_order_ref("trim", D1, "NVDA"), parent=0),
            "order_type": "LMT", "lmt_price": 169.0, "aux_price": 0.0}
    b = broker([pos()], [trim], quotes={"NVDA": (150.0, False)})
    fixes = fm.plan_fixes(check(ledger, b), submit=True)
    assert fixes[0]["qty"] == 80 and "capped at 80" in fixes[0]["note"]


def test_manual_position_is_never_modified(ledger):
    filled_trade(ledger)
    b = broker([pos(), pos("AAPL", 10, 200.0)], [stop_order()], quotes={"AAPL": (100.0, False)})
    rows, fixes, _ = fm.run(b, ledger, submit=True, client_id=16, now=RTH, exit_state={})
    assert one(rows, "AAPL").action == fm.REPORT and fixes == []
    assert no_orders_sent(b) and "AAPL" not in b.quote_calls


def test_manual_touched_trade_is_report_only(ledger):
    filled_trade(ledger)
    b = broker([pos(shares=200)], [], quotes={"NVDA": (140.0, False)})   # user added 80 shares
    rows, fixes, report = fm.run(b, ledger, submit=True, client_id=16, now=RTH, exit_state={})
    assert ledger.trade_status(D1, "NVDA") == "manual_touched"
    assert one(rows).action == fm.REPORT and fixes == [] and no_orders_sent(b)


def test_default_run_never_sends(ledger):
    filled_trade(ledger)
    b = broker([pos(shares=120)], [stop_order(qty=60)], submit=False, quotes={"NVDA": (140.0, False)})
    rows, fixes, _ = fm.run(b, ledger, submit=False, client_id=17, now=RTH, exit_state={})
    assert one(rows).action == fm.FLATTEN
    assert [f["action"] for f in fixes] == ["ALERT_FLATTEN"]
    assert no_orders_sent(b)
    with pytest.raises(PermissionError):
        fm.apply_fixes(fixes, b, ledger)


def test_report_csv(ledger, tmp_path):
    filled_trade(ledger)
    b = broker([pos()], [], submit=False, quotes={"NVDA": (150.0, False)})
    rows, fixes, _ = fm.run(b, ledger, submit=False, client_id=17, now=RTH, exit_state={})
    path = fm.write_report(rows, tmp_path / "p.csv", fixes=fixes, now=RTH)
    df = pd.read_csv(path)
    assert list(df.columns) == fm.REPORT_COLUMNS
    assert df.loc[0, "action"] == fm.PLACE_STOP and df.loc[0, "fix"].startswith("ALERT_PLACE_STOP")


# ─── CLI guards ───

def test_main_holiday_exits_before_connecting(monkeypatch):
    monkeypatch.setattr(fm, "now_et", lambda: datetime(2026, 11, 26, 10, 0))   # Thanksgiving
    import order_broker
    monkeypatch.setattr(order_broker, "open_session", lambda *a, **k: pytest.fail("connected"))
    assert fm.main(["--account", "DUX1"]) == 0


def test_main_refuses_live_port(monkeypatch, tmp_path):
    monkeypatch.setattr(fm, "now_et", lambda: RTH)
    monkeypatch.setattr(ol, "ORDERS_DIR", tmp_path)
    assert fm.main(["--account", "DUX1", "--port", "7496"]) == 2


def test_main_submit_without_config_is_read_only(monkeypatch, tmp_path):
    monkeypatch.setattr(fm, "now_et", lambda: RTH)
    monkeypatch.setattr(ol, "ORDERS_DIR", tmp_path)
    monkeypatch.setattr(config, "AUTO_SUBMIT_ENABLED", False, raising=False)
    seen = {}
    import order_broker

    def fake_open(port, account, cli_submit, client_id):
        seen["client_id"] = client_id
        return broker([], [], submit=False)
    monkeypatch.setattr(order_broker, "open_session", fake_open)
    assert fm.main(["--account", "DUX1", "--submit"]) == 2      # refused -> 2
    assert seen["client_id"] == 17


def test_main_lock_held_exits_3(monkeypatch, tmp_path):
    monkeypatch.setattr(fm, "now_et", lambda: RTH)
    monkeypatch.setattr(ol, "ORDERS_DIR", tmp_path)
    with ol.ledger_lock(tmp_path):
        assert fm.main(["--account", "DUX1"]) == 3


# ─── a short the bot's own SELLs caused (OCA type 3 has no overfill block) ───

def _overfilled(ledger, sold=240):
    """Entry 120 filled; the stop and an exit MOO both filled 120 at the open -> short 120."""
    filled_trade(ledger)
    key = LedgerKey(D1, "NVDA", "PIVOT_BREAKOUT")
    ref = make_order_ref("exit", D1, "NVDA")
    ledger.record_intent(key, "exit", ref, 1050, action="SELL", order_type="MKT", tif="OPG", qty=120,
                         client_id=16, account=ACCT, oca_group=OCA)
    base = {**key.as_fields(), "client_id": 16, "role": "exit", "order_ref": ref, "order_id": 1050}
    ledger.record({**base, "state": "submitted"})
    ledger.record({**base, "state": "filled", "filled_qty": 120})
    ledger.record({**key.as_fields(), "client_id": 16, "role": "stop", "order_ref": STOP_REF, "order_id": 1044,
                   "state": "filled", "filled_qty": sold - 120})


def test_bot_caused_short_is_covered_in_the_session(ledger):
    _overfilled(ledger)
    b = broker([pos(shares=-120.0)], [])
    r = one(check(ledger, b))
    assert r.action == fm.COVER_SHORT and r.bot_trade == KEY
    fixes = fm.plan_fixes([r], submit=True)
    fm.apply_fixes(fixes, b, ledger)
    _, contract, order = b.placed[0]
    assert (order.action, order.orderType, order.tif, order.totalQuantity) == ("BUY", "MKT", "DAY", 120)
    assert isinstance(order.totalQuantity, int) and order.orderRef == make_order_ref("cover", D1, "NVDA")
    row = ledger.current().query("role == 'cover'").iloc[-1]
    assert row["state"] == "working" and row["action"] == "BUY"
    # once it fills, the ledger trade nets to zero
    ledger.record({**row.to_dict(), "state": "filled", "filled_qty": 120.0})
    assert ledger.open_trades()[0]["net_qty"] == 0


def test_bot_short_outside_session_or_read_only_only_alerts(ledger):
    _overfilled(ledger)
    b = broker([pos(shares=-120.0)], [])
    assert one(check(ledger, b, now=AFTER)).action == fm.ALERT_COVER_SHORT
    r = one(check(ledger, b))
    assert fm.plan_fixes([r], submit=False)[0]["action"] == "ALERT_COVER_SHORT"
    assert no_orders_sent(b)


def test_bot_short_with_a_working_buy_is_reported(ledger):
    _overfilled(ledger)
    buy = {**stop_order(oid=77, ref="", client_id=0), "action": "BUY", "order_type": "LMT"}
    assert one(check(ledger, broker([pos(shares=-120.0)], [buy]))).action == fm.REPORT


def test_short_not_matching_the_ledger_is_manual(ledger):
    _overfilled(ledger)
    r = one(check(ledger, broker([pos(shares=-50.0)], [])))    # not what the bot's fills explain
    assert r.action != fm.COVER_SHORT and r.bot_trade is None


def test_cover_never_buys_more_than_the_live_short(ledger):
    _overfilled(ledger)
    b = broker([pos(shares=-120.0)], [])
    fixes = fm.plan_fixes([one(check(ledger, b))], submit=True)
    b.state().positions[0]["shares"] = 0.0                   # covered by hand meanwhile
    fm.apply_fixes(fixes, b, ledger)
    assert fixes[0]["result"].startswith("refused") and not b.placed


def test_resize_keeps_the_filled_part_of_the_stop(ledger):
    filled_trade(ledger)
    b = broker([pos(shares=120)], [{**stop_order(qty=80), "filled": 20.0}], quotes={"NVDA": (148.0, False)})
    r = one(check(ledger, b))
    assert r.action == fm.RESIZE_STOP
    fixes = fm.plan_fixes([r], submit=True)
    fm.apply_fixes(fixes, b, ledger)
    # totalQuantity is the whole order at TWS: the 20 filled + the new remaining quantity
    assert b.modified[0][2].totalQuantity == 20 + fixes[0]["new_qty"]


@pytest.mark.parametrize("module", ["fill_monitor", "exit_manager", "order_entry"])
def test_order_clis_refuse_past_the_holiday_table(monkeypatch, module):
    """From 2028 every weekday would read as a trading day: the order modules fail closed."""
    import importlib
    import market_time
    import order_broker
    mod = importlib.import_module(module)
    year = max(h.year for h in market_time.NYSE_HOLIDAYS) + 1
    monkeypatch.setattr(mod, "now_et", lambda: datetime(year, 1, 3, 17, 0))
    monkeypatch.setattr(order_broker, "open_session", lambda *a, **k: pytest.fail("connected"))
    args = ["--account", "DUX1"] + (["--decisions", "/nonexistent.csv", "--plan-only"] if module == "order_entry" else [])
    assert mod.main(args) == 2
    assert market_time.calendar_covers(date(year - 1, 12, 31)) and not market_time.calendar_covers(date(year, 1, 3))


def test_a_runs_alerts_post_one_banner_not_one_per_position(ledger, monkeypatch, caplog):
    # 3 manual positions with no stop: 3 ALERT log lines, but a single (sounded) notification
    import notify
    sent = []
    monkeypatch.setattr(notify, "notify", lambda *a, **k: sent.append(a))
    b = broker([pos("AAPL"), pos("TSLA"), pos("EQT")], [], submit=False,
               quotes={s: (150.0, False) for s in ("AAPL", "TSLA", "EQT")})
    rows, _, _ = fm.run(b, ledger, submit=False, client_id=17, now=RTH, exit_state={})
    assert caplog.text.count("ALERT: ") == len(rows) == 3
    assert len(sent) == 1 and sent[0][0] == "Fill monitor" and sent[0][1].startswith("3 alerts: ")


# ─── TRIM_STOP_MODE split_stop: the trim block's own stop ───

TSTOP_REF = make_order_ref("trim_stop", D1, "NVDA")
TRIM_OCA = ol.make_trim_oca_group(D1, "NVDA")


def test_split_blocks_count_as_covered(ledger):
    filled_trade(ledger)
    tstop = {**stop_order(qty=40, oid=2001, ref=TSTOP_REF, parent=0), "oca_group": TRIM_OCA}
    b = broker([pos()], [stop_order(qty=80), tstop])
    r = one(check(ledger, b))
    assert (r.action, r.protected, r.unprotected) == (fm.OK, 120, 0)


def test_trim_stop_left_after_its_trim_filled_is_the_one_shrunk(ledger):
    """TWS didn't reduce the trim block's stop when the trim filled: 80 held, stops 80 + 40."""
    filled_trade(ledger)
    tstop = {**stop_order(qty=40, oid=2001, ref=TSTOP_REF, parent=0), "oca_group": TRIM_OCA}
    b = broker([pos(shares=80)], [stop_order(qty=80), tstop])
    r = one(check(ledger, b))
    assert r.action == fm.SHRINK_STOP and r.target_order['order_id'] == 2001


def test_main_stop_is_the_one_grown_on_a_partial_cover(ledger):
    filled_trade(ledger)
    tstop = {**stop_order(qty=40, oid=2001, ref=TSTOP_REF, parent=0), "oca_group": TRIM_OCA}
    b = broker([pos()], [stop_order(qty=60), tstop], quotes={"NVDA": (148.0, False)})
    r = one(check(ledger, b))
    assert r.action == fm.RESIZE_STOP and r.target_order['order_id'] == 1044


def test_no_main_stop_places_a_stop_fix_in_the_main_group(ledger):
    """Main stop gone, trim block stop 40 left on 120 held: a stop_fix in BOT-OCA, the trim_stop untouched."""
    filled_trade(ledger)
    ledger.record_intent(LedgerKey(D1, "NVDA", "PIVOT_BREAKOUT"), "trim_stop", TSTOP_REF, 2001, action="SELL",
                         order_type="STP", aux_price=141.0, tif="GTC", oca_group=TRIM_OCA, client_id=16,
                         account=ACCT, qty=40)
    tstop = {**stop_order(qty=40, oid=2001, ref=TSTOP_REF, parent=0), "oca_group": TRIM_OCA}
    b = broker([pos()], [tstop], quotes={"NVDA": (148.0, False)})
    r = one(check(ledger, b))
    assert r.action == fm.PLACE_STOP and r.oca_group == OCA and r.unprotected == 80
