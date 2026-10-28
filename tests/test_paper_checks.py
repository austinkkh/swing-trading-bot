"""scripts/paper_checks.py: gates, 'yes' prompts, cleanup and isolation (offline: a simulated TWS)."""

import importlib.util
import itertools
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

import config
import order_broker as ob
import order_ledger as ol
from helpers import FakeConnector

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("paper_checks", ROOT / "scripts" / "paper_checks.py")
pc = importlib.util.module_from_spec(_spec)
sys.modules["paper_checks"] = pc          # dataclasses look the module up while the file executes
_spec.loader.exec_module(pc)

RTH = datetime(2026, 11, 4, 10, 30)        # Wednesday, regular session (naive ET)
PREOPEN = datetime(2026, 11, 4, 9, 5)


class SimBroker:
    """
    A small simulated paper TWS. Marketable LMT orders fill at once (partial=n fills only n of a
    PARENTB bracket parent); STP orders never trigger (they rest 25%+ away). oca_mode: 'reduce'
    (ocaType 3), 'cancel' or 'ignore' (the two outcomes check 1 must catch). list_staged:
    transmit=False orders show in open orders. partial_child: what a partly filled parent's child
    does when the parent is cancelled ('resize' / 'keep' / 'cancel'); orphan_child: the child stays
    working even when the parent is cancelled unfilled. cancel_fails: ids whose cancel raises;
    cancel_ignored: ids whose cancel is accepted but never takes. delayed: quotes flagged delayed.
    """

    def __init__(self, prices=None, positions=(), open_orders=(), oca_mode="reduce", list_staged=True,
                 partial=None, partial_child="resize", messages=(399,), ack=None, submit_enabled=True,
                 account="DUX1", client_id=16, ref_prefix="PAPERCHECK-", orphan_child=False,
                 cancel_fails=(), cancel_ignored=(), delayed=False, completed_orders=()):
        self.account, self.submit_enabled, self.client_id, self.ref_prefix = account, submit_enabled, client_id, ref_prefix
        self.prices = dict(prices or {"F": 10.0, "T": 25.0, "BAC": 40.0, "PFE": 30.0})
        self._state = ob.BrokerState(positions=[dict(p) for p in positions],
                                     open_orders=[dict(o) for o in open_orders], executions=[],
                                     completed_orders=[dict(o) for o in completed_orders])
        self.orphan_child, self.cancel_fails, self.cancel_ignored = orphan_child, set(cancel_fails), set(cancel_ignored)
        self.delayed = delayed
        self.oca_mode, self.list_staged, self.partial, self.partial_child = oca_mode, list_staged, partial, partial_child
        self.messages, self.ack_override = list(messages), ack
        self.order_messages = {}
        self.hidden = {}            # staged orders TWS doesn't list
        self.placed, self.cancelled = [], []
        self.next_id = 500
        self.closed = False
        self.raise_on_ack = None
        self.exec_n = 0

    # ─── Broker protocol ───
    def state(self):
        return self._state

    def next_order_ids(self, n):
        ids = list(range(self.next_id, self.next_id + n))
        self.next_id += n
        return ids

    def last_price(self, symbol):
        return self.prices.get(symbol), self.delayed

    def daily_bars(self, symbol, duration="6 M", include_forming=False):
        return pd.DataFrame(columns=['date', 'open', 'high', 'low', 'close', 'volume'])

    def place(self, oid, contract, order):
        if not self.submit_enabled:
            raise PermissionError("submission disabled")
        if not str(order.orderRef).startswith(self.ref_prefix):
            raise PermissionError("wrong orderRef prefix")
        order.account = self.account
        self.placed.append((oid, contract.symbol, order))
        row = {"order_id": oid, "perm_id": oid * 10, "client_id": self.client_id, "account": self.account,
               "symbol": contract.symbol, "sec_type": "STK", "action": order.action,
               "order_type": order.orderType, "quantity": float(order.totalQuantity), "filled": 0.0,
               "aux_price": order.auxPrice, "lmt_price": order.lmtPrice, "status": "PreSubmitted",
               "order_ref": order.orderRef, "parent_id": order.parentId, "tif": order.tif,
               "oca_group": order.ocaGroup, "transmit": order.transmit}
        if not order.transmit:
            (self._state.open_orders if self.list_staged else self.hidden.setdefault('x', [])).append(row)
            return
        self._state.open_orders.append(row)
        if self.messages and order.orderType == "STP LMT":
            self.order_messages[oid] = [{"code": c, "msg": f"code {c}"} for c in self.messages]
        if order.parentId:
            parent = self._find(order.parentId)
            if parent:
                self._try_fill(parent)
        else:
            self._try_fill(row)

    def _find(self, oid):
        for o in self._state.open_orders + self.hidden.get('x', []):
            if o["order_id"] == oid:
                return o
        return None

    def _pos(self, symbol, delta):
        for p in self._state.positions:
            if p["symbol"] == symbol and p["account"] == self.account:
                p["shares"] += delta
                return
        self._state.positions.append({"account": self.account, "symbol": symbol, "sec_type": "STK",
                                      "shares": float(delta), "avg_cost": self.prices.get(symbol, 0)})

    def _complete(self, row, status):
        if row in self._state.open_orders:
            self._state.open_orders.remove(row)
        self._state.completed_orders.append({**row, "status": status})

    def _try_fill(self, row):
        price = self.prices[row["symbol"]]
        if row["order_type"] != "LMT":
            return
        marketable = row["lmt_price"] >= price if row["action"] == "BUY" else row["lmt_price"] <= price
        if not marketable:
            return
        qty = row["quantity"] - row["filled"]
        if self.partial and "PARENTB" in row["order_ref"]:
            qty = min(qty, self.partial)
        self.exec_n += 1
        self._state.executions.append({"exec_id": f"e{self.exec_n}", "order_id": row["order_id"],
                                       "order_ref": row["order_ref"], "symbol": row["symbol"],
                                       "side": "BOT" if row["action"] == "BUY" else "SLD", "shares": qty,
                                       "price": price, "account": self.account})
        self._pos(row["symbol"], qty if row["action"] == "BUY" else -qty)
        row["filled"] += qty
        if row["filled"] >= row["quantity"]:
            self._complete(row, "Filled")
        if row["oca_group"]:
            for other in list(self._state.open_orders):
                if other is row or other["oca_group"] != row["oca_group"]:
                    continue
                if self.oca_mode == "cancel":
                    self._complete(other, "Cancelled")
                elif self.oca_mode == "ignore":
                    continue
                else:
                    other["quantity"] -= qty
                    if other["quantity"] <= 0:
                        self._complete(other, "Cancelled")

    def cancel(self, oid):
        if not self.submit_enabled:
            raise PermissionError("submission disabled")
        if oid in self.cancel_fails:
            raise RuntimeError(f"cancel {oid}: TWS connection dropped")
        self.cancelled.append(oid)
        row = self._find(oid)
        if row is None or oid in self.cancel_ignored:
            return
        if row in self.hidden.get('x', []):
            self.hidden['x'].remove(row)
            self._state.completed_orders.append({**row, "status": "Cancelled"})
        else:
            self._complete(row, "Cancelled")
        for child in [o for o in self._state.open_orders if o.get("parent_id") == oid]:
            if row["filled"] <= 0 and self.orphan_child:
                continue
            if row["filled"] <= 0 or self.partial_child == "cancel":
                self._complete(child, "Cancelled")
            elif self.partial_child == "resize":
                child["quantity"] = row["filled"]

    def wait_ack(self, oid, timeout=10.0):
        if self.raise_on_ack:
            raise self.raise_on_ack
        if self.ack_override:
            return self.ack_override
        return ("working", "PreSubmitted")

    def close(self):
        self.closed = True


def args(*extra):
    return pc.parse_args(["--account", "DUX1", "--port", "7497", *extra])


def go(monkeypatch, tmp_path, broker, argv=("--submit",), answers=None, now=RTH, sleep=None, auto=True):
    """Run paper_checks with the sim; answers: list of replies (default: always yes)."""
    monkeypatch.setattr(config, "AUTO_SUBMIT_ENABLED", auto, raising=False)
    calls = {"factory": [], "prompts": [], "out": []}

    def factory(port, account, cli_submit, client_id, ref_prefix):
        calls["factory"].append(dict(port=port, account=account, cli_submit=cli_submit, client_id=client_id,
                                     ref_prefix=ref_prefix))
        broker.submit_enabled = broker.submit_enabled and cli_submit
        return broker

    replies = iter(answers) if answers is not None else itertools.repeat("yes")

    def ask(prompt):
        calls["prompts"].append(prompt)
        return next(replies, "no")

    clock = itertools.count(0, 1.0)
    rc = pc.run(args(*argv), broker_factory=factory, ask=ask, out=calls["out"].append,
                sleep=sleep or (lambda s: None), clock=lambda: next(clock), now_fn=lambda: now,
                orders_dir=tmp_path, fill_timeout=5, settle_sec=0)
    return rc, calls


def results(tmp_path, day="2026-11-04"):
    data = json.loads((tmp_path / f"paper_checks_{day}.json").read_text())
    return {r["check"]: r for r in data["runs"][-1]["results"]}, data["runs"][-1]


def own_open(broker):
    return [o for o in broker.state().open_orders if str(o["order_ref"]).startswith("PAPERCHECK-")]


def position(broker, sym):
    return sum(p["shares"] for p in broker.state().positions if p["symbol"] == sym)


# ─── gates ───

def test_explain_never_connects(monkeypatch, tmp_path):
    def factory(*a, **k):
        raise AssertionError("must not connect")
    out = []
    assert pc.run(pc.parse_args(["--explain"]), broker_factory=factory, out=out.append, orders_dir=tmp_path) == 0
    text = "\n".join(out)
    assert "Check 1" in text and "Check 5" in text and "PAPERCHECK" in text
    assert not list(tmp_path.glob("paper_checks_*.json"))


@pytest.mark.parametrize("port", [7496, 4001])
def test_live_port_refused_before_connecting(monkeypatch, tmp_path, port):
    broker = SimBroker()
    rc, calls = go(monkeypatch, tmp_path, broker, argv=("--submit", "--port", str(port)))
    assert rc == 2 and calls["factory"] == [] and broker.placed == []
    assert any("LIVE" in line for line in calls["out"])


@pytest.mark.parametrize("auto,argv", [(False, ("--submit",)), (True, ()), ("True", ("--submit",))])
def test_refused_without_both_gates(monkeypatch, tmp_path, auto, argv):
    broker = SimBroker()
    rc, calls = go(monkeypatch, tmp_path, broker, argv=argv, auto=auto)
    assert rc == 2 and calls["factory"] == [] and broker.placed == []
    assert any("AUTO_SUBMIT_ENABLED" in line for line in calls["out"])


def test_non_paper_account_refused(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "AUTO_SUBMIT_ENABLED", True, raising=False)
    out = []
    rc = pc.run(pc.parse_args(["--account", "U123", "--submit"]), broker_factory=lambda *a, **k: 1 / 0,
                out=out.append, orders_dir=tmp_path, now_fn=lambda: RTH)
    assert rc == 2


def test_weekend_refused(monkeypatch, tmp_path):
    broker = SimBroker()
    rc, calls = go(monkeypatch, tmp_path, broker, now=datetime(2026, 11, 7, 10, 30))
    assert rc == 2 and calls["factory"] == []


def test_session_uses_order_client_and_paper_check_prefix(monkeypatch, tmp_path):
    broker = SimBroker()
    rc, calls = go(monkeypatch, tmp_path, broker, argv=("--submit", "--check", "4"))
    assert rc == 0
    assert calls["factory"] == [dict(port=7497, account="DUX1", cli_submit=True, client_id=16,
                                     ref_prefix="PAPERCHECK-")]
    assert broker.closed


def test_ledger_lock_held_exits_3(monkeypatch, tmp_path):
    with ol.ledger_lock(tmp_path):
        rc, calls = go(monkeypatch, tmp_path, SimBroker(), argv=("--submit", "--check", "4"))
    assert rc == 3 and calls["factory"] == []


def test_order_broker_session_prefix_is_exclusive(monkeypatch):
    """open_session(ref_prefix=PAPERCHECK-) places only PAPERCHECK orders; a bot session never one."""
    from ibapi.client import EClient
    from ibapi.order import Order
    sent = []
    monkeypatch.setattr(EClient, "placeOrder", lambda self, *a, **k: sent.append(a))
    monkeypatch.setattr(config, "AUTO_SUBMIT_ENABLED", True, raising=False)

    def order(ref):
        o = Order()
        o.action, o.orderType, o.totalQuantity, o.orderRef = "BUY", "LMT", 1, ref
        return o
    chk = ob.open_session(7497, "DUX1", True, 16, connector_factory=FakeConnector(), ref_prefix="PAPERCHECK-")
    bot = ob.open_session(7497, "DUX1", True, 16, connector_factory=FakeConnector())
    with pytest.raises(PermissionError):
        chk.place(1, ob.stock_contract("F"), order("BOT-ENTRY-20261104-F"))
    with pytest.raises(PermissionError):
        bot.place(2, ob.stock_contract("F"), order("PAPERCHECK-4-STAGED-x-F"))
    chk.place(3, ob.stock_contract("F"), order("PAPERCHECK-4-STAGED-x-F"))
    bot.place(4, ob.stock_contract("F"), order("BOT-ENTRY-20261104-F"))
    assert [a[0] for a in sent] == [3, 4]
    with pytest.raises(PermissionError):
        ob.open_session(7497, "DUX1", True, 16, connector_factory=FakeConnector(), ref_prefix="MANUAL-")
    with pytest.raises(PermissionError, match="LIVE"):
        ob.open_session(7496, "DUX1", True, 16, connector_factory=FakeConnector(), ref_prefix="PAPERCHECK-")


# ─── 'yes' required ───

@pytest.mark.parametrize("reply", ["no", "y", "YES please", ""])
def test_nothing_sent_without_yes(monkeypatch, tmp_path, reply):
    broker = SimBroker()
    rc, calls = go(monkeypatch, tmp_path, broker, answers=itertools.repeat(reply))
    assert rc == 0 and broker.placed == [] and broker.cancelled == []
    res, run = results(tmp_path)
    assert {n: r["result"] for n, r in res.items()} == {n: "INCONCLUSIVE" for n in (1, 2, 3, 4, 5)}
    assert run["symbol"] == "F"


def test_yes_accepts_case_and_spaces(monkeypatch, tmp_path):
    broker = SimBroker()
    rc, _ = go(monkeypatch, tmp_path, broker, argv=("--submit", "--check", "4"), answers=[" Yes ", "yes"])
    assert rc == 0 and len(broker.placed) == 1


def test_order_size_capped():
    s = pc.CheckSession(SimBroker(), "F", "t", ask=lambda p: "yes", out=lambda t: None)
    with pytest.raises(ValueError):
        s.place(1, pc.OrderSpec("buy", "BUY", pc.MAX_SHARES + 1, "LMT", lmt=10.3))


# ─── checks ───

def test_check1_pass_when_oca_reduces_and_cleanup_flattens(monkeypatch, tmp_path):
    broker = SimBroker(oca_mode="reduce")
    rc, _ = go(monkeypatch, tmp_path, broker, argv=("--submit", "--check", "1"))
    res, _ = results(tmp_path)
    assert rc == 0 and res[1]["result"] == "PASS", res[1]["summary"]
    assert res[1]["observations"]["stop_after_trim"]["remaining"] == 2
    buy, stop, trim = (o for _, _, o in broker.placed[:3])
    assert (buy.action, buy.orderType, buy.totalQuantity, buy.lmtPrice) == ("BUY", "LMT", 3, 10.3)
    assert (stop.orderType, stop.auxPrice, stop.tif, stop.ocaType) == ("STP", 7.5, "GTC", 3)
    assert trim.ocaGroup == stop.ocaGroup and trim.totalQuantity == 1
    assert own_open(broker) == [] and position(broker, "F") == 0     # stop cancelled, 2 sold


def test_check1_fail_when_trim_cancels_the_stop(monkeypatch, tmp_path):
    broker = SimBroker(oca_mode="cancel")
    go(monkeypatch, tmp_path, broker, argv=("--submit", "--check", "1"))
    res, _ = results(tmp_path)
    assert res[1]["result"] == "FAIL" and "CANCELLED" in res[1]["summary"]
    assert position(broker, "F") == 0


def test_check1_needs_regular_session(monkeypatch, tmp_path):
    broker = SimBroker()
    go(monkeypatch, tmp_path, broker, argv=("--submit", "--check", "1"), now=PREOPEN)
    res, _ = results(tmp_path)
    assert res[1]["result"] == "INCONCLUSIVE" and broker.placed == []


@pytest.mark.parametrize("listed,expect", [(True, "PASS"), (False, "FAIL")])
def test_check4_staged_parent(monkeypatch, tmp_path, listed, expect):
    broker = SimBroker(list_staged=listed)
    go(monkeypatch, tmp_path, broker, argv=("--submit", "--check", "4"))
    res, _ = results(tmp_path)
    assert res[4]["result"] == expect
    (oid, _, order), = broker.placed
    assert order.transmit is False and order.lmtPrice == 7.5 and oid in broker.cancelled


def test_check5_preopen_399_is_routine(monkeypatch, tmp_path):
    broker = SimBroker(messages=(399,))
    go(monkeypatch, tmp_path, broker, argv=("--submit", "--check", "5"), now=PREOPEN)
    res, _ = results(tmp_path)
    assert res[5]["result"] == "PASS" and "399 seen" in res[5]["summary"]
    (oid, _, order), = broker.placed
    assert (order.orderType, order.auxPrice, order.lmtPrice, order.tif) == ("STP LMT", 12.5, 12.75, "DAY")
    assert oid in broker.cancelled and own_open(broker) == []


def test_check5_rejection_fails(monkeypatch, tmp_path):
    broker = SimBroker(messages=(201,), ack=("rejected", "TWS 201: rejected"))
    go(monkeypatch, tmp_path, broker, argv=("--submit", "--check", "5"), now=PREOPEN)
    assert results(tmp_path)[0][5]["result"] == "FAIL"


def test_check5_skipped_after_cutoff(monkeypatch, tmp_path):
    broker = SimBroker()
    go(monkeypatch, tmp_path, broker, argv=("--submit", "--check", "5"), now=datetime(2026, 11, 4, 9, 29, 45))
    res, _ = results(tmp_path)
    assert res[5]["result"] == "INCONCLUSIVE" and broker.placed == []


def test_check3_part_a_and_partial_fill_resized_child(monkeypatch, tmp_path):
    broker = SimBroker(partial=2, partial_child="resize")
    go(monkeypatch, tmp_path, broker, argv=("--submit", "--check", "3"))
    res, _ = results(tmp_path)
    assert res[3]["result"] == "PASS", res[3]["summary"]
    assert res[3]["observations"]["b_filled"] == 2
    assert own_open(broker) == [] and position(broker, "F") == 0


def test_check3_oversized_child_fails(monkeypatch, tmp_path):
    broker = SimBroker(partial=2, partial_child="keep")
    go(monkeypatch, tmp_path, broker, argv=("--submit", "--check", "3"))
    assert results(tmp_path)[0][3]["result"] == "FAIL"
    assert own_open(broker) == [] and position(broker, "F") == 0


def test_check2_read_only_without_gates(monkeypatch, tmp_path):
    broker = SimBroker()
    rc, calls = go(monkeypatch, tmp_path, broker, argv=("--check", "2"), auto=False)
    assert rc == 0 and calls["factory"][0]["cli_submit"] is False and calls["factory"][0]["client_id"] == 17
    assert results(tmp_path)[0][2]["result"] == "INCONCLUSIVE" and broker.placed == []


def _ledger_rows(stop_filled, moo_filled, moo_px=9.0, stop_px=9.5):
    base = dict(trade_date="2026-11-03", symbol="NVDA", setup="PIVOT_BREAKOUT", oca_group="BOT-OCA-20261103-NVDA")
    return pd.DataFrame([
        {**base, "role": "stop", "order_ref": "BOT-STOP-20261103-NVDA", "tif": "GTC", "filled_qty": stop_filled,
         "aux_price": stop_px, "avg_fill_price": None},
        {**base, "role": "exit", "order_ref": "BOT-EXIT-20261103-NVDA", "tif": "OPG", "filled_qty": moo_filled,
         "aux_price": None, "avg_fill_price": moo_px}])


def test_observe_gap_outcomes():
    assert pc.observe_gap(_ledger_rows(10, 10), [])[0] == "FAIL"
    assert pc.observe_gap(_ledger_rows(0, 10), [])[0] == "PASS"
    assert pc.observe_gap(_ledger_rows(0, 10, moo_px=10.0), [])[0] == "INCONCLUSIVE"   # no gap below the stop
    assert pc.observe_gap(pd.DataFrame(columns=ol.VIEW_COLUMNS), [])[0] == "INCONCLUSIVE"
    execs = [{"exec_id": "1", "side": "SLD", "order_ref": "BOT-STOP-20261103-NVDA", "shares": 10},
             {"exec_id": "2", "side": "SLD", "order_ref": "BOT-EXIT-20261103-NVDA", "shares": 10}]
    assert pc.observe_gap(pd.DataFrame(columns=ol.VIEW_COLUMNS), execs)[0] == "FAIL"


# ─── cleanup on Ctrl-C / errors ───

def test_ctrl_c_mid_check_cancels_own_orders_and_flattens(monkeypatch, tmp_path):
    broker = SimBroker()
    calls = {"n": 0}

    def sleep(_):
        # Ctrl-C while check 1 waits for the trim fill (after the BUY filled and the stop is working)
        if len(broker.placed) >= 2 and calls["n"] == 0:
            calls["n"] += 1
            raise KeyboardInterrupt
    broker.prices["F"] = 10.0
    # trim never fills: make it non-marketable by moving the price up after the buy
    orig = broker._try_fill

    def try_fill(row):
        if "TRIM" in row["order_ref"]:
            return
        orig(row)
    broker._try_fill = try_fill
    rc, _ = go(monkeypatch, tmp_path, broker, argv=("--submit", "--check", "1"), sleep=sleep)
    assert rc == 130
    res, run = results(tmp_path)
    assert run["interrupted"] is True and res[1]["result"] == "INCONCLUSIVE"
    placed_ids = {oid for oid, _, _ in broker.placed}
    assert set(broker.cancelled) <= placed_ids
    assert own_open(broker) == [] and position(broker, "F") == 0


def test_exception_cancels_own_orders(monkeypatch, tmp_path):
    broker = SimBroker()
    broker.raise_on_ack = RuntimeError("TWS dropped")
    rc, _ = go(monkeypatch, tmp_path, broker, argv=("--submit", "--check", "1"))
    res, _ = results(tmp_path)
    assert rc == 0 and res[1]["result"] == "INCONCLUSIVE" and "TWS dropped" in res[1]["summary"]
    stop_id = broker.placed[1][0]
    assert stop_id in broker.cancelled and own_open(broker) == [] and position(broker, "F") == 0


def test_declined_flatten_leaves_shares_and_says_so(monkeypatch, tmp_path):
    broker = SimBroker()
    # yes to check 1, then no to the cleanup sell (every later prompt: no)
    rc, calls = go(monkeypatch, tmp_path, broker, argv=("--submit", "--check", "1"), answers=["yes"])
    assert position(broker, "F") == 2 and own_open(broker) == []
    assert any("sell it by hand" in line for line in calls["out"])


# ─── never touches other orders / positions ───

def test_other_orders_and_positions_untouched(monkeypatch, tmp_path):
    manual_f = {"order_id": 7, "client_id": 0, "account": "DUX1", "symbol": "F", "action": "BUY",
                "order_type": "LMT", "quantity": 100.0, "order_ref": "", "status": "Submitted", "oca_group": ""}
    bot_stop = {"order_id": 42, "client_id": 16, "account": "DUX1", "symbol": "NVDA", "action": "SELL",
                "order_type": "STP", "quantity": 10.0, "order_ref": "BOT-STOP-20261103-NVDA",
                "status": "Submitted", "oca_group": "BOT-OCA-20261103-NVDA"}
    held = [{"account": "DUX1", "symbol": "T", "sec_type": "STK", "shares": 50.0, "avg_cost": 25.0},
            {"account": "DUX1", "symbol": "NVDA", "sec_type": "STK", "shares": 10.0, "avg_cost": 100.0}]
    broker = SimBroker(positions=held, open_orders=[manual_f, bot_stop])
    rc, _ = go(monkeypatch, tmp_path, broker)
    _, run = results(tmp_path)
    assert run["symbol"] == "BAC"                     # F has an order, T a position
    assert {o for _, sym, o in broker.placed} and all(sym == "BAC" for _, sym, _ in broker.placed)
    assert 7 not in broker.cancelled and 42 not in broker.cancelled
    assert set(broker.cancelled) <= {oid for oid, _, _ in broker.placed}
    assert manual_f in broker.state().open_orders and bot_stop in broker.state().open_orders
    assert position(broker, "T") == 50 and position(broker, "NVDA") == 10 and position(broker, "BAC") == 0


def test_cancel_refuses_an_id_it_did_not_place():
    s = pc.CheckSession(SimBroker(), "F", "t", ask=lambda p: "yes", out=lambda t: None)
    with pytest.raises(PermissionError):
        s.cancel(42)


def test_flatten_never_sells_more_than_its_own_fills(monkeypatch):
    broker = SimBroker(positions=[{"account": "DUX1", "symbol": "F", "sec_type": "STK", "shares": 100.0,
                                   "avg_cost": 9.0}])
    s = pc.CheckSession(broker, "F", "t", ask=lambda p: "yes", out=lambda t: None, now_fn=lambda: RTH,
                        sleep=lambda x: None, fill_timeout=1, settle_sec=0)
    assert s.flatten()["result"] == "flat" and broker.placed == []    # owner's 100 shares: not ours
    s.place(1, pc.OrderSpec("buy", "BUY", 3, "LMT", lmt=10.3))
    out = s.flatten()
    assert out["result"] == "sold" and out["qty"] == 3 and position(broker, "F") == 100


def test_leftover_paper_check_orders_need_yes(monkeypatch, tmp_path):
    left = {"order_id": 300, "client_id": 16, "account": "DUX1", "symbol": "PFE", "action": "BUY",
            "order_type": "LMT", "quantity": 1.0, "order_ref": "PAPERCHECK-4-STAGED-x-PFE",
            "status": "Submitted", "oca_group": ""}
    broker = SimBroker(open_orders=[left])
    rc, _ = go(monkeypatch, tmp_path, broker, answers=["no"])
    assert rc == 2 and broker.placed == [] and broker.cancelled == []
    broker2 = SimBroker(open_orders=[left])
    rc, _ = go(monkeypatch, tmp_path, broker2, argv=("--submit", "--check", "4"), answers=["yes", "yes", "yes"])
    assert rc == 0 and 300 in broker2.cancelled


# ─── review fixes (2026-10-26) ───

def test_check1_fail_when_stop_not_reduced(monkeypatch, tmp_path):
    broker = SimBroker(oca_mode="ignore")
    go(monkeypatch, tmp_path, broker, argv=("--submit", "--check", "1"))
    res, _ = results(tmp_path)
    assert res[1]["result"] == "FAIL" and "still 3" in res[1]["summary"]
    assert own_open(broker) == [] and position(broker, "F") == 0


def test_check3_part_a_fail_when_child_orphaned(monkeypatch, tmp_path):
    broker = SimBroker(orphan_child=True)
    go(monkeypatch, tmp_path, broker, argv=("--submit", "--check", "3"))
    res, _ = results(tmp_path)
    assert res[3]["result"] == "FAIL" and "part A" in res[3]["summary"]
    assert own_open(broker) == [] and broker.placed[2:] == []      # part B never sent


def _session_with_stop(broker):
    s = pc.CheckSession(broker, "F", "t", ask=lambda p: "yes", out=lambda t: None, now_fn=lambda: RTH,
                        sleep=lambda x: None, clock=itertools.count(0, 1.0).__next__, fill_timeout=5, settle_sec=0)
    s.place(1, pc.OrderSpec("buy", "BUY", 3, "LMT", lmt=10.3))
    stop = s.place(1, pc.OrderSpec("stop", "SELL", 3, "STP", aux=7.5, tif="GTC"))
    return s, stop


@pytest.mark.parametrize("mode", ["fails", "ignored"])
def test_cleanup_never_sells_while_own_stop_may_be_live(mode):
    """A stop whose cancel raised, or was accepted but never took: no flatten SELL beside it."""
    broker = SimBroker()
    broker.next_id = 500
    s, stop = _session_with_stop(broker)
    (broker.cancel_fails if mode == "fails" else broker.cancel_ignored).add(stop)
    record = s.cleanup("test")
    assert record["flatten"]["result"] == "own_order_still_live" and stop in record["flatten"]["order_ids"]
    assert [o.action for _, _, o in broker.placed] == ["BUY", "SELL"]      # no flatten order
    assert position(broker, "F") == 3
    assert any("NOT selling" in line for line in s.log)


def test_cleanup_flattens_once_stop_is_cancelled():
    broker = SimBroker()
    s, stop = _session_with_stop(broker)
    record = s.cleanup("test")
    assert record["cancelled"] == [stop] and record["flatten"]["result"] == "sold"
    assert position(broker, "F") == 0


def test_no_intraday_quote_sends_no_fill_wanted_orders(monkeypatch, tmp_path):
    broker = SimBroker()
    broker.last_price = lambda symbol: (None, True)
    broker.daily_bars = lambda symbol, duration="6 M", include_forming=False: pd.DataFrame({"close": [10.0]})
    go(monkeypatch, tmp_path, broker, argv=("--submit", "--check", "1", "--check", "3"))
    res, _ = results(tmp_path)
    assert res[1]["result"] == "INCONCLUSIVE" and "no intraday quote" in res[1]["summary"]
    assert "part B not sent" in res[3]["summary"]
    assert all(o.action != "BUY" or not o.transmit for _, _, o in broker.placed)   # only part A's staged parent


def test_delayed_quote_is_flagged_before_yes(monkeypatch, tmp_path):
    broker = SimBroker(delayed=True)
    _, calls = go(monkeypatch, tmp_path, broker, argv=("--submit", "--check", "1"))
    assert any("delayed or stale" in line for line in calls["out"])
    assert results(tmp_path)[0][1]["result"] == "PASS"


def test_ids_start_above_working_orders_of_the_same_client(monkeypatch, tmp_path):
    bot_stop = {"order_id": 900, "client_id": 16, "account": "DUX1", "symbol": "NVDA", "action": "SELL",
                "order_type": "STP", "quantity": 10.0, "order_ref": "BOT-STOP-20261103-NVDA",
                "status": "Submitted", "oca_group": ""}
    broker = SimBroker(open_orders=[bot_stop])         # sim's nextValidId (500) is below the bot's 900
    go(monkeypatch, tmp_path, broker, argv=("--submit", "--check", "4"))
    assert [oid for oid, _, _ in broker.placed] == [901] and 900 not in broker.cancelled


def test_cancel_refuses_an_id_tws_lists_under_a_bot_ref():
    broker = SimBroker()
    s = pc.CheckSession(broker, "F", "t", ask=lambda p: "yes", out=lambda t: None)
    s.placed[42] = {"ref": "PAPERCHECK-4-STAGED-t-F", "spec": {}, "check": 4, "cancel_requested": False}
    broker._state.open_orders.append({"order_id": 42, "client_id": 16, "symbol": "NVDA", "action": "SELL",
                                      "order_ref": "BOT-STOP-20261103-NVDA", "status": "Submitted"})
    assert s.cancel(42, broker.state()) is False and broker.cancelled == []


def test_leftover_with_shares_on_another_symbol_is_reported(monkeypatch, tmp_path):
    left_stop = {"order_id": 300, "client_id": 16, "account": "DUX1", "symbol": "F", "action": "SELL",
                 "order_type": "STP", "quantity": 2.0, "filled": 0.0, "order_ref": "PAPERCHECK-1-STOP-x-F",
                 "status": "Submitted", "oca_group": ""}
    held = [{"account": "DUX1", "symbol": "F", "sec_type": "STK", "shares": 2.0, "avg_cost": 10.0}]
    broker = SimBroker(open_orders=[left_stop], positions=held)
    rc, calls = go(monkeypatch, tmp_path, broker, argv=("--submit", "--check", "4"))
    _, run = results(tmp_path)
    assert rc == 0 and run["symbol"] == "T" and 300 in broker.cancelled
    assert run["leftover_positions"] == {"F": 2.0} and position(broker, "F") == 2      # never sold by the run
    assert any("NO stop now" in line for line in calls["out"])
