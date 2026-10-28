"""Order broker paper guard and submission switch (offline: no socket is ever opened)."""

from types import SimpleNamespace

import pytest
from ibapi.client import EClient

import config
import order_broker as ob
from helpers import FakeConnector


@pytest.fixture
def sent(monkeypatch):
    """Record what would reach the socket; the real IBKRConnector read-only guard runs first."""
    calls = []
    monkeypatch.setattr(EClient, "placeOrder", lambda self, *a, **k: calls.append(a))
    monkeypatch.setattr(EClient, "cancelOrder", lambda self, *a, **k: calls.append(("cancel",) + a))
    return calls


def bot_order(ref="BOT-ENTRY-20261102-NVDA"):
    from ibapi.order import Order
    o = Order()
    o.action, o.orderType, o.totalQuantity, o.orderRef = "BUY", "STP LMT", 10, ref
    return o


def session(monkeypatch, auto=True, cli=True, accounts=("DUX1",), account="DUX1", port=7497):
    monkeypatch.setattr(config, "AUTO_SUBMIT_ENABLED", auto, raising=False)
    return ob.open_session(port, account, cli, 16, connector_factory=FakeConnector(accounts=accounts))


@pytest.mark.parametrize("port", [7496, 4001])
def test_live_ports_refused_before_connecting(monkeypatch, port):
    factory = FakeConnector()
    monkeypatch.setattr(config, "AUTO_SUBMIT_ENABLED", True, raising=False)
    with pytest.raises(PermissionError, match="LIVE"):
        ob.open_session(port, "DUX1", True, 16, connector_factory=factory)
    assert factory.instances == []          # no connector was even built


def test_unknown_port_refused(monkeypatch):
    with pytest.raises(PermissionError, match="not a paper port"):
        session(monkeypatch, port=5000)


def test_non_paper_account_in_session_refused(monkeypatch):
    with pytest.raises(PermissionError, match="Non-paper"):
        session(monkeypatch, accounts=("DUX1", "U123"))


def test_requested_account_must_be_in_session(monkeypatch):
    with pytest.raises(PermissionError, match="not in this TWS session"):
        session(monkeypatch, account="DUX2")
    with pytest.raises(PermissionError, match="paper"):
        session(monkeypatch, account="U999")
    with pytest.raises(PermissionError):
        session(monkeypatch, account="")


def test_config_off_with_submit_flag_is_read_only(monkeypatch, sent):
    b = session(monkeypatch, auto=False, cli=True)
    assert b.connector.read_only and not b.submit_enabled
    with pytest.raises(PermissionError):
        b.place(1, ob.stock_contract("NVDA"), bot_order())
    with pytest.raises(PermissionError):         # the connector itself refuses too
        b.connector.placeOrder(1, ob.stock_contract("NVDA"), bot_order())
    assert sent == []


def test_config_on_without_submit_flag_is_read_only(monkeypatch, sent):
    b = session(monkeypatch, auto=True, cli=False)
    assert b.connector.read_only and not b.submit_enabled
    with pytest.raises(PermissionError):
        b.cancel(5)
    assert sent == []


@pytest.mark.parametrize("value", ["True", 1, "yes"])
def test_truthy_non_bool_config_does_not_count(monkeypatch, value):
    monkeypatch.setattr(config, "AUTO_SUBMIT_ENABLED", value, raising=False)
    assert ob.submission_enabled(True) is False


def test_missing_config_key_means_off(monkeypatch):
    monkeypatch.delattr(config, "AUTO_SUBMIT_ENABLED", raising=False)
    assert ob.submission_enabled(True) is False


def test_both_gates_allow_a_bot_order_and_stamp_the_account(monkeypatch, sent):
    b = session(monkeypatch)
    order = bot_order()
    b.place(7, ob.stock_contract("NVDA"), order)
    assert len(sent) == 1 and order.account == "DUX1"
    assert not b.connector.read_only and b.connector.connect_calls[0][1:] == (7497, 16)


def test_non_bot_order_ref_refused(monkeypatch, sent):
    b = session(monkeypatch)
    for ref in ("", "ALERT", "manual"):
        with pytest.raises(PermissionError, match="orderRef"):
            b.place(7, ob.stock_contract("NVDA"), bot_order(ref))
    assert sent == []


def state(positions=(), orders=()):
    return ob.BrokerState(positions=list(positions), open_orders=list(orders))


def test_oversell_guard():
    pos = [{"symbol": "NVDA", "shares": 100, "sec_type": "STK", "account": "DUX1"}]
    stop = {"symbol": "NVDA", "action": "SELL", "quantity": 100, "status": "Submitted", "perm_id": 9}
    ob.assert_no_oversell(state(pos), "NVDA", 100)
    with pytest.raises(ValueError, match="oversell"):
        ob.assert_no_oversell(state(pos, [stop, dict(stop)]), "NVDA", 34)   # repeat counted once
    ob.assert_no_oversell(state(pos, [{**stop, "status": "Cancelled"}]), "NVDA", 100)
    with pytest.raises(ValueError):
        ob.assert_no_oversell(state(), "NVDA", 10)                          # no position
    ob.assert_no_oversell(state(), "NVDA", 10, pending_buy_qty=10)          # new bracket child


def test_oversell_guard_pools_one_oca_group():
    """exit_manager: a partial-trim LMT beside the full-size stop in the same OCA group is one exposure."""
    pos = [{"symbol": "NVDA", "shares": 300, "sec_type": "STK", "account": "DUX1"}]
    stop = {"symbol": "NVDA", "action": "SELL", "quantity": 300, "status": "Submitted", "perm_id": 9,
            "oca_group": "BOT-OCA-20261102-NVDA"}
    with pytest.raises(ValueError):
        ob.assert_no_oversell(state(pos, [stop]), "NVDA", 100)                     # default: summed
    ob.assert_no_oversell(state(pos, [stop]), "NVDA", 100, oca_group="BOT-OCA-20261102-NVDA")
    ob.assert_no_oversell(state(pos, [stop]), "NVDA", 300, oca_group="BOT-OCA-20261102-NVDA")
    with pytest.raises(ValueError):                                                  # bigger than the position
        ob.assert_no_oversell(state(pos, [stop]), "NVDA", 301, oca_group="BOT-OCA-20261102-NVDA")
    manual = {"symbol": "NVDA", "action": "SELL", "quantity": 50, "status": "Submitted", "perm_id": 10}
    with pytest.raises(ValueError, match="oversell"):                                # manual SELL still adds
        ob.assert_no_oversell(state(pos, [stop, manual]), "NVDA", 100, oca_group="BOT-OCA-20261102-NVDA")
    with pytest.raises(ValueError):                                                  # another group adds
        ob.assert_no_oversell(state(pos, [stop]), "NVDA", 100, oca_group="BOT-OCA-20261103-NVDA")


def test_daily_bars_drop_the_forming_bar_unless_asked(monkeypatch):
    from datetime import datetime
    bars = [SimpleNamespace(date=d, open=10, high=11, low=9, close=10.5, volume=1e6)
            for d in ("20261103", "20261104")]
    conn = SimpleNamespace(read_only=True, fetch_historical_data_ex=lambda *a, **k: (bars, "ok"))
    b = ob.IBKRBroker(conn, "DUX1", submit_enabled=False, client_id=17)
    monkeypatch.setattr(ob, "now_et", lambda: datetime(2026, 11, 4, 12, 30))
    assert b.daily_bars("NVDA")['date'].dt.date.astype(str).tolist() == ["2026-11-03"]
    assert len(b.daily_bars("NVDA", include_forming=True)) == 2
    monkeypatch.setattr(ob, "now_et", lambda: datetime(2026, 11, 4, 16, 30))
    assert len(b.daily_bars("NVDA")) == 2
    conn.fetch_historical_data_ex = lambda *a, **k: (None, "timeout")
    assert b.daily_bars("NVDA").empty


def test_last_price_flags_stale_or_delayed_bars(monkeypatch):
    from datetime import datetime
    bars = [SimpleNamespace(date="20261104 12:28:00 US/Eastern", close=150.25)]
    conn = SimpleNamespace(data_delayed=False, read_only=True,
                           fetch_historical_data_ex=lambda *a, **k: (bars, "ok"))
    b = ob.IBKRBroker(conn, "DUX1", submit_enabled=False, client_id=17)
    monkeypatch.setattr(ob, "now_et", lambda: datetime(2026, 11, 4, 12, 30))
    assert b.last_price("NVDA") == (150.25, False)
    monkeypatch.setattr(ob, "now_et", lambda: datetime(2026, 11, 4, 12, 45))   # 16 min old
    assert b.last_price("NVDA") == (150.25, True)
    conn.data_delayed = True
    monkeypatch.setattr(ob, "now_et", lambda: datetime(2026, 11, 4, 12, 30))
    assert b.last_price("NVDA")[1] is True
    conn.fetch_historical_data_ex = lambda *a, **k: (None, "timeout")
    assert b.last_price("NVDA") == (None, True)


# ─── wait_ack through a real (offline) IBKRConnector ───
# FakeBroker returns a constant ack; these drive IBKRBroker.wait_ack with the connector's own
# error() / orderStatus() callbacks, ibapi 10.x argument order (reqId, errorTime, code, msg).

ERR_TIME = 1_793_000_000_000


def _status(conn, oid, status):
    conn.orderStatus(oid, status, 0, 10, 0.0, 555, 0, 0.0, 16, "", 0.0)


def test_399_warning_then_presubmitted_is_working(monkeypatch, sent):
    b = session(monkeypatch)
    b.place(7, ob.stock_contract("NVDA"), bot_order())
    b.connector.error(7, ERR_TIME, 399, "Order Message: Warning: your order will not be placed at the "
                                        "exchange until 2026-11-02 09:30:00 US/Eastern")
    _status(b.connector, 7, "PreSubmitted")
    assert b.wait_ack(7, timeout=0.5, poll=0.01) == ("working", "PreSubmitted")
    assert 7 not in b.connector.order_errors and 7 not in b.connector.errors


def test_399_warning_alone_is_not_a_rejection(monkeypatch, sent):
    b = session(monkeypatch)
    b.place(7, ob.stock_contract("NVDA"), bot_order())
    b.connector.error(7, ERR_TIME, 399, "Order Message: Warning")
    assert b.wait_ack(7, timeout=0.2, poll=0.01)[0] == "unknown"


@pytest.mark.parametrize("code", [201, 110, 10268])
def test_rejecting_codes_any_size_reject(monkeypatch, sent, code):
    monkeypatch.setattr(ob, "REJECT_GRACE_SEC", 0.02)
    b = session(monkeypatch)
    b.place(7, ob.stock_contract("NVDA"), bot_order())
    b.connector.error(7, ERR_TIME, code, "Order rejected")
    ack, why = b.wait_ack(7, timeout=1.0, poll=0.01)
    assert ack == "rejected" and str(code) in why


def test_rejection_with_cancelled_status_is_immediate(monkeypatch, sent):
    b = session(monkeypatch)
    b.place(7, ob.stock_contract("NVDA"), bot_order())
    b.connector.error(7, ERR_TIME, 201, "Order rejected - reason: margin")
    _status(b.connector, 7, "Cancelled")
    ack, why = b.wait_ack(7, timeout=5, poll=0.01)
    assert ack == "rejected" and "201" in why


def test_request_ids_never_collide_with_order_ids(monkeypatch, sent):
    """A stale 162 for a historical request must not reject order 7 (both used to start at 1)."""
    b = session(monkeypatch)
    rid = b.connector.get_next_req_id()
    assert rid > ob.REQ_ID_BASE
    b.place(7, ob.stock_contract("NVDA"), bot_order())
    b.connector.error(rid, ERR_TIME, 162, "Historical Market Data Service error message")
    assert b.connector.errors[rid]["code"] == 162 and b.connector.order_errors == {}
    _status(b.connector, 7, "PreSubmitted")
    assert b.wait_ack(7, timeout=0.5, poll=0.01)[0] == "working"


def test_other_clients_fills_come_from_executions(monkeypatch):
    """A client-17 run sees client 16's partly filled stop at its unfilled size, not the full size."""
    conn = SimpleNamespace(
        read_only=True, order_status={},
        fetch_account_state=lambda timeout=10: {
            'positions': [], 'account_values': {},
            'open_orders': [{'order_id': 5, 'perm_id': 900, 'client_id': 16, 'quantity': 100.0}]},
        fetch_executions=lambda timeout=10: [
            {'exec_id': 'a', 'perm_id': 900, 'shares': 30.0}, {'exec_id': 'a', 'perm_id': 900, 'shares': 30.0},
            {'exec_id': 'b', 'perm_id': 900, 'shares': 10.0}],
        fetch_completed_orders=lambda timeout=10: [])
    st = ob.IBKRBroker(conn, "DUX1", submit_enabled=False, client_id=17).state()
    assert st.open_orders[0]['filled'] == 40.0


def test_oversell_guard_pools_each_bot_group_once():
    """split_stop: main block (stop 200) + trim block (trim_stop 100, trim 100) = 300, not 400."""
    pos = [{"symbol": "NVDA", "shares": 300, "sec_type": "STK", "account": "DUX1"}]
    main, trim_g = "BOT-OCA-20261102-NVDA", "BOT-OCAT-20261102-NVDA"
    orders = [{"symbol": "NVDA", "action": "SELL", "quantity": 200, "status": "Submitted", "perm_id": 1, "oca_group": main},
              {"symbol": "NVDA", "action": "SELL", "quantity": 100, "status": "Submitted", "perm_id": 2, "oca_group": trim_g},
              {"symbol": "NVDA", "action": "SELL", "quantity": 100, "status": "Submitted", "perm_id": 3, "oca_group": trim_g}]
    ob.assert_no_oversell(state(pos, orders), "NVDA", 200, oca_group=main, pool_bot_groups=True)      # the main block's exit
    ob.assert_no_oversell(state(pos, orders), "NVDA", 100, oca_group=trim_g, pool_bot_groups=True)    # the trim block's exit
    with pytest.raises(ValueError):
        ob.assert_no_oversell(state(pos, orders), "NVDA", 201, oca_group=main, pool_bot_groups=True)
    manual_g = {**orders[1], "perm_id": 4, "oca_group": "MY-OCA"}               # a non-bot group still adds up
    with pytest.raises(ValueError):
        ob.assert_no_oversell(state(pos, orders[:1] + [manual_g, {**manual_g, "perm_id": 5}]), "NVDA", 200, oca_group=main, pool_bot_groups=True)
    with pytest.raises(ValueError):                                             # default (shared_oca): groups add up
        ob.assert_no_oversell(state(pos, orders), "NVDA", 200, oca_group=main)
