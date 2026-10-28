"""Historical fetches wait for the end-of-data signal and parse ibapi 10.x errors. No TWS needed."""

import threading
import time
from types import SimpleNamespace

from ibkr_connector import IBKRConnector


def connector_with_fake_feed(emit):
    """IBKRConnector whose reqHistoricalData starts `emit(conn, req_id)` on another thread."""
    conn = IBKRConnector()
    conn.serverVersion = lambda: 219
    conn.reqHistoricalData = lambda req_id, *a, **k: threading.Thread(target=emit, args=(conn, req_id)).start()
    return conn


def test_waits_for_all_bars_not_the_first():
    def emit(conn, req_id):
        for i in range(300):
            conn.historicalData(req_id, SimpleNamespace(close=float(i)))
            if i % 50 == 0:
                time.sleep(0.02)  # bars trickle in, as on the real reader thread
        conn.historicalDataEnd(req_id, "", "")

    conn = connector_with_fake_feed(emit)
    bars = conn.fetch_historical_data("TEST", timeout=5)
    assert len(bars) == 300
    assert conn.historical_data == {} and conn.historical_done == set()  # memory freed


def test_request_error_returns_quickly():
    def emit(conn, req_id):
        conn.error(req_id, 1792715947541, 162, "HMDS query returned no data", "")  # ibapi 10.x signature

    conn = connector_with_fake_feed(emit)
    started = time.time()
    assert conn.fetch_historical_data("DELISTED", timeout=10) is None
    assert time.time() - started < 2


def test_error_parsing_handles_both_signatures():
    conn = IBKRConnector()
    conn.error(7, 1792715947541, 200, "No security definition", "")  # 10.x: errorTime first
    conn.error(8, 162, "HMDS query returned no data")                 # older: code first
    assert conn.errors[7]['code'] == 200
    assert conn.errors[8]['code'] == 162


def test_timeout_discards_partial_bars():
    def emit(conn, req_id):
        conn.historicalData(req_id, SimpleNamespace(close=1.0))  # never sends historicalDataEnd

    conn = connector_with_fake_feed(emit)
    assert conn.fetch_historical_data("SLOW", timeout=0.5) is None


# ─── Paper guard (shared by every tool) ───
import pytest
from types import SimpleNamespace
import ibkr_connector as ic


class Accounts:
    def __init__(self, accounts):
        self._a = accounts

    def fetch_managed_accounts(self, timeout=8):
        return self._a


def test_live_ports_are_refused_before_connecting():
    for port in (7496, 4001, 7777):
        conn = ic.IBKRConnector()
        conn.connect = lambda *a, **k: pytest.fail("must not connect")
        assert conn.connect_and_run("127.0.0.1", port, 99) is False


def test_mixed_paper_and_live_session_is_refused():
    with pytest.raises(PermissionError, match="Non-paper"):
        ic.confirm_paper_session(Accounts(["DUX1", "U2"]), 7497)
    with pytest.raises(PermissionError, match="not a paper port"):
        ic.confirm_paper_session(Accounts(["DUX1"]), 7496)
    with pytest.raises(PermissionError, match="no accounts"):
        ic.confirm_paper_session(Accounts([]), 7497)
    assert ic.confirm_paper_session(Accounts(["DUX1"]), 4002) == ["DUX1"]


def test_order_calls_raise_on_read_only_connectors():
    conn = ic.IBKRConnector()
    for call in (lambda: conn.placeOrder(1, None, None), lambda: conn.cancelOrder(1), lambda: conn.reqGlobalCancel()):
        with pytest.raises(PermissionError):
            call()


def test_executor_refuses_a_mixed_session(monkeypatch):
    import ibkr_paper_executor as ex

    class FakeConn:
        def __init__(self, *a, **k): pass
        def connect_and_run(self, *a): return True
        def fetch_managed_accounts(self, timeout=8): return ["DUX1", "U2"]
        def disconnect_and_stop(self): pass

    monkeypatch.setattr(ex, "IBKRConnector", FakeConn)
    with pytest.raises(PermissionError, match="Non-paper"):
        ex._confirm_paper_account_with_broker(ex.PaperExecutorConfig(account="DUX1"))


# ─── Data-farm / server-link state (2026-10-26: scan hit TWS mid-reconnect after a wake) ───
import logging


@pytest.mark.parametrize("signature", ["10.x", "old"])
def test_farm_broken_then_ok_is_tracked(signature):
    conn = ic.IBKRConnector()

    def send(code, msg):
        if signature == "10.x":
            conn.error(-1, 1792715947541, code, msg, "")
        else:
            conn.error(-1, code, msg)

    send(2105, "HMDS data farm connection is broken:ushmds")
    assert conn.farm_status["ushmds"] is False and not conn.data_farms_ready()
    assert "ushmds broken since" in conn.farm_status_summary()
    send(2106, "HMDS data farm connection is OK:ushmds")
    assert conn.farm_status["ushmds"] is True and conn.data_farms_ready()
    send(2104, "Market data farm connection is OK:usfarm.nj")
    assert conn.farm_status["usfarm.nj"] is True                  # dotted farm ids survive
    send(2107, "HMDS data farm connection is inactive but should be available upon demand.ushmds")
    assert conn.farm_status["ushmds"] is True                     # inactive-on-demand is usable
    assert conn.errors == {}                                      # status codes are not request errors


def test_server_link_down_and_restored():
    conn = ic.IBKRConnector()
    conn.error(-1, 1792715947541, 2110, "Connectivity between Trader Workstation and server is broken.", "")
    assert conn.server_link_ok is False and not conn.data_farms_ready(required=())
    conn.error(-1, 1792715947541, 1102, "Connectivity between IB and TWS has been restored - data maintained.", "")
    assert conn.server_link_ok is True
    conn.error(-1, 1100, "Connectivity between IB and Trader Workstation has been lost.")
    assert conn.server_link_ok is False
    conn.error(-1, 2106, "HMDS data farm connection is OK:ushmds")   # TWS often sends no 1102
    assert conn.server_link_ok is True


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def now(self):
        return self.t

    def sleep(self, s):
        self.t += s


def test_wait_for_farms_returns_at_once_when_nothing_is_broken(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(ic, "_monotonic", clock.now)
    monkeypatch.setattr(ic, "_sleep", clock.sleep)
    assert ic.IBKRConnector().wait_for_data_farms(max_wait=600) is True
    assert clock.t == 0


def test_wait_for_farms_recovers_mid_wait():
    conn = ic.IBKRConnector()
    conn.error(-1, 2105, "HMDS data farm connection is broken:ushmds")
    threading.Timer(0.2, lambda: conn.error(-1, 2106, "HMDS data farm connection is OK:ushmds")).start()
    started = time.monotonic()
    assert conn.wait_for_data_farms(max_wait=5, poll=0.02) is True
    assert time.monotonic() - started < 2


def test_wait_for_farms_gives_up_after_max_wait(monkeypatch, caplog):
    clock = FakeClock()
    monkeypatch.setattr(ic, "_monotonic", clock.now)
    monkeypatch.setattr(ic, "_sleep", clock.sleep)
    conn = ic.IBKRConnector()
    conn.error(-1, 2105, "HMDS data farm connection is broken:ushmds")
    with caplog.at_level(logging.INFO, logger="ibkr_connector"):
        assert conn.wait_for_data_farms(max_wait=600, poll=1.0) is False
    assert 600 <= clock.t <= 601
    errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
    assert errors and "ushmds" in errors[0] and "not OK after 600s" in errors[0]
    assert any("Waiting for ushmds" in r.getMessage() for r in caplog.records)   # progress lines


# ─── Retries on critical fetches ───
def scripted_connector(script, monkeypatch):
    """Connector whose n-th historical request follows script[n]: 'timeout', 'ok', or an error tuple."""
    conn = ic.IBKRConnector()
    conn.serverVersion = lambda: 219
    calls, cancelled, slept = [], [], []
    monkeypatch.setattr(ic, "_sleep", slept.append)

    def req(req_id, *a, **k):
        step = script[len(calls)]
        calls.append(req_id)

        def emit():
            if step == "ok":
                conn.historicalData(req_id, SimpleNamespace(close=1.0))
                conn.historicalDataEnd(req_id, "", "")
            elif isinstance(step, tuple):
                conn.error(req_id, 1792715947541, step[0], step[1], "")
        threading.Thread(target=emit).start()

    conn.reqHistoricalData = req
    conn.cancelHistoricalData = cancelled.append
    return conn, calls, cancelled, slept


def test_timeout_is_retried_and_cancelled(monkeypatch):
    conn, calls, cancelled, slept = scripted_connector(["timeout", "ok"], monkeypatch)
    bars, status = conn.fetch_historical_data_ex("SPY", timeout=0.3, retries=2)
    assert status == "ok" and len(bars) == 1
    assert len(calls) == 2 and cancelled == [calls[0]]       # the timed-out request was cancelled
    assert slept and slept[0] >= 15                           # pacing-safe gap before the retry


def test_default_is_a_single_try(monkeypatch):
    conn, calls, _, _ = scripted_connector(["timeout", "ok"], monkeypatch)
    assert conn.fetch_historical_data("MU", timeout=0.2) is None
    assert len(calls) == 1


@pytest.mark.parametrize("err, status", [((162, "HMDS query returned no data"), "no_data"),
                                         ((200, "No security definition has been found"), "no_secdef")])
def test_hard_errors_are_not_retried(monkeypatch, err, status):
    conn, calls, _, slept = scripted_connector([err, "ok"], monkeypatch)
    assert conn.fetch_historical_data_ex("DEAD", timeout=5, retries=3) == (None, status)
    assert len(calls) == 1 and slept == []


def test_pacing_violation_is_retried(monkeypatch):
    pacing = (162, "Historical Market Data Service error message:Historical data request pacing violation")
    conn, calls, _, _ = scripted_connector([pacing, "ok"], monkeypatch)
    bars, status = conn.fetch_historical_data_ex("SPY", timeout=5, retries=1)
    assert status == "ok" and len(calls) == 2


def test_late_bars_for_a_cancelled_request_are_dropped(monkeypatch):
    conn, calls, _, _ = scripted_connector(["timeout"], monkeypatch)
    assert conn.fetch_historical_data("SLOW", timeout=0.2) is None
    late = calls[0]
    conn.historicalData(late, SimpleNamespace(close=1.0))
    conn.historicalDataEnd(late, "", "")
    conn.error(late, 1792715947541, 366, "No historical data query found for ticker id", "")
    assert conn.historical_data == {} and conn.historical_done == set() and conn.errors == {}


# ─── Review L1-L3 ───
def test_dotted_hmds_variant_counts_as_the_required_farm():
    import ibkr_connector as ic
    conn = ic.IBKRConnector()
    conn.error(-1, 2105, "HMDS data farm connection is broken:ushmds.nj")
    assert conn.data_farms_ready() is False
    assert "ushmds.nj broken" in conn.farm_status_summary()
    conn.error(-1, 2106, "HMDS data farm connection is OK:ushmds.nj")
    assert conn.data_farms_ready() is True
    conn.error(-1, 2105, "HMDS data farm connection is broken:euhmds")   # a farm the scan doesn't use
    assert conn.data_farms_ready() is True


def test_contract_details_returns_on_error_200_without_waiting(monkeypatch):
    import time as _t
    import ibkr_connector as ic
    conn = ic.IBKRConnector()

    def req(req_id, contract):
        conn.error(req_id, 200, "No security definition has been found for the request")
    monkeypatch.setattr(conn, "reqContractDetails", req)
    start = _t.monotonic()
    assert conn.fetch_contract_details("DEAD", timeout=5) is None
    assert _t.monotonic() - start < 1
    assert conn.errors == {} and conn.contract_details == {}


def test_contract_details_found_frees_its_entry(monkeypatch):
    import ibkr_connector as ic
    conn = ic.IBKRConnector()
    monkeypatch.setattr(conn, "reqContractDetails", lambda rid, c: conn.contractDetails(rid, "DETAILS"))
    assert conn.fetch_contract_details("MU", timeout=5) == "DETAILS"
    assert conn.contract_details == {}


def test_cancelled_request_ids_are_bounded():
    import ibkr_connector as ic
    conn = ic.IBKRConnector()
    for rid in range(conn.CANCELLED_KEEP + 100):
        conn._mark_cancelled(rid)
    assert len(conn._cancelled_hist) == conn.CANCELLED_KEEP
    assert 0 not in conn._cancelled_hist and conn.CANCELLED_KEEP + 99 in conn._cancelled_hist
    conn.historicalData(conn.CANCELLED_KEEP + 99, "late bar")        # recent: still dropped
    assert conn.historical_data == {}


# ─── Read-only order lifecycle callbacks (order ledger reconcile) ───
def _order(**kw):
    base = dict(permId=7, clientId=16, account="DUX1", action="SELL", orderType="STP", totalQuantity=10,
                auxPrice=95.0, lmtPrice=0.0, trailStopPrice=0.0, orderRef="BOT-STOP-20261102-NVDA",
                parentId=41, tif="GTC", ocaGroup="BOT-OCA-20261102-NVDA", transmit=True, orderId=42)
    base.update(kw)
    return SimpleNamespace(**base)


CONTRACT = SimpleNamespace(symbol="NVDA", secType="STK", conId=1)


def test_open_order_keeps_bracket_fields():
    conn = IBKRConnector()
    conn.openOrder(42, CONTRACT, _order(), SimpleNamespace(status="PreSubmitted"))
    o = conn.open_orders[0]
    assert (o["parent_id"], o["tif"], o["oca_group"], o["transmit"]) == (41, "GTC", "BOT-OCA-20261102-NVDA", True)


def test_order_status_execution_and_completed_callbacks():
    conn = IBKRConnector()
    conn.orderStatus(42, "Submitted", 0, 10, 0.0, 7, 41, 0.0, 16, "", 0.0)
    conn.orderStatus(42, "Filled", 10, 0, 95.1, 7, 41, 95.1, 16, "", 0.0)
    assert conn.order_status[42]["status"] == "Filled" and conn.order_status[42]["filled"] == 10.0
    ex = SimpleNamespace(execId="e1", orderId=42, permId=7, clientId=16, orderRef="BOT-STOP-20261102-NVDA",
                         acctNumber="DUX1", side="SLD", shares=10, price=95.1, time="20261102 10:00:00")
    conn.execDetails(1, CONTRACT, ex)
    assert conn.executions[0]["shares"] == 10.0 and conn.executions[0]["order_ref"].startswith("BOT-")
    conn.completedOrder(CONTRACT, _order(filledQuantity=1.7976931348623157e308),
                        SimpleNamespace(status="Cancelled", completedStatus="Expired"))
    assert conn.completed_orders[0]["filled"] == 0.0 and conn.completed_orders[0]["completed_status"] == "Expired"


def test_fetch_executions_and_completed_wait_for_end():
    conn = IBKRConnector()
    conn.reqExecutions = lambda req_id, flt: conn.execDetailsEnd(req_id)
    conn.reqCompletedOrders = lambda api_only: conn.completedOrdersEnd()
    assert conn.fetch_executions(timeout=1) == [] and conn.fetch_completed_orders(timeout=1) == []
    conn.reqExecutions = lambda req_id, flt: None          # never ends: timeout -> None, not []
    assert conn.fetch_executions(timeout=0.1) is None
