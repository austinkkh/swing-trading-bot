"""Order ledger: intents before placing, idempotent entries, forward-only states, reconcile (offline)."""

import json
from datetime import date, datetime
from types import SimpleNamespace

import pandas as pd
import pytest

import order_ledger as ol
from order_ledger import (DuplicateEntry, LedgerCorrupt, LedgerKey, LedgerLocked, OrderLedger,
                          ledger_lock, make_order_ref, parse_order_ref)

D1 = date(2026, 11, 2)
D2 = date(2026, 11, 3)
ACCT = "DUXXXXXXX"


class Clock:
    """Settable fake ET clock."""
    def __init__(self, now=datetime(2026, 11, 2, 9, 24, 11)):
        self.now = now

    def __call__(self):
        return self.now


def make_ledger(tmp_path, clock=None, run_id="test-run"):
    return OrderLedger(tmp_path / "ledger.jsonl", run_id=run_id, clock=clock or Clock())


def state(positions=(), open_orders=(), executions=(), completed=(), complete=True):
    """Same attributes as order_broker.BrokerState."""
    return SimpleNamespace(positions=list(positions), open_orders=list(open_orders),
                           executions=list(executions), completed_orders=list(completed),
                           account_values={}, fetched_at=datetime(2026, 11, 2, 9, 30),
                           complete=complete)


def intend_bracket(ledger, symbol="NVDA", d=D1, entry_id=1043, qty=120, setup="PIVOT_BREAKOUT"):
    key = LedgerKey(d, symbol, setup)
    common = dict(client_id=16, account=ACCT, qty=qty)
    ledger.record_intent(key, "entry", make_order_ref("entry", d, symbol), entry_id, action="BUY",
                         order_type="STP LMT", aux_price=150.0, lmt_price=153.0, tif="DAY",
                         plan={"pivot": 150.0, "stop": 141.0}, **common)
    ledger.record_intent(key, "stop", make_order_ref("stop", d, symbol), entry_id + 1, action="SELL",
                         order_type="STP", aux_price=141.0, tif="GTC", parent_order_id=entry_id,
                         oca_group=ol.make_oca_group(d, symbol), **common)
    return key


def tws_order(ref, order_id, status="Submitted", client_id=16, perm_id=0, symbol="NVDA",
              action="BUY", qty=120, **extra):
    return {'order_id': order_id, 'perm_id': perm_id, 'client_id': client_id, 'order_ref': ref,
            'symbol': symbol, 'sec_type': 'STK', 'account': ACCT, 'action': action,
            'order_type': 'STP LMT' if action == 'BUY' else 'STP', 'quantity': float(qty),
            'aux_price': 150.0, 'lmt_price': 153.0, 'status': status, **extra}


def latest(ledger, ref):
    cur = ledger.current()
    return cur[cur['order_ref'] == ref].iloc[-1]


# ─── refs ───

def test_order_ref_round_trip():
    ref = make_order_ref("stop_fix", D1, "brk b")
    assert ref == "BOT-STOPFIX-20261102-BRK B"
    assert parse_order_ref(ref) == ("stop_fix", "2026-11-02", "BRK B")
    assert parse_order_ref("ALERT") is None
    assert parse_order_ref("BOT-NOPE-20261102-X") is None
    assert parse_order_ref("") is None


# ─── writing ───

def test_intent_survives_when_place_raises(tmp_path):
    ledger = make_ledger(tmp_path)

    def fake_place(*_):
        raise ConnectionError("TWS dropped")

    intend_bracket(ledger)
    with pytest.raises(ConnectionError):
        fake_place(1043)
    on_disk = OrderLedger(tmp_path / "ledger.jsonl").events()
    assert [(e['role'], e['state']) for e in on_disk] == [("entry", "intended"), ("stop", "intended")]
    assert on_disk[0]['plan'] == {"pivot": 150.0, "stop": 141.0}
    assert on_disk[0]['placed_ts'] == "2026-11-02T09:24:11"


def test_duplicate_entry_refused_even_with_another_setup(tmp_path):
    ledger = make_ledger(tmp_path)
    intend_bracket(ledger)
    with pytest.raises(DuplicateEntry):
        intend_bracket(ledger, entry_id=2000)
    with pytest.raises(DuplicateEntry):
        intend_bracket(ledger, entry_id=2000, setup="ORB")
    # Another day is a new slot
    intend_bracket(ledger, d=D2, entry_id=2000)
    assert ledger.entries_count(D1) == 1 and ledger.entries_count(D2) == 1


def test_illegal_transition_not_applied(tmp_path):
    ledger = make_ledger(tmp_path)
    intend_bracket(ledger)
    ref = make_order_ref("entry", D1, "NVDA")
    base = {'trade_date': D1, 'symbol': 'NVDA', 'setup': 'PIVOT_BREAKOUT', 'role': 'entry',
            'order_ref': ref, 'order_id': 1043}
    assert ledger.record({**base, 'state': 'filled'}) is None          # intended -> filled: illegal
    assert latest(ledger, ref)['state'] == 'intended'
    assert ledger.record({**base, 'state': 'submitted'})['state'] == 'submitted'
    assert ledger.record({**base, 'state': 'cancelled'})['state'] == 'cancelled'
    assert ledger.record({**base, 'state': 'working'}) is None         # terminal stays terminal
    n = len(ledger.events())
    # A brand-new order can't start mid-life
    assert ledger.record({**base, 'order_id': 9999, 'state': 'working'}) is None
    assert len(ledger.events()) == n


def test_same_state_update_with_nothing_new_is_not_written(tmp_path):
    ledger = make_ledger(tmp_path)
    intend_bracket(ledger)
    base = {'trade_date': D1, 'symbol': 'NVDA', 'setup': 'PIVOT_BREAKOUT', 'role': 'entry',
            'order_ref': make_order_ref("entry", D1, "NVDA"), 'order_id': 1043}
    ledger.record({**base, 'state': 'submitted'})
    ledger.record({**base, 'state': 'working'})
    n = len(ledger.events())
    ledger.record({**base, 'state': 'working'})
    assert len(ledger.events()) == n
    ledger.record({**base, 'state': 'working', 'perm_id': 555})
    assert len(ledger.events()) == n + 1


def test_truncated_last_line_tolerated_and_repaired(tmp_path):
    ledger = make_ledger(tmp_path)
    intend_bracket(ledger)
    with open(ledger.path, 'a') as fh:
        fh.write('{"ts": "2026-11-02T09:25:00", "trade_da')       # crash mid-write
    fresh = make_ledger(tmp_path)
    assert len(fresh.events()) == 2
    # The next write moves the fragment aside instead of gluing it into a corrupt middle line
    intend_bracket(fresh, d=D2, entry_id=2000)
    assert len(make_ledger(tmp_path).events()) == 4
    assert list(tmp_path.glob("ledger.jsonl.partial-*"))


def test_corrupt_middle_line_raises(tmp_path):
    ledger = make_ledger(tmp_path)
    intend_bracket(ledger)
    lines = ledger.path.read_text().splitlines()
    ledger.path.write_text(lines[0] + "\nnot json\n" + lines[1] + "\n")
    with pytest.raises(LedgerCorrupt):
        make_ledger(tmp_path).events()


def test_second_lock_raises(tmp_path):
    with ledger_lock(tmp_path):
        with pytest.raises(LedgerLocked):
            with ledger_lock(tmp_path):
                pass
    with ledger_lock(tmp_path):     # released after the first run
        pass


def test_view_csv_matches_events(tmp_path):
    clock = Clock()
    ledger = make_ledger(tmp_path, clock)
    intend_bracket(ledger)
    base = {'trade_date': D1, 'symbol': 'NVDA', 'setup': 'PIVOT_BREAKOUT', 'role': 'entry',
            'order_ref': make_order_ref("entry", D1, "NVDA"), 'order_id': 1043}
    ledger.record({**base, 'state': 'submitted'})
    view = pd.read_csv(tmp_path / "ledger_view_2026-11-02.csv")
    assert len(view) == 2
    by_role = view.set_index('role')
    assert by_role.loc['entry', 'state'] == 'submitted'
    assert by_role.loc['stop', 'state'] == 'intended'
    assert json.loads(by_role.loc['entry', 'plan']) == {"pivot": 150.0, "stop": 141.0}
    assert list(view.columns[:len(ol.VIEW_COLUMNS)]) == ol.VIEW_COLUMNS
    assert ledger.highest_order_id() == 1044


# ─── reconcile ───

def test_intended_never_at_tws_is_cancelled_and_slot_kept(tmp_path):
    ledger = make_ledger(tmp_path)
    intend_bracket(ledger)
    report = ledger.reconcile(state(), D1)
    cur = ledger.current()
    assert set(cur['state']) == {'cancelled'}
    assert set(cur['reason']) == {'never_reached_tws'}
    assert len(report.applied) == 2
    assert ledger.has_entry(D1, "NVDA") and ledger.entries_count(D1) == 1
    with pytest.raises(DuplicateEntry):
        intend_bracket(ledger, entry_id=3000)


def test_incomplete_reads_never_mark_never_reached(tmp_path):
    ledger = make_ledger(tmp_path)
    intend_bracket(ledger)
    ledger.reconcile(state(complete=False), D1)
    assert set(ledger.current()['state']) == {'intended'}


def test_intended_found_at_tws_goes_submitted_then_working(tmp_path):
    ledger = make_ledger(tmp_path)
    intend_bracket(ledger)
    oo = [tws_order(make_order_ref("entry", D1, "NVDA"), 1043, perm_id=9001),
          tws_order(make_order_ref("stop", D1, "NVDA"), 1044, status="PreSubmitted", action="SELL",
                    perm_id=9002, parent_id=1043)]
    report = ledger.reconcile(state(open_orders=oo), D1)
    assert set(ledger.current()['state']) == {'working'}
    assert latest(ledger, make_order_ref("entry", D1, "NVDA"))['perm_id'] == 9001
    assert report.clean


def test_working_entry_filled_from_executions(tmp_path):
    ledger = make_ledger(tmp_path)
    intend_bracket(ledger)
    entry_ref = make_order_ref("entry", D1, "NVDA")
    oo = [tws_order(entry_ref, 1043, perm_id=9001),
          tws_order(make_order_ref("stop", D1, "NVDA"), 1044, action="SELL", perm_id=9002, parent_id=1043)]
    ledger.reconcile(state(open_orders=oo), D1)
    # Entry fills in two executions (matched by perm id); the order leaves the open-order list
    ex = [{'exec_id': 'e1', 'perm_id': 9001, 'order_id': 1043, 'client_id': 16, 'order_ref': entry_ref,
           'symbol': 'NVDA', 'side': 'BOT', 'shares': 100, 'price': 151.0, 'account': ACCT},
          {'exec_id': 'e2', 'perm_id': 9001, 'order_id': 1043, 'client_id': 16, 'order_ref': entry_ref,
           'symbol': 'NVDA', 'side': 'BOT', 'shares': 20, 'price': 152.0, 'account': ACCT},
          {'exec_id': 'e2', 'perm_id': 9001, 'order_id': 1043, 'client_id': 16, 'order_ref': entry_ref,
           'symbol': 'NVDA', 'side': 'BOT', 'shares': 20, 'price': 152.0, 'account': ACCT}]  # repeat
    pos = [{'account': ACCT, 'symbol': 'NVDA', 'sec_type': 'STK', 'shares': 120.0, 'avg_cost': 151.17}]
    report = ledger.reconcile(state(positions=pos, open_orders=oo[1:], executions=ex), D1)
    e = latest(ledger, entry_ref)
    assert e['state'] == 'filled' and e['filled_qty'] == 120
    assert e['avg_fill_price'] == pytest.approx((100 * 151 + 20 * 152) / 120, abs=1e-4)
    trades = ledger.open_trades()
    assert len(trades) == 1 and trades[0]['net_qty'] == 120 and trades[0]['trade_status'] == 'open'
    assert not report.mismatches


def test_partial_fill_then_rest(tmp_path):
    ledger = make_ledger(tmp_path)
    intend_bracket(ledger)
    entry_ref = make_order_ref("entry", D1, "NVDA")
    oo = [tws_order(entry_ref, 1043, filled=40),
          tws_order(make_order_ref("stop", D1, "NVDA"), 1044, action="SELL", parent_id=1043)]
    pos = [{'account': ACCT, 'symbol': 'NVDA', 'sec_type': 'STK', 'shares': 40.0}]
    ledger.reconcile(state(positions=pos, open_orders=oo), D1)
    e = latest(ledger, entry_ref)
    assert e['state'] == 'partially_filled' and e['filled_qty'] == 40


def test_day_order_gone_next_day_expires(tmp_path):
    clock = Clock()
    ledger = make_ledger(tmp_path, clock)
    intend_bracket(ledger)
    entry_ref = make_order_ref("entry", D1, "NVDA")
    oo = [tws_order(entry_ref, 1043),
          tws_order(make_order_ref("stop", D1, "NVDA"), 1044, action="SELL", parent_id=1043)]
    ledger.reconcile(state(open_orders=oo), D1)
    clock.now = datetime(2026, 11, 3, 9, 0)
    report = ledger.reconcile(state(), D2)
    assert latest(ledger, entry_ref)['state'] == 'expired'
    # The GTC child isn't a DAY order: gone without a status is a mismatch, left alone
    assert latest(ledger, make_order_ref("stop", D1, "NVDA"))['state'] == 'working'
    assert [m['role'] for m in report.mismatches] == ['stop']


def test_day_order_gone_same_day_is_mismatch_not_expired(tmp_path):
    ledger = make_ledger(tmp_path)
    intend_bracket(ledger)
    entry_ref = make_order_ref("entry", D1, "NVDA")
    ledger.reconcile(state(open_orders=[tws_order(entry_ref, 1043)]), D1)
    report = ledger.reconcile(state(), D1)
    assert latest(ledger, entry_ref)['state'] == 'working'
    assert any(m['order_ref'] == entry_ref for m in report.mismatches)


def test_exit_with_old_entry_date_is_not_expired_on_its_own_day(tmp_path):
    """Exit refs carry the ENTRY date; expiry goes by when the order was placed."""
    clock = Clock(datetime(2026, 11, 5, 16, 20))
    ledger = make_ledger(tmp_path, clock)
    key = LedgerKey(D1, "NVDA", "PIVOT_BREAKOUT")
    ref = make_order_ref("exit", D1, "NVDA")
    ledger.record_intent(key, "exit", ref, 1100, client_id=16, qty=120, tif="OPG", action="SELL")
    ledger.record({**key.as_fields(), 'role': 'exit', 'order_ref': ref, 'order_id': 1100,
                   'state': 'submitted'})
    ledger.reconcile(state(), date(2026, 11, 5))
    assert latest(ledger, ref)['state'] == 'submitted'
    clock.now = datetime(2026, 11, 6, 16, 20)
    ledger.reconcile(state(), date(2026, 11, 6))
    assert latest(ledger, ref)['state'] == 'expired'


def test_completed_cancel_is_applied(tmp_path):
    ledger = make_ledger(tmp_path)
    intend_bracket(ledger)
    entry_ref = make_order_ref("entry", D1, "NVDA")
    ledger.reconcile(state(open_orders=[tws_order(entry_ref, 1043),
                                        tws_order(make_order_ref("stop", D1, "NVDA"), 1044,
                                                  action="SELL", parent_id=1043)]), D1)
    done = [tws_order(entry_ref, 1043, status="Cancelled"),
            tws_order(make_order_ref("stop", D1, "NVDA"), 1044, status="Cancelled", action="SELL")]
    ledger.reconcile(state(completed=done), D1)
    assert set(ledger.current()['state']) == {'cancelled'}


def test_orphan_bot_order_reported_not_cancelled(tmp_path):
    ledger = make_ledger(tmp_path)
    oo = [tws_order("BOT-TRIM-20261102-AMD", 77, symbol="AMD", action="SELL"),
          tws_order("BOT-TRIM-20261102-AMD", 77, symbol="AMD", action="SELL"),   # openOrder repeat
          tws_order("ALERT", 5, symbol="MSFT", qty=1),                          # alert: not the bot's
          tws_order("", 6, symbol="AAPL")]                                       # manual
    report = ledger.reconcile(state(open_orders=oo), D1)
    assert [(o['role'], o['symbol'], o['setup']) for o in report.orphans] == [("trim", "AMD", "UNKNOWN")]
    assert latest(ledger, "BOT-TRIM-20261102-AMD")['state'] == 'working'
    # Known now: a second reconcile reports no orphan
    assert not ledger.reconcile(state(open_orders=oo), D1).orphans


def test_orphan_entry_uses_the_days_slot(tmp_path):
    ledger = make_ledger(tmp_path)
    oo = [tws_order("BOT-ENTRY-20261102-AMD", 77, symbol="AMD"),
          tws_order("BOT-STOP-20261102-AMD", 78, symbol="AMD", action="SELL", parent_id=77)]
    ledger.reconcile(state(open_orders=oo), D1)
    assert ledger.has_entry(D1, "AMD") and ledger.entries_count(D1) == 1


def test_staged_parent_with_no_child_is_incomplete_bracket(tmp_path):
    ledger = make_ledger(tmp_path)
    intend_bracket(ledger)
    entry_ref = make_order_ref("entry", D1, "NVDA")
    report = ledger.reconcile(state(open_orders=[tws_order(entry_ref, 1043, status="PreSubmitted",
                                                           transmit=False)]), D1)
    assert [(b['order_ref'], b['staged']) for b in report.incomplete_brackets] == [(entry_ref, True)]
    # The child was intended but never reached TWS
    assert latest(ledger, make_order_ref("stop", D1, "NVDA"))['reason'] == 'never_reached_tws'
    # Rerun: the slot is still used, nothing is re-planned
    with pytest.raises(DuplicateEntry):
        intend_bracket(ledger, entry_id=5000)


def _filled_trade(ledger, qty=120):
    intend_bracket(ledger, qty=qty)
    entry_ref = make_order_ref("entry", D1, "NVDA")
    stop_ref = make_order_ref("stop", D1, "NVDA")
    oo = [tws_order(stop_ref, 1044, action="SELL", parent_id=1043, qty=qty)]
    pos = [{'account': ACCT, 'symbol': 'NVDA', 'sec_type': 'STK', 'shares': float(qty)}]
    done = [tws_order(entry_ref, 1043, status="Filled", filled=qty, avg_fill_price=151.0, qty=qty)]
    ledger.reconcile(state(positions=pos, open_orders=oo, completed=done), D1)
    return entry_ref, stop_ref, oo


def test_position_mismatch_marks_manual_touched(tmp_path):
    ledger = make_ledger(tmp_path)
    _, _, oo = _filled_trade(ledger)
    assert ledger.open_trades()[0]['trade_status'] == 'open'
    pos = [{'account': ACCT, 'symbol': 'NVDA', 'sec_type': 'STK', 'shares': 200.0}]   # user added 80
    report = ledger.reconcile(state(positions=pos, open_orders=oo), D1)
    assert [m['issue'] for m in report.mismatches] == ['position_mismatch']
    assert ledger.open_trades()[0]['trade_status'] == 'manual_touched'
    assert ledger.trade_status(D1, "NVDA") == 'manual_touched'
    # Not re-recorded on every run
    n = len(ledger.events())
    ledger.reconcile(state(positions=pos, open_orders=oo), D1)
    assert len(ledger.events()) == n


def test_position_gone_closes_trade(tmp_path):
    ledger = make_ledger(tmp_path)
    _filled_trade(ledger)
    ledger.reconcile(state(), D1)            # paper reset / stop-out not yet in executions
    assert ledger.open_trades() == []
    closed = latest(ledger, make_order_ref("trade", D1, "NVDA"))
    assert closed['state'] == 'closed' and closed['reason'] == 'position_gone'


def test_stop_out_fill_closes_trade_flat(tmp_path):
    ledger = make_ledger(tmp_path)
    _, stop_ref, _ = _filled_trade(ledger)
    ex = [{'exec_id': 's1', 'order_id': 1044, 'client_id': 16, 'order_ref': stop_ref, 'symbol': 'NVDA',
           'side': 'SLD', 'shares': 120, 'price': 140.9, 'account': ACCT}]
    ledger.reconcile(state(executions=ex), D1)
    assert latest(ledger, stop_ref)['state'] == 'filled'
    assert latest(ledger, make_order_ref("trade", D1, "NVDA"))['reason'] == 'flat'
    assert ledger.open_trades() == []


def test_manual_position_without_ledger_trade_is_ignored(tmp_path):
    ledger = make_ledger(tmp_path)
    pos = [{'account': ACCT, 'symbol': 'AAPL', 'sec_type': 'STK', 'shares': 10.0}]
    report = ledger.reconcile(state(positions=pos), D1)
    assert report.clean and ledger.events() == []


def test_same_state_modify_request_is_written(tmp_path):
    """fill_monitor / exit_manager record a resize, stop move or cancel request before sending it."""
    ledger = make_ledger(tmp_path)
    intend_bracket(ledger)
    base = {'trade_date': D1, 'symbol': 'NVDA', 'setup': 'PIVOT_BREAKOUT', 'role': 'stop',
            'order_ref': make_order_ref("stop", D1, "NVDA"), 'order_id': 1044, 'state': 'intended'}
    n = len(ledger.events())
    ledger.record({**base, 'qty': 60, 'reason': 'resize_stop_intent 120->60'})
    ledger.record({**base, 'cancel_requested': True})
    assert len(ledger.events()) == n + 2
    assert latest(ledger, base['order_ref'])['qty'] == 60


def test_closed_trade_stop_gone_expires_without_repeat_alerts(tmp_path):
    """The GTC stop filled on a day with no bot run: TWS no longer shows the order or its execution."""
    ledger = make_ledger(tmp_path)
    _, stop_ref, _ = _filled_trade(ledger)
    report = ledger.reconcile(state(), D2)
    assert ledger.open_trades() == []
    stop = latest(ledger, stop_ref)
    assert stop['state'] == 'expired' and stop['reason'] == 'trade_closed'
    assert [m for m in report.mismatches if m.get('issue') == 'not_found_at_tws'] == []
    assert ledger.reconcile(state(), D2).mismatches == []           # quiet on every later run


def test_closed_trade_order_still_at_tws_is_left_alone(tmp_path):
    ledger = make_ledger(tmp_path)
    _, stop_ref, oo = _filled_trade(ledger)
    ledger.reconcile(state(open_orders=oo), D2)                    # position gone, stop still listed
    assert ledger.open_trades() == []
    assert latest(ledger, stop_ref)['state'] == 'working'


def test_execution_without_shares_key_does_not_crash(tmp_path):
    ledger = make_ledger(tmp_path)
    intend_bracket(ledger)
    entry_ref = make_order_ref("entry", D1, "NVDA")
    base = {'order_id': 1043, 'client_id': 16, 'order_ref': entry_ref, 'symbol': 'NVDA', 'side': 'BOT',
            'price': 150.5, 'account': ACCT}
    ex = [{**base, 'exec_id': 'e1', 'shares': 50}, {**base, 'exec_id': 'e2'}]     # e2: malformed
    ledger.reconcile(state(open_orders=[tws_order(entry_ref, 1043)], executions=ex), D1)
    assert float(latest(ledger, entry_ref)['filled_qty']) == 50.0
