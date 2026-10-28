"""exit_manager: parity with exit_rules.simulate_trade, order mapping, submit gates (offline only)."""

import json
from datetime import date, datetime, time

import pandas as pd
import pytest

import config
import exit_manager as em
import exit_rules as er
import order_ledger as ol
from helpers import FakeBroker, price_path
from order_ledger import LedgerKey, OrderLedger, make_oca_group, make_order_ref

ACCT = "DUX1"
ENTRY, STOP, SHARES = 50.0, 47.0, 300
R = ENTRY - STOP
TARGET = ENTRY + er.TRIM_R * R                 # the trim limit
TRIM_QTY = int(round(SHARES * er.TRIM_FRACTION))
REST = SHARES - TRIM_QTY
FAST = er.FAST_SESSIONS                        # the resting trim is first placed on bar T + FAST
T = 60                                         # entry bar index in every path
HISTORY = [(59, 50.0)]                         # 60 bars rising 40 -> 50 before the entry bar
GAP_OPEN = TARGET + 2 * R

# Leg ends are relative to the target so the scenarios follow strategy_params; the coverage test
# below checks each path still exercises the branch it is named for.
PATHS = {
    'fast_trail': [(FAST, TARGET + R / 2), (6, TARGET + 2.5 * R), (3, TARGET), (3, TARGET - R / 2)],
    # +R by the last fast session, target the bar after: the resting trim fills on bar T + FAST + 1
    'slow_trim': [(FAST, ENTRY + R / 2), (1, ENTRY + R), (1, TARGET + R),
                  (5, TARGET + 1.5 * R), (4, ENTRY + R), (2, ENTRY + 2 * R / 3)],
    'time_stop': [(er.TIME_STOP_SESSIONS + 4, ENTRY + R / 3)],      # never the target: time stop
    'stop_out': [(3, 49), (2, 46), (2, 45)],
    'breakeven_stop': [(FAST, TARGET + R / 2), (1, 49.5), (3, 49)],  # target fast, then a gap below entry
    'gap_through_trim': [(FAST + 1, ENTRY + R), (1, GAP_OPEN), (4, GAP_OPEN + 1),
                         (3, TARGET + R / 2), (2, TARGET - R / 2)],  # opens above the trim limit
    'open_at_end': [(FAST, TARGET + R / 2), (3, TARGET + 2.5 * R)],  # still open on the last bar
}


@pytest.fixture(autouse=True)
def _quiet(monkeypatch):
    monkeypatch.setenv("TRADING_NOTIFY", "0")      # no macOS notifications from tests


def path(name):
    return price_path(HISTORY + PATHS[name], start=40.0)


def first_hit(name):
    """Bar index on which the path's high first reaches the target."""
    df = path(name)
    return next(i for i in range(T, len(df)) if df['high'].iat[i] >= TARGET)


def day(df, i):
    return df['date'].iat[i].date()


class SimBroker(FakeBroker):
    """
    FakeBroker that fills the bot's SELLs bar by bar the way simulate_trade assumes:
    MOO at the open, then stops at min(open, stop) (at the stop on the entry bar), then LMTs at
    max(open, limit); MOC at the close. OCA type 3: a fill reduces the other orders in its group
    by the filled quantity. Filled / cancelled / expired orders move to completed_orders, like TWS.
    """

    def __init__(self, df, symbol="NVDA", oca="quantity", **kw):
        super().__init__(account=ACCT, submit_enabled=True, first_id=2000, **kw)
        self.df, self.symbol, self.i = df, symbol, T
        # What a full fill does to the rest of its OCA group (paper check 1 decides which TWS does):
        # quantity = reduce by the shares filled, proportional = by the filled fraction, cancel = cancel
        self.oca = oca
        self.fills = []                          # (bar index, qty, price, order_ref)
        self._exec = 0

    def daily_bars(self, symbol, duration="6 M", include_forming=False):
        self.bar_calls.append((symbol, include_forming))
        return self.df.iloc[:self.i + 1].copy()

    def cancel(self, order_id):
        super().cancel(order_id)
        for o in list(self._state.open_orders):
            if o['order_id'] == order_id:
                self._close(o, 'Cancelled')

    def _close(self, o, status):
        self._state.open_orders.remove(o)
        self._state.completed_orders.append({**o, 'status': status})

    def _fill(self, o, qty, price):
        self._exec += 1
        self._state.executions.append({'exec_id': f"e{self._exec}", 'order_id': o['order_id'], 'perm_id': 0,
                                       'client_id': o['client_id'], 'order_ref': o['order_ref'],
                                       'symbol': o['symbol'], 'side': 'SLD', 'shares': qty, 'price': price,
                                       'account': ACCT})
        self.fills.append((self.i, qty, price, o['order_ref']))
        pos = self._state.positions[0]
        pos['shares'] -= qty
        frac = qty / float(o['quantity'])
        o['filled'] = float(o.get('filled') or 0) + qty
        self._close(o, 'Filled')
        for other in list(self._state.open_orders):      # OCA type 3: reduce the rest of the group
            if o.get('oca_group') and other.get('oca_group') == o['oca_group']:
                if self.oca == 'cancel':
                    other['quantity'] = 0.0
                elif self.oca == 'proportional':
                    other['quantity'] = float(round(float(other['quantity']) * (1 - frac)))
                else:
                    other['quantity'] = max(float(other['quantity']) - qty, 0.0)
                if other['quantity'] <= 1e-9:
                    self._close(other, 'Cancelled')

    def _sells(self, kind):
        def is_kind(o):
            if kind == 'moo':
                return o['order_type'] == 'MKT' and o['tif'] == 'OPG'
            if kind == 'moc':
                return o['order_type'] == 'MOC'
            return o['order_type'] == kind
        return [o for o in list(self._state.open_orders) if o['action'] == 'SELL' and is_kind(o)]

    def intraday(self, i):
        self.i = i
        o_, h, l = (float(self.df[k].iat[i]) for k in ('open', 'high', 'low'))
        for o in self._sells('moo'):
            if o in self._state.open_orders:
                self._fill(o, float(o['quantity']), o_)
        for o in self._sells('STP'):
            if o in self._state.open_orders and l <= o['aux_price']:
                self._fill(o, float(o['quantity']), o['aux_price'] if i == T else min(o_, o['aux_price']))
        for o in self._sells('LMT'):
            if o in self._state.open_orders and h >= o['lmt_price']:
                self._fill(o, float(o['quantity']), max(o_, o['lmt_price']))
        for o in self._sells('moo'):              # an OPG order that didn't fill is dead after the open
            self._close(o, 'Cancelled')

    def at_close(self, i):
        c = float(self.df['close'].iat[i])
        for o in self._sells('moc'):
            self._fill(o, float(o['quantity']), c)


def open_trade(ledger, broker, d0, symbol="NVDA", shares=SHARES, entry=ENTRY, stop=STOP, client_id=16):
    """A bot bracket filled at `entry` on d0, its child stop working at TWS."""
    key = LedgerKey(d0, symbol, "PIVOT_BREAKOUT")
    eref, sref, oca = make_order_ref("entry", d0, symbol), make_order_ref("stop", d0, symbol), make_oca_group(d0, symbol)
    common = dict(client_id=client_id, account=ACCT, qty=shares)
    ledger.record_intent(key, "entry", eref, 1043, action="BUY", order_type="STP LMT", aux_price=entry,
                         lmt_price=round(entry * 1.02, 2), tif="DAY", plan={'pivot': entry, 'stop': stop}, **common)
    ledger.record_intent(key, "stop", sref, 1044, action="SELL", order_type="STP", aux_price=stop, tif="GTC",
                         parent_order_id=1043, oca_group=oca, **common)
    base = {**key.as_fields(), "client_id": client_id}
    for role, ref, oid in (("entry", eref, 1043), ("stop", sref, 1044)):
        ledger.record({**base, "role": role, "order_ref": ref, "order_id": oid, "state": "submitted"})
        ledger.record({**base, "role": role, "order_ref": ref, "order_id": oid, "state": "working"})
    ledger.record({**base, "role": "entry", "order_ref": eref, "order_id": 1043, "state": "filled",
                   "filled_qty": shares, "avg_fill_price": entry})
    broker._state.positions.append({"account": ACCT, "symbol": symbol, "sec_type": "STK", "shares": float(shares),
                                    "avg_cost": entry})
    broker._state.open_orders.append({
        "order_id": 1044, "perm_id": 0, "client_id": client_id, "account": ACCT, "symbol": symbol,
        "sec_type": "STK", "action": "SELL", "order_type": "STP", "quantity": float(shares), "aux_price": stop,
        "lmt_price": 0.0, "status": "PreSubmitted", "order_ref": sref, "parent_id": 1043, "tif": "GTC",
        "oca_group": oca, "transmit": True})
    return key


class Clock:
    def __init__(self, ts):
        self.ts = ts

    def __call__(self):
        return self.ts


def stop_cover(b):
    """Shares the working bot stops would sell, and the position."""
    stops = sum(float(o['quantity']) for o in b._state.open_orders if o['action'] == 'SELL' and o['order_type'] == 'STP')
    return stops, b._state.positions[0]['shares']


def replay(df, tmp_path, close_window=True, oca="quantity", check=None):
    """Day-by-day: intraday fills -> close-window run -> MOC fills -> nightly run, until flat.
    check(b, when) runs after every step (intraday fills, each run, MOC fills)."""
    clock = Clock(datetime.combine(day(df, T), time(9, 0)))
    ledger = OrderLedger(tmp_path / "ledger.jsonl", run_id="t", clock=clock, source="exit_manager")
    b = SimBroker(df, oca=oca)
    open_trade(ledger, b, day(df, T))
    exit_state = {'trades': {}}
    check = check or (lambda b, when: None)
    for i in range(T, len(df)):
        d = day(df, i)
        clock.ts = datetime.combine(d, time(10, 0))
        b.intraday(i)
        check(b, f"bar {i} intraday")
        if close_window:
            clock.ts = datetime.combine(d, time(15, 40))
            em.run(b, ledger, exit_state, clock.ts, submit=True, close_window=True)
            check(b, f"bar {i} close window")
            b.at_close(i)
            check(b, f"bar {i} close")
        clock.ts = datetime.combine(d, time(16, 30))
        em.run(b, ledger, exit_state, clock.ts, submit=True)
        check(b, f"bar {i} nightly")
        if b._state.positions[0]['shares'] <= 0:
            break
    r = ENTRY - STOP
    pnl = sum(q * (px - ENTRY) for _, q, px, _ in b.fills)
    left = b._state.positions[0]['shares']
    pnl += left * (float(df['close'].iat[-1]) - ENTRY)
    exit_bar = b.fills[-1][0] if left <= 0 else len(df) - 1
    trade = next(iter(exit_state['trades'].values()))
    return pnl / (SHARES * r), exit_bar, trade['trimmed'], b, ledger, trade


# ─── Parity: the key test ───

@pytest.mark.parametrize("name", list(PATHS))
def test_parity_with_simulate_trade(name, tmp_path):
    df = path(name)
    expected = er.simulate_trade(df, T, ENTRY, STOP, er.respected_ma(df, T))
    got_r, got_bar, got_trim, b, ledger, trade = replay(df, tmp_path)
    assert got_r == pytest.approx(expected[0], abs=1e-9)
    assert got_bar == expected[1]
    assert got_trim == expected[2]
    # no bot SELL is left working once the trade is flat (OCA cleaned up after itself)
    if b._state.positions[0]['shares'] <= 0:
        assert not [o for o in b._state.open_orders if o['action'] == 'SELL']
        assert trade['status'] == 'closed'


def test_parity_cases_cover_every_exit_path(tmp_path):
    """Guard against the path set silently drifting: each case exercises the branch it is named for."""
    out = {n: er.simulate_trade(path(n), T, ENTRY, STOP, er.respected_ma(path(n), T)) for n in PATHS}
    assert out['slow_trim'][2] and out['gap_through_trim'][2]
    assert first_hit('slow_trim') == T + FAST + 1 and first_hit('gap_through_trim') == T + FAST + 1
    assert first_hit('fast_trail') <= T + FAST and first_hit('breakeven_stop') <= T + FAST
    assert not out['fast_trail'][2] and out['fast_trail'][0] > er.TRIM_R
    assert out['stop_out'][0] < -1 + 1e-9
    assert out['breakeven_stop'][0] == pytest.approx((49.5 - ENTRY) / R)
    assert out['time_stop'][1] == T + er.TIME_STOP_SESSIONS
    assert out['open_at_end'][1] == len(path('open_at_end')) - 1
    _, _, _, b, _, _ = replay(path('gap_through_trim'), tmp_path)
    trim_fill = [f for f in b.fills if f[3].startswith("BOT-TRIM")]
    assert trim_fill and trim_fill[0][2] == pytest.approx(GAP_OPEN)      # max(open, target) on the gap


def test_time_stop_without_close_window_is_a_tagged_next_open_deviation(tmp_path):
    df = path('time_stop')
    _, bar, _, b, ledger, trade = replay(df, tmp_path, close_window=False)
    assert bar == T + er.TIME_STOP_SESSIONS + 1            # one bar late: sold at the next open
    exits = [e for e in ledger.events() if e.get('role') == 'exit']
    assert exits and exits[0]['reason'] == em.DEVIATION_TIME_STOP_NEXT_OPEN


# ─── Single-night behaviour ───

D0 = date(2025, 3, 27)                       # bar T of every path (price_path starts 2025-01-02)


@pytest.fixture
def ledger(tmp_path):
    return OrderLedger(tmp_path / "ledger.jsonl", run_id="t", clock=lambda: datetime(2025, 4, 30, 16, 30),
                       source="exit_manager")


def night(df, i, ledger, broker, exit_state, submit=True):
    broker.i = i
    return em.run(broker, ledger, exit_state, datetime.combine(day(df, i), time(16, 30)), submit=submit)[0]


def setup_trade(name, ledger):
    df = path(name)
    assert day(df, T) == D0
    b = SimBroker(df)
    open_trade(ledger, b, D0)
    return df, b, {'trades': {}}


def actions_of(rows, kind):
    return [a for a in rows if a['action'] == kind]


def test_stale_bars_change_nothing(ledger):
    df, b, st = setup_trade('slow_trim', ledger)
    b.i = T + 3
    rows = em.run(b, ledger, st, datetime.combine(day(df, T + 4), time(16, 30)), submit=True)[0]
    assert [a['action'] for a in rows] == [em.ALERT_STALE_BARS]
    assert not (b.placed or b.modified or b.cancelled)
    trade = next(iter(st['trades'].values()))
    assert trade['history'] == [] and trade['current_stop'] == STOP


def test_trim_order_only_after_the_fast_window_and_once(ledger):
    df, b, st = setup_trade('slow_trim', ledger)
    for i in range(T, T + FAST):               # inside the fast window: no resting trim yet
        assert not actions_of(night(df, i, ledger, b, st), em.PLACE_TRIM)
    rows = night(df, T + FAST, ledger, b, st)  # first session after it
    trims = actions_of(rows, em.PLACE_TRIM)
    assert len(trims) == 1 and trims[0]['qty'] == TRIM_QTY and trims[0]['price'] == round(TARGET, 2)
    placed = b.placed[-1][2]
    assert (placed.orderType, placed.tif, placed.lmtPrice, placed.ocaGroup, placed.ocaType) == \
        ("LMT", "GTC", round(TARGET, 2), make_oca_group(D0, "NVDA"), 3)
    assert not actions_of(night(df, T + FAST, ledger, b, st), em.PLACE_TRIM)    # rerun: already working


def test_no_trim_ever_when_the_target_comes_fast(ledger):
    df, b, st = setup_trade('fast_trail', ledger)
    for i in range(T, T + 12):
        b.intraday(i)
        rows = night(df, i, ledger, b, st)
        assert not actions_of(rows, em.PLACE_TRIM) and not actions_of(rows, em.PLACE_TRIM_MOO)
        if b._state.positions[0]['shares'] <= 0:
            break
    assert next(iter(st['trades'].values()))['hold_full']


def test_breakeven_modifies_the_child_stop_at_the_open_quantity(ledger):
    df, b, st = setup_trade('fast_trail', ledger)
    for i in range(T, first_hit('fast_trail') + 1):
        b.intraday(i)
        rows = night(df, i, ledger, b, st)
    mods = actions_of(rows, em.MODIFY_STOP)
    assert len(mods) == 1 and mods[0]['price'] == ENTRY and mods[0]['target_order_id'] == 1044
    oid, _, order = b.modified[-1]
    assert oid == 1044 and order.auxPrice == ENTRY and float(order.totalQuantity) == SHARES
    assert order.parentId == 1043 and order.tif == "GTC"
    stop_row = [e for e in ledger.events() if e.get('role') == 'stop'][-1]
    assert stop_row['aux_price'] == ENTRY                      # recorded before the modify


def test_trim_fill_is_synced_before_manage(ledger):
    """A filled trim makes trimmed True before manage() runs: no extra next-open trim."""
    df, b, st = setup_trade('slow_trim', ledger)
    for i in range(T, T + FAST + 2):
        b.intraday(i)
        rows = night(df, i, ledger, b, st)
    trade = next(iter(st['trades'].values()))
    assert trade['trimmed'] and trade['shares_open'] == REST
    assert not actions_of(rows, em.PLACE_TRIM_MOO)
    mod = actions_of(rows, em.MODIFY_STOP)[0]
    assert mod['qty'] == REST                                   # OCA already reduced the stop


def test_missing_trim_falls_back_to_a_tagged_next_open_sale(ledger):
    df, b, st = setup_trade('slow_trim', ledger)
    trade_rows = []
    for i in range(T, T + FAST + 2):
        b.intraday(i)
        trade_rows = night(df, i, ledger, b, st, submit=(i < T + FAST))   # trim never placed: dry runs
        for o in [o for o in b._state.open_orders if o['order_ref'].startswith("BOT-TRIM")]:
            b.cancel(o['order_id'])
    moo = actions_of(trade_rows, em.PLACE_TRIM_MOO)
    assert len(moo) == 1 and moo[0]['tag'] == em.DEVIATION_TRIM_NEXT_OPEN and moo[0]['qty'] == TRIM_QTY
    assert (moo[0]['order_type'], moo[0]['tif']) == ('MKT', 'OPG')


def test_exit_cancels_the_trim_and_sizes_the_moo_to_shares_open(ledger):
    df, b, st = setup_trade('slow_trim', ledger)
    for i in range(T, T + FAST + 1):
        b.intraday(i)
        night(df, i, ledger, b, st)
    assert any(o['order_ref'].startswith("BOT-TRIM") for o in b._state.open_orders)
    trade = next(iter(st['trades'].values()))
    trade['reached_2r'] = True                 # force the trail exit with the trim still resting
    trade['pending_exit'] = True
    rows = night(df, T + FAST, ledger, b, st)
    assert [a['action'] for a in rows if a['action'] in (em.CANCEL_TRIM, em.PLACE_EXIT_MOO)] == \
        [em.CANCEL_TRIM, em.PLACE_EXIT_MOO]
    moo = actions_of(rows, em.PLACE_EXIT_MOO)[0]
    assert moo['qty'] == SHARES and (moo['order_type'], moo['tif']) == ('MKT', 'OPG')
    assert b.cancelled and trade['status'] == 'closing'
    assert not actions_of(night(df, T + FAST, ledger, b, st), em.PLACE_EXIT_MOO)   # no second exit


def test_time_stop_uses_moc_in_the_close_window(ledger):
    df, b, st = setup_trade('time_stop', ledger)
    for i in range(T, T + er.TIME_STOP_SESSIONS):
        b.intraday(i)
        night(df, i, ledger, b, st)
        b.i = i + 1
        cw = em.run(b, ledger, st, datetime.combine(day(df, i + 1), time(15, 40)), submit=True, close_window=True)[0]
        if i + 1 < T + er.TIME_STOP_SESSIONS:
            assert cw == []
    moc = actions_of(cw, em.PLACE_EXIT_MOC)
    assert len(moc) == 1 and moc[0]['qty'] == SHARES and moc[0]['tif'] == 'DAY' and moc[0]['tag'] == ''
    assert b.bar_calls[-1] == ("NVDA", True)                   # close window asks for the forming bar


def test_manual_touched_trade_is_report_only(ledger):
    df, b, st = setup_trade('slow_trim', ledger)
    b._state.positions[0]['shares'] = 250.0                     # someone sold 50 by hand
    rows = night(df, T + 6, ledger, b, st)
    assert [a['action'] for a in rows] == [em.REPORT]
    assert not (b.placed or b.modified or b.cancelled)
    assert next(iter(st['trades'].values()))['status'] == 'manual_touched'


def test_dry_run_never_touches_orders_and_marks_results(ledger):
    df, b, st = setup_trade('fast_trail', ledger)
    b.submit_enabled = False
    rows = []
    for i in range(T, T + 12):
        b.intraday(i)
        rows += night(df, i, ledger, b, st, submit=False)
    assert not (b.placed or b.modified or b.cancelled)
    assert actions_of(rows, em.MODIFY_STOP) and all(a['result'] == 'dry_run' for a in rows
                                                     if a['action'] in em.ORDER_ACTIONS)


def test_stop_placed_by_another_client_is_alert_only(ledger, tmp_path):
    df = path('fast_trail')
    b = SimBroker(df)
    open_trade(ledger, b, D0, client_id=12)
    st = {'trades': {}}
    for i in range(T, T + 4):
        b.intraday(i)
        rows = night(df, i, ledger, b, st)
    assert actions_of(rows, em.ALERT_OTHER_CLIENT) and not b.modified


def test_missing_stop_places_a_stop_fix_at_night(ledger):
    """No bot stop at night (a full trim fill may have cancelled it, a partial entry's child may have
    died with its parent): the nightly run places one instead of only alerting."""
    df, b, st = setup_trade('slow_trim', ledger)
    b._state.open_orders.clear()
    rows = night(df, T, ledger, b, st)
    fix = actions_of(rows, em.PLACE_STOP_FIX)
    assert len(fix) == 1 and fix[0]['qty'] == SHARES and fix[0]['price'] == STOP
    assert fix[0]['result'].startswith('working')
    order = b.placed[-1][2]
    assert (order.orderType, order.tif, order.auxPrice, order.ocaGroup, order.ocaType) == \
        ("STP", "GTC", STOP, make_oca_group(D0, "NVDA"), 3)
    assert order.totalQuantity == SHARES and isinstance(order.totalQuantity, int)
    assert order.orderRef == make_order_ref('stop_fix', D0, "NVDA")
    assert not actions_of(night(df, T, ledger, b, st), em.PLACE_STOP_FIX)    # rerun: it is working


def test_missing_stop_dry_run_only_plans(ledger):
    df, b, st = setup_trade('slow_trim', ledger)
    b._state.open_orders.clear()
    b.submit_enabled = False
    rows = night(df, T, ledger, b, st, submit=False)
    assert actions_of(rows, em.PLACE_STOP_FIX)[0]['result'] == 'dry_run' and not b.placed


def test_stop_fix_sent_but_not_at_tws_is_an_alert(ledger):
    df, b, st = setup_trade('slow_trim', ledger)
    b._state.open_orders.clear()
    b.ack = ("unknown", "no order status within 10s")
    place = b.place

    def place_unlisted(oid, contract, order):       # sent, but TWS doesn't list it (yet)
        place(oid, contract, order)
        b._state.open_orders.pop()
    b.place = place_unlisted
    night(df, T, ledger, b, st)
    rows = night(df, T, ledger, b, st)
    assert actions_of(rows, em.ALERT_NO_STOP) and not actions_of(rows, em.PLACE_STOP_FIX)


def _trim_lmt(b):
    return next(o for o in b._state.open_orders if o['order_ref'].startswith("BOT-TRIM"))


def test_unfilled_trim_lmt_at_the_target_is_replaced_by_a_next_open_sale(ledger):
    """The target printed but the resting LMT didn't fill: cancel it AND sell the trim at the next open
    in the same run; trimmed stays False until the sale fills."""
    df, b, st = setup_trade('slow_trim', ledger)
    for i in range(T, T + FAST + 1):
        b.intraday(i)
        night(df, i, ledger, b, st)
    _trim_lmt(b)['lmt_price'] = 1e9                  # the high touched the target but the LMT didn't fill
    b.intraday(T + FAST + 1)
    rows = night(df, T + FAST + 1, ledger, b, st)
    trade = next(iter(st['trades'].values()))
    assert [a['action'] for a in rows if a['action'] in (em.CANCEL_TRIM, em.PLACE_TRIM_MOO)] == \
        [em.CANCEL_TRIM, em.PLACE_TRIM_MOO]
    moo = actions_of(rows, em.PLACE_TRIM_MOO)[0]
    assert moo['qty'] == TRIM_QTY and moo['tag'] == em.DEVIATION_TRIM_NEXT_OPEN and moo['result'].startswith('working')
    assert not trade['trimmed']
    b.intraday(T + FAST + 2)                         # the MOO fills at the open
    rows = night(df, T + FAST + 2, ledger, b, st)
    assert trade['trimmed'] and trade['shares_open'] == REST
    assert not actions_of(rows, em.PLACE_TRIM_MOO) and not actions_of(rows, em.PLACE_TRIM)


def test_partial_trim_fill_sells_the_rest_next_open(ledger):
    df, b, st = setup_trade('slow_trim', ledger)
    for i in range(T, T + FAST + 1):
        b.intraday(i)
        night(df, i, ledger, b, st)
    lmt = _trim_lmt(b)
    lmt['lmt_price'] = 1e9
    b.intraday(T + FAST + 1)
    # 40 of the trim LMT fill; OCA type 3 reduces the stop by 40
    b._exec += 1
    b._state.executions.append({'exec_id': 'p1', 'order_id': lmt['order_id'], 'perm_id': 0, 'client_id': 16,
                                'order_ref': lmt['order_ref'], 'symbol': 'NVDA', 'side': 'SLD', 'shares': 40.0,
                                'price': round(TARGET, 2), 'account': ACCT})
    lmt['filled'] = 40.0
    b._state.positions[0]['shares'] -= 40
    stop = next(o for o in b._state.open_orders if o['order_ref'].startswith("BOT-STOP"))
    stop['quantity'] -= 40
    rows = night(df, T + FAST + 1, ledger, b, st)
    trade = next(iter(st['trades'].values()))
    assert not trade['trimmed'] and trade['trim_filled'] == 40 and trade['shares_open'] == SHARES - 40
    assert actions_of(rows, em.CANCEL_TRIM)
    assert actions_of(rows, em.PLACE_TRIM_MOO)[0]['qty'] == TRIM_QTY - 40


def test_rejected_trim_moo_is_retried_the_next_night(ledger):
    df, b, st = setup_trade('slow_trim', ledger)
    for i in range(T, T + FAST + 1):
        b.intraday(i)
        night(df, i, ledger, b, st, submit=(i < T + FAST))   # no resting LMT ever placed
    b.intraday(T + FAST + 1)
    b.ack = ("rejected", "TWS 201: rejected")
    rows = night(df, T + FAST + 1, ledger, b, st)
    assert actions_of(rows, em.PLACE_TRIM_MOO)[0]['result'].startswith('rejected')
    for o in [o for o in b._state.open_orders if o['order_ref'].startswith("BOT-TRIM")]:
        b._close(o, 'Cancelled')                             # TWS dropped the rejected order
    trade = next(iter(st['trades'].values()))
    assert not trade['trimmed']
    b.ack = ("working", "PreSubmitted")
    rows = night(df, T + FAST + 1, ledger, b, st)
    assert actions_of(rows, em.PLACE_TRIM_MOO)[0]['qty'] == TRIM_QTY


def test_oversized_stop_is_resized_at_night(ledger):
    """Entry filled 120 of 300 and expired; the child still covers 300: shrink it to 120."""
    df = path('slow_trim')
    b = SimBroker(df)
    open_trade(ledger, b, D0, shares=120)
    stop = next(o for o in b._state.open_orders if o['order_ref'].startswith("BOT-STOP"))
    stop['quantity'] = 300.0
    rows = night(df, T, ledger, b, {'trades': {}})
    mod = actions_of(rows, em.MODIFY_STOP)
    assert len(mod) == 1 and mod[0]['qty'] == 120 and mod[0]['price'] == STOP
    assert b.modified[-1][2].totalQuantity == 120


def test_modify_keeps_the_filled_part_of_a_partly_filled_stop(ledger):
    b = FakeBroker(account=ACCT)
    trade = {'entry_date': D0.isoformat(), 'symbol': 'NVDA', 'setup': 'PIVOT_BREAKOUT',
             'oca_group': make_oca_group(D0, 'NVDA')}
    target = {'order_id': 1044, 'client_id': 16, 'quantity': 300.0, 'filled': 100.0, 'aux_price': 47.0,
              'order_ref': make_order_ref('stop', D0, 'NVDA'), 'parent_id': 1043, 'tif': 'GTC',
              'order_type': 'STP', 'oca_group': make_oca_group(D0, 'NVDA')}
    a = em._row(trade, em.MODIFY_STOP, role='stop', order_type='STP', tif='GTC', qty=200.0, price=50.0,
                order_ref=target['order_ref'], target=target)
    em._modify_stop(a, b, ledger)
    order = b.modified[-1][2]
    assert order.totalQuantity == 300 and isinstance(order.totalQuantity, int) and order.auxPrice == 50.0


def test_new_trade_uses_the_real_fill_and_the_child_stop(ledger):
    df, b, st = setup_trade('slow_trim', ledger)
    night(df, T, ledger, b, st)
    t = st['trades'][f"{D0.isoformat()}:NVDA:PIVOT_BREAKOUT"]
    assert (t['entry_price'], t['initial_stop'], t['shares_entered'], t['trim_qty']) == (ENTRY, STOP, SHARES, TRIM_QTY)
    assert t['trail_line'] == er.respected_ma(df, T) and t['oca_group'] == make_oca_group(D0, "NVDA")


def test_tiny_position_has_no_trim():
    t = em.new_trade({'trade_date': '2026-11-02', 'symbol': 'X', 'setup': 'S', 'filled_qty': 1,
                      'avg_fill_price': 10.0, 'plan': {'stop': 9.0}},
                     OrderLedger(path=__import__('pathlib').Path('/nonexistent/l.jsonl')))
    assert t['trim_qty'] == 0


# ─── exit_state file ───

def test_exit_state_write_is_atomic(tmp_path, monkeypatch):
    p = tmp_path / "exit_state.json"
    em.save_exit_state({'trades': {'a': {'x': 1}}}, p)
    real_dump = json.dump

    def boom(obj, fh, **kw):
        fh.write('{"partial": ')
        raise OSError("disk full")
    monkeypatch.setattr(em.json, "dump", boom)
    with pytest.raises(OSError):
        em.save_exit_state({'trades': {'b': {}}}, p)
    monkeypatch.setattr(em.json, "dump", real_dump)
    assert em.load_exit_state(p)['trades'] == {'a': {'x': 1}}
    assert not (tmp_path / "exit_state.json.tmp").exists()


def test_unreadable_exit_state_refuses(tmp_path):
    p = tmp_path / "exit_state.json"
    p.write_text("{not json")
    with pytest.raises(RuntimeError):
        em.load_exit_state(p)


# ─── Time gates and CLI ───

def test_time_gates():
    assert em.time_gate(datetime(2026, 11, 4, 15, 59), close_window=False)
    assert em.time_gate(datetime(2026, 11, 4, 16, 5), close_window=False) == ""
    assert em.time_gate(datetime(2026, 11, 27, 13, 5), close_window=False) == ""     # early close
    assert em.time_gate(datetime(2026, 11, 4, 15, 40), close_window=True) == ""
    assert em.time_gate(datetime(2026, 11, 4, 15, 49), close_window=True)
    assert em.time_gate(datetime(2026, 11, 27, 12, 40), close_window=True) == ""
    assert em.time_gate(datetime(2026, 11, 27, 15, 40), close_window=True)


def _main_env(monkeypatch, tmp_path, now):
    monkeypatch.setattr(ol, "ORDERS_DIR", tmp_path)
    monkeypatch.setattr(em, "now_et", lambda: now)
    monkeypatch.setenv("TRADING_NOTIFY", "0")


def test_main_holiday_and_window_refusals(monkeypatch, tmp_path):
    import order_broker
    monkeypatch.setattr(order_broker, "open_session", lambda *a, **k: pytest.fail("must not connect"))
    _main_env(monkeypatch, tmp_path, datetime(2026, 11, 26, 17, 0))          # Thanksgiving
    assert em.main(["--account", ACCT]) == 0
    _main_env(monkeypatch, tmp_path, datetime(2026, 11, 4, 14, 0))
    assert em.main(["--account", ACCT]) == 2
    assert em.main(["--account", ACCT, "--close-window"]) == 2


def test_main_submit_without_config_runs_read_only(monkeypatch, tmp_path):
    import order_broker
    calls = []

    def fake_open(port, account, cli_submit, client_id):
        calls.append((port, cli_submit, client_id))
        return FakeBroker(account=ACCT, submit_enabled=False)
    monkeypatch.setattr(order_broker, "open_session", fake_open)
    monkeypatch.setattr(config, "AUTO_SUBMIT_ENABLED", False, raising=False)
    _main_env(monkeypatch, tmp_path, datetime(2026, 11, 4, 16, 30))
    assert em.main(["--account", ACCT, "--submit"]) == 2
    assert calls == [(7497, True, 17)]
    assert not (tmp_path / "exit_state.json").exists()              # dry runs never write exit_state
    assert (tmp_path / "exit_plan_2026-11-04.csv").exists()


def test_main_lock_held_exits_3(monkeypatch, tmp_path):
    import order_broker
    monkeypatch.setattr(order_broker, "open_session", lambda *a, **k: pytest.fail("must not connect"))
    _main_env(monkeypatch, tmp_path, datetime(2026, 11, 4, 16, 30))
    with ol.ledger_lock(tmp_path):
        assert em.main(["--account", ACCT]) == 3


def test_exit_plan_csv_columns(tmp_path):
    a = em._row({'entry_date': '2026-11-02', 'symbol': 'NVDA', 'setup': 'PIVOT_BREAKOUT'}, em.HOLD)
    p = em.write_exit_plan([a], tmp_path / "plan.csv", now=datetime(2026, 11, 4, 16, 30))
    em.write_exit_plan([a], p, now=datetime(2026, 11, 4, 16, 31))
    df = pd.read_csv(p)
    assert list(df.columns) == em.PLAN_COLUMNS and len(df) == 2


# ─── TRIM_STOP_MODE split_stop (the fallback if paper check 1 shows TWS cancels the stop) ───

OCA_SEMANTICS = ('quantity', 'proportional', 'cancel')


@pytest.fixture
def split_mode(monkeypatch):
    monkeypatch.setattr(config, 'TRIM_STOP_MODE', 'split_stop', raising=False)


def _always_covered(b, when):
    stops, held = stop_cover(b)
    assert stops == pytest.approx(held), f"{when}: stops cover {stops:g}, position {held:g}"
    assert held >= 0, f"{when}: short {held:g}"


@pytest.mark.parametrize("oca", OCA_SEMANTICS)
@pytest.mark.parametrize("name", list(PATHS))
def test_split_stop_parity_under_every_oca_behaviour(name, oca, tmp_path, split_mode):
    """Same R, exit bar and trim as simulate_trade, and the stops cover exactly the position after
    every step, whether a full OCA fill reduces by quantity, by fraction or cancels the group."""
    df = path(name)
    expected = er.simulate_trade(df, T, ENTRY, STOP, er.respected_ma(df, T))
    got_r, got_bar, got_trim, b, ledger, trade = replay(df, tmp_path, oca=oca, check=_always_covered)
    assert got_r == pytest.approx(expected[0], abs=1e-9)
    assert got_bar == expected[1]
    assert got_trim == expected[2]
    if b._state.positions[0]['shares'] <= 0:
        assert not [o for o in b._state.open_orders if o['action'] == 'SELL']
        assert trade['status'] == 'closed'


def test_shared_oca_leaves_the_position_naked_if_tws_cancels(tmp_path):
    """Why split_stop exists: under cancel semantics the trim fill kills the shared stop and the
    shares left have no stop for the rest of the session, until the nightly run replaces it."""
    df = path('slow_trim')
    naked = []

    def watch(b, when):
        stops, held = stop_cover(b)
        if held > 0 and stops < held:
            naked.append(when)
    replay(df, tmp_path, oca='cancel', check=watch)
    assert naked and naked[0].endswith('intraday') and not any('nightly' in w for w in naked)


def test_split_builds_the_trim_block_in_order(ledger, split_mode):
    df, b, st = setup_trade('slow_trim', ledger)
    for i in range(T, T + FAST):
        night(df, i, ledger, b, st)
    rows = night(df, T + FAST, ledger, b, st)                 # first session after the fast window: the resting trim
    acts = [(a['action'], a['qty']) for a in rows if a['action'] in em.ORDER_ACTIONS]
    # the trim block's stop first, then the shrink: a failure in between covers twice, never naked
    assert acts == [(em.PLACE_TRIM_STOP, TRIM_QTY), (em.MODIFY_STOP, REST), (em.PLACE_TRIM, TRIM_QTY)]
    trim_oca = ol.make_trim_oca_group(D0, "NVDA")
    tstop, trim = b.placed[-2][2], b.placed[-1][2]
    assert (tstop.orderType, tstop.auxPrice, tstop.ocaGroup, tstop.ocaType, tstop.tif) == ("STP", STOP, trim_oca, 3, "GTC")
    assert tstop.orderRef == make_order_ref('trim_stop', D0, "NVDA")
    assert (trim.ocaGroup, trim.lmtPrice) == (trim_oca, round(TARGET, 2))
    child = next(o for o in b._state.open_orders if o['order_id'] == 1044)
    assert child['quantity'] == REST and child['oca_group'] == make_oca_group(D0, "NVDA")
    again = night(df, T + FAST, ledger, b, st)                # rerun: nothing new
    assert not [a for a in again if a['action'] in em.ORDER_ACTIONS]


def test_split_breakeven_moves_both_block_stops(ledger, split_mode):
    df, b, st = setup_trade('slow_trim', ledger)
    for i in range(T, T + FAST + 1):
        night(df, i, ledger, b, st)
    trade = next(iter(st['trades'].values()))
    trade['hold_full'] = True                              # keep the trim block, then move the stop
    trade['current_stop'] = 49.0
    b._state.open_orders = [o for o in b._state.open_orders if not o['order_ref'].startswith("BOT-TRIM-")]
    decisions = [{'decision': em.STOP_TO, 'price': 50.0}, {'decision': em.NO_TRIM}]
    rows = em.plan_orders(trade, decisions, b._state.open_orders, ledger)
    mods = sorted((a['role'], a['qty'], a['price']) for a in rows if a['action'] == em.MODIFY_STOP)
    assert mods == [('stop', REST, 50.0), ('trim_stop', TRIM_QTY, 50.0)]


def test_split_exit_is_one_order_per_block(ledger, split_mode):
    df, b, st = setup_trade('slow_trim', ledger)
    for i in range(T, T + FAST + 1):
        night(df, i, ledger, b, st)
    trade = next(iter(st['trades'].values()))
    trade.update(reached_2r=True, pending_exit=True)
    rows = night(df, T + FAST, ledger, b, st)
    exits = sorted((a['qty'], a['oca_group']) for a in rows if a['action'] == em.PLACE_EXIT_MOO)
    assert exits == sorted([(TRIM_QTY, ol.make_trim_oca_group(D0, "NVDA")), (REST, make_oca_group(D0, "NVDA"))])
    assert actions_of(rows, em.CANCEL_TRIM)
    assert not actions_of(night(df, T + FAST, ledger, b, st), em.PLACE_EXIT_MOO)   # both working: none again


def test_split_cancels_a_trim_stop_left_after_the_trim_filled(ledger, split_mode):
    """If TWS doesn't reduce the trim block's stop at all, the nightly run cancels it."""
    df, b, st = setup_trade('slow_trim', ledger)
    for i in range(T, T + FAST + 1):
        night(df, i, ledger, b, st)
    tstop = next(o for o in b._state.open_orders if o['order_ref'].startswith("BOT-TSTOP"))
    trim = next(o for o in b._state.open_orders if o['order_ref'].startswith("BOT-TRIM-"))
    b.i = T + FAST + 1
    b._fill(trim, float(TRIM_QTY), round(TARGET, 2))
    b._state.open_orders.append({**tstop, 'quantity': float(TRIM_QTY)})   # "no reduce": the stop is still there
    rows = night(df, T + FAST + 1, ledger, b, st)
    assert [a['action'] for a in rows if a['action'] == em.CANCEL_TRIM_STOP] == [em.CANCEL_TRIM_STOP]
    assert not actions_of(rows, em.PLACE_TRIM_STOP)


def test_split_refuses_the_trim_if_the_main_stop_belongs_to_another_client(ledger, tmp_path, split_mode):
    df = path('slow_trim')
    b = SimBroker(df)
    open_trade(ledger, b, D0, client_id=99)
    st = {'trades': {}}
    for i in range(T, T + FAST):
        night(df, i, ledger, b, st)
    rows = night(df, T + FAST, ledger, b, st)
    assert not actions_of(rows, em.PLACE_TRIM) and not actions_of(rows, em.PLACE_TRIM_STOP)
    assert actions_of(rows, em.ALERT_OTHER_CLIENT) and actions_of(rows, em.ALERT_NO_STOP)


def test_unknown_trim_stop_mode_is_refused(monkeypatch):
    monkeypatch.setattr(config, 'TRIM_STOP_MODE', 'split', raising=False)
    with pytest.raises(ValueError):
        em.trim_stop_mode()


def _build_block(ledger, split_mode_on=True):
    df, b, st = setup_trade('slow_trim', ledger)
    for i in range(T, T + FAST + 1):
        night(df, i, ledger, b, st)
    return df, b, st


def test_split_keeps_the_trim_stop_when_it_is_all_that_is_left(ledger, split_mode):
    """Main block sold (its MOO filled), the trim block's MOO died: the trim_stop is the only stop and stays."""
    df, b, st = _build_block(ledger)
    trade = next(iter(st['trades'].values()))
    trade.update(reached_2r=True, pending_exit=True)
    night(df, T + FAST, ledger, b, st)
    b.i = T + FAST + 1
    main_exit = next(o for o in b._state.open_orders if o['order_ref'].startswith("BOT-EXIT")
                     and o['oca_group'] == make_oca_group(D0, "NVDA"))
    b._fill(main_exit, main_exit['quantity'], 55.0)
    for o in [o for o in b._state.open_orders if o['order_ref'].startswith("BOT-EXIT")]:
        b._close(o, 'Cancelled')                                 # the trim block's MOO died unfilled
    rows = night(df, T + FAST + 1, ledger, b, st)
    assert not actions_of(rows, em.CANCEL_TRIM_STOP) and not actions_of(rows, em.PLACE_STOP_FIX)
    assert not [a for a in rows if str(a['result']).startswith('refused')]
    stops, held = stop_cover(b)
    assert held == TRIM_QTY and stops == TRIM_QTY
    exits = actions_of(rows, em.PLACE_EXIT_MOO)
    assert [(a['qty'], a['oca_group']) for a in exits] == [(TRIM_QTY, ol.make_trim_oca_group(D0, "NVDA"))]


def test_split_is_not_started_when_the_guard_would_refuse_it(ledger, split_mode):
    """A manual SELL makes the trim block's stop an oversell: the main stop is NOT shrunk."""
    df, b, st = setup_trade('slow_trim', ledger)
    b._state.open_orders.append({"order_id": 77, "perm_id": 77, "client_id": 0, "account": ACCT, "symbol": "NVDA",
                                 "sec_type": "STK", "action": "SELL", "order_type": "LMT", "quantity": 10.0,
                                 "aux_price": 0.0, "lmt_price": 80.0, "status": "Submitted", "order_ref": "",
                                 "parent_id": 0, "tif": "GTC", "oca_group": "", "transmit": True})
    for i in range(T, T + FAST + 1):
        rows = night(df, i, ledger, b, st)
    assert actions_of(rows, em.PLACE_TRIM_STOP)[0]['result'].startswith('refused')
    assert all(a['result'].startswith('skipped') for a in rows
               if a['action'] in (em.PLACE_TRIM, em.MODIFY_STOP))
    child = next(o for o in b._state.open_orders if o['order_id'] == 1044)
    assert child['quantity'] == SHARES and not b.modified


@pytest.mark.parametrize("ack", [("rejected", "201 test reject"), ("unknown", "no order status within 10s")])
def test_split_never_shrinks_until_the_trim_stop_is_working(ledger, split_mode, ack):
    """Rejected or unconfirmed trim block stop: the main stop keeps its full size, no trim goes out.
    (unknown: the trim stop may be live, so stops cover more than held until the next run, never fewer)"""
    df, b, st = setup_trade('slow_trim', ledger)
    for i in range(T, T + FAST):
        night(df, i, ledger, b, st)
    b.ack = ack
    rows = night(df, T + FAST, ledger, b, st)
    assert actions_of(rows, em.PLACE_TRIM_STOP)[0]['result'].startswith(ack[0])
    assert actions_of(rows, em.PLACE_TRIM)[0]['result'].startswith('skipped')
    child = next(o for o in b._state.open_orders if o['order_id'] == 1044)
    assert child['quantity'] == SHARES and not b.modified
    assert stop_cover(b)[0] >= SHARES


def test_split_connection_loss_on_the_trim_stop_leaves_the_main_stop_whole(ledger, split_mode):
    df, b, st = setup_trade('slow_trim', ledger)
    for i in range(T, T + FAST):
        night(df, i, ledger, b, st)
    b.fail_on_place, b.place_calls = 1, 0
    rows = night(df, T + FAST, ledger, b, st)
    assert actions_of(rows, em.PLACE_TRIM_STOP)[0]['result'].startswith('failed')
    assert not b.modified and next(o for o in b._state.open_orders if o['order_id'] == 1044)['quantity'] == SHARES


def test_split_refused_still_moves_the_stop_price(ledger, split_mode):
    """A refused trim block must not cost the breakeven move merged into the same modify."""
    df, b, st = setup_trade('slow_trim', ledger)
    b._state.open_orders.append({"order_id": 77, "perm_id": 77, "client_id": 0, "account": ACCT, "symbol": "NVDA",
                                 "sec_type": "STK", "action": "SELL", "order_type": "LMT", "quantity": 10.0,
                                 "aux_price": 0.0, "lmt_price": 80.0, "status": "Submitted", "order_ref": "",
                                 "parent_id": 0, "tif": "GTC", "oca_group": "", "transmit": True})
    for i in range(T, T + FAST):
        night(df, i, ledger, b, st)
    trade = next(iter(st['trades'].values()))
    trade['current_stop'] = 49.0
    acts = em.plan_orders(trade, [{'decision': em.STOP_TO, 'price': 49.0},
                                  {'decision': em.TRIM_RESTING, 'price': round(TARGET, 2), 'qty': TRIM_QTY}], b._state.open_orders, ledger)
    em.apply_actions(acts, b, ledger)
    child = next(o for o in b._state.open_orders if o['order_id'] == 1044)
    assert child['aux_price'] == 49.0 and child['quantity'] == SHARES


def test_split_regrows_a_main_stop_left_shrunk(ledger, split_mode):
    """A shrink whose trim block never got built (crash in between) is healed the next night."""
    df, b, st = setup_trade('slow_trim', ledger)
    for i in range(T, T + FAST):
        night(df, i, ledger, b, st)
    next(o for o in b._state.open_orders if o['order_id'] == 1044)['quantity'] = float(REST)
    trade = next(iter(st['trades'].values()))
    trade['hold_full'] = True                                    # no trim tonight: just the heal
    rows = em.plan_orders(trade, [{'decision': em.STOP_TO, 'price': STOP}, {'decision': em.NO_TRIM}],
                          b._state.open_orders, ledger)
    mods = actions_of(rows, em.MODIFY_STOP)
    assert [(m['qty'], m['target_order_id']) for m in mods] == [(SHARES, 1044)]


def test_shared_oca_default_keeps_the_old_guard_and_no_wait(ledger, monkeypatch):
    """Default mode: no pooling flag, no split marks, so nothing waits or prechecks."""
    df, b, st = setup_trade('slow_trim', ledger)
    for i in range(T, T + FAST + 1):
        rows = night(df, i, ledger, b, st)
    assert not [a for a in rows if a.get('pool_groups') or a.get('split_shrink')]


def test_close_window_never_builds_a_new_trim_block(ledger, split_mode):
    """15:40 run, trim resting, its block stop gone, main stop full size: no second stop during regular hours."""
    df, b, st = _build_block(ledger)
    for o in [o for o in b._state.open_orders if o['order_ref'].startswith("BOT-TSTOP")]:
        b._close(o, 'Cancelled')
    next(o for o in b._state.open_orders if o['order_id'] == 1044)['quantity'] = float(SHARES)
    trade = next(iter(st['trades'].values()))
    acts = em.plan_orders(trade, [{'decision': em.EXIT_AT_CLOSE, 'qty': SHARES, 'tag': '', 'note': ''}],
                          b._state.open_orders, ledger)
    assert not actions_of(acts, em.PLACE_TRIM_STOP) and not [a for a in acts if a.get('split_shrink')]
    assert [a['qty'] for a in actions_of(acts, em.PLACE_EXIT_MOC)] == [SHARES]
