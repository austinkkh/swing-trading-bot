#!/usr/bin/env python3
"""
paper_checks.py: guided, interactive runner for the broker-behaviour checks on the PAPER account
(see README, "Broker behaviour checks"), run by hand during market hours. Not scheduled.

Each check prints what it will do and the exact orders (symbol, side, qty, type, prices), waits for
you to type 'yes' before anything is transmitted, records what TWS reports, cleans up after itself
and writes PASS / FAIL / INCONCLUSIVE per check to Data/orders/paper_checks_<date>.json.

  1  Full fill of the partial trim: does ocaType 3 REDUCE the shared-OCA stop (what the code assumes)
     or CANCEL it? Needs the regular session; buys 3 shares.
  2  Gap-down open with stop + market-on-open (MOO) exit in one OCA group: can both fill? Can't be
     forced. Read-only: inspects the ledger and today's TWS executions for a real gap; INCONCLUSIVE
     until one has happened (procedure in --explain).
  3  Child stop when the parent fills in part and is then cancelled (stands in for "expires at the
     close"). Part A (always): unfilled parent cancelled -> is the child gone? Part B (regular
     session, optional): parent at the last price; a partial fill can't be forced, so B is often
     INCONCLUSIVE.
  4  Does reqAllOpenOrders list a parent staged with transmit=False and no child?
  5  Pre-open entry message codes: is 399 the routine message (order stays working)? Only before
     09:29:30 ET on a trading day; INCONCLUSIVE otherwise.

PASS = TWS behaves the way the order code assumes. FAIL = it doesn't: stop and fix the code before
enabling submission. INCONCLUSIVE = not observed (wrong time of day, no fill, you said no).

Safety:
  - goes through order_broker.open_session: paper ports only (live 7496 / 4001 refused before any
    connection), DU account only, and orders only if config AUTO_SUBMIT_ENABLED is the bool True AND
    --submit is passed. Check 2 alone runs read-only (client 17) without either.
  - client ORDER_CLIENT_ID (16), the bot's order client, and the order-ledger lock for the whole run,
    so no bot order run can start meanwhile. The bot ledger is never written.
  - every order has orderRef PAPERCHECK-...; the broker session can place nothing else. Size 1-5
    shares (MAX_SHARES) in one liquid low-priced name (SYMBOLS) that has no position and no open
    order in the account. Orders that must not fill rest FAR_PCT (25%) from the last price;
    orders that must fill are marketable limits MARKETABLE_PCT (3%) through it.
  - cleanup (after every check, and in a finally on Ctrl-C or any error) cancels only the order
    ids this run placed (plus PAPERCHECK leftovers you agree to cancel) and sells only the shares
    this run's own PAPERCHECK fills bought, after you type 'yes'. Never another order or position.

TWS precautionary settings may reject a limit 25% away or a marketable limit; that is recorded as
what TWS reported. Set AUTO_SUBMIT_ENABLED back to False after the session unless you are moving on
to the README's enable steps.

Usage:
  python scripts/paper_checks.py --explain                       # print the plan, no connection
  python scripts/paper_checks.py --account DU... --submit        # all checks, guided
  python scripts/paper_checks.py --account DU... --submit --check 1 --check 4
  python scripts/paper_checks.py --account DU... --check 2       # read-only gap observation
Exit codes: 0 ran (see the results file) · 1 error · 2 refused by a guard · 3 ledger lock held ·
130 interrupted (cleanup ran).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, time as dt_time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

import config  # noqa: E402
import order_ledger as ol  # noqa: E402
from ibkr_connector import LIVE_PORTS, ORDER_WARNING_CODES, PAPER_PORTS  # noqa: E402
from market_time import calendar_covers, in_regular_session, is_trading_day, now_et  # noqa: E402
from order_broker import (ORDER_CLIENT_ID, ORDER_READONLY_CLIENT_ID, PAPERCHECK_REF_PREFIX,  # noqa: E402
                          SUBMIT_REFUSED, open_session, stock_contract, submission_enabled)

logger = logging.getLogger(__name__)

SYMBOLS = ("F", "T", "BAC", "PFE")    # liquid, low-priced; the first one that is free is used
MAX_PRICE = 60.0                      # skip a candidate above this (keeps a 5-share test small)
MAX_SHARES = 5
FAR_PCT = 25.0                        # resting orders this far from the last price: no fill wanted
MARKETABLE_PCT = 3.0                  # limit this far through the last price: fill wanted
FILL_TIMEOUT_SEC = 60.0
SETTLE_SEC = 3.0                      # let TWS send status / OCA updates before reading them
POLL_SEC = 2.0
ACK_TIMEOUT_SEC = 10.0
PREOPEN_CUTOFF_ET = dt_time(9, 29, 30)
EPS = 1e-9
LIVE_SOURCE = 'last'                  # reference_price source of a current (not delayed) 1-min bar

PASS, FAIL, INCONCLUSIVE = "PASS", "FAIL", "INCONCLUSIVE"
CHECK_NAMES = {
    1: "Full fill of the partial trim: OCA type 3 reduces or cancels the stop",
    2: "Gap-down open: stop + MOO in one OCA, can both fill",
    3: "Child stop when the parent fills in part, then is cancelled / expires",
    4: "reqAllOpenOrders lists a staged (transmit=False) parent with no child",
    5: "Pre-open entry: 399 is a routine message, not a rejection",
}
CHECK_ORDER = (5, 4, 1, 3, 2)          # time-critical pre-open check first, read-only check last
TERMINAL_TWS = {'Filled', 'Cancelled', 'ApiCancelled', 'Inactive'}

GAP_PROCEDURE = (
    "Can't be forced. When a bot trade holds its GTC stop and exit_manager has placed a SELL MKT OPG "
    "(MOO) exit in the same OCA group, and the next open gaps below the stop, run "
    "`python scripts/paper_checks.py --account DU... --check 2` after 09:35 ET (and again after the "
    "next exit_manager / fill_monitor run has reconciled the ledger). FAIL = both the stop and the "
    "MOO filled (the account went short; fill_monitor --submit buys it back in the session). "
    "PASS = only one filled on a gap below the stop.")


def _px(value: float) -> float:
    """Price to the cent (these names trade in cents)."""
    return round(float(value) + 1e-9, 2)


@dataclass
class OrderSpec:
    """One order a check may send, printed for the owner before anything is transmitted."""
    role: str
    action: str
    qty: int
    order_type: str
    lmt: Optional[float] = None
    aux: Optional[float] = None
    tif: str = "DAY"
    transmit: bool = True
    oca_group: str = ""
    oca_type: int = 0
    note: str = ""

    def describe(self, symbol: str) -> str:
        prices = " ".join(p for p in (f"trigger {self.aux:.2f}" if self.aux is not None else "",
                                      f"limit {self.lmt:.2f}" if self.lmt is not None else "") if p)
        extra = [self.tif]
        if self.oca_group:
            extra.append(f"OCA {self.oca_group} type {self.oca_type}")
        if not self.transmit:
            extra.append("transmit=False (staged)")
        text = f"{self.action} {self.qty} {symbol} {self.order_type} {prices} ({', '.join(extra)})"
        return " ".join(text.split()) + (f"  [{self.note}]" if self.note else "")


@dataclass
class CheckResult:
    check: int
    name: str
    result: str = INCONCLUSIVE
    summary: str = ""
    orders: List[str] = field(default_factory=list)
    observations: Dict[str, object] = field(default_factory=dict)
    started: str = ""
    finished: str = ""

    def done(self, result: str, summary: str) -> "CheckResult":
        self.result, self.summary = result, summary
        return self


def _tws_status(broker, order_id: int) -> Optional[dict]:
    """Latest orderStatus TWS sent for this client's order (IBKRBroker: connector.order_status)."""
    conn = getattr(broker, 'connector', None)
    statuses = getattr(conn, 'order_status', None) if conn is not None else getattr(broker, 'order_status', None)
    return (statuses or {}).get(order_id)


def _order_messages(broker, order_id: int) -> List[dict]:
    """Every TWS message on the order, warnings included (connector.order_messages)."""
    conn = getattr(broker, 'connector', None)
    msgs = getattr(conn, 'order_messages', None) if conn is not None else getattr(broker, 'order_messages', None)
    return [{'code': m.get('code'), 'msg': m.get('msg')} for m in (msgs or {}).get(order_id, [])]


class CheckSession:
    """
    One guided session on one symbol. Places, cancels and sells ONLY its own PAPERCHECK orders and
    the shares its own fills bought. ask / out / sleep / clock / now_fn are injectable for tests.
    """

    def __init__(self, broker, symbol: str, run_tag: str, ask: Callable[[str], str] = input,
                 out: Callable[[str], None] = print, sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic, now_fn: Callable[[], datetime] = now_et,
                 id_floor: int = 0, fill_timeout: float = FILL_TIMEOUT_SEC, settle_sec: float = SETTLE_SEC):
        self.broker = broker
        self.symbol = symbol.upper()
        self.run_tag = run_tag
        self.ask, self.out, self.sleep, self.clock, self.now = ask, out, sleep, clock, now_fn
        self.id_floor = int(id_floor)
        self.fill_timeout, self.settle_sec = fill_timeout, settle_sec
        self.placed: Dict[int, dict] = {}       # order id -> {ref, spec, cancel_requested}
        self.log: List[str] = []
        self.price: Optional[float] = None       # the reference price the current check is priced off
        self.price_source: str = 'none'          # reference_price's source for it ('last' = live 1-min bar)

    # ─── owner prompts ───

    def say(self, text: str) -> None:
        self.log.append(text)
        self.out(text)

    def confirm(self, question: str, suffix: str = "Type yes to go ahead (anything else = no): ") -> bool:
        """True only for an explicit 'yes'. Ctrl-C / end of input at the prompt counts as no."""
        try:
            answer = self.ask(f"{question} {suffix}")
        except (KeyboardInterrupt, EOFError):
            self.say("  (no answer: treated as no)")
            return False
        ok = str(answer or '').strip().lower() == 'yes'
        self.log.append(f"{question} -> {'yes' if ok else 'no'}")
        return ok

    def confirm_orders(self, check: int, specs: Sequence[OrderSpec], what: str) -> bool:
        self.say(f"\nCheck {check}: {CHECK_NAMES[check]}")
        self.say(f"  {what}")
        if self.price is not None:
            self.say(f"  Priced off {self.symbol} {self.price:.2f} ({self.price_source})")
            if self.price_source != LIVE_SOURCE:
                self.say("  WARNING: that quote is delayed or stale: compare it with the TWS window before "
                         "typing yes (a marketable limit never fills worse than its limit; it may just not fill)")
        self.say("  Orders this check may transmit:")
        for spec in specs:
            self.say(f"    - {spec.describe(self.symbol)}")
        return self.confirm(f"Transmit check {check}'s orders on {self.broker.account}?")

    # ─── orders ───

    def ref(self, check: int, role: str) -> str:
        return f"{PAPERCHECK_REF_PREFIX}{check}-{role.upper()}-{self.run_tag}-{self.symbol}"

    def place(self, check: int, spec: OrderSpec, parent_id: int = 0) -> int:
        from ibapi.order import Order
        if not 1 <= spec.qty <= MAX_SHARES:
            raise ValueError(f"paper checks send 1-{MAX_SHARES} shares, not {spec.qty}")
        order = Order()
        order.action, order.orderType, order.totalQuantity = spec.action, spec.order_type, spec.qty
        if spec.lmt is not None:
            order.lmtPrice = spec.lmt
        if spec.aux is not None:
            order.auxPrice = spec.aux
        order.tif, order.outsideRth, order.transmit = spec.tif, False, spec.transmit
        if spec.oca_group:
            order.ocaGroup, order.ocaType = spec.oca_group, spec.oca_type
        if parent_id:
            order.parentId = parent_id
        order.orderRef = self.ref(check, spec.role)
        oid = max(self.broker.next_order_ids(1)[0], self.id_floor)
        self.id_floor = oid + 1
        # Record before sending: if place() raises, cleanup still tries to cancel this id
        self.placed[oid] = {'ref': order.orderRef, 'spec': asdict(spec), 'check': check, 'cancel_requested': False}
        self.broker.place(oid, stock_contract(self.symbol), order)
        self.say(f"  sent {oid}: {spec.describe(self.symbol)} ref {order.orderRef}")
        return oid

    def adopt(self, order: dict) -> None:
        """A PAPERCHECK order left by an earlier run (owner agreed to cancel it)."""
        oid = int(order['order_id'])
        self.placed.setdefault(oid, {'ref': order.get('order_ref'), 'spec': {}, 'check': 0,
                                     'cancel_requested': False, 'leftover': True})

    def cancel(self, oid: int, state=None) -> bool:
        """
        Cancel one of this run's orders. order_broker.cancel has no orderRef check (cancelOrder takes
        only an id), so with a state this also refuses an id TWS lists for this client under a
        non-PAPERCHECK ref: client 16 also places the bot's BOT- orders.
        """
        if oid not in self.placed:
            raise PermissionError(f"order {oid} was not placed by this paper-check run: not cancelling it")
        if state is not None:
            foreign = [o for o in state.open_orders
                       if o.get('order_id') == oid and o.get('client_id') in (None, self.broker.client_id)
                       and not str(o.get('order_ref') or '').startswith(PAPERCHECK_REF_PREFIX)]
            if foreign:
                self.say(f"  NOT cancelling {oid}: TWS lists it as {foreign[0].get('order_ref')!r}, not a "
                         "PAPERCHECK order")
                return False
        try:
            self.broker.cancel(int(oid))
            self.placed[oid]['cancel_requested'] = True
            self.say(f"  cancel requested: {oid} ({self.placed[oid]['ref']})")
            return True
        except Exception as exc:
            self.say(f"  cancel of {oid} FAILED ({exc}): cancel it in TWS by hand")
            return False

    def ack(self, oid: int) -> Tuple[str, str]:
        return self.broker.wait_ack(oid, timeout=ACK_TIMEOUT_SEC)

    # ─── reading TWS ───

    def state(self):
        return self.broker.state()

    def view(self, oid: int, state=None) -> dict:
        """What TWS reports for one of our orders: open list, completed list, status only, or missing."""
        ref = self.placed[oid]['ref']
        state = state if state is not None else self.state()
        tws = _tws_status(self.broker, oid)
        opens = [o for o in state.open_orders if o.get('order_ref') == ref]
        if opens:
            o = opens[-1]
            qty, filled = float(o.get('quantity') or 0), float(o.get('filled') or 0)
            remaining = float(tws['remaining']) if tws and tws.get('remaining') is not None else qty - filled
            return {'where': 'open', 'status': o.get('status') or (tws or {}).get('status', ''),
                    'quantity': qty, 'filled': filled, 'remaining': remaining,
                    'oca_group': o.get('oca_group', ''), 'tws_status': tws}
        done = [c for c in state.completed_orders if c.get('order_ref') == ref]
        if done:
            c = done[-1]
            return {'where': 'completed', 'status': c.get('status', ''), 'quantity': float(c.get('quantity') or 0),
                    'filled': float(c.get('filled') or 0), 'remaining': 0.0,
                    'completed_status': c.get('completed_status', ''), 'tws_status': tws}
        if tws:
            return {'where': 'status_only', 'status': tws.get('status', ''), 'filled': tws.get('filled'),
                    'remaining': tws.get('remaining'), 'tws_status': tws}
        return {'where': 'missing', 'status': ''}

    @staticmethod
    def is_live(view: dict) -> bool:
        return view.get('where') in ('open', 'status_only') and view.get('status') not in TERMINAL_TWS

    def position(self, state=None) -> float:
        state = state if state is not None else self.state()
        return sum(float(p.get('shares') or 0) for p in state.positions
                   if str(p.get('symbol', '')).upper() == self.symbol and p.get('sec_type', 'STK') == 'STK'
                   and p.get('account') in (None, self.broker.account))

    def own_net_shares(self, state=None) -> float:
        """Shares this run's own fills bought minus sold in self.symbol (executions tagged with our refs;
        adopted leftovers can be on another symbol)."""
        state = state if state is not None else self.state()
        refs = {v['ref'] for v in self.placed.values()}
        seen, net = set(), 0.0
        for e in state.executions:
            if e.get('order_ref') not in refs or e.get('exec_id') in seen:
                continue
            if e.get('symbol') and str(e['symbol']).upper() != self.symbol:
                continue
            seen.add(e.get('exec_id'))
            side = str(e.get('side', '')).upper()
            net += float(e.get('shares') or 0) * (1 if side in ('BOT', 'BUY') else -1)
        return net

    def wait_for(self, predicate: Callable[[object], bool], timeout: Optional[float] = None) -> bool:
        timeout = self.fill_timeout if timeout is None else timeout
        start = self.clock()
        while True:
            if predicate(self.state()):
                return True
            if self.clock() - start >= timeout:
                return False
            self.sleep(POLL_SEC)

    def settle(self) -> None:
        self.sleep(self.settle_sec)

    def quote(self, fallback: Optional[float] = None) -> Optional[float]:
        """Fresh reference price for the next check (sets price / price_source); else the fallback
        (the price the symbol was picked at, treated as stale)."""
        price, source = reference_price(self.broker, self.symbol)
        if not price:
            price, source = fallback, 'quote at symbol pick (stale)'
        self.price, self.price_source = price, source
        return price

    def no_intraday_quote(self) -> bool:
        """True when the check's price is not from an intraday bar (daily close / pick-time fallback):
        orders that must fill are not sent off it."""
        return not str(self.price_source).startswith('last')

    # ─── cleanup ───

    def cancel_working(self) -> Tuple[List[int], List[int]]:
        """Cancel every order this run placed that TWS doesn't report as finished:
        (cancel requested, cancel failed or refused)."""
        cancelled, failed = [], []
        try:
            state = self.state()
        except Exception as exc:
            self.say(f"  state unavailable ({exc}): cancelling every order this run placed")
            state = None
        for oid in list(self.placed):
            view = self.view(oid, state) if state is not None else {'where': 'unknown'}
            if view.get('where') == 'completed' or view.get('status') in TERMINAL_TWS:
                continue
            if view.get('where') == 'missing' and self.placed[oid]['cancel_requested']:
                continue      # already cancelled, and TWS lists it nowhere
            # 'missing' too: an untransmitted (staged) order may not be listed but still sits in TWS
            (cancelled if self.cancel(oid, state) else failed).append(oid)
        return cancelled, failed

    def flatten(self) -> dict:
        """Sell the shares this run's own fills bought (never more than the position), after a 'yes'."""
        try:
            state = self.state()
            qty = min(self.position(state), self.own_net_shares(state))
        except Exception as exc:
            return {'result': 'state_unavailable', 'note': str(exc)}
        if qty <= EPS:
            return {'result': 'flat', 'qty': 0}
        qty = int(round(qty))
        now = self.now()
        if not in_regular_session(now):
            self.say(f"  {qty} {self.symbol} from this run is still held and it's outside the regular session: "
                     f"sell it by hand in TWS ({qty} sh, orderRef PAPERCHECK).")
            return {'result': 'left_open_outside_session', 'qty': qty}
        # A limit MARKETABLE_PCT under the reference price: if that price is stale and the sell
        # doesn't fill, it is cancelled after fill_timeout and you sell by hand (never a market order)
        price, source = reference_price(self.broker, self.symbol)
        if not price or not source.startswith('last'):
            self.say(f"  no intraday quote for {self.symbol} ({source}): sell the {qty} test shares by hand in TWS.")
            return {'result': 'no_quote', 'qty': qty, 'source': source}
        spec = OrderSpec('flat', 'SELL', qty, 'LMT', lmt=_px(price * (1 - MARKETABLE_PCT / 100)),
                         note='marketable limit: sells this run\'s test shares')
        self.say(f"  Cleanup: {qty} {self.symbol} bought by this run is still held.")
        self.say(f"    - {spec.describe(self.symbol)}  (priced off {price:.2f}, {source})")
        if source != LIVE_SOURCE:
            self.say("    WARNING: delayed or stale quote: compare it with TWS before typing yes")
        if not self.confirm(f"Sell this run's {qty} {self.symbol} test shares?"):
            self.say(f"  Not sold: {qty} {self.symbol} stays in the paper account; sell it by hand.")
            return {'result': 'owner_declined', 'qty': qty}
        before = self.position()
        oid = self.place(0, spec)
        sold = self.wait_for(lambda st: self.position(st) <= before - qty + EPS)
        if not sold:
            self.cancel(oid)
            self.say(f"  The cleanup sell did not fill within {self.fill_timeout:g}s and was cancelled: "
                     f"sell {qty} {self.symbol} by hand.")
        return {'result': 'sold' if sold else 'sell_not_filled', 'qty': qty, 'order_id': oid}

    def cleanup(self, label: str = "") -> dict:
        self.say(f"  Cleanup{(' after ' + label) if label else ''}: cancelling this run's working orders")
        record: dict = {}
        failed: List[int] = []
        try:
            record['cancelled'], failed = self.cancel_working()
            if failed:
                record['cancel_failed'] = failed
        except Exception as exc:
            record['cancel_error'] = str(exc)
            self.say(f"  cancel step failed ({exc}): check TWS for PAPERCHECK orders")
        still_live: List[int] = []
        if record.get('cancelled'):
            # A sell must not go out while a cancelled stop could still be working
            def live_ids(st) -> List[int]:
                return [o for o in record['cancelled'] if self.is_live(self.view(o, st))]
            try:
                if not self.wait_for(lambda st: not live_ids(st), timeout=max(self.settle_sec * 3, 5)):
                    still_live = live_ids(self.state())
            except Exception as exc:
                self.say(f"  could not confirm the cancels ({exc})")
                still_live = list(record['cancelled'])
        unresolved = sorted(set(failed) | set(still_live))
        if unresolved or 'cancel_error' in record:
            # Two SELLs for the same shares (the flatten and a stop that didn't cancel) would leave a short
            record['flatten'] = {'result': 'own_order_still_live', 'order_ids': unresolved}
            self.say(f"  NOT selling: this run's order(s) {unresolved or '(unknown)'} may still be working "
                     f"(a stop could sell the same shares). Cancel every PAPERCHECK order in TWS, then sell "
                     f"any {self.symbol} test shares by hand.")
            return record
        try:
            record['flatten'] = self.flatten()
        except Exception as exc:
            record['flatten'] = {'result': 'error', 'note': str(exc)}
            self.say(f"  flatten step failed ({exc}): check the {self.symbol} position in TWS")
        return record


# ─── checks ───

def check_trim_oca(s: CheckSession, price: float) -> CheckResult:
    """1: BUY 3, SELL STP 3 (GTC, OCA type 3), then a marketable SELL LMT 1 in the same OCA."""
    res = CheckResult(1, CHECK_NAMES[1])
    oca = f"{PAPERCHECK_REF_PREFIX}OCA1-{s.run_tag}-{s.symbol}"
    specs = [OrderSpec('buy', 'BUY', 3, 'LMT', lmt=_px(price * (1 + MARKETABLE_PCT / 100)), note='fill wanted'),
             OrderSpec('stop', 'SELL', 3, 'STP', aux=_px(price * (1 - FAR_PCT / 100)), tif='GTC',
                       oca_group=oca, oca_type=3, note='never triggers'),
             OrderSpec('trim', 'SELL', 1, 'LMT', lmt=_px(price * (1 - MARKETABLE_PCT / 100)),
                       oca_group=oca, oca_type=3, note='fill wanted: the full partial trim')]
    res.orders = [sp.describe(s.symbol) for sp in specs]
    if not in_regular_session(s.now()):
        return res.done(INCONCLUSIVE, "needs the regular session (09:30-16:00 ET): not run")
    if s.no_intraday_quote():
        return res.done(INCONCLUSIVE, f"no intraday quote ({s.price_source}): the fill-wanted orders were not sent")
    if not s.confirm_orders(1, specs, "Buys 3 shares, rests a far stop for 3 and sells 1 into the same OCA "
                                      "group; then reads the stop. Cleanup sells the 2 left (asks first)."):
        return res.done(INCONCLUSIVE, "owner did not confirm: nothing sent")
    start = s.position()
    res.observations['start_position'] = start
    s.place(1, specs[0])
    if not s.wait_for(lambda st: s.position(st) >= start + 3 - EPS):
        return res.done(INCONCLUSIVE, f"the BUY did not fill within {s.fill_timeout:g}s")
    stop_id = s.place(1, specs[1])
    res.observations['stop_ack'] = list(s.ack(stop_id))
    if res.observations['stop_ack'][0] != 'working':
        return res.done(INCONCLUSIVE, f"the stop was not confirmed working: {res.observations['stop_ack']}")
    s.place(1, specs[2])
    if not s.wait_for(lambda st: s.position(st) <= start + 2 + EPS):
        return res.done(INCONCLUSIVE, f"the trim did not fill within {s.fill_timeout:g}s")
    s.settle()
    view = s.view(stop_id)
    res.observations['stop_after_trim'] = view
    res.observations['position_after_trim'] = s.position()
    remaining = view.get('remaining')
    if not s.is_live(view):
        return res.done(FAIL, f"the trim's full fill CANCELLED the stop (TWS: {view.get('where')} "
                              f"{view.get('status')}): the position has no stop until the nightly run "
                              "replaces it. Fix the code before enabling submission.")
    if remaining is not None and abs(float(remaining) - 2) <= EPS:
        return res.done(PASS, "ocaType 3 reduced the stop from 3 to 2 shares (what the code assumes)")
    if remaining is not None and float(remaining) >= 3 - EPS:
        return res.done(FAIL, f"the stop is still {float(remaining):g} shares for 2 held: not reduced, a stop-out "
                              "would oversell. Fix the code before enabling submission.")
    return res.done(INCONCLUSIVE, f"stop remaining {remaining!r} is neither 2 nor 3: look at TWS")


def check_partial_parent(s: CheckSession, price: float) -> CheckResult:
    """3: bracket children when the parent is cancelled unfilled (A) or after a partial fill (B)."""
    res = CheckResult(3, CHECK_NAMES[3])
    stop_px = _px(price * (1 - FAR_PCT / 100))
    a_specs = [OrderSpec('parenta', 'BUY', 5, 'LMT', lmt=_px(price * (1 - FAR_PCT / 100)), transmit=False,
                         note='never fills'),
               OrderSpec('childa', 'SELL', 5, 'STP', aux=_px(price * (1 - 2 * FAR_PCT / 100)), tif='GTC',
                         oca_group=f"{PAPERCHECK_REF_PREFIX}OCA3A-{s.run_tag}-{s.symbol}", oca_type=3,
                         note='child of the parent above')]
    b_specs = [OrderSpec('parentb', 'BUY', 5, 'LMT', lmt=_px(price), transmit=False,
                         note='at the last price: may fill in full, in part or not at all'),
               OrderSpec('childb', 'SELL', 5, 'STP', aux=stop_px, tif='GTC',
                         oca_group=f"{PAPERCHECK_REF_PREFIX}OCA3B-{s.run_tag}-{s.symbol}", oca_type=3,
                         note='child of the parent above')]
    res.orders = [sp.describe(s.symbol) for sp in a_specs + b_specs]
    if not s.confirm_orders(3, a_specs, "Part A: a bracket whose parent can't fill; the parent is cancelled "
                                        "and the child is read (it must go with the parent)."):
        return res.done(INCONCLUSIVE, "owner did not confirm part A: nothing sent")
    parent = s.place(3, a_specs[0])
    child = s.place(3, a_specs[1], parent_id=parent)
    res.observations['a_child_ack'] = list(s.ack(child))
    s.cancel(parent)
    s.settle()
    view_a = s.view(child)
    res.observations['a_child_after_parent_cancel'] = view_a
    part_a = FAIL if s.is_live(view_a) else PASS
    res.observations['part_a'] = part_a
    if part_a == FAIL:
        return res.done(FAIL, "part A: the child stop stayed working after its unfilled parent was cancelled "
                              "(a SELL with no position: an orphan that would open a short)")

    if not in_regular_session(s.now()):
        return res.done(INCONCLUSIVE, "part A PASS (child cancelled with its unfilled parent); part B needs the "
                                      "regular session: not run")
    if s.no_intraday_quote():
        return res.done(INCONCLUSIVE, f"part A PASS; part B not sent: no intraday quote ({s.price_source})")
    if not s.confirm_orders(3, b_specs, "Part B: may BUY up to 5 shares. If it fills only in part within "
                                        f"{s.fill_timeout:g}s the parent is cancelled and the child is read. "
                                        "Cleanup sells whatever filled (asks first)."):
        return res.done(INCONCLUSIVE, "part A PASS; owner skipped part B")
    start = s.position()
    parent = s.place(3, b_specs[0])
    child = s.place(3, b_specs[1], parent_id=parent)
    s.wait_for(lambda st: s.position(st) >= start + 5 - EPS)
    filled = s.position() - start
    res.observations['b_filled'] = filled
    if filled >= 5 - EPS:
        return res.done(INCONCLUSIVE, "part A PASS; part B filled in full (5), no partial fill to observe")
    s.cancel(parent)
    s.settle()
    view_b = s.view(child)
    res.observations['b_child_after_parent_cancel'] = view_b
    if filled <= EPS:
        return res.done(INCONCLUSIVE, "part A PASS; part B got no fill, so no partial fill to observe")
    if not s.is_live(view_b):
        return res.done(FAIL, f"part B: after a {filled:g}-share partial fill and a parent cancel the child "
                              "stop is gone: the shares are unprotected")
    remaining = float(view_b.get('remaining') or 0)
    if abs(remaining - filled) <= EPS:
        return res.done(PASS, f"part A PASS; part B: the child stop shrank to the {filled:g} shares filled")
    return res.done(FAIL, f"part B: child stop is {remaining:g} shares for {filled:g} held (oversized: a stop-out "
                          "would oversell until fill_monitor's SHRINK_STOP runs)")


def check_staged_parent(s: CheckSession, price: float) -> CheckResult:
    """4: one BUY LMT far below the market, transmit=False, no child: is it in reqAllOpenOrders?"""
    res = CheckResult(4, CHECK_NAMES[4])
    spec = OrderSpec('staged', 'BUY', 1, 'LMT', lmt=_px(price * (1 - FAR_PCT / 100)), transmit=False,
                     note='never transmitted, never fills')
    res.orders = [spec.describe(s.symbol)]
    if not s.confirm_orders(4, [spec], "Stages one untransmitted parent (no child), reads TWS's open orders, "
                                       "then cancels it."):
        return res.done(INCONCLUSIVE, "owner did not confirm: nothing sent")
    oid = s.place(4, spec)
    s.settle()
    state = s.state()
    view = s.view(oid, state)
    listed = any(o.get('order_ref') == s.placed[oid]['ref'] for o in state.open_orders)
    res.observations.update(listed_in_open_orders=listed, view=view)
    s.cancel(oid)
    s.settle()
    res.observations['after_cancel'] = s.view(oid)
    gone = s.confirm(f"Look at the TWS window: is the staged {s.symbol} BUY (ref {s.placed[oid]['ref']}) gone? "
                     "If not, cancel it there now.", suffix="Type yes if it is gone: ")
    res.observations['owner_confirmed_gone_in_tws'] = gone
    if listed:
        return res.done(PASS, "reqAllOpenOrders lists a staged untransmitted parent: reconcile can see and "
                              "cancel one left by a failed child")
    return res.done(FAIL, "reqAllOpenOrders does NOT list a staged parent: one left by a failed child shows "
                          "only in the TWS window (the code then only alerts you to cancel it by hand)")


def check_preopen_399(s: CheckSession, price: float) -> CheckResult:
    """5: a pre-open BUY STP LMT far above the market; which TWS messages does it get?"""
    res = CheckResult(5, CHECK_NAMES[5])
    aux = _px(price * (1 + FAR_PCT / 100))
    spec = OrderSpec('preopen', 'BUY', 1, 'STP LMT', aux=aux, lmt=_px(aux * 1.02), note='never triggers')
    res.orders = [spec.describe(s.symbol)]
    now = s.now()
    if not is_trading_day(now.date()) or now.time() >= PREOPEN_CUTOFF_ET:
        return res.done(INCONCLUSIVE, f"only runs before {PREOPEN_CUTOFF_ET:%H:%M:%S} ET on a trading day "
                                      f"(now {now:%a %H:%M} ET): not run")
    if not s.confirm_orders(5, [spec], "Sends one pre-open entry like order_entry's (trigger 25% above the "
                                       "market), records TWS's messages, then cancels it before the open."):
        return res.done(INCONCLUSIVE, "owner did not confirm: nothing sent")
    oid = s.place(5, spec)
    ack = s.ack(oid)
    s.settle()
    msgs = _order_messages(s.broker, oid)
    codes = [m['code'] for m in msgs]
    res.observations.update(ack=list(ack), messages=msgs, view=s.view(oid))
    s.cancel(oid)
    routine = all(c in ORDER_WARNING_CODES or (isinstance(c, int) and 2100 <= c < 2200) for c in codes)
    if ack[0] == 'working' and routine:
        seen = "399 seen" if 399 in codes else f"no 399 (codes {codes or 'none'})"
        return res.done(PASS, f"the pre-open entry stayed working ({ack[1]}); {seen}; every message is a warning")
    if ack[0] == 'working':
        return res.done(FAIL, f"working, but TWS sent codes the connector treats as rejections: {codes}")
    if ack[0] == 'rejected':
        return res.done(FAIL, f"TWS rejected the pre-open entry: {ack[1]} (codes {codes})")
    return res.done(INCONCLUSIVE, f"no order status within {ACK_TIMEOUT_SEC:g}s: {ack[1]} (codes {codes})")


def _f(value) -> Optional[float]:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(v) or v >= 1e300 else v


def observe_gap(rows: pd.DataFrame, executions: Sequence[dict]) -> Tuple[str, str, dict]:
    """
    2, read-only: (result, summary, details) from the ledger's latest rows and today's executions.
    FAIL  a trade where the stop AND the MOO/MOC exit (same OCA group) both filled
    PASS  a MOO filled at or below the stop's trigger (both could have fired) and the stop did not
    INCONCLUSIVE otherwise (no gap-down open seen yet)
    """
    both, blocked = [], []
    if rows is not None and not rows.empty:
        for (td, sym), g in rows.groupby(['trade_date', 'symbol']):
            stops = g[g['role'].isin(['stop', 'stop_fix'])]
            opens = g[g['role'].isin(['exit', 'trim']) & (g['tif'].astype(str).str.upper() == 'OPG')]
            for _, m in opens.iterrows():
                m_filled = _f(m.get('filled_qty')) or 0.0
                if m_filled <= 0:
                    continue
                oca = m.get('oca_group')
                same = stops[stops['oca_group'] == oca] if oca else stops
                for _, st in same.iterrows():
                    s_filled = _f(st.get('filled_qty')) or 0.0
                    trigger, fill = _f(st.get('aux_price')), _f(m.get('avg_fill_price'))
                    item = {'trade_date': td, 'symbol': sym, 'stop_ref': st.get('order_ref'),
                            'moo_ref': m.get('order_ref'), 'stop_filled': s_filled, 'moo_filled': m_filled,
                            'stop_trigger': trigger, 'moo_fill_price': fill}
                    if s_filled > 0:
                        both.append(item)
                    elif trigger is not None and fill is not None and fill <= trigger + EPS:
                        blocked.append(item)
    # Today's executions, before any reconcile has written them to the ledger
    sold: Dict[tuple, Dict[str, float]] = {}
    seen = set()
    for e in executions or []:
        if e.get('exec_id') in seen or str(e.get('side', '')).upper() not in ('SLD', 'SELL'):
            continue
        seen.add(e.get('exec_id'))
        parsed = ol.parse_order_ref(e.get('order_ref') or '')
        if not parsed or parsed[0] not in ('stop', 'stop_fix', 'exit'):
            continue
        kind = 'exit' if parsed[0] == 'exit' else 'stop'
        bucket = sold.setdefault((parsed[1], parsed[2]), {})
        bucket[kind] = bucket.get(kind, 0.0) + float(e.get('shares') or 0)
    for (td, sym), kinds in sold.items():
        if kinds.get('stop') and kinds.get('exit'):
            both.append({'trade_date': td, 'symbol': sym, 'source': 'executions today', **kinds})
    details = {'both_filled': both, 'one_blocked_other': blocked}
    if both:
        return FAIL, f"stop AND market-on-open exit both filled for {len(both)} trade(s): the account went short", details
    if blocked:
        return PASS, f"{len(blocked)} gap-down open(s) below the stop: only one OCA member filled", details
    return INCONCLUSIVE, "no gap-down open with a stop + MOO in one OCA seen yet. " + GAP_PROCEDURE, details


def check_gap_observe(broker, ledger: ol.OrderLedger) -> CheckResult:
    res = CheckResult(2, CHECK_NAMES[2])
    try:
        rows = ledger.current()
    except Exception as exc:
        return res.done(INCONCLUSIVE, f"ledger unreadable ({exc})")
    try:
        execs = broker.state().executions if broker is not None else []
    except Exception as exc:
        execs = []
        res.observations['tws_error'] = str(exc)
    result, summary, details = observe_gap(rows, execs)
    res.observations.update(details)
    return res.done(result, summary)


ORDER_CHECKS: Dict[int, Callable[[CheckSession, float], CheckResult]] = {
    1: check_trim_oca, 3: check_partial_parent, 4: check_staged_parent, 5: check_preopen_399}


# ─── setup ───

def reference_price(broker, symbol: str) -> Tuple[Optional[float], str]:
    """(price, source): the last 1-min bar ('last', or 'last (delayed/stale)' when order_broker
    flags it delayed), else the last daily close ('daily close'; pre-open there are no intraday
    bars). Good enough to set prices FAR_PCT away. Orders that must fill are sent only off a 1-min
    bar; a delayed one is shown with a warning before the 'yes' (a marketable limit never fills
    worse than its limit, a stale one may just not fill)."""
    try:
        price, delayed = broker.last_price(symbol)
        if price:
            return float(price), 'last (delayed/stale)' if delayed else 'last'
    except Exception as exc:
        logger.warning(f"{symbol}: quote failed ({exc})")
    try:
        bars = broker.daily_bars(symbol, duration="5 D", include_forming=True)
        if bars is not None and len(bars):
            return float(bars['close'].iloc[-1]), 'daily close'
    except Exception as exc:
        logger.warning(f"{symbol}: daily bars failed ({exc})")
    return None, 'none'


def pick_symbol(broker, candidates: Sequence[str] = SYMBOLS, max_price: float = MAX_PRICE
                ) -> Tuple[Optional[str], Optional[float], List[str]]:
    """First candidate with no position and no open order (any client) in the account, and a quote."""
    state = broker.state()
    notes = []
    for sym in candidates:
        sym = sym.upper()
        if any(str(p.get('symbol', '')).upper() == sym and abs(float(p.get('shares') or 0)) > EPS
               for p in state.positions):
            notes.append(f"{sym}: position held, skipped")
            continue
        if any(str(o.get('symbol', '')).upper() == sym for o in state.open_orders):
            notes.append(f"{sym}: open order(s), skipped")
            continue
        price, source = reference_price(broker, sym)
        if not price:
            notes.append(f"{sym}: no quote")
            continue
        if price > max_price:
            notes.append(f"{sym}: {price:.2f} > {max_price:g}")
            continue
        notes.append(f"{sym}: chosen at {price:.2f} ({source})")
        return sym, float(price), notes
    return None, None, notes


def results_path(today, orders_dir: Optional[Path] = None) -> Path:
    return Path(orders_dir or ol.ORDERS_DIR) / f"paper_checks_{today:%Y-%m-%d}.json"


def write_results(record: dict, path: Path) -> Path:
    """Append this run to the day's file (several runs a day keep every run)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {'date': record['date'], 'runs': []}
    if path.exists():
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            path = path.with_name(path.stem + f"_{int(time.time())}.json")
    data.setdefault('runs', []).append(record)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, default=str))
    tmp.replace(path)
    return path


def explain(out: Callable[[str], None] = print, checks: Sequence[int] = tuple(sorted(CHECK_NAMES))) -> None:
    p = 10.00   # illustrative price
    out("Paper checks: the plan (nothing connects; prices below are for a $10.00 stock)\n")
    out("Gates: paper port 7497/4002 only (7496/4001 refused before connecting), --account DU..., config "
        "AUTO_SUBMIT_ENABLED = True and --submit. Client 16 + the order-ledger lock. orderRef PAPERCHECK-...")
    out(f"Symbol: first of {', '.join(SYMBOLS)} with no position, no open order and a quote <= ${MAX_PRICE:g}.")
    out("Every check asks you to type 'yes' before it transmits; cleanup cancels this run's orders and asks "
        "before selling this run's shares.\n")
    plans = {
        1: [f"BUY 3 LMT {_px(p * 1.03):.2f} DAY (fill wanted)",
            f"SELL 3 STP {_px(p * 0.75):.2f} GTC, OCA type 3 (never triggers)",
            f"SELL 1 LMT {_px(p * 0.97):.2f} DAY, same OCA (the full partial trim fills)",
            "PASS: stop reduced to 2. FAIL: stop cancelled, or still 3. Regular session only."],
        2: ["No orders. Reads the ledger and today's TWS executions.", GAP_PROCEDURE],
        3: [f"A: BUY 5 LMT {_px(p * 0.75):.2f} (staged) + child SELL 5 STP {_px(p * 0.5):.2f} GTC; cancel the "
            "parent. PASS: the child goes too. FAIL: the child stays working (orphan sell).",
            f"B (regular session, asks again): BUY 5 LMT {_px(p):.2f} + child SELL 5 STP {_px(p * 0.75):.2f} GTC; "
            f"after a partial fill within {FILL_TIMEOUT_SEC:g}s cancel the parent. PASS: child = shares filled. "
            "FAIL: child gone or bigger. Full / no fill: INCONCLUSIVE."],
        4: [f"BUY 1 LMT {_px(p * 0.75):.2f} DAY transmit=False, no child; read reqAllOpenOrders; cancel; you "
            "confirm in the TWS window. PASS: listed. FAIL: not listed."],
        5: [f"Before {PREOPEN_CUTOFF_ET:%H:%M:%S} ET only: BUY 1 STP LMT trigger {_px(p * 1.25):.2f} limit "
            f"{_px(_px(p * 1.25) * 1.02):.2f} DAY; record TWS messages; cancel. PASS: working and only warning "
            "codes (399 expected). FAIL: rejected. Otherwise INCONCLUSIVE."],
    }
    for n in checks:
        out(f"Check {n}: {CHECK_NAMES[n]}")
        for line in plans[n]:
            out(f"    {line}")
    out(f"\nResults: {results_path(now_et().date())} (PASS / FAIL / INCONCLUSIVE per check).")
    out("Not covered here (README item): a BUY STP LMT whose trigger is below the market.")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Guided paper-account checks of TWS order behaviour (paper only)")
    parser.add_argument('--account', default=None, help="Paper account DU... (default config IBKR_ACCOUNT)")
    parser.add_argument('--port', type=int, default=int(getattr(config, 'TWS_PORT', 7497)))
    parser.add_argument('--submit', action='store_true', help="Allow orders (also needs AUTO_SUBMIT_ENABLED=True)")
    parser.add_argument('--check', type=int, action='append', choices=sorted(CHECK_NAMES),
                        help="Run only this check (repeatable); default all")
    parser.add_argument('--symbol', choices=SYMBOLS, help="Use this candidate instead of the first free one")
    parser.add_argument('--explain', action='store_true', help="Print the plan and exit; never connects")
    return parser.parse_args(argv)


def run(args: argparse.Namespace, *, broker_factory=open_session, ask: Callable[[str], str] = input,
        out: Callable[[str], None] = print, sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic, now_fn: Callable[[], datetime] = now_et,
        orders_dir: Optional[Path] = None, fill_timeout: float = FILL_TIMEOUT_SEC,
        settle_sec: float = SETTLE_SEC) -> int:
    checks = sorted(set(args.check or CHECK_NAMES))
    if args.explain:
        explain(out, checks)
        return 0
    port = int(args.port)
    if port in LIVE_PORTS:
        out(f"REFUSED: port {port} is a LIVE port {sorted(LIVE_PORTS)}. Paper checks never connect to it.")
        return 2
    if port not in PAPER_PORTS:
        out(f"REFUSED: port {port} is not a paper port {sorted(PAPER_PORTS)}.")
        return 2
    account = (args.account or str(getattr(config, 'IBKR_ACCOUNT', '') or '')).strip().upper()
    if not account.startswith('DU'):
        out(f"REFUSED: --account DU... is required (got {account or 'none'}): paper accounts only.")
        return 2
    read_only = checks == [2]
    if not read_only and not submission_enabled(args.submit):
        out(f"REFUSED: {SUBMIT_REFUSED}. Paper checks send real (paper) orders: set AUTO_SUBMIT_ENABLED = True "
            "in config.py for this session and pass --submit (only --check 2 runs without them).")
        return 2
    now = now_fn()
    if not read_only and (not calendar_covers(now.date()) or not is_trading_day(now.date())):
        out(f"REFUSED: {now.date()} is not a trading day (or is past the holiday table).")
        return 2

    orders_dir = Path(orders_dir or ol.ORDERS_DIR)
    record = {'date': now.date().isoformat(), 'started': now.isoformat(timespec='seconds'), 'account': account,
              'port': port, 'client_id': ORDER_READONLY_CLIENT_ID if read_only else ORDER_CLIENT_ID,
              'checks_requested': checks, 'symbol': None, 'results': [], 'interrupted': False}
    rc = 0
    try:
        with ol.ledger_lock(orders_dir):
            try:
                broker = broker_factory(port, account, cli_submit=bool(args.submit) and not read_only,
                                        client_id=record['client_id'], ref_prefix=PAPERCHECK_REF_PREFIX)
            except PermissionError as exc:
                out(f"REFUSED by the paper guard: {exc}")
                return 2
            except ConnectionError as exc:
                out(f"Could not connect: {exc}")
                return 1
            ledger = ol.OrderLedger(path=orders_dir / "ledger.jsonl", source='paper_checks')
            if not read_only and not broker.submit_enabled:
                out(f"REFUSED: the session is read-only ({SUBMIT_REFUSED}).")
                broker.close()
                return 2
            session: Optional[CheckSession] = None
            holder: dict = {}
            try:
                rc = _run_checks(broker, ledger, checks, args, record, ask, out, sleep, clock, now_fn,
                                 fill_timeout, settle_sec, holder)
                session = holder.get('session')
            except KeyboardInterrupt:
                record['interrupted'] = True
                session = holder.get('session')
                out("\nInterrupted: cleaning up this run's orders and shares")
                rc = 130
            except Exception as exc:
                record['error'] = str(exc)
                session = holder.get('session')
                logger.exception(f"paper checks failed: {exc}")
                rc = 1
            finally:
                if session is not None:
                    try:
                        record['final_cleanup'] = session.cleanup("the run")
                    except KeyboardInterrupt:
                        record['final_cleanup'] = {'result': 'interrupted'}
                        out("Cleanup interrupted: check TWS for PAPERCHECK orders and test shares by hand.")
                    record['placed_orders'] = {str(k): {kk: vv for kk, vv in v.items() if kk != 'spec'}
                                               for k, v in session.placed.items()}
                    record['transcript'] = session.log
                broker.close()
                record['finished'] = now_fn().isoformat(timespec='seconds')
                path = write_results(record, results_path(now.date(), orders_dir))
                out("\nResults:")
                for r in record['results']:
                    out(f"  {r['check']}  {r['result']:<12} {r['summary']}")
                out(f"Written to {path}")
    except ol.LedgerLocked as exc:
        out(f"Another order run holds the ledger lock ({exc}); try again when it is done.")
        return 3
    return rc


def _run_checks(broker, ledger, checks, args, record, ask, out, sleep, clock, now_fn,
                fill_timeout, settle_sec, holder: dict) -> int:
    order_checks = [n for n in CHECK_ORDER if n in checks and n != 2]
    if order_checks:
        state = broker.state()
        leftovers = [o for o in state.open_orders if str(o.get('order_ref') or '').startswith(PAPERCHECK_REF_PREFIX)]
        sym, price, notes = pick_symbol(broker, (args.symbol,) if args.symbol else SYMBOLS)
        record['symbol_notes'] = notes
        for line in notes:
            out(f"  {line}")
        run_tag = now_fn().strftime('%Y%m%d%H%M%S')
        # New ids start above the ledger AND above every order TWS lists for this client: client 16 also
        # places the bot's orders, and placeOrder on a working id would MODIFY that order
        own_ids = [int(o['order_id']) for o in state.open_orders + state.completed_orders
                   if o.get('client_id') == broker.client_id and o.get('order_id') is not None]
        session = CheckSession(broker, sym or 'NONE', run_tag, ask=ask, out=out, sleep=sleep, clock=clock,
                               now_fn=now_fn, id_floor=max([ledger.highest_order_id()] + own_ids) + 1,
                               fill_timeout=fill_timeout, settle_sec=settle_sec)
        holder['session'] = session
        if leftovers:
            out("PAPERCHECK orders from an earlier run are still working:")
            for o in leftovers:
                out(f"    {o.get('order_id')} {o.get('action')} {o.get('quantity')} {o.get('symbol')} "
                    f"{o.get('order_type')} ref {o.get('order_ref')} (client {o.get('client_id')})")
            mine = [o for o in leftovers if o.get('client_id') == broker.client_id]
            if len(mine) != len(leftovers) or not session.confirm("Cancel them before the checks?"):
                out("REFUSED: cancel the leftover PAPERCHECK orders first (TWS window, or rerun and answer yes).")
                record['refused'] = 'leftover PAPERCHECK orders'
                return 2
            for o in mine:
                session.adopt(o)
            session.cleanup("leftovers")
            # The cleanup sells only the new symbol's shares: a leftover's own shares (an earlier run's
            # test buy whose stop was just cancelled) are now unprotected; the owner sells them by hand
            held = {}
            try:
                for p in broker.state().positions:
                    psym = str(p.get('symbol', '')).upper()
                    if (psym in {str(o.get('symbol', '')).upper() for o in mine} and psym != session.symbol
                            and abs(float(p.get('shares') or 0)) > EPS):
                        held[psym] = held.get(psym, 0.0) + float(p.get('shares') or 0)
            except Exception as exc:
                out(f"  could not read positions after the leftover cleanup ({exc}): check TWS")
            for psym, shares in held.items():
                out(f"  WARNING: {shares:g} {psym} held and its leftover PAPERCHECK order(s) were just cancelled: "
                    f"if these are test shares they have NO stop now; sell them by hand in TWS.")
            if held:
                record['leftover_positions'] = held
        record['symbol'] = sym
        for n in order_checks:
            res = CheckResult(n, CHECK_NAMES[n])
            res.started = now_fn().isoformat(timespec='seconds')
            if sym is None:
                res.done(INCONCLUSIVE, "no candidate symbol was free with a quote: " + "; ".join(notes))
            else:
                started, interrupted = res.started, False
                try:
                    res = ORDER_CHECKS[n](session, session.quote(price))
                    res.started = started
                except KeyboardInterrupt:
                    # res keeps whatever the check had; its orders are cancelled just below
                    res.done(INCONCLUSIVE, "interrupted (Ctrl-C)")
                    interrupted = True
                except Exception as exc:
                    logger.exception(f"check {n} failed")
                    res.done(INCONCLUSIVE, f"error: {exc}")
                finally:
                    res.observations['cleanup'] = session.cleanup(f"check {n}")
                if interrupted:
                    res.finished = now_fn().isoformat(timespec='seconds')
                    record['results'].append(asdict(res))
                    raise KeyboardInterrupt
            res.finished = now_fn().isoformat(timespec='seconds')
            record['results'].append(asdict(res))
            out(f"  => check {n}: {res.result}: {res.summary}")
    if 2 in checks:
        res = check_gap_observe(broker, ledger)
        res.started = res.finished = now_fn().isoformat(timespec='seconds')
        record['results'].append(asdict(res))
        out(f"  => check 2: {res.result}: {res.summary}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s", datefmt="%H:%M:%S")
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
