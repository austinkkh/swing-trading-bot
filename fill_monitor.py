#!/usr/bin/env python3
"""
Fill / stop monitor: checks that every position is covered by its stop, and repairs the bot's own
trades (paper only).

For EVERY position it measures stop coverage with account_snapshot.position_risks() (same working-
status, contract and account rules as the snapshot), then decides an action. Only the bot's own
open trades (ledger entries with fills, not manual_touched) are ever changed; any other position is
reported, never touched.

  covered == shares                              OK
  bot child stop covers less (partial fill)      RESIZE_STOP   modify the bot stop up to the position
  no bot stop working                            PLACE_STOP    SELL STP GTC at the exit_state stop
                                                               (else the ledger stop), BOT-STOPFIX ref,
                                                               the trade's OCA group
  last <= stop, live quote, regular session      FLATTEN       SELL MKT DAY for the unprotected shares
  last <= stop, delayed quote / no quote / closed ALERT_FLATTEN only
  stops cover more than the position            SHRINK_STOP (bot stop) / REPORT_OVERSIZED (manual):
                                                 a stop bigger than the position flips the account short
  bot SELL working with no position              CANCEL_ORPHAN_SELL
  short that the bot's own SELLs caused          COVER_SHORT   BUY MKT DAY to flat, regular session
    (ledger net == position < 0)                               only (else ALERT_COVER_SHORT); a
                                                               working BUY for the stock -> REPORT.
                                                               OCA type 3 has no overfill block: a
                                                               stop and an exit MOO can both fill
  manual position, or a manual_touched trade     OK / REPORT / REPORT_OVERSIZED (never modified)

A stop that covers more than the position is expected while the bot entry is still working (a
bracket child covers the parent's unfilled rest), so the parent's remaining quantity is allowed.
Every new SELL is capped at the "sell room" (long shares minus every working SELL, bot and manual)
and re-checked with order_broker.assert_no_oversell, so a fix can never leave the account short.

Submission is DISABLED by default. Fixes are sent only if config AUTO_SUBMIT_ENABLED is the bool
True AND --submit is passed; otherwise every action is downgraded to ALERT_<action> and only goes
into the report and the alert. Every fix is written to the ledger as an intent before it is sent.
IBKR only lets the client that placed an order modify or cancel it, so a modify / cancel of an
order placed by another client id is downgraded to an alert too.

Client IDs: read-only runs use ORDER_READONLY_CLIENT_ID (17); --submit with the config switch uses
ORDER_CLIENT_ID (16), the client that placed the bot's brackets.

Exit codes: 0 ok (also nothing to do / holiday) · 1 error · 2 refused by a guard, or a position
needs attention that this run did not fix · 3 another order run holds the ledger lock.

Suggested times (no launchd job is installed): 09:45, 12:30 and 15:30 ET.

Usage:
  python fill_monitor.py --account DUXXXXXXX                 # read-only report + alerts
  python fill_monitor.py --account DUXXXXXXX --submit        # paper, both gates: applies fixes
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

import account_snapshot as acct
import config
import notify as notifier
import order_ledger as ol
from market_time import calendar_covers, in_regular_session, is_trading_day, now_et
from order_broker import (ORDER_CLIENT_ID, ORDER_READONLY_CLIENT_ID, SUBMIT_REFUSED, Broker, BrokerState,
                          WORKING as BROKER_WORKING, assert_no_oversell, stock_contract, submission_enabled)
from order_ledger import LedgerKey, OrderLedger, make_oca_group, make_order_ref, parse_order_ref

logger = logging.getLogger(__name__)

# Actions
OK = "OK"
REPORT = "REPORT"
REPORT_OVERSIZED = "REPORT_OVERSIZED"
RESIZE_STOP = "RESIZE_STOP"
PLACE_STOP = "PLACE_STOP"
FLATTEN = "FLATTEN"
ALERT_FLATTEN = "ALERT_FLATTEN"
SHRINK_STOP = "SHRINK_STOP"
CANCEL_ORPHAN_SELL = "CANCEL_ORPHAN_SELL"
COVER_SHORT = "COVER_SHORT"
ALERT_COVER_SHORT = "ALERT_COVER_SHORT"
FIX_ACTIONS = (RESIZE_STOP, PLACE_STOP, FLATTEN, SHRINK_STOP, CANCEL_ORPHAN_SELL, COVER_SHORT)
ALERT_PREFIX = "ALERT_"
# Anything other than OK means a human should look (alerted and exit 2 unless the run fixed it)
QUIET_ACTIONS = {OK}

STOP_ROLES = ('stop', 'stop_fix', 'trim_stop')     # trim_stop: exit_manager TRIM_STOP_MODE split_stop
LIVE_LEDGER_STATES = {'intended', 'submitted', 'working', 'partially_filled'}
EPS = 1e-9

REPORT_COLUMNS = ['run_ts', 'mode', 'account', 'symbol', 'sec_type', 'shares', 'protected', 'unprotected',
                  'oversized', 'stop_qty', 'sell_room', 'pending_entry_qty', 'bot_trade', 'trade_status',
                  'stop', 'ledger_stop', 'last', 'delayed', 'action', 'fix', 'note']


def _orders_dir() -> Path:
    return ol.ORDERS_DIR


@dataclass
class Protection:
    """One position (or one orphan bot SELL) and what the monitor decided for it."""
    symbol: str
    shares: float
    protected: float
    unprotected: float
    oversized: float
    bot_trade: Optional[str]            # "<trade_date>:<SYM>:<setup>" for the bot's own open trades
    stop: Optional[float]               # the stop level the trade should have (exit_state > ledger > TWS)
    ledger_stop: Optional[float]
    last: Optional[float]
    delayed: bool
    action: str
    note: str = ""
    account: Optional[str] = None
    sec_type: str = "STK"
    stop_qty: float = 0.0               # working protective stop shares (bot + manual), uncapped
    sell_room: float = 0.0              # long shares minus every working SELL (bot + manual)
    pending_entry_qty: float = 0.0      # unfilled rest of a working bot entry (its child covers it)
    trade_status: str = ""              # open / manual_touched / '' (not a bot trade)
    trade_date: Optional[str] = None
    setup: Optional[str] = None
    oca_group: str = ""
    # The bot stop (or orphan SELL) order at TWS this row would modify / cancel; not in the CSV
    target_order: Optional[dict] = field(default=None, repr=False)


# ─── Reading state ───

def load_exit_state(path: Optional[Path] = None) -> dict:
    """Data/orders/exit_state.json (written by exit_manager). Missing or unreadable -> {} with a warning:
    the monitor then falls back to the ledger's stop, which is never above the trailed stop."""
    path = Path(path) if path else _orders_dir() / "exit_state.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(f"exit_state {path} unreadable ({exc}); using the ledger stops")
        return {}


def _dedupe(orders: List[dict]) -> List[dict]:
    """openOrder repeats on every status update: one copy per order, the newest wins (same identity
    rule as account_snapshot.position_risks)."""
    out: Dict[tuple, dict] = {}
    for o in orders:
        out[o.get('perm_id') or (o.get('client_id'), o.get('order_id'), o.get('symbol'))] = o
    return list(out.values())


def _is_working(o: dict) -> bool:
    status = o.get('status')
    return not status or status in BROKER_WORKING


def _remaining(o: dict) -> float:
    return max(float(o.get('quantity') or 0) - float(o.get('filled') or 0), 0.0)


def _same_account(o: dict, account: Optional[str]) -> bool:
    return account is None or o.get('account') in (None, account)


def _protective_stops(position: dict, orders: List[dict]) -> List[dict]:
    """Working stop orders on the protective side of this position (SELL stops for a long)."""
    long = float(position['shares']) > 0
    return [o for o in orders
            if acct.stop_price(o) is not None
            and (not o.get('status') or o.get('status') in acct.WORKING_STATUSES)
            and acct._same_contract(position, o)
            and o.get('account', position.get('account')) == position.get('account')
            and str(o.get('action', '')).upper() == ('SELL' if long else 'BUY')]


def _working_sell_qty(symbol: str, account: Optional[str], orders: List[dict]) -> float:
    """Every working SELL for the stock (stops, trims, exits; bot and manual), unfilled part only."""
    return sum(_remaining(o) for o in orders
               if str(o.get('symbol', '')).upper() == symbol and str(o.get('action', '')).upper() == 'SELL'
               and o.get('sec_type', 'STK') == 'STK' and _same_account(o, account) and _is_working(o))


def _trade_key(t: dict) -> str:
    return f"{t['trade_date']}:{t['symbol']}:{t['setup']}"


def _ledger_stop(ledger_rows: pd.DataFrame, trade_date: str, symbol: str) -> Optional[float]:
    """The trade's latest live stop / stop_fix level in the ledger (else the latest one at all)."""
    if ledger_rows.empty:
        return None
    rows = ledger_rows[(ledger_rows['trade_date'] == trade_date) & (ledger_rows['symbol'] == symbol)
                       & ledger_rows['role'].isin(STOP_ROLES)].sort_values('ts', kind='stable')
    if rows.empty:
        return None
    live = rows[rows['state'].isin(LIVE_LEDGER_STATES)]
    px = (live if not live.empty else rows)['aux_price'].iloc[-1]
    try:
        return float(px) if px is not None and float(px) > 0 else None
    except (TypeError, ValueError):
        return None


def _ledger_oca(ledger_rows: pd.DataFrame, trade_date: str, symbol: str) -> str:
    if not ledger_rows.empty:
        rows = ledger_rows[(ledger_rows['trade_date'] == trade_date) & (ledger_rows['symbol'] == symbol)]
        # the main block's group: a split_stop trade also has its trim block's BOT-OCAT group
        groups = [g for g in rows['oca_group'].tolist() if isinstance(g, str) and g
                  and g != ol.make_trim_oca_group(trade_date, symbol)]
        if groups:
            return groups[-1]
    return make_oca_group(trade_date, symbol)


def _exit_stop(exit_state: dict, key: str) -> Tuple[Optional[float], Optional[str]]:
    """(current_stop, status) for a trade in exit_state.json, or (None, None)."""
    t = (exit_state or {}).get('trades', {}).get(key) or {}
    try:
        stop = float(t['current_stop']) if t.get('current_stop') else None
    except (TypeError, ValueError):
        stop = None
    return stop, t.get('status')


def _bot_orders_for(trade_date: str, symbol: str, orders: List[dict], roles) -> List[dict]:
    out = []
    for o in orders:
        parsed = parse_order_ref(o.get('order_ref'))
        if parsed and parsed[0] in roles and parsed[1] == trade_date and parsed[2] == symbol:
            out.append(o)
    return out


# ─── Deciding ───

def check_protection(state: BrokerState, ledger: OrderLedger, exit_state: dict,
                     quote: Callable[[str], Tuple[Optional[float], bool]] = lambda s: (None, True),
                     now: Optional[datetime] = None, account: Optional[str] = None) -> List[Protection]:
    """
    One Protection row per position (all of them, bot or not) plus one per orphan bot SELL.
    quote(symbol) -> (last, delayed) is called only for bot trades with unprotected shares, the one
    case where the price decides between PLACE_STOP / RESIZE_STOP and FLATTEN.
    """
    now = now or now_et()
    positions = [p for p in state.positions if float(p.get('shares') or 0) != 0
                 and (account is None or p.get('account') in (None, account))]
    orders = [o for o in _dedupe(state.open_orders) if _same_account(o, account)]
    risks = acct.position_risks(positions, orders)
    ledger_rows = ledger.current()
    trades = {}
    for t in ledger.open_trades():
        trades.setdefault((t.get('account'), t['symbol']), t)

    rows: List[Protection] = []
    for p, risk in zip(positions, risks):
        sym = str(p['symbol']).upper()
        shares = float(p['shares'])
        stops = _protective_stops(p, orders)
        stop_qty = sum(_remaining(o) for o in stops)
        long_stk = shares > 0 and p.get('sec_type', 'STK') == 'STK'
        sell_room = max(shares - _working_sell_qty(sym, p.get('account'), orders), 0.0) if long_stk else 0.0
        # The bot only goes long in stock: anything else is never a bot trade, except a short its
        # own SELLs caused (ledger net == the position)
        any_trade = trades.get((p.get('account'), sym)) or trades.get((None, sym))
        trade = any_trade if long_stk else None
        row = Protection(symbol=sym, shares=shares, protected=float(risk['protected_shares']),
                         unprotected=float(risk['unprotected_shares']), oversized=0.0, bot_trade=None,
                         stop=risk['stop'], ledger_stop=None, last=None, delayed=True, action=OK,
                         account=p.get('account'), sec_type=p.get('sec_type', 'STK'), stop_qty=stop_qty,
                         sell_room=sell_room)
        if (trade is None and any_trade is not None and shares < 0 and p.get('sec_type', 'STK') == 'STK'
                and any_trade['trade_status'] != 'manual_touched'
                and abs(float(any_trade['net_qty']) - shares) < EPS):
            _decide_bot_short(row, any_trade, orders, now)
            rows.append(row)
            continue
        if trade is None:
            _decide_manual(row)
            rows.append(row)
            continue

        d = trade['trade_date']
        key = _trade_key(trade)
        exit_stop, exit_status = _exit_stop(exit_state, key)
        row.bot_trade, row.trade_date, row.setup = key, d, trade['setup']
        row.trade_status = 'manual_touched' if exit_status == 'manual_touched' else trade['trade_status']
        row.ledger_stop = _ledger_stop(ledger_rows, d, sym)
        row.stop = exit_stop or row.ledger_stop or row.stop
        row.oca_group = _ledger_oca(ledger_rows, d, sym)
        # A working bot entry's child stop legitimately covers the parent's unfilled rest
        row.pending_entry_qty = sum(_remaining(o) for o in _bot_orders_for(d, sym, orders, ('entry',))
                                    if _is_working(o))
        row.oversized = max(stop_qty - abs(shares) - row.pending_entry_qty, 0.0)
        bot_stops = [o for o in stops if (parse_order_ref(o.get('order_ref')) or ('',))[0] in STOP_ROLES
                     and parse_order_ref(o.get('order_ref'))[1:] == (d, sym)]
        # The bot stop to modify: the largest one (normally there is exactly one). split_stop: a resize
        # grows the main stop; a shrink takes a trim-block stop whose trim is no longer working first
        # (the trim filled and TWS left its stop), else the largest stop
        main = [o for o in bot_stops if parse_order_ref(o.get('order_ref'))[0] != 'trim_stop']
        lone_tstops = [o for o in bot_stops if parse_order_ref(o.get('order_ref'))[0] == 'trim_stop'
                       and not any(_is_working(t) for t in _bot_orders_for(d, sym, orders, ('trim',)))]
        # No main stop: a RESIZE must not grow the trim block's stop (its group would then cover
        # shares outside the block); PLACE_STOP puts a stop_fix in the main group instead
        row.target_order = max(main, key=_remaining) if main else None
        if row.oversized > EPS and bot_stops:
            row.target_order = max(lone_tstops or main or bot_stops, key=_remaining)
        if row.trade_status == 'manual_touched':
            _decide_manual(row)
            row.note = ("manual_touched: the bot no longer manages this trade; " + row.note).rstrip('; ')
        else:
            _decide_bot(row, quote, now)
        rows.append(row)

    rows.extend(_orphan_sells(positions, orders, account))
    return rows


def _decide_manual(row: Protection) -> None:
    """Manual (or manual_touched) positions are only reported."""
    row.oversized = max(row.stop_qty - abs(row.shares), 0.0)
    if row.oversized > EPS:
        row.action = REPORT_OVERSIZED
        row.note = (f"stops cover {row.stop_qty:g} > {abs(row.shares):g} held: a full stop-out would leave "
                    f"the account {'short' if row.shares > 0 else 'long'}; fix by hand")
    elif row.unprotected > EPS:
        row.action = REPORT
        row.note = f"{row.unprotected:g} of {abs(row.shares):g} shares have no working stop (not a bot trade)"
    else:
        row.action = OK


def _decide_bot_short(row: Protection, trade: dict, orders: List[dict], now: datetime) -> None:
    """The bot's own SELLs overfilled into a short (two OCA members filled together): buy it back
    at market in the regular session; outside it, or with a BUY already working, alert only."""
    row.bot_trade, row.trade_date, row.setup = _trade_key(trade), trade['trade_date'], trade['setup']
    row.trade_status = trade['trade_status']
    row.oca_group = ''
    buys = [o for o in orders if str(o.get('symbol', '')).upper() == row.symbol and _is_working(o)
            and str(o.get('action', '')).upper() == 'BUY' and _same_account(o, row.account)]
    if buys:
        row.action = REPORT
        row.note = (f"bot SELLs left the account short {abs(row.shares):g}; a BUY order is already working "
                    f"({buys[0].get('order_ref') or buys[0].get('order_id')}): check it by hand")
    elif in_regular_session(now):
        row.action = COVER_SHORT
        row.note = f"bot SELLs left the account short {abs(row.shares):g}: buy to cover at market"
    else:
        row.action = ALERT_COVER_SHORT
        row.note = (f"bot SELLs left the account short {abs(row.shares):g}; market closed: cover by hand "
                    f"or let the next regular-session run do it")


def _decide_bot(row: Protection, quote: Callable[[str], Tuple[Optional[float], bool]], now: datetime) -> None:
    if row.oversized > EPS:
        if row.target_order is not None:
            row.action = SHRINK_STOP
            row.note = f"stops cover {row.stop_qty:g} > {row.shares:g} held: shrink the bot stop"
        else:
            row.action = REPORT_OVERSIZED
            row.note = f"manual stops cover {row.stop_qty:g} > {row.shares:g} held: fix by hand"
        return
    if row.unprotected <= EPS:
        row.action = OK
        return
    if row.stop is None:
        row.action = REPORT
        row.note = f"{row.unprotected:g} shares unprotected and no stop level known (ledger / exit_state)"
        return
    row.last, row.delayed = quote(row.symbol)
    if row.last is not None and row.last <= row.stop:
        live = not row.delayed and in_regular_session(now)
        row.action = FLATTEN if live else ALERT_FLATTEN
        row.note = (f"last {row.last:g} <= stop {row.stop:g} with {row.unprotected:g} shares unprotected"
                    + ("" if live else ("; quote delayed" if row.delayed else "; market closed")
                       + ": alert only, flatten by hand"))
        return
    if row.target_order is not None:
        row.action = RESIZE_STOP
        row.note = (f"bot stop covers {_remaining(row.target_order):g} of {row.shares:g} (partial fill?): "
                    f"resize to the position")
    else:
        row.action = PLACE_STOP
        row.note = f"no bot stop working for {row.unprotected:g} shares: place one at {row.stop:g}"
    if row.last is None:
        row.note += "; no quote"


def _orphan_sells(positions: List[dict], orders: List[dict], account: Optional[str]) -> List[Protection]:
    """Bot SELL orders working with no long position behind them (a stop-out left the trim, a paper
    reset, ...). A bracket child whose parent is still working is NOT an orphan."""
    held = {}
    for p in positions:
        if p.get('sec_type', 'STK') == 'STK':
            k = str(p['symbol']).upper()
            held[k] = held.get(k, 0.0) + float(p['shares'])
    working_ids = {(o.get('client_id'), o.get('order_id')) for o in orders if _is_working(o)}
    out = []
    for o in orders:
        parsed = parse_order_ref(o.get('order_ref'))
        if (not parsed or parsed[0] not in ol.SELL_ROLES or not _is_working(o)
                or str(o.get('action', '')).upper() != 'SELL'):
            continue
        sym = str(o.get('symbol', '')).upper()
        if held.get(sym, 0.0) > 0:
            continue
        parent = o.get('parent_id') or 0
        entry_working = any(_is_working(e) for e in _bot_orders_for(parsed[1], sym, orders, ('entry',)))
        if (parent and (o.get('client_id'), parent) in working_ids) or entry_working:
            continue           # child of a bracket still waiting for its parent
        out.append(Protection(
            symbol=sym, shares=held.get(sym, 0.0), protected=0.0, unprotected=0.0, oversized=_remaining(o),
            bot_trade=None, stop=acct.stop_price(o), ledger_stop=None, last=None, delayed=True,
            action=CANCEL_ORPHAN_SELL, account=o.get('account') or account, trade_date=parsed[1],
            note=f"bot SELL {o.get('order_ref')} (order {o.get('order_id')}) for {_remaining(o):g} shares "
                 f"with no long position: it would open a short", oca_group=o.get('oca_group') or '',
            target_order=o))
    return out


def plan_fixes(rows: List[Protection], submit: bool, client_id: int = ORDER_CLIENT_ID) -> List[dict]:
    """
    One fix dict per row whose action is a fix. Without both submit gates every fix becomes
    ALERT_<action> (report + alert only). A modify / cancel of an order another client placed is an
    alert too (IBKR refuses it). New SELL quantities are capped at the row's sell room.
    """
    fixes = []
    for i, r in enumerate(rows):
        if r.action not in FIX_ACTIONS:
            continue
        target = r.target_order or {}
        fix = {'row': i, 'action': r.action, 'symbol': r.symbol, 'account': r.account, 'trade_date': r.trade_date,
               'setup': r.setup, 'bot_trade': r.bot_trade, 'stop': r.stop, 'oca_group': r.oca_group,
               'qty': 0.0, 'new_qty': None, 'order_id': target.get('order_id'),
               'order_ref': target.get('order_ref'), 'client_id': target.get('client_id'),
               'target': target or None, 'note': r.note}
        if r.action in (PLACE_STOP, FLATTEN):
            fix['qty'] = float(min(r.unprotected, r.sell_room))
            fix['order_ref'] = make_order_ref('stop_fix' if r.action == PLACE_STOP else 'flatten',
                                              r.trade_date, r.symbol)
            fix['order_id'] = fix['client_id'] = None
            if fix['qty'] < r.unprotected - EPS:
                fix['note'] += (f"; capped at {fix['qty']:g} by other working SELLs (trim / exit): "
                                f"{r.unprotected - fix['qty']:g} shares stay unprotected")
        elif r.action == RESIZE_STOP:
            cur = _remaining(target)
            fix['new_qty'] = float(cur + min(r.unprotected, r.sell_room))
            fix['qty'] = fix['new_qty'] - cur
        elif r.action == SHRINK_STOP:
            cur = _remaining(target)
            fix['new_qty'] = float(max(cur - r.oversized, 0.0))
            fix['qty'] = fix['new_qty'] - cur
        elif r.action == CANCEL_ORPHAN_SELL:
            fix['new_qty'] = 0.0
        elif r.action == COVER_SHORT:
            fix['qty'] = float(abs(r.shares))
            fix['order_ref'] = make_order_ref(ol.COVER_ROLE, r.trade_date, r.symbol)
            fix['order_id'] = fix['client_id'] = None

        if fix['qty'] <= EPS and r.action in (PLACE_STOP, FLATTEN, RESIZE_STOP):
            fix['action'] = ALERT_PREFIX + r.action
            fix['note'] += "; no sell room left (other working SELLs cover the position)"
        elif r.action in (RESIZE_STOP, SHRINK_STOP, CANCEL_ORPHAN_SELL) and fix['client_id'] != client_id:
            fix['action'] = ALERT_PREFIX + r.action
            fix['note'] += (f"; order was placed by client {fix['client_id']}, only that client can "
                            f"modify / cancel it")
        if not submit and not fix['action'].startswith(ALERT_PREFIX):
            fix['action'] = ALERT_PREFIX + fix['action']
        fixes.append(fix)
    return fixes


# ─── Acting (submit mode only) ───

def _sell_order(fix: dict, order_type: str, qty: float, account: str, aux: Optional[float] = None,
                tif: str = "GTC", order_ref: Optional[str] = None, parent_id: int = 0):
    from ibapi.order import Order
    o = Order()
    # whole shares as an int: a float goes over the wire as "100.0"
    o.action, o.orderType, o.totalQuantity = "SELL", order_type, int(round(float(qty)))
    if aux is not None:
        o.auxPrice = aux
    o.tif, o.outsideRth, o.transmit = tif, False, True
    o.ocaGroup, o.ocaType = fix['oca_group'] or make_oca_group(fix['trade_date'], fix['symbol']), 3
    o.orderRef, o.account = order_ref or fix['order_ref'], account
    if parent_id:
        o.parentId = parent_id
    return o


def _ledger_row_for(ledger: OrderLedger, target: dict) -> Optional[dict]:
    """The ledger's latest event for a TWS order (orderRef + order id + client)."""
    for ev in reversed(ledger.events()):
        if (ev.get('order_ref') == target.get('order_ref') and ev.get('order_id') == target.get('order_id')
                and ev.get('client_id') == target.get('client_id')):
            return ev
    return None


def _ident(ev: dict) -> dict:
    return {k: ev[k] for k in ol.ORDER_ID_FIELDS}


def apply_fixes(fixes: List[dict], broker: Broker, ledger: OrderLedger, client_id: int = ORDER_CLIENT_ID,
                ack_timeout: float = 10.0) -> List[dict]:
    """
    Send the non-ALERT fixes (caller holds ledger_lock). New orders: fresh id -> ledger intent ->
    oversell guard -> place -> submitted -> ack. Modifies / cancels: the request is written onto the
    order's ledger row first. A TWS failure stops the run (nothing resent; reconcile settles it).
    Returns the fixes with 'result' set.
    """
    if not broker.submit_enabled:
        raise PermissionError(SUBMIT_REFUSED)
    stopped = False
    for fix in fixes:
        if fix['action'].startswith(ALERT_PREFIX):
            fix.setdefault('result', 'alert_only')
            continue
        if stopped:
            fix['result'] = 'not_sent_after_failure'
            continue
        try:
            if fix['action'] in (PLACE_STOP, FLATTEN):
                fix['result'] = _place_fix(fix, broker, ledger, client_id, ack_timeout)
            elif fix['action'] == COVER_SHORT:
                fix['result'] = _cover_fix(fix, broker, ledger, client_id, ack_timeout)
            elif fix['action'] in (RESIZE_STOP, SHRINK_STOP) and fix['new_qty'] and fix['new_qty'] > EPS:
                fix['result'] = _modify_fix(fix, broker, ledger)
            else:
                fix['result'] = _cancel_fix(fix, broker, ledger)
        except (ConnectionError, OSError, RuntimeError) as exc:
            logger.error(f"{fix['symbol']}: {fix['action']} failed ({exc}); stopping, no resend")
            fix['result'], stopped = f"failed: {exc}", True
        if not str(fix['result']).startswith(('working', 'submitted', 'modified', 'cancel_requested')):
            notify(f"{fix['symbol']} {fix['action']}: {fix['result']}")
    return fixes


def _place_fix(fix: dict, broker: Broker, ledger: OrderLedger, client_id: int, ack_timeout: float) -> str:
    sym, qty = fix['symbol'], fix['qty']
    role = 'stop_fix' if fix['action'] == PLACE_STOP else 'flatten'
    if fix['action'] == PLACE_STOP:
        order = _sell_order(fix, "STP", qty, broker.account, aux=fix['stop'], tif="GTC")
    else:
        order = _sell_order(fix, "MKT", qty, broker.account, tif="DAY")
    oid = max(broker.next_order_ids(1)[0], ledger.highest_order_id() + 1)
    key = LedgerKey(date.fromisoformat(fix['trade_date']), sym, fix['setup'])
    stored = ledger.record_intent(key, role, fix['order_ref'], oid, action='SELL', order_type=order.orderType,
                                  qty=qty, aux_price=fix['stop'] if role == 'stop_fix' else None,
                                  tif=order.tif, oca_group=order.ocaGroup, client_id=client_id,
                                  account=broker.account, reason=fix['action'])
    ident = _ident(stored)
    try:
        assert_no_oversell(broker.state(), sym, qty, broker.account)
    except ValueError as exc:
        ledger.record({**ident, 'state': 'cancelled', 'reason': 'oversell_guard'})
        return f"refused: {exc}"
    try:
        broker.place(oid, stock_contract(sym), order)
    except Exception:
        raise ConnectionError(f"place {fix['order_ref']} failed; left 'intended' for reconcile")
    ledger.record({**ident, 'state': 'submitted'})
    ack, why = broker.wait_ack(oid, timeout=ack_timeout)
    if ack in ('working', 'rejected'):
        ledger.record({**ident, 'state': ack, 'reason': why if ack == 'rejected' else ''})
    fix['order_id'] = oid
    return f"{ack}: {why}" if ack != 'working' else f"working (order {oid})"


def _cover_fix(fix: dict, broker: Broker, ledger: OrderLedger, client_id: int, ack_timeout: float) -> str:
    """BUY MKT DAY back to flat. Re-checks the live position first: never buys more than the short."""
    from ibapi.order import Order
    sym = fix['symbol']
    st = broker.state()
    held = sum(float(p.get('shares') or 0) for p in st.positions
               if str(p.get('symbol', '')).upper() == sym and p.get('sec_type', 'STK') == 'STK'
               and p.get('account') in (None, broker.account))
    qty = int(round(min(fix['qty'], max(-held, 0.0))))
    if qty <= 0:
        return f"refused: no short left ({held:g} held)"
    order = Order()
    order.action, order.orderType, order.totalQuantity = "BUY", "MKT", qty
    order.tif, order.outsideRth, order.transmit = "DAY", False, True
    order.orderRef, order.account = fix['order_ref'], broker.account
    oid = max(broker.next_order_ids(1)[0], ledger.highest_order_id() + 1)
    key = LedgerKey(date.fromisoformat(fix['trade_date']), sym, fix['setup'])
    stored = ledger.record_intent(key, ol.COVER_ROLE, fix['order_ref'], oid, action='BUY', order_type='MKT',
                                  qty=qty, tif='DAY', client_id=client_id, account=broker.account,
                                  reason=fix['action'])
    ident = _ident(stored)
    try:
        broker.place(oid, stock_contract(sym), order)
    except Exception:
        raise ConnectionError(f"place {fix['order_ref']} failed; left 'intended' for reconcile")
    ledger.record({**ident, 'state': 'submitted'})
    ack, why = broker.wait_ack(oid, timeout=ack_timeout)
    if ack in ('working', 'rejected'):
        ledger.record({**ident, 'state': ack, 'reason': why if ack == 'rejected' else ''})
    fix['order_id'] = oid
    return f"{ack}: {why}" if ack != 'working' else f"working (order {oid})"


def _modify_fix(fix: dict, broker: Broker, ledger: OrderLedger) -> str:
    t = fix['target']
    row = _ledger_row_for(ledger, t)
    old_qty = _remaining(t)
    if fix['new_qty'] > old_qty:
        try:
            assert_no_oversell(broker.state(), fix['symbol'], fix['new_qty'] - old_qty, broker.account)
        except ValueError as exc:
            return f"refused: {exc}"
    aux = acct.stop_price(t) or fix['stop']
    # TWS reads totalQuantity as the whole order: filled part + the new remaining quantity
    order = _sell_order(fix, str(t.get('order_type') or 'STP'), float(t.get('filled') or 0) + fix['new_qty'],
                        broker.account, aux=aux,
                        tif=str(t.get('tif') or 'GTC'), order_ref=t.get('order_ref'),
                        parent_id=int(t.get('parent_id') or 0))
    if t.get('oca_group'):
        order.ocaGroup = t['oca_group']   # a modify must keep the order's own OCA group
    if row is not None:
        ledger.record({**_ident(row), 'state': row['state'], 'qty': float(order.totalQuantity),
                       'reason': f"{fix['action'].lower()}_intent {old_qty:g}->{fix['new_qty']:g}"})
    try:
        broker.modify(int(t['order_id']), stock_contract(fix['symbol']), order)
    except Exception as exc:
        if row is not None:
            ledger.record({**_ident(row), 'state': row['state'], 'qty': row.get('qty'),
                           'reason': f"modify_failed: {exc}"})
        raise ConnectionError(f"modify {t.get('order_ref')} failed: {exc}")
    return f"modified {old_qty:g}->{fix['new_qty']:g}"


def _cancel_fix(fix: dict, broker: Broker, ledger: OrderLedger) -> str:
    t = fix['target']
    row = _ledger_row_for(ledger, t)
    if row is not None:
        ledger.record({**_ident(row), 'state': row['state'], 'cancel_requested': True,
                       'reason': f"{fix['action'].lower()}: cancel requested"})
    broker.cancel(int(t['order_id']))
    return f"cancel_requested (order {t['order_id']})"


# ─── Output ───

def write_report(rows: List[Protection], path: Optional[Path] = None, fixes: Optional[List[dict]] = None,
                 mode: str = "read-only", now: Optional[datetime] = None) -> Path:
    """Data/orders/protection_<yyyy-mm-dd>_<HHMM>.csv: one row per position / orphan SELL."""
    now = now or now_et()
    path = Path(path) if path else _orders_dir() / f"protection_{now:%Y-%m-%d_%H%M}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    by_row = {f['row']: f for f in fixes or []}
    out = []
    for i, r in enumerate(rows):
        d = asdict(r)
        d.pop('target_order', None)
        f = by_row.get(i)
        d['fix'] = (f"{f['action']} {f.get('result', '')}".strip()) if f else ''
        out.append({'run_ts': now.isoformat(timespec='seconds'), 'mode': mode, **d})
    pd.DataFrame(out, columns=REPORT_COLUMNS).to_csv(path, index=False)
    return path


def notify(message: str) -> None:
    """Log, plus a macOS notification (notify.py: rate-limited, off with NOTIFY_ENABLED / TRADING_NOTIFY=0)."""
    logger.warning(f"ALERT: {message}")
    notifier.notify("Fill monitor", message, 'warning')


def notify_all(messages: List[str]) -> None:
    """Log each alert, plus ONE macOS notification for the batch (8 naked positions = 1 banner, not 8)."""
    for message in messages:
        logger.warning(f"ALERT: {message}")
    notifier.notify_batch("Fill monitor", messages, 'warning')


def _summary(rows: List[Protection]) -> str:
    counts: Dict[str, int] = {}
    for r in rows:
        counts[r.action] = counts.get(r.action, 0) + 1
    return ", ".join(f"{k} {v}" for k, v in sorted(counts.items())) or "no positions"


def run(broker: Broker, ledger: OrderLedger, submit: bool, client_id: int, now: datetime,
        exit_state: Optional[dict] = None) -> Tuple[List[Protection], List[dict], ol.ReconcileReport]:
    """One monitor pass on an open session (caller holds ledger_lock): reconcile -> check -> fix -> report."""
    state = broker.state()
    report = ledger.reconcile(state, now.date())
    alerts = [f"ledger reconcile: {item.get('order_ref')} {item.get('issue') or item.get('reason')}"
              for item in report.orphans + report.mismatches]
    exit_state = load_exit_state() if exit_state is None else exit_state
    rows = check_protection(state, ledger, exit_state, quote=broker.last_price, now=now, account=broker.account)
    fixes = plan_fixes(rows, submit=submit and broker.submit_enabled, client_id=client_id)
    if submit and broker.submit_enabled and any(not f['action'].startswith(ALERT_PREFIX) for f in fixes):
        apply_fixes(fixes, broker, ledger, client_id=client_id)
        ledger.reconcile(broker.state(), now.date())
    alerts += [f"{f['symbol']} {f['action']}: {f['note']}" for f in fixes if f['action'].startswith(ALERT_PREFIX)]
    alerts += [f"{r.symbol} {r.action}: {r.note}" for r in rows
               if r.action in (REPORT, REPORT_OVERSIZED, ALERT_FLATTEN, ALERT_COVER_SHORT)]
    notify_all(alerts)
    return rows, fixes, report


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check every position's stop coverage; with both gates, "
                                                 "repair the bot's own trades. Paper only.")
    parser.add_argument('--account', default=None, help="Paper account DU... (default config IBKR_ACCOUNT)")
    parser.add_argument('--port', type=int, default=int(getattr(config, 'TWS_PORT', 7497)))
    parser.add_argument('--submit', action='store_true',
                        help="Apply fixes (only if config AUTO_SUBMIT_ENABLED is True)")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s", datefmt="%H:%M:%S")
    args = parse_args(argv)
    now = now_et()
    if not calendar_covers(now.date()):
        logger.error(f"{now.date()} is past the NYSE holiday table in market_time.py: add that year's holidays and early closes first")
        return 2
    if not is_trading_day(now.date()):
        logger.info(f"{now.date()} is not a trading day: nothing to do")
        return 0
    account = (args.account or str(getattr(config, 'IBKR_ACCOUNT', '') or '')).strip().upper() or None
    if not account:
        logger.error("--account DU... (or config IBKR_ACCOUNT) is required")
        return 2
    submit = submission_enabled(args.submit)
    refused = args.submit and not submit
    if refused:
        logger.error(f"--submit refused: {SUBMIT_REFUSED}; running read-only (alerts only)")
    client_id = ORDER_CLIENT_ID if submit else ORDER_READONLY_CLIENT_ID
    from order_broker import open_session
    try:
        with ol.ledger_lock():
            ledger = OrderLedger(source='fill_monitor')
            try:
                broker = open_session(args.port, account, args.submit, client_id)
            except PermissionError as exc:
                logger.error(f"Paper guard refused the session: {exc}")
                return 2
            try:
                rows, fixes, _ = run(broker, ledger, submit, client_id, now)
            finally:
                broker.close()
            mode = 'submit' if submit else ('submit-refused' if refused else 'read-only')
            path = write_report(rows, fixes=fixes, mode=mode, now=now)
            logger.info(f"Protection ({mode}): {_summary(rows)} -> {path}")
    except ol.LedgerLocked as exc:
        logger.error(str(exc))
        return 3
    except Exception as exc:
        logger.exception(f"fill_monitor failed: {exc}")
        return 1
    unresolved = [r for r in rows if r.action not in QUIET_ACTIONS] if not submit else (
        [r for r in rows if r.action not in QUIET_ACTIONS and r.action not in FIX_ACTIONS]
        + [f for f in fixes if not str(f.get('result', '')).startswith(('working', 'modified', 'cancel_requested'))])
    return 2 if refused or unresolved else 0


if __name__ == "__main__":
    raise SystemExit(notifier.notify_exit_code(main(), "Fill monitor"))
