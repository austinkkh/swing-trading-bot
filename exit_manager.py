#!/usr/bin/env python3
"""
Exit manager: runs the adopted exit plan (exit_rules.manage, unchanged) on the bot's own open
trades and turns its decisions into the next session's orders (paper only).

Two runs a day (no launchd job is installed; scheduling is the owner's call):
  nightly          after market_close_time(today) ET. Complete daily bars -> manage() -> orders for
                   the next session (stop to breakeven, resting trim, market-on-open exits).
  --close-window   inside CLOSE_WINDOW_ET (default 15:35-15:48 ET, shifted to 12:35-12:48 on early
                   closes). If today is the session after TIME_STOP_SESSIONS and the intraday high is still
                   below +TRIM_R, a SELL MOC (market-on-close) for the open shares: exit_rules books the
                   time stop at that session's close, and a nightly run can only sell at the next open.

manage() output -> order (TRIM_STOP_MODE shared_oca, the default: one OCA group per trade, ocaType 3,
every bot SELL in it; split_stop: see TRIM_STOP_MODES, the partial trim gets its own stop and group):
  stop_price above the working stop   modify the bracket's child stop (same id, client 16); also
                                      when the stop covers more than shares_open (partial entry)
  no bot stop working                 SELL STP GTC for shares_open at the current stop, BOT-STOPFIX
                                      (a full trim fill may have cancelled the stop if OCA cancels
                                      instead of reducing; a partial entry's child may have died with
                                      the parent). A close below it means a sale at the next open,
                                      as simulate_trade books a gap through the stop.
  take_profit_order set               a SELL LMT GTC for trim_qty at +TRIM_R is working (placed once);
                                      it fills at max(open, target), as simulate_trade assumes
  +TRIM_R within FAST_SESSIONS (hold_full)   no trim order ever exists
  SELL_TRIM_AT_TARGET (the resting LMT     cancel any unfilled trim LMT; SELL MKT OPG for the unfilled
    was missing or didn't fill)       part of trim_qty, tagged DEVIATION_TRIM_NEXT_OPEN (the
                                      simulator sold at max(open, target) that bar)
  EXIT_NEXT_OPEN                      cancel the working trim; SELL MKT OPG (MOO) for shares_open
  TIME_STOP_AT_CLOSE                  --close-window placed a MOC that day; if the trade is still
                                      open at night, SELL MOO next open, tagged
                                      DEVIATION_TIME_STOP_NEXT_OPEN

Fills are synced from the ledger (reconcile) BEFORE manage() runs, so a trim LMT that filled today
counts as trimmed exactly like simulate_trade's resting-order fill before manage() on that bar.
'trimmed' comes ONLY from trim fills (all of trim_qty): manage() sets it when it asks for the sale,
which is not persisted, so an unfilled or rejected trim is asked for again the next night.
tests/test_exit_manager.py replays historical price paths through these functions and a fake
broker and checks the R, exit bar and trimmed flag equal exit_rules.simulate_trade's.

Data/orders/exit_state.json holds one entry per bot trade (see new_trade()). It is written
atomically (tmp file + os.replace) under the ledger lock, and only by --submit runs: a dry run
must not record a trim or a stop move that never happened. fill_monitor reads current_stop.

Submission is DISABLED by default. Orders are sent only if config AUTO_SUBMIT_ENABLED is the bool
True AND --submit is passed; otherwise the run writes Data/orders/exit_plan_<date>.csv and alerts.
Every order is written to the ledger as an intent before it is sent, and every new SELL passes
order_broker.assert_no_oversell with its OCA group pooled (a trim beside a full-size stop is one
exposure, not two).

Client IDs: read-only runs use ORDER_READONLY_CLIENT_ID (17); --submit with the config switch uses
ORDER_CLIENT_ID (16), the client that placed the bracket (only it can modify the child stop).

Exit codes: 0 ok (also nothing to do / holiday) · 1 error · 2 refused by a guard (time window,
paper guard, --submit without the config switch) or an action needs a human · 3 another order run
holds the ledger lock.

Usage:
  python exit_manager.py --account DUXXXXXXX                       # nightly dry run: plan + alerts
  python exit_manager.py --account DUXXXXXXX --submit              # paper, both gates: sends orders
  python exit_manager.py --account DUXXXXXXX --close-window --submit   # ~15:40 ET time-stop MOC
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
from datetime import date, datetime, time as dtime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

import config
import notify as notifier
import exit_rules
import order_ledger as ol
from market_time import REGULAR_CLOSE, calendar_covers, is_trading_day, market_close_time, now_et
from order_broker import (ORDER_CLIENT_ID, ORDER_READONLY_CLIENT_ID, SUBMIT_REFUSED, Broker, BrokerState,
                          WORKING as BROKER_WORKING, assert_no_oversell, stock_contract, submission_enabled)
from order_ledger import LedgerKey, OrderLedger, make_oca_group, make_order_ref, make_trim_oca_group, parse_order_ref

logger = logging.getLogger(__name__)

STATE_VERSION = 1
EPS = 1e-9
PRICE_TICK = 0.005               # a stop modify smaller than half a cent is noise

# Decisions (what manage() says), before they become orders
STOP_TO = "STOP_TO"
TRIM_RESTING = "TRIM_RESTING"
NO_TRIM = "NO_TRIM"              # no resting trim should exist (reached +TRIM_R, hold_full, exiting)
TRIM_NEXT_OPEN = "TRIM_NEXT_OPEN"
EXIT_NEXT_OPEN = "EXIT_NEXT_OPEN"
TIME_STOP_NEXT_OPEN = "TIME_STOP_NEXT_OPEN"
EXIT_AT_CLOSE = "EXIT_AT_CLOSE"
STALE_BARS = "STALE_BARS"

# Order actions (exit plan rows)
MODIFY_STOP = "MODIFY_STOP"
PLACE_TRIM = "PLACE_TRIM"
CANCEL_TRIM = "CANCEL_TRIM"
PLACE_TRIM_MOO = "PLACE_TRIM_MOO"
PLACE_EXIT_MOO = "PLACE_EXIT_MOO"
PLACE_EXIT_MOC = "PLACE_EXIT_MOC"
PLACE_STOP_FIX = "PLACE_STOP_FIX"   # no bot stop working at night: a new GTC stop for shares_open
PLACE_TRIM_STOP = "PLACE_TRIM_STOP"   # split_stop: the trim block's own stop, in the trim's OCA group
CANCEL_TRIM_STOP = "CANCEL_TRIM_STOP"
HOLD = "HOLD"
REPORT = "REPORT"                # manual_touched / closed: report only
ALERT_STALE_BARS = "ALERT_STALE_BARS"
ALERT_NO_STOP = "ALERT_NO_STOP"
ALERT_OTHER_CLIENT = "ALERT_OTHER_CLIENT"
ORDER_ACTIONS = (MODIFY_STOP, PLACE_TRIM, CANCEL_TRIM, PLACE_TRIM_MOO, PLACE_EXIT_MOO, PLACE_EXIT_MOC,
                 PLACE_STOP_FIX, PLACE_TRIM_STOP, CANCEL_TRIM_STOP)
ALERT_ACTIONS = (ALERT_STALE_BARS, ALERT_NO_STOP, ALERT_OTHER_CLIENT)

DEVIATION_TRIM_NEXT_OPEN = "DEVIATION_TRIM_NEXT_OPEN"
DEVIATION_TIME_STOP_NEXT_OPEN = "DEVIATION_TIME_STOP_NEXT_OPEN"
DEVIATION_EXIT_RETRY = "DEVIATION_EXIT_RETRY"

STOP_ROLES = ('stop', 'stop_fix')        # the main block's stops (all of the position in shared_oca)
TRIM_STOP_ROLE = 'trim_stop'
LIVE_LEDGER_STATES = {'intended', 'submitted', 'working', 'partially_filled'}
OPEN_STATUSES = ('open', 'closing')

# How the resting trim sits beside the stop (config TRIM_STOP_MODE; paper check 1 decides):
#   shared_oca  one OCA group (ocaType 3) for the stop, the trim and the exits: assumes a trim fill
#               REDUCES the stop by the shares sold. If TWS cancels the stop instead, the position has
#               no stop until fill_monitor / the nightly run replaces it.
#   split_stop  two blocks, each in its own OCA group and every order in a group sized to that block:
#               main block (shares - trim): the stop (bracket child) and its exit share
#               trim block (trim shares):   a trim_stop, the trim LMT / MOO and its exit share
#               A member that fills in full leaves nothing to protect in its group, so cancel,
#               proportional and quantity reduce all end correct. Only partial fills differ; the
#               nightly run re-sizes the trim block's stop to the trim still working.
TRIM_STOP_MODES = ('shared_oca', 'split_stop')

PLAN_COLUMNS = ['run_ts', 'mode', 'trade', 'symbol', 'action', 'role', 'order_type', 'tif', 'qty', 'price',
                'order_ref', 'order_id', 'target_order_id', 'oca_group', 'tag', 'note', 'result']


def _orders_dir() -> Path:
    return ol.ORDERS_DIR


def trim_stop_mode() -> str:
    """config TRIM_STOP_MODE (default shared_oca); anything else is refused, never guessed."""
    mode = _cfg('TRIM_STOP_MODE', 'shared_oca')
    if mode not in TRIM_STOP_MODES:
        raise ValueError(f"config TRIM_STOP_MODE {mode!r} must be one of {TRIM_STOP_MODES}")
    return mode


def _cfg(key: str, default):
    return getattr(config, key, default)


def _trade_key(trade_date: str, symbol: str, setup: str) -> str:
    return f"{trade_date}:{symbol}:{setup}"


# ─── exit_state.json ───

def load_exit_state(path: Optional[Path] = None) -> dict:
    """{'version', 'updated', 'trades': {key: trade}}. Missing -> empty. Unreadable -> RuntimeError:
    managing trades without their state (trimmed / hold_full) could repeat a trim."""
    path = Path(path) if path else _orders_dir() / "exit_state.json"
    if not path.exists():
        return {'version': STATE_VERSION, 'updated': None, 'trades': {}}
    try:
        state = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"exit_state {path} unreadable ({exc}): fix or restore it by hand") from exc
    state.setdefault('trades', {})
    return state


def save_exit_state(state: dict, path: Optional[Path] = None, now: Optional[datetime] = None) -> Path:
    """Atomic write: a tmp file in the same folder, fsynced, then os.replace. A crash mid-write
    leaves the previous file untouched."""
    path = Path(path) if path else _orders_dir() / "exit_state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {**state, 'version': STATE_VERSION,
             'updated': (now or now_et()).isoformat(timespec='seconds')}
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, 'w') as fh:
            json.dump(state, fh, indent=1, default=str)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return path


def new_trade(entry: dict, ledger: OrderLedger) -> dict:
    """
    exit_state entry for a ledger entry with fills (created on the first fill). entry_price is the
    actual average fill, so R is measured from the real fill; initial_stop is the bracket child's
    stop as sent. trail_line is filled from the bars on the first run (respected_ma, the same call
    the scanner makes for Trail_MA) unless the ledger plan carries one.
    """
    d, sym = entry['trade_date'], entry['symbol']
    stop = _initial_stop(ledger, d, sym) or float((entry.get('plan') or {}).get('stop') or 0)
    shares = float(entry.get('filled_qty') or 0)
    trim = int(round(shares * exit_rules.TRIM_FRACTION))
    return {'symbol': sym, 'setup': entry['setup'], 'account': entry.get('account'), 'entry_date': d,
            'entry_price': float(entry.get('avg_fill_price') or (entry.get('plan') or {}).get('pivot') or 0),
            'initial_stop': stop, 'trail_line': (entry.get('plan') or {}).get('trail_line'),
            'trimmed': False, 'hold_full': False, 'reached_2r': False, 'pending_exit': False,
            'shares_entered': shares, 'shares_open': shares, 'trim_qty': trim if trim >= 1 else 0,
            'current_stop': stop, 'stop_order_id': None, 'stop_perm_id': None,
            'oca_group': make_oca_group(d, sym), 'trim_order_id': None, 'status': 'open',
            'last_bar_date': None, 'history': []}


def _initial_stop(ledger: OrderLedger, trade_date: str, symbol: str) -> Optional[float]:
    """The bracket child's stop as first recorded (the intent), before any breakeven move."""
    for ev in ledger.events():
        if ev.get('role') == 'stop' and ev.get('trade_date') == trade_date and ev.get('symbol') == symbol:
            try:
                return float(ev['aux_price'])
            except (KeyError, TypeError, ValueError):
                return None
    return None


def to_exit_state(t: dict) -> exit_rules.ExitState:
    return exit_rules.ExitState(entry_price=float(t['entry_price']), initial_stop=float(t['initial_stop']),
                                entry_date=pd.Timestamp(t['entry_date']),
                                trail_line=t.get('trail_line') or 'ema21', trimmed=bool(t.get('trimmed')),
                                hold_full=bool(t.get('hold_full')), reached_2r=bool(t.get('reached_2r')))


def from_exit_state(s: exit_rules.ExitState, t: dict) -> dict:
    t.update(trimmed=bool(s.trimmed), hold_full=bool(s.hold_full), reached_2r=bool(s.reached_2r))
    return t


def ensure_trades(exit_state: dict, ledger: OrderLedger) -> List[str]:
    """Add an exit_state entry for every ledger trade with fills that has none yet. Returns new keys."""
    trades = exit_state.setdefault('trades', {})
    added = []
    for e in ledger.open_trades():
        key = _trade_key(e['trade_date'], e['symbol'], e['setup'])
        if key not in trades:
            trades[key] = new_trade(e, ledger)
            added.append(key)
            logger.info(f"{e['symbol']}: new bot trade {key} ({e['filled_qty']:g} @ {e.get('avg_fill_price')})")
    return added


def sync_fills(trade: dict, ledger: OrderLedger, state: Optional[BrokerState] = None) -> None:
    """
    BEFORE manage(): bring the trade in line with the (just reconciled) ledger.
      trim filled in full  -> trimmed=True (mirrors simulate_trade's resting-LMT fill before manage)
      shares_open          -> ledger net qty (entry fills - bot SELL fills)
      ledger closed        -> status closed (stop-out, exit fill, paper reset)
      ledger manual_touched-> status manual_touched (position != ledger: report only from now on)
      exit order filled    -> closed (reconcile closes it too once the position is gone)
    """
    d, sym = trade['entry_date'], trade['symbol']
    status = ledger.trade_status(d, sym)
    if status == 'closed':
        trade['status'] = 'closed'
        trade['shares_open'] = 0.0
        return
    if status == 'manual_touched':
        trade['status'] = 'manual_touched'
    rows = _ledger_rows(ledger, d, sym)
    entry = next((r for r in rows if r['role'] == 'entry'), None)
    if entry is not None:
        filled = float(entry.get('filled_qty') or 0)
        if filled > trade.get('shares_entered', 0) + EPS:
            # A partial entry fill grew after the trade was created (DAY order, same session)
            trade['shares_entered'] = filled
            trim = int(round(filled * exit_rules.TRIM_FRACTION))
            trade['trim_qty'] = trim if trim >= 1 else 0
        sold = sum(float(r.get('filled_qty') or 0) for r in rows if r['role'] in ol.SELL_ROLES)
        covered = sum(float(r.get('filled_qty') or 0) for r in rows if r['role'] == ol.COVER_ROLE)
        trade['shares_open'] = max(float(entry.get('filled_qty') or 0) - sold + covered, 0.0)
    # Trimmed = the whole trim quantity filled (a 40-of-100 partial is not a trim: the rest is sold next)
    trim_filled = sum(float(r.get('filled_qty') or 0) for r in rows if r['role'] == 'trim')
    trade['trim_filled'] = trim_filled
    if trim_filled > EPS and trim_filled >= float(trade.get('trim_qty') or 0) - EPS:
        trade['trimmed'] = True
    stops = [r for r in rows if r['role'] in STOP_ROLES and r['state'] in LIVE_LEDGER_STATES]
    if stops:
        trade['stop_order_id'] = stops[-1].get('order_id')
        trade['stop_perm_id'] = stops[-1].get('perm_id')
    trims = [r for r in rows if r['role'] == 'trim' and r['state'] in LIVE_LEDGER_STATES]
    trade['trim_order_id'] = trims[-1].get('order_id') if trims else None
    if trade['status'] in OPEN_STATUSES and trade['shares_open'] <= EPS and entry is not None:
        trade['status'] = 'closed'
    elif trade['status'] == 'closing' and not _live_ledger(rows, 'exit'):
        trade['status'] = 'open'     # the exit order died unfilled (rejected / expired): manage again


def _ledger_rows(ledger: OrderLedger, trade_date: str, symbol: str) -> List[dict]:
    """Latest event per order for one trade, oldest first."""
    cur = ledger.current()
    if cur.empty:
        return []
    rows = cur[(cur['trade_date'] == trade_date) & (cur['symbol'] == symbol) & (cur['role'] != ol.TRADE_ROLE)]
    return rows.sort_values('ts', kind='stable').to_dict('records')


def _live_ledger(rows: List[dict], role: str) -> List[dict]:
    return [r for r in rows if r['role'] == role and r['state'] in LIVE_LEDGER_STATES]


# ─── Deciding ───

def _bars_through(bars: pd.DataFrame, today: date) -> Tuple[pd.DataFrame, str]:
    """(bars with normalized dates, '' if the last bar is today's, else why not)."""
    if bars is None or bars.empty:
        return pd.DataFrame(), "no daily bars"
    b = bars.copy()
    b['date'] = pd.to_datetime(b['date']).dt.normalize()
    last = b['date'].iat[-1].date()
    if last != today:
        return b, f"last daily bar is {last}, not today {today}"
    return b.reset_index(drop=True), ""


def _set_trail_line(trade: dict, bars: pd.DataFrame) -> None:
    """First run: the MA the stock respected in the RESPECT_LOOKBACK sessions before entry (scanner's Trail_MA)."""
    if trade.get('trail_line') in exit_rules.TRAIL_LINES:
        return
    idx = bars.index[bars['date'] >= pd.Timestamp(trade['entry_date'])]
    trade['trail_line'] = exit_rules.respected_ma(bars, int(idx[0])) if len(idx) else 'ema21'


def nightly_actions(trade: dict, bars: pd.DataFrame, today: date) -> List[dict]:
    """
    After today's close: exit_rules.manage() on complete daily bars, unchanged, and the decisions
    it implies for the next session. Refuses (one STALE_BARS decision, nothing changed) unless the
    bars' last date is today. Updates trade in place (flags, current_stop, pending_exit, history).
    """
    b, why = _bars_through(bars, today)
    if why:
        return [{'decision': STALE_BARS, 'note': why}]
    _set_trail_line(trade, b)
    trimmed_by_fill = bool(trade.get('trimmed'))
    st = to_exit_state(trade)
    acts = exit_rules.manage(st, b)
    from_exit_state(st, trade)
    # manage() sets trimmed when it ASKS for the trim sale; only a fill (sync_fills) makes it true.
    # Persisting manage()'s flag would silence the trim for good if that order never fills.
    trade['trimmed'] = trimmed_by_fill
    trim_left = max(int(round(float(trade.get('trim_qty') or 0) - float(trade.get('trim_filled') or 0))), 0)
    since = b[b['date'] >= pd.Timestamp(trade['entry_date'])]
    out: List[dict] = []
    notes = list(acts.notes)

    if acts.stop_price > float(trade['current_stop']) + EPS:
        notes.append(f"stop {trade['current_stop']:g} -> {acts.stop_price:g} (breakeven at +TRIM_R)")
    trade['current_stop'] = max(float(trade['current_stop']), float(acts.stop_price))
    out.append({'decision': STOP_TO, 'price': round(float(trade['current_stop']), 2)})

    exiting = 'EXIT_NEXT_OPEN' in acts.actions or 'TIME_STOP_AT_CLOSE' in acts.actions or trade.get('pending_exit')
    if acts.take_profit_order is not None and not exiting and trim_left >= 1:
        out.append({'decision': TRIM_RESTING, 'price': float(acts.take_profit_order), 'qty': trim_left})
    else:
        out.append({'decision': NO_TRIM})
    if 'SELL_TRIM_AT_TARGET' in acts.actions and trim_left >= 1 and not exiting:
        # +TRIM_R printed and the trim is not (fully) filled: the resting LMT was missing, didn't fill at
        # the target, or filled in part. Repeats every night until a fill makes trimmed True.
        out.append({'decision': TRIM_NEXT_OPEN, 'qty': trim_left, 'tag': DEVIATION_TRIM_NEXT_OPEN})
    if 'EXIT_NEXT_OPEN' in acts.actions:
        trade['pending_exit'] = True
        out.append({'decision': EXIT_NEXT_OPEN, 'qty': trade['shares_open'], 'tag': ''})
    elif 'TIME_STOP_AT_CLOSE' in acts.actions:
        # --close-window should have sold at today's close; still open -> next open, a deviation
        trade['pending_exit'] = True
        out.append({'decision': TIME_STOP_NEXT_OPEN, 'qty': trade['shares_open'],
                    'tag': DEVIATION_TIME_STOP_NEXT_OPEN})
    elif trade.get('pending_exit'):
        # An exit decided earlier never filled (rejected / missed open): simulate_trade would be out
        out.append({'decision': EXIT_NEXT_OPEN, 'qty': trade['shares_open'], 'tag': DEVIATION_EXIT_RETRY})

    trade['last_bar_date'] = today.isoformat()
    trade.setdefault('history', []).append({
        'date': today.isoformat(), 'sessions': int(len(since)), 'actions': list(acts.actions),
        'notes': notes, 'stop': trade['current_stop']})
    for d in out:
        d.setdefault('note', '; '.join(notes))
    return out


def close_window_actions(trade: dict, bars_today: pd.DataFrame, today: date) -> List[dict]:
    """
    Inside the close window, with today's forming bar: if manage() on today's bars so far says
    TIME_STOP_AT_CLOSE (today is the session after TIME_STOP_SESSIONS and the intraday high is still below +TRIM_R), one
    EXIT_AT_CLOSE decision (SELL MOC shares_open). manage() runs on a COPY of the state: the
    nightly run makes the real update from the complete bar.
    Known gap: +TRIM_R printed between this run and the close would have cancelled the time stop in
    simulate_trade; the MOC still sells (the trim LMT filling first reduces it via the OCA group).
    """
    if trade.get('status') not in OPEN_STATUSES or trade.get('shares_open', 0) <= EPS:
        return []
    b, why = _bars_through(bars_today, today)
    if why:
        return [{'decision': STALE_BARS, 'note': why}]
    t = copy.deepcopy(trade)
    _set_trail_line(t, b)
    st = to_exit_state(t)
    acts = exit_rules.manage(st, b)
    if 'TIME_STOP_AT_CLOSE' in acts.actions and not st.reached_2r:
        return [{'decision': EXIT_AT_CLOSE, 'qty': trade['shares_open'], 'tag': '',
                 'note': '; '.join(acts.notes) + " (market-on-close, as exit_rules books it)"}]
    return []


# ─── Decisions -> orders ───

def _is_working(o: dict) -> bool:
    return not o.get('status') or o.get('status') in BROKER_WORKING


def _remaining(o: dict) -> float:
    return max(float(o.get('quantity') or 0) - float(o.get('filled') or 0), 0.0)


def _tws_orders(open_orders: List[dict], trade: dict, roles) -> List[dict]:
    """Working bot orders at TWS for this trade with one of these roles (deduped, newest wins)."""
    out: Dict[tuple, dict] = {}
    for o in open_orders:
        parsed = parse_order_ref(o.get('order_ref'))
        if (parsed and parsed[0] in roles and parsed[1] == trade['entry_date'] and parsed[2] == trade['symbol']
                and _is_working(o) and str(o.get('action', '')).upper() == 'SELL'):
            out[o.get('perm_id') or (o.get('client_id'), o.get('order_id'))] = o
    return list(out.values())


def _stop_price(o: dict) -> Optional[float]:
    try:
        px = float(o.get('aux_price'))
    except (TypeError, ValueError):
        return None
    return px if 0 < px < 1e300 else None


def _row(trade: dict, action: str, role: str = '', order_type: str = '', tif: str = '', qty: float = 0.0,
         price: Optional[float] = None, order_ref: str = '', target: Optional[dict] = None, tag: str = '',
         note: str = '') -> dict:
    return {'trade': _trade_key(trade['entry_date'], trade['symbol'], trade['setup']), 'symbol': trade['symbol'],
            'setup': trade['setup'], 'trade_date': trade['entry_date'], 'action': action, 'role': role,
            'order_type': order_type, 'tif': tif, 'qty': float(qty), 'price': price, 'order_ref': order_ref,
            'order_id': None, 'target': target, 'target_order_id': (target or {}).get('order_id'),
            'target_client_id': (target or {}).get('client_id'),
            'oca_group': trade.get('oca_group') or make_oca_group(trade['entry_date'], trade['symbol']),
            'tag': tag, 'note': note, 'result': ''}


def plan_orders(trade: dict, decisions: List[dict], open_orders: List[dict], ledger: OrderLedger,
                client_id: int = ORDER_CLIENT_ID, mode: Optional[str] = None) -> List[dict]:
    """
    Concrete order actions for one trade, idempotent against what is already working at TWS and
    in the ledger (a rerun never places a second trim or exit). A modify / cancel of an order
    another client placed becomes ALERT_OTHER_CLIENT (IBKR refuses it). mode = TRIM_STOP_MODE
    (default: config); split_stop, or a trim_stop already working, adds _plan_trim_block().
    """
    mode = mode or trim_stop_mode()
    if trade.get('status') == 'manual_touched':
        return [_row(trade, REPORT, note="manual_touched: position != ledger, the bot no longer manages it")]
    if trade.get('status') not in OPEN_STATUSES:
        return []
    rows = _ledger_rows(ledger, trade['entry_date'], trade['symbol'])
    stops = _tws_orders(open_orders, trade, STOP_ROLES)
    trims = _tws_orders(open_orders, trade, ('trim',))
    exits_live = _tws_orders(open_orders, trade, ('exit',)) or _live_ledger(rows, 'exit')
    trim_stops = _tws_orders(open_orders, trade, (TRIM_STOP_ROLE,))
    split = mode == 'split_stop' or bool(trim_stops)
    exit_req: Optional[dict] = None           # split: the exit decision, sized per block afterwards
    shares = float(trade.get('shares_open') or 0)
    out: List[dict] = []

    def other_client(target: dict, action: str, note: str) -> Optional[dict]:
        if target.get('client_id') != client_id:
            return _row(trade, ALERT_OTHER_CLIENT, target=target,
                        note=f"{action}: {note}; order {target.get('order_id')} was placed by client "
                             f"{target.get('client_id')}, only that client can change it")
        return None

    exiting = any(d['decision'] in (EXIT_NEXT_OPEN, TIME_STOP_NEXT_OPEN, EXIT_AT_CLOSE) for d in decisions)
    cancelling = set()          # (client, order id) of trims this plan cancels
    for d in decisions:
        kind, note = d['decision'], d.get('note', '')
        if kind == STALE_BARS:
            out.append(_row(trade, ALERT_STALE_BARS, note=f"{note}: no changes, the working stop stays"))
        elif kind == STOP_TO:
            if shares <= EPS:
                continue
            if not stops:
                pending = [r for r in rows if r['role'] in STOP_ROLES and r['state'] in ('intended', 'submitted')]
                if pending:
                    out.append(_row(trade, ALERT_NO_STOP, price=d['price'],
                                    note=f"no bot stop working for {shares:g} shares; {pending[-1]['order_ref']} "
                                         f"is {pending[-1]['state']} but not at TWS: check TWS"))
                    continue
                out.append(_row(trade, PLACE_STOP_FIX, role='stop_fix', order_type='STP', tif='GTC', qty=shares,
                                price=d['price'], order_ref=make_order_ref('stop_fix', trade['entry_date'], trade['symbol']),
                                note=f"no bot stop working for {shares:g} shares: new stop at {d['price']:g}"))
                continue
            target = max(stops, key=_remaining)
            cur = _stop_price(target)
            move = cur is None or d['price'] > cur + PRICE_TICK          # never lowered
            # A stop bigger than the position (partial entry fill, its parent expired) would flip the
            # account short on a stop-out: resize it even when the price stays
            oversized = _remaining(target) > shares + EPS
            if not move and not oversized:
                continue
            price = d['price'] if move else cur
            alert = other_client(target, MODIFY_STOP, f"move stop to {price:g} for {shares:g} shares")
            # qty = the shares still open: after a trim fill OCA type 3 already reduced the stop; a
            # modify at the original size would make it bigger than the position
            out.append(alert or _row(trade, MODIFY_STOP, role='stop', order_type=str(target.get('order_type') or 'STP'),
                                     tif=str(target.get('tif') or 'GTC'), qty=min(_remaining(target), shares) or shares,
                                     price=price, order_ref=str(target.get('order_ref')), target=target,
                                     note=(f"stop {cur if cur is not None else '?'} -> {price:g}"
                                           + (f", {_remaining(target):g} -> {shares:g} shares" if oversized else '')
                                           + f"; {note}").strip('; ')))
        elif kind == TRIM_RESTING:
            if trims or _live_ledger(rows, 'trim') or trade.get('trimmed') or exiting:
                continue
            out.append(_row(trade, PLACE_TRIM, role='trim', order_type='LMT', tif='GTC', qty=d['qty'],
                            price=round(d['price'], 2), order_ref=make_order_ref('trim', trade['entry_date'], trade['symbol']),
                            note="resting trim sale at +TRIM_R (after the fast window)"))
        elif kind == NO_TRIM:
            for t in trims:
                alert = other_client(t, CANCEL_TRIM, "cancel the resting trim")
                if not alert:
                    cancelling.add((t.get('client_id'), t.get('order_id')))
                out.append(alert or _row(trade, CANCEL_TRIM, role='trim', qty=_remaining(t), target=t,
                                         order_ref=str(t.get('order_ref')),
                                         note="no resting trim now (+TRIM_R reached, hold_full or exiting)"))
        elif kind == TRIM_NEXT_OPEN:
            # A trim LMT this plan cancels (it didn't fill although +TRIM_R printed, or filled in part)
            # doesn't block the next-open sale; any other live trim does (no double trim). The LMT is
            # RTH-only, so the cancel lands before the open the MOO sells at.
            if any((t.get('client_id'), t.get('order_id')) not in cancelling for t in trims) or any(
                    (r.get('client_id'), r.get('order_id')) not in cancelling for r in _live_ledger(rows, 'trim')):
                continue
            out.append(_row(trade, PLACE_TRIM_MOO, role='trim', order_type='MKT', tif='OPG', qty=d['qty'],
                            order_ref=make_order_ref('trim', trade['entry_date'], trade['symbol']), tag=d['tag'],
                            note="+TRIM_R printed with no resting trim filled: sell the trim at the next open"))
        elif kind in (EXIT_NEXT_OPEN, TIME_STOP_NEXT_OPEN, EXIT_AT_CLOSE):
            if split:
                exit_req = exit_req or d
                continue
            if exits_live or shares <= EPS:
                continue
            moc = kind == EXIT_AT_CLOSE
            out.append(_row(trade, PLACE_EXIT_MOC if moc else PLACE_EXIT_MOO, role='exit',
                            order_type='MOC' if moc else 'MKT', tif='DAY' if moc else 'OPG', qty=shares,
                            order_ref=make_order_ref('exit', trade['entry_date'], trade['symbol']),
                            tag=d.get('tag', ''), note=note))
    if split:
        out = _plan_trim_block(trade, out, decisions, stops, trim_stops, trims, cancelling, rows, open_orders,
                               exit_req, other_client)
        for r in out:
            r['pool_groups'] = True     # oversell guard: each bot OCA block counted once (split only)
    if not out:
        out.append(_row(trade, HOLD, note=next((d.get('note') for d in decisions if d.get('note')), '')))
    return out


def _role_of(o: dict) -> str:
    return (parse_order_ref(o.get('order_ref')) or ('',))[0]


# Execution order inside one trade's plan: shrink first, so every placement passes the oversell guard
_SPLIT_ORDER = {MODIFY_STOP: 0, PLACE_STOP_FIX: 0, CANCEL_TRIM_STOP: 1, CANCEL_TRIM: 2, PLACE_TRIM_STOP: 3,
                PLACE_TRIM: 4, PLACE_TRIM_MOO: 4, PLACE_EXIT_MOO: 5, PLACE_EXIT_MOC: 5}


def _plan_trim_block(trade: dict, out: List[dict], decisions: List[dict], stops: List[dict],
                     trim_stops: List[dict], trims: List[dict], cancelling: set, rows: List[dict],
                     open_orders: List[dict], exit_req: Optional[dict], other_client) -> List[dict]:
    """
    split_stop: size the two blocks (see TRIM_STOP_MODES) around what plan_orders decided.
      trim block = the trim still working after this plan (kept + new), else the trim_stop already
                   working (trim cancelled: hold_full / exiting), else 0 once the trim has filled
      main block = shares_open - trim block: the main stop is shrunk to it (never grown here;
                   fill_monitor RESIZE_STOP does that), a PLACE_STOP_FIX is sized to it
    New trim orders join the trim group. The exit becomes one order per block, each in its block's
    group, so an exit fill can never leave the other group's orders behind it. If the main stop
    can't be shrunk (another client placed it) no trim block is built and no new trim is placed.
    """
    shares = float(trade.get('shares_open') or 0)
    d0, sym = trade['entry_date'], trade['symbol']
    main_oca = trade.get('oca_group') or make_oca_group(d0, sym)
    trim_oca = make_trim_oca_group(d0, sym)
    new_trims = [r for r in out if r['action'] in (PLACE_TRIM, PLACE_TRIM_MOO)]
    kept = [t for t in trims if (t.get('client_id'), t.get('order_id')) not in cancelling]
    a_qty = sum(_remaining(o) for o in trim_stops)
    trim_after = sum(_remaining(t) for t in kept) + sum(float(r['qty']) for r in new_trims)
    want = trim_after if trim_after > EPS else (0.0 if trade.get('trimmed') else a_qty)
    want = min(want, shares)
    # A trim resting in the main group (placed under shared_oca) keeps the old layout until it is gone;
    # a NEW trim as big as the position needs no second block. A trim block that is all that is left
    # (the main block's stop or exit filled) stays: its trim_stop is then the position's only stop.
    if any(t.get('oca_group') != trim_oca for t in kept):
        want = 0.0
    elif shares - want < 1 - EPS:
        want = min(a_qty, shares)
    stop_to = next((d['price'] for d in decisions if d['decision'] == STOP_TO), None)
    if stop_to is None:
        # Only the nightly run (after the close: no stop can trigger) builds a NEW trim block; the
        # close-window run is in regular hours, where placing the block's stop before the shrink
        # would briefly cover the trim shares twice. Here only existing blocks are sized / exited.
        want = min(want, a_qty)
    price = float(stop_to if stop_to is not None else trade.get('current_stop') or 0)
    b_target = shares - want
    new: List[dict] = []

    # Main block: its stop(s) cover b_target at most
    b_total = sum(_remaining(o) for o in stops)
    b_ok = not any(r['action'] == ALERT_OTHER_CLIENT and r.get('target') in stops for r in out)
    mod_b = next((r for r in out if r['action'] == MODIFY_STOP and r.get('target') in stops), None)
    if stops and b_total < b_target - EPS and b_target >= 1 - EPS:
        # A shrink whose trim block never got built (refused / failed placement): grow it back
        target = mod_b['target'] if mod_b else max(stops, key=_remaining)
        q = _remaining(target) + (b_target - b_total)
        if mod_b is not None:
            mod_b['qty'] = q
            mod_b['note'] += f"; main block back to {q:g} shares"
        else:
            alert = other_client(target, MODIFY_STOP, f"grow the stop to {q:g} shares")
            new.append(alert or _row(trade, MODIFY_STOP, role='stop', order_type=str(target.get('order_type') or 'STP'),
                                     tif=str(target.get('tif') or 'GTC'), qty=q, price=_stop_price(target),
                                     order_ref=str(target.get('order_ref')), target=target,
                                     note=f"stop {_remaining(target):g} -> {q:g} shares: no trim block beside it"))
    elif stops and b_total > b_target + EPS:
        target = mod_b['target'] if mod_b else max(stops, key=_remaining)
        q = _remaining(target) - (b_total - b_target)
        if q < 1 - EPS:
            b_ok = False
            new.append(_row(trade, ALERT_NO_STOP, price=price,
                            note=f"main stops cover {b_total:g}, the main block is {b_target:g}: resize by hand"))
        elif mod_b is not None:
            mod_b['qty'] = min(float(mod_b['qty']), q)
            mod_b['note'] += f"; main block {q:g} shares (trim block {want:g})"
            mod_b['split_shrink'] = trim_oca if want > a_qty + EPS else ''
        else:
            alert = other_client(target, MODIFY_STOP, f"resize the stop to {q:g} shares for the trim block")
            if alert:
                b_ok = False
                new.append(alert)
            else:
                new.append(_row(trade, MODIFY_STOP, role='stop', order_type=str(target.get('order_type') or 'STP'),
                                tif=str(target.get('tif') or 'GTC'), qty=q, price=_stop_price(target),
                                order_ref=str(target.get('order_ref')), target=target,
                                note=f"stop {_remaining(target):g} -> {q:g} shares: the trim block gets its own stop"))
                new[-1]['split_shrink'] = trim_oca if want > a_qty + EPS else ''
    for r in out:
        if r['action'] == PLACE_STOP_FIX:
            r['qty'] = b_target
            r['note'] = f"no main-block stop working for {b_target:g} shares: new stop at {r['price']:g}"
    out = [r for r in out if not (r['action'] == PLACE_STOP_FIX and r['qty'] < 1 - EPS)]

    # Trim block: its stop(s) equal `want`, and follow the stop price up
    if want > a_qty + EPS:
        if b_ok:
            row = _row(trade, PLACE_TRIM_STOP, role=TRIM_STOP_ROLE, order_type='STP', tif='GTC', qty=want - a_qty,
                       price=price, order_ref=make_order_ref(TRIM_STOP_ROLE, d0, sym),
                       note=f"trim block stop for {want - a_qty:g} shares at {price:g}, OCA with the trim")
            row['oca_group'] = trim_oca
            new.append(row)
        else:
            for r in new_trims:          # a trim with no stop beside it would leave its block naked
                out.remove(r)
                new.append(_row(trade, ALERT_NO_STOP, price=price,
                                note=f"{r['action']} not placed: the main stop can't be shrunk for the trim block"))
            new_trims = []
    left = a_qty - want                  # shares of trim_stop to remove
    for o in sorted(trim_stops, key=_remaining):
        rem, cur = _remaining(o), _stop_price(o)
        cut = min(rem, max(left, 0.0))
        left -= cut
        if rem - cut < 1 - EPS:
            alert = other_client(o, CANCEL_TRIM_STOP, "cancel the trim block stop")
            new.append(alert or _row(trade, CANCEL_TRIM_STOP, role=TRIM_STOP_ROLE, qty=rem, target=o,
                                     order_ref=str(o.get('order_ref')),
                                     note="trim block gone (trim filled, or no trim left)"))
            continue
        move = stop_to is not None and (cur is None or price > cur + PRICE_TICK)
        if move or cut > EPS:
            px = price if move else cur
            alert = other_client(o, MODIFY_STOP, f"trim block stop to {px:g} for {rem - cut:g} shares")
            new.append(alert or _row(trade, MODIFY_STOP, role=TRIM_STOP_ROLE, order_type=str(o.get('order_type') or 'STP'),
                                     tif=str(o.get('tif') or 'GTC'), qty=rem - cut, price=px,
                                     order_ref=str(o.get('order_ref')), target=o,
                                     note=f"trim block stop {cur if cur is not None else '?'} -> {px:g}, {rem - cut:g} shares"))
    for r in new_trims:
        r['oca_group'] = trim_oca if want > EPS else main_oca

    # Exits: one per block, in its block's group; a block whose exit is already working is skipped
    if exit_req is not None and shares > EPS:
        live: Dict[str, float] = {}
        for o in _tws_orders(open_orders, trade, ('exit',)):
            live[o.get('oca_group') or main_oca] = live.get(o.get('oca_group') or main_oca, 0.0) + _remaining(o)
        for r in _live_ledger(rows, 'exit'):
            g = r.get('oca_group') or main_oca
            live[g] = max(live.get(g, 0.0), float(r.get('qty') or 0) - float(r.get('filled_qty') or 0), EPS * 2)
        moc = exit_req['decision'] == EXIT_AT_CLOSE
        for g, q in ((main_oca, b_target), (trim_oca, want)):
            if q < 1 - EPS or live.get(g, 0.0) > EPS:
                continue
            row = _row(trade, PLACE_EXIT_MOC if moc else PLACE_EXIT_MOO, role='exit',
                       order_type='MOC' if moc else 'MKT', tif='DAY' if moc else 'OPG', qty=q,
                       order_ref=make_order_ref('exit', d0, sym), tag=exit_req.get('tag', ''),
                       note=(exit_req.get('note', '') + ('; trim block' if g == trim_oca else '')).strip('; '))
            row['oca_group'] = g
            new.append(row)
    merged = out + new
    # The shrink that makes room for a NEW trim block stop goes after that stop (key 3.5): every failure
    # then leaves the trim shares covered twice until the next run, never uncovered
    shrink = next((r for r in merged if r.get('split_shrink')), None)
    for r in merged:
        if r['action'] == PLACE_TRIM_STOP and shrink is not None:
            r['guard_allowance'] = max(_remaining(shrink['target']) - float(shrink['qty']), 0.0)
    return sorted(merged, key=lambda r: 3.5 if r.get('split_shrink') else _SPLIT_ORDER.get(r['action'], 6))


# ─── Acting (submit mode only) ───

def _sell_order(a: dict, account: str, parent_id: int = 0):
    from ibapi.order import Order
    o = Order()
    # whole shares as an int: a float goes over the wire as "100.0"
    o.action, o.orderType, o.totalQuantity = "SELL", a['order_type'], int(round(float(a['qty'])))
    if a['order_type'] in ('STP', 'STP LMT'):
        o.auxPrice = a['price']
    elif a['order_type'] == 'LMT':
        o.lmtPrice = a['price']
    o.tif, o.outsideRth, o.transmit = a['tif'], False, True
    o.ocaGroup, o.ocaType = a['oca_group'], 3
    o.orderRef, o.account = a['order_ref'], account
    if parent_id:
        o.parentId = parent_id
    return o


def _ledger_row_for(ledger: OrderLedger, target: dict) -> Optional[dict]:
    for ev in reversed(ledger.events()):
        if (ev.get('order_ref') == target.get('order_ref') and ev.get('order_id') == target.get('order_id')
                and ev.get('client_id') == target.get('client_id')):
            return ev
    return None


def _ident(ev: dict) -> dict:
    return {k: ev[k] for k in ol.ORDER_ID_FIELDS}


def apply_actions(actions: List[dict], broker: Broker, ledger: OrderLedger, client_id: int = ORDER_CLIENT_ID,
                  ack_timeout: float = 10.0, exit_state: Optional[dict] = None) -> List[dict]:
    """
    Send the order actions (caller holds ledger_lock). New orders: fresh id -> ledger intent ->
    oversell guard (OCA pooled) -> place -> submitted -> ack. Modifies / cancels: the request is
    written onto the order's ledger row first. A TWS failure stops the run; nothing is resent and
    reconcile settles what reached TWS. Sets 'result' on each action.
    """
    if not broker.submit_enabled:
        raise PermissionError(SUBMIT_REFUSED)
    trades = (exit_state or {}).get('trades', {})
    stopped = False
    no_block: Dict[str, str] = {}     # split_stop: trade -> why its trim block was not built
    for a in actions:
        if a['action'] not in ORDER_ACTIONS:
            a['result'] = a['result'] or ('alert_only' if a['action'] in ALERT_ACTIONS else '')
            continue
        if stopped:
            a['result'] = 'not_sent_after_failure'
            continue
        if a['trade'] in no_block and a['action'] in (PLACE_TRIM, PLACE_TRIM_MOO):
            a['result'] = f"skipped: {no_block[a['trade']]}"
            notify(f"{a['symbol']} {a['action']}: {a['result']}")
            continue
        try:
            if a['action'] == MODIFY_STOP and a.get('split_shrink'):
                # split_stop: runs AFTER the trim block's stop is placed (see _plan_trim_block). Shrink
                # only once that stop is confirmed working; otherwise keep the size and move the price only
                target, cur = a['target'], _stop_price(a['target'])
                if a['trade'] in no_block:
                    if cur is not None and float(a['price']) <= cur + PRICE_TICK:
                        a['result'] = f"skipped: {no_block[a['trade']]}"
                        notify(f"{a['symbol']} {a['action']}: {a['result']}")
                        continue
                    a['qty'] = _remaining(target)
                    a['note'] += "; size kept (no trim block)"
                    a['result'] = _modify_stop(a, broker, ledger)
                else:
                    a['result'] = _modify_stop(a, broker, ledger)
                    # the guard must see the shrink before the trim order goes out beside it
                    _await_remaining(broker, target, float(a['qty']), ack_timeout)
            elif a['action'] == MODIFY_STOP:
                a['result'] = _modify_stop(a, broker, ledger)
            elif a['action'] in (CANCEL_TRIM, CANCEL_TRIM_STOP):
                a['result'] = _cancel(a, broker, ledger)
            else:
                a['result'] = _place(a, broker, ledger, client_id, ack_timeout)
                if a['action'] == PLACE_TRIM_STOP and not a['result'].startswith('working'):
                    # not confirmed (refused / rejected / unknown): no shrink, no trim. At worst the
                    # trim shares are covered twice until the next run, never left without a stop
                    no_block[a['trade']] = f"trim block stop not confirmed ({a['result']})"
                if a['action'] in (PLACE_EXIT_MOO, PLACE_EXIT_MOC) and a['result'].startswith(('working', 'submitted')):
                    t = trades.get(a['trade'])
                    if t is not None:
                        t['status'] = 'closing'
        except (ConnectionError, OSError, RuntimeError) as exc:
            logger.error(f"{a['symbol']}: {a['action']} failed ({exc}); stopping, no resend")
            a['result'], stopped = f"failed: {exc}", True
        if not str(a['result']).startswith(('working', 'submitted', 'modified', 'cancel_requested')):
            notify(f"{a['symbol']} {a['action']}: {a['result']}")
    return actions


def _place(a: dict, broker: Broker, ledger: OrderLedger, client_id: int, ack_timeout: float) -> str:
    sym, qty = a['symbol'], a['qty']
    order = _sell_order(a, broker.account)
    oid = max(broker.next_order_ids(1)[0], ledger.highest_order_id() + 1)
    key = LedgerKey(date.fromisoformat(a['trade_date']), sym, a['setup'])
    stored = ledger.record_intent(key, a['role'], a['order_ref'], oid, action='SELL', order_type=a['order_type'],
                                  qty=qty, aux_price=a['price'] if a['order_type'] == 'STP' else None,
                                  lmt_price=a['price'] if a['order_type'] == 'LMT' else None, tif=a['tif'],
                                  oca_group=a['oca_group'], client_id=client_id, account=broker.account,
                                  reason=a['tag'] or a['action'])
    ident = _ident(stored)
    try:
        # guard_allowance: split_stop's trim block stop goes out before the main stop shrinks by the
        # same shares (nightly, after the close: stops can't trigger before the shrink lands)
        assert_no_oversell(broker.state(), sym, qty, broker.account, oca_group=a['oca_group'],
                           pending_buy_qty=float(a.get('guard_allowance') or 0.0),
                           pool_bot_groups=bool(a.get('pool_groups')))
    except ValueError as exc:
        ledger.record({**ident, 'state': 'cancelled', 'reason': 'oversell_guard'})
        return f"refused: {exc}"
    try:
        broker.place(oid, stock_contract(sym), order)
    except Exception:
        raise ConnectionError(f"place {a['order_ref']} failed; left 'intended' for reconcile")
    ledger.record({**ident, 'state': 'submitted'})
    a['order_id'] = oid
    ack, why = broker.wait_ack(oid, timeout=ack_timeout)
    if ack in ('working', 'rejected'):
        ledger.record({**ident, 'state': ack, 'reason': why if ack == 'rejected' else ''})
    return f"working (order {oid})" if ack == 'working' else f"{ack}: {why}"


def _modify_stop(a: dict, broker: Broker, ledger: OrderLedger) -> str:
    t = a['target']
    row = _ledger_row_for(ledger, t)
    old = _stop_price(t)
    order = _sell_order(a, broker.account, parent_id=int(t.get('parent_id') or 0))
    # TWS reads totalQuantity as the whole order (filled + remaining): a partly filled stop keeps
    # its filled part, or the modify would shrink it by that much
    total = int(round(float(t.get('filled') or 0) + float(a['qty'])))
    order.totalQuantity = total
    if t.get('oca_group'):
        order.ocaGroup = t['oca_group']          # a modify keeps the order's own OCA group
    if row is not None:
        ledger.record({**_ident(row), 'state': row['state'], 'aux_price': a['price'], 'qty': total,
                       'reason': f"stop_move_intent {old}->{a['price']:g}"})
    try:
        broker.modify(int(t['order_id']), stock_contract(a['symbol']), order)
    except Exception as exc:
        if row is not None:
            ledger.record({**_ident(row), 'state': row['state'], 'aux_price': row.get('aux_price'),
                           'qty': row.get('qty'), 'reason': f"modify_failed: {exc}"})
        raise ConnectionError(f"modify {t.get('order_ref')} failed: {exc}")
    return f"modified stop {old}->{a['price']:g}"


def _await_remaining(broker: Broker, target: dict, qty: float, timeout: float, poll: float = 0.2) -> None:
    """Wait until TWS shows the modified order at <= qty remaining; RuntimeError (stops the run) if not."""
    import time as _time
    deadline = _time.monotonic() + timeout
    while True:
        for o in broker.state().open_orders:
            if o.get('order_id') == target.get('order_id') and o.get('client_id') == target.get('client_id'):
                if _remaining(o) <= qty + EPS:
                    return
                break
        else:
            return                      # gone from the open orders (filled / cancelled): nothing left to shrink
        if _time.monotonic() >= deadline:
            raise RuntimeError(f"{target.get('order_ref')} still shows more than {qty:g} shares after the modify")
        _time.sleep(poll)


def _cancel(a: dict, broker: Broker, ledger: OrderLedger) -> str:
    t = a['target']
    row = _ledger_row_for(ledger, t)
    if row is not None:
        ledger.record({**_ident(row), 'state': row['state'], 'cancel_requested': True,
                       'reason': f"{a['action'].lower()}: cancel requested"})
    broker.cancel(int(t['order_id']))
    return f"cancel_requested (order {t['order_id']})"


# ─── Runs ───

def _manage_one(trade: dict, broker: Broker, today: date, close_window: bool) -> List[dict]:
    try:
        bars = broker.daily_bars(trade['symbol'], include_forming=close_window)
    except Exception as exc:
        logger.warning(f"{trade['symbol']}: daily bars failed ({exc})")
        bars = pd.DataFrame()
    if close_window:
        return close_window_actions(trade, bars, today)
    return nightly_actions(trade, bars, today)


def run(broker: Broker, ledger: OrderLedger, exit_state: dict, now: datetime, submit: bool,
        close_window: bool = False, client_id: int = ORDER_CLIENT_ID) -> Tuple[List[dict], ol.ReconcileReport]:
    """
    One pass on an open session (caller holds ledger_lock): reconcile -> new trades -> sync fills ->
    manage (nightly) or the time-stop check (close window) -> orders -> (submit) apply -> reconcile.
    Updates exit_state in memory; the caller saves it (submit runs only).
    """
    today = now.date()
    state = broker.state()
    report = ledger.reconcile(state, today)
    alerts = [f"ledger reconcile: {item.get('order_ref')} {item.get('issue') or item.get('reason')}"
              for item in report.orphans + report.mismatches]
    ensure_trades(exit_state, ledger)
    actions: List[dict] = []
    for key, trade in exit_state['trades'].items():
        if trade.get('status') == 'closed':
            continue
        sync_fills(trade, ledger, state)
        if trade['status'] == 'closed':
            logger.info(f"{trade['symbol']}: trade {key} closed")
            continue
        if trade['status'] == 'manual_touched':
            actions.extend(plan_orders(trade, [], state.open_orders, ledger, client_id))
            continue
        decisions = _manage_one(trade, broker, today, close_window)
        actions.extend(plan_orders(trade, decisions, state.open_orders, ledger, client_id)
                       if decisions or not close_window else [])
    if submit and broker.submit_enabled and any(a['action'] in ORDER_ACTIONS for a in actions):
        apply_actions(actions, broker, ledger, client_id=client_id, exit_state=exit_state)
        ledger.reconcile(broker.state(), today)
    else:
        for a in actions:
            if a['action'] in ORDER_ACTIONS:
                a['result'] = 'dry_run'
    alerts += [f"{a['symbol']} {a['action']}{' ' + a['tag'] if a['tag'] else ''}: {a['note']}" for a in actions
               if a['action'] in ALERT_ACTIONS + (REPORT,) or (not submit and a['action'] in ORDER_ACTIONS)]
    notify_all(alerts)
    return actions, report


def write_exit_plan(actions: List[dict], path: Optional[Path] = None, mode: str = "dry-run",
                    now: Optional[datetime] = None) -> Path:
    """Data/orders/exit_plan_<yyyy-mm-dd>.csv; each run appends its block."""
    now = now or now_et()
    path = Path(path) if path else _orders_dir() / f"exit_plan_{now:%Y-%m-%d}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{'run_ts': now.isoformat(timespec='seconds'), 'mode': mode,
             **{k: a.get(k) for k in PLAN_COLUMNS if k not in ('run_ts', 'mode')}} for a in actions]
    df = pd.DataFrame(rows, columns=PLAN_COLUMNS)
    df.to_csv(path, mode='a', header=not path.exists() or path.stat().st_size == 0, index=False)
    return path


def notify(message: str) -> None:
    """Log, plus a macOS notification (notify.py: rate-limited, off with NOTIFY_ENABLED / TRADING_NOTIFY=0)."""
    logger.warning(f"ALERT: {message}")
    notifier.notify("Exit manager", message, 'warning')


def notify_all(messages: List[str]) -> None:
    """Log each alert, plus ONE macOS notification for the batch (8 naked positions = 1 banner, not 8)."""
    for message in messages:
        logger.warning(f"ALERT: {message}")
    notifier.notify_batch("Exit manager", messages, 'warning')


def close_window_bounds(d: date) -> Tuple[dtime, dtime]:
    """CLOSE_WINDOW_ET shifted by the day's early close (15:35-15:48 -> 12:35-12:48 on a 13:00 close)."""
    start_s, end_s = _cfg('CLOSE_WINDOW_ET', ("15:35", "15:48"))
    shift = (datetime.combine(d, market_close_time(d)) - datetime.combine(d, REGULAR_CLOSE))
    start = (datetime.combine(d, dtime.fromisoformat(start_s)) + shift).time()
    end = (datetime.combine(d, dtime.fromisoformat(end_s)) + shift).time()
    return start, end


def time_gate(now: datetime, close_window: bool) -> str:
    """'' if this run may go now, else why not."""
    d = now.date()
    if close_window:
        start, end = close_window_bounds(d)
        if not start <= now.time() <= end:
            return f"--close-window runs only {start:%H:%M}-{end:%H:%M} ET (now {now:%H:%M})"
        return ""
    if now.time() < market_close_time(d):
        return f"the nightly run needs complete bars: after {market_close_time(d):%H:%M} ET (now {now:%H:%M})"
    return ""


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the adopted exit plan on the bot's open trades. Paper only; "
                                                 "orders are sent only with --submit AND config AUTO_SUBMIT_ENABLED.")
    parser.add_argument('--account', default=None, help="Paper account DU... (default config IBKR_ACCOUNT)")
    parser.add_argument('--port', type=int, default=int(getattr(config, 'TWS_PORT', 7497)))
    parser.add_argument('--submit', action='store_true', help="Send orders (only if config AUTO_SUBMIT_ENABLED is True)")
    parser.add_argument('--close-window', action='store_true',
                        help="Time-stop MOC check inside CLOSE_WINDOW_ET (default 15:35-15:48 ET)")
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
    why = time_gate(now, args.close_window)
    if why:
        logger.error(why)
        return 2
    account = (args.account or str(getattr(config, 'IBKR_ACCOUNT', '') or '')).strip().upper() or None
    if not account:
        logger.error("--account DU... (or config IBKR_ACCOUNT) is required")
        return 2
    submit = submission_enabled(args.submit)
    refused = args.submit and not submit
    if refused:
        logger.error(f"--submit refused: {SUBMIT_REFUSED}; running read-only (plan + alerts only)")
    client_id = ORDER_CLIENT_ID if submit else ORDER_READONLY_CLIENT_ID
    from order_broker import open_session
    try:
        with ol.ledger_lock():
            ledger = OrderLedger(source='exit_manager')
            exit_state = load_exit_state()
            try:
                broker = open_session(args.port, account, args.submit, client_id)
            except PermissionError as exc:
                logger.error(f"Paper guard refused the session: {exc}")
                return 2
            try:
                actions, _ = run(broker, ledger, exit_state, now, submit, close_window=args.close_window,
                                 client_id=client_id)
            finally:
                broker.close()
            if submit:
                save_exit_state(exit_state, now=now)
            mode = ('submit' if submit else 'submit-refused' if refused else 'dry-run') + (
                '/close-window' if args.close_window else '/nightly')
            path = write_exit_plan(actions, mode=mode, now=now)
            counts: Dict[str, int] = {}
            for a in actions:
                counts[a['action']] = counts.get(a['action'], 0) + 1
            logger.info(f"Exit plan ({mode}): {', '.join(f'{k} {v}' for k, v in sorted(counts.items())) or 'no open trades'}"
                        f" -> {path}")
    except ol.LedgerLocked as exc:
        logger.error(str(exc))
        return 3
    except Exception as exc:
        logger.exception(f"exit_manager failed: {exc}")
        return 1
    needs_human = [a for a in actions if a['action'] in ALERT_ACTIONS + (REPORT,)
                   or (submit and a['action'] in ORDER_ACTIONS
                       and not str(a['result']).startswith(('working', 'submitted', 'modified', 'cancel_requested')))]
    return 2 if refused or needs_human else 0


if __name__ == "__main__":
    raise SystemExit(notifier.notify_exit_code(main(), "Exit manager"))
