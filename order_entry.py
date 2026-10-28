#!/usr/bin/env python3
"""
Order entry: turns this run's execution-engine BUY rows into bracket orders (paper only).

Each auto entry is a bracket:
  parent  BUY STP LMT at the VCP pivot, limit = pivot * (1 + ENTRY_CHASE_CAP_PCT/100)   [decision a]
          TIF DAY, outsideRth False, transmit False (sent together with the child)
  child   SELL STP at the setup's stop, same quantity, TIF GTC (a DAY stop would expire at the
          first close and leave the position naked overnight), one OCA group per trade so later
          bot SELLs (trim / exits) can't both fill, transmit True
Sizing and the stop-band check use the WORST-case fill, the limit: the stop % is measured from
the limit, the risk budget is divided by that %, and if it falls outside the setup's band the row is
skipped for manual review (lowering the limit to fit would be a rule change).

Submission is DISABLED by default. Orders are transmitted only if config AUTO_SUBMIT_ENABLED is
the bool True AND --submit is passed (order_broker.submission_enabled); every other run is a dry
run that writes the planned orders to Data/orders/plan_<yyyy-mm-dd>.csv.

Modes:
  --plan-only   never connects to TWS; uses the account snapshot JSON (what the pipeline runs)
  (default)     connects read-only (client ORDER_READONLY_CLIENT_ID, 17) for fresh state, reconciles
                the ledger, writes the plan
  --submit      client ORDER_CLIENT_ID (16); transmits only with the config switch on. Without it:
                logs SUBMIT_REFUSED, writes the plan read-only and exits 2

Halts (submit mode): fewer than 5 sessions of equity history since the last
reset means the loss breakers are blind -> no entries. An unprotected position halts too.
v1 auto-enters only Entry_Type PIVOT_BREAKOUT rows with a VCP_Pivot (other BUY rows are listed as
NOT_AUTO_NO_PIVOT), at most MAX_AUTO_ENTRIES_PER_DAY (1) per session, and only before
ENTRY_SUBMIT_CUTOFF_ET (09:29).

Exit codes: 0 ok (also nothing to do / holiday) · 1 error · 2 refused by a guard or halted ·
3 another order run holds the ledger lock.

Usage:
  python order_entry.py --decisions Scans/scan_2026-11-02_0920_execution_decisions.csv \\
      --account-snapshot Scans/paper_runs/account_snapshot_2026-11-02_0920.json --plan-only
  python order_entry.py --decisions <file> --account DUXXXXXXX                # read-only plan
  python order_entry.py --decisions <file> --account DUXXXXXXX --submit       # paper, both gates
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time as dt_time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

import config
import notify as notifier
import order_ledger as ol
import risk_manager
import strategy_params as sp
from execution_engine import BUY, ExecutionPolicy, _to_float, _to_int
from ibkr_connector import ALERT_ORDER_REF
from market_time import calendar_covers, is_trading_day, now_et
from order_broker import (ORDER_CLIENT_ID, ORDER_READONLY_CLIENT_ID, SUBMIT_REFUSED, Broker, BrokerState,
                          assert_no_oversell, stock_contract, submission_enabled)
from order_ledger import LedgerKey, OrderLedger

logger = logging.getLogger(__name__)

PIVOT_ENTRY_TYPE = "PIVOT_BREAKOUT"

# Plan statuses. PLANNED = passes every check (a dry run stops here).
PLANNED = "PLANNED"
SUBMITTED = "SUBMITTED"
WORKING = "WORKING"
REJECTED = "REJECTED"
HALTED_NOT_TRADING_DAY = "HALTED_NOT_TRADING_DAY"
HALTED_AFTER_CUTOFF = "HALTED_AFTER_CUTOFF"
HALTED_NO_BREAKER_HISTORY = "HALTED_NO_BREAKER_HISTORY"
HALTED_UNPROTECTED = "HALTED_UNPROTECTED"
HALTED_NO_SNAPSHOT = "HALTED_NO_SNAPSHOT"
NOT_AUTO_NO_PIVOT = "NOT_AUTO_NO_PIVOT"
SKIP_PIVOT_EXCEEDED = "SKIP_PIVOT_EXCEEDED"
SKIP_BAD_STOP = "SKIP_BAD_STOP"
SKIP_STOP_BAND = "SKIP_STOP_BAND"                    # tighter than the band's minimum at the pivot
SKIP_STOP_BAND_AT_LIMIT = "SKIP_STOP_BAND_AT_LIMIT"  # wider than the band's maximum at the limit
SKIP_ZERO_SHARES = "SKIP_ZERO_SHARES"
SKIP_MISSING_FIELDS = "SKIP_MISSING_FIELDS"
SKIP_HELD = "SKIP_HELD"
SKIP_IN_LEDGER = "SKIP_IN_LEDGER"
SKIP_OPEN_BUY_ORDER = "SKIP_OPEN_BUY_ORDER"
SKIP_OPEN_RISK_CAP = "SKIP_OPEN_RISK_CAP"
SKIP_GROUP_RISK_CAP = "SKIP_GROUP_RISK_CAP"
SKIP_MAX_POSITIONS = "SKIP_MAX_POSITIONS"
SKIP_DAILY_ENTRY_CAP = "SKIP_DAILY_ENTRY_CAP"
SKIP_OVERSELL_GUARD = "SKIP_OVERSELL_GUARD"
FAILED_PARENT = "FAILED_PARENT"
FAILED_CHILD = "FAILED_CHILD"
FAILED_CHILD_ACK = "FAILED_CHILD_ACK"   # TWS did not confirm the child stop: parent cancelled
NOT_SENT = "NOT_SENT_AFTER_FAILURE"
HALT_STATUSES = {HALTED_NOT_TRADING_DAY, HALTED_AFTER_CUTOFF, HALTED_NO_BREAKER_HISTORY,
                 HALTED_UNPROTECTED, HALTED_NO_SNAPSHOT}
# A run with any of these exits 2 (a human must look at TWS), like a halt
FAILURE_STATUSES = {FAILED_PARENT, FAILED_CHILD, FAILED_CHILD_ACK, REJECTED}
NOTE_ABOVE_PIVOT = "ABOVE_PIVOT_TRIGGERS_AT_OPEN"

PLAN_COLUMNS = ['run_ts', 'run_id', 'mode', 'symbol', 'setup', 'tier', 'status', 'reason', 'pivot',
                'aux_price', 'lmt_price', 'stop_price', 'last_close', 'stop_pct_at_pivot',
                'stop_pct_at_limit', 'stop_band', 'risk_pct_budget', 'position_cap_pct', 'position_pct',
                'risk_pct_at_limit', 'net_liq', 'shares', 'group', 'parent_ref', 'child_ref', 'oca_group',
                'parent_id', 'child_id']


def _cfg(key: str, default):
    """Config read at call time, so tests (and a changed config.py) take effect without a reload."""
    return getattr(config, key, default)


def _orders_dir() -> Path:
    return ol.ORDERS_DIR


@dataclass
class EntryPlan:
    symbol: str
    setup: str
    tier: str = ""
    status: str = PLANNED
    reason: str = ""
    pivot: float = 0.0
    aux_price: float = 0.0
    lmt_price: float = 0.0
    stop_price: float = 0.0
    last_close: Optional[float] = None
    stop_pct_at_pivot: Optional[float] = None
    stop_pct_at_limit: Optional[float] = None
    stop_band: Tuple[float, float] = (0.0, 0.0)
    risk_pct_budget: float = 0.0
    position_cap_pct: float = 0.0
    position_pct: float = 0.0
    risk_pct_at_limit: float = 0.0
    net_liq: float = 0.0
    shares: int = 0
    group: Optional[int] = None
    parent_ref: str = ""
    child_ref: str = ""
    oca_group: str = ""
    parent_id: Optional[int] = None
    child_id: Optional[int] = None
    notes: List[str] = field(default_factory=list)

    def fail(self, status: str, reason: str) -> "EntryPlan":
        """First failing check wins: later checks never overwrite it."""
        if self.status == PLANNED:
            self.status, self.reason = status, reason
        return self

    def note(self, text: str) -> None:
        self.notes.append(text)

    @property
    def ok(self) -> bool:
        return self.status == PLANNED


# ─── Inputs ───

def load_buy_rows(path: Path, now: Optional[datetime] = None,
                  max_age_min: Optional[float] = None) -> pd.DataFrame:
    """
    BUY rows of one execution-decisions file. Refuses the file (ValueError) if Scan_Timestamp is
    missing, not today, or older than ORDER_MAX_SCAN_AGE_MIN: yesterday's plan never trades today.
    Scan_Timestamp is the scanning Mac's local clock, so `now` is local time (as in the engine).
    """
    now = now or datetime.now()
    max_age_min = float(max_age_min if max_age_min is not None else _cfg('ORDER_MAX_SCAN_AGE_MIN', 120))
    df = pd.read_csv(path)
    if 'Decision' not in df.columns:
        raise ValueError(f"{path} has no Decision column: not an execution-decisions file")
    if 'Scan_Timestamp' not in df.columns:
        raise ValueError(f"{path} has no Scan_Timestamp: its age can't be checked")
    buys = df[df['Decision'].astype(str).str.upper() == BUY].copy()
    stamps = pd.to_datetime(df['Scan_Timestamp'], errors='coerce')
    if stamps.isna().all():
        raise ValueError(f"{path}: Scan_Timestamp missing or unreadable")
    if len(buys) and pd.to_datetime(buys['Scan_Timestamp'], errors='coerce').isna().any():
        raise ValueError(f"{path}: a BUY row has no Scan_Timestamp")
    newest, oldest = stamps.max(), stamps.min()
    if oldest.date() != now.date() or newest.date() != now.date():
        raise ValueError(f"{path}: scan is from {oldest:%Y-%m-%d}, not today {now:%Y-%m-%d}")
    age = (now - oldest).total_seconds() / 60
    if age > max_age_min:
        raise ValueError(f"{path}: scan is {age:.0f} min old (max {max_age_min:g})")
    if age < -5:
        raise ValueError(f"{path}: Scan_Timestamp is {-age:.0f} min in the future; clock or file is wrong")
    return buys.reset_index(drop=True)


def entry_prices(pivot: float, chase_cap_pct: Optional[float] = None) -> Tuple[float, float]:
    """
    (aux, lmt) for the buy stop-limit: aux = pivot to the cent; lmt = aux * (1 + cap/100) floored to
    the cent, so the limit never exceeds the cap (e.g. a 3% cap: 33.33 -> 34.32, never above the cap).
    """
    cap = float(chase_cap_pct if chase_cap_pct is not None else _cfg('ENTRY_CHASE_CAP_PCT', sp.ENTRY_CHASE_CAP_PCT))
    aux = round(float(pivot), 2)
    # +1e-9: e.g. 150 * 1.02 is 152.99999999999997 in floats; without it the floor would lose a cent
    lmt = math.floor(aux * (1 + cap / 100) * 100 + 1e-9) / 100
    return aux, round(lmt, 2)


def _row_get(row: pd.Series, col: str):
    return row[col] if col in row.index and not pd.isna(row[col]) else None


def size_at_worst_fill(row: pd.Series, net_liq: float, policy: ExecutionPolicy,
                       chase_cap_pct: Optional[float] = None) -> EntryPlan:
    """
    Plan one pivot row at its worst-case fill (the limit):
      stop % at the limit -> band check (minimum at the pivot, maximum at the limit)
      -> risk_manager.size_position(Risk_Pct_Budget, stop % at limit, Position_Cap_Pct)
      -> shares = floor(net_liq * position % / 100 / limit); under 1 share = skip.
    The pivot band is the engine's PIVOT_BREAKOUT band (policy.min_stop_pct .. max_stop_pct_pivot).
    """
    symbol = str(_row_get(row, 'Symbol') or '').strip().upper()
    setup = str(_row_get(row, 'Entry_Type') or '').strip()
    plan = EntryPlan(symbol=symbol, setup=setup, tier=str(_row_get(row, 'Tier') or ''),
                     net_liq=float(net_liq or 0), group=_to_int(_row_get(row, 'IBD_Group_Rank')))
    pivot = _to_float(_row_get(row, 'VCP_Pivot')) or 0.0
    plan.pivot = pivot
    plan.last_close = _to_float(_row_get(row, 'Entry_Price'))
    if setup != PIVOT_ENTRY_TYPE or pivot <= 0:
        return plan.fail(NOT_AUTO_NO_PIVOT, f"{setup or 'blank'} entry: v1 auto-enters VCP pivot breakouts only")
    plan.aux_price, plan.lmt_price = entry_prices(pivot, chase_cap_pct)
    plan.stop_band = (policy.min_stop_pct, policy.max_stop_pct_pivot)

    # Freshness: a close already above the limit means the breakout ran away without us
    if plan.last_close is not None and plan.last_close > plan.lmt_price:
        return plan.fail(SKIP_PIVOT_EXCEEDED, f"last close {plan.last_close:.2f} > limit {plan.lmt_price:.2f}")
    if plan.last_close is not None and plan.last_close >= plan.aux_price:
        plan.note(NOTE_ABOVE_PIVOT)

    stop = _to_float(_row_get(row, 'Stop_Price'))
    if stop is None and plan.last_close and _to_float(_row_get(row, 'Stop_Pct')):
        stop = plan.last_close * (1 - float(row['Stop_Pct']) / 100)
    if stop is None or stop <= 0:
        return plan.fail(SKIP_BAD_STOP, "no stop price")
    plan.stop_price = round(stop, 2)
    if plan.stop_price >= plan.aux_price:
        return plan.fail(SKIP_BAD_STOP, f"stop {plan.stop_price:.2f} >= trigger {plan.aux_price:.2f}")
    plan.stop_pct_at_pivot = round((plan.aux_price - plan.stop_price) / plan.aux_price * 100, 4)
    plan.stop_pct_at_limit = round((plan.lmt_price - plan.stop_price) / plan.lmt_price * 100, 4)
    lo, hi = plan.stop_band
    if plan.stop_pct_at_pivot < lo:
        return plan.fail(SKIP_STOP_BAND, f"stop {plan.stop_pct_at_pivot:.2f}% at the pivot < {lo:g}% minimum")
    if plan.stop_pct_at_limit > hi:
        return plan.fail(SKIP_STOP_BAND_AT_LIMIT,
                         f"stop {plan.stop_pct_at_limit:.2f}% at the limit {plan.lmt_price:.2f} > {hi:g}% "
                         f"max for {setup} (worst-case fill); manual review")

    budget = _to_float(_row_get(row, 'Risk_Pct_Budget'))
    cap = _to_float(_row_get(row, 'Position_Cap_Pct'))
    if budget is None or cap is None:
        return plan.fail(SKIP_MISSING_FIELDS, "decisions file lacks Risk_Pct_Budget / Position_Cap_Pct "
                                              "(rerun the engine)")
    plan.risk_pct_budget, plan.position_cap_pct = budget, cap
    if not plan.net_liq or plan.net_liq <= 0:
        return plan.fail(HALTED_NO_SNAPSHOT, "net liquidation unknown: can't size")
    position_pct, _ = risk_manager.size_position(budget, plan.stop_pct_at_limit, cap)
    plan.shares = int(math.floor(plan.net_liq * position_pct / 100 / plan.lmt_price + 1e-9))
    if plan.shares < 1:
        return plan.fail(SKIP_ZERO_SHARES, f"{position_pct:.2f}% of ${plan.net_liq:,.0f} is under 1 share "
                                           f"at {plan.lmt_price:.2f}")
    # What the whole-share order actually carries at the worst fill
    plan.position_pct = round(plan.shares * plan.lmt_price / plan.net_liq * 100, 4)
    plan.risk_pct_at_limit = round(plan.shares * (plan.lmt_price - plan.stop_price) / plan.net_liq * 100, 4)
    return plan


def _pending_bot_entries(ledger: OrderLedger) -> List[dict]:
    """Bot entry orders still live (not terminal, nothing filled yet): their planned risk counts."""
    cur = ledger.current()
    if cur.empty:
        return []
    live = cur[(cur['role'] == 'entry') & (~cur['state'].isin(ol.TERMINAL))]
    return [r for r in live.to_dict('records') if float(r.get('filled_qty') or 0) == 0]


def build_plan(rows: pd.DataFrame, snapshot: Optional[dict], ledger: OrderLedger, open_orders: List[dict],
               policy: ExecutionPolicy, now: datetime, submit: bool) -> List[EntryPlan]:
    """
    One EntryPlan per BUY row, checks in priority order (first failure sets the status; caps count
    only the rows planned before it). `now` is ET. snapshot is an account_snapshot dict (fresh from
    TWS, or the pipeline's JSON); None halts everything (caps can't be checked blind).
    """
    today = now.date()
    net_liq = float((snapshot or {}).get('net_liquidation') or 0)
    plans = [size_at_worst_fill(row, net_liq, policy) for _, row in rows.iterrows()]

    # 1-3: run-wide halts (applied to every row, overriding row-level results)
    halt: Optional[Tuple[str, str]] = None
    cutoff = dt_time.fromisoformat(str(_cfg('ENTRY_SUBMIT_CUTOFF_ET', '09:29')))
    min_hist = policy.min_breaker_history_sessions
    sessions = (snapshot or {}).get('breaker_history_sessions')
    if not is_trading_day(today):
        halt = (HALTED_NOT_TRADING_DAY, f"{today} is not a trading day")
    elif submit and now.time() >= cutoff:
        halt = (HALTED_AFTER_CUTOFF, f"{now:%H:%M} ET is past the {cutoff:%H:%M} entry cutoff")
    elif snapshot is None:
        halt = (HALTED_NO_SNAPSHOT, "no fresh account snapshot: open positions and caps unknown")
    elif submit and (sessions is None or int(sessions) < min_hist):
        have = 'unknown' if sessions is None else int(sessions)
        halt = (HALTED_NO_BREAKER_HISTORY,
                f"Circuit breakers have {have} of {min_hist} sessions since the last reset: no auto entries")
    elif snapshot.get('unprotected_symbols'):
        halt = (HALTED_UNPROTECTED, f"position(s) without a full stop: {', '.join(snapshot['unprotected_symbols'])}")
    if halt:
        for p in plans:
            p.status, p.reason = halt
        return plans
    if sessions is None or int(sessions) < min_hist:
        for p in plans:   # dry run: plan anyway, but say a submit run would halt
            p.note(f"submit would halt: breakers have {'unknown' if sessions is None else int(sessions)} "
                   f"of {min_hist} sessions")

    held = {str(s).upper() for s in snapshot.get('held_symbols', [])}
    pending = _pending_bot_entries(ledger)
    pending_syms = {str(r['symbol']).upper() for r in pending} | {
        str(s).upper() for s in snapshot.get('pending_entry_orders', [])}
    pending_risk = sum(float((r.get('plan') or {}).get('risk_pct') or 0) for r in pending)
    open_risk = float(snapshot.get('total_open_risk_pct') or 0) + pending_risk
    groups: Dict[str, float] = {str(k): float(v) for k, v in (snapshot.get('group_open_risk_pct') or {}).items()}
    for r in pending:
        g = (r.get('plan') or {}).get('group')
        if g is not None:
            groups[str(int(g))] = groups.get(str(int(g)), 0.0) + float((r.get('plan') or {}).get('risk_pct') or 0)
    positions = len(held | pending_syms)
    entries_today = ledger.entries_count(today)
    max_auto = int(_cfg('MAX_AUTO_ENTRIES_PER_DAY', 1))
    planned = 0

    for p in plans:
        if not p.ok:
            continue
        # 7: duplicates
        if p.symbol in held:
            p.fail(SKIP_HELD, "already holding this symbol")
            continue
        if ledger.has_entry(today, p.symbol):
            p.fail(SKIP_IN_LEDGER, "ledger already has an entry for this symbol today (no same-day resend)")
            continue
        buys = [o for o in open_orders if str(o.get('symbol', '')).upper() == p.symbol
                and str(o.get('action', '')).upper() == 'BUY']
        if any(str(o.get('order_ref') or '') != ALERT_ORDER_REF for o in buys):
            p.fail(SKIP_OPEN_BUY_ORDER, "an open BUY order for this symbol already exists (manual or bot)")
            continue
        if buys:
            p.note("staged ALERT order(s) for this symbol ignored; cancel them by hand if not wanted")
        # 8: caps, cumulative over this run's planned rows
        if open_risk + p.risk_pct_at_limit > policy.max_total_open_risk_pct + 1e-9:
            p.fail(SKIP_OPEN_RISK_CAP, f"open risk {open_risk:.2f}% + {p.risk_pct_at_limit:.2f}% > "
                                       f"{policy.max_total_open_risk_pct:g}%")
            continue
        gkey = str(p.group) if p.group is not None else None
        if gkey is not None and groups.get(gkey, 0.0) + p.risk_pct_at_limit > policy.max_group_open_risk_pct + 1e-9:
            p.fail(SKIP_GROUP_RISK_CAP, f"group #{gkey} risk {groups.get(gkey, 0.0):.2f}% + "
                                        f"{p.risk_pct_at_limit:.2f}% > {policy.max_group_open_risk_pct:g}%")
            continue
        if positions + 1 > policy.max_open_positions:
            p.fail(SKIP_MAX_POSITIONS, f"{positions} held/pending, max {policy.max_open_positions}")
            continue
        if entries_today + planned >= max_auto:
            p.fail(SKIP_DAILY_ENTRY_CAP, f"auto entries today {entries_today + planned} of {max_auto}")
            continue
        open_risk += p.risk_pct_at_limit
        if gkey is not None:
            groups[gkey] = groups.get(gkey, 0.0) + p.risk_pct_at_limit
        positions += 1
        planned += 1
        p.parent_ref = ol.make_order_ref('entry', today, p.symbol)
        p.child_ref = ol.make_order_ref('stop', today, p.symbol)
        p.oca_group = ol.make_oca_group(today, p.symbol)
        p.reason = "meets entry rules"
    return plans


# ─── Orders ───

def build_bracket(plan: EntryPlan, parent_id: int, account: str):
    """(parent, child) ibapi Orders for one plan. The child's parentId ties it to the parent."""
    from ibapi.order import Order
    parent = Order()
    parent.orderId = parent_id
    parent.action, parent.orderType, parent.totalQuantity = "BUY", "STP LMT", plan.shares
    parent.auxPrice, parent.lmtPrice = plan.aux_price, plan.lmt_price
    parent.tif, parent.outsideRth, parent.transmit = "DAY", False, False
    parent.orderRef, parent.account = plan.parent_ref, account

    child = Order()
    child.orderId = parent_id + 1
    child.action, child.orderType, child.totalQuantity = "SELL", "STP", plan.shares
    child.auxPrice, child.parentId = plan.stop_price, parent_id
    # GTC: the stop must outlive the entry day (F3). ocaType 3 = reduce the others, no block (F4)
    child.tif, child.outsideRth, child.transmit = "GTC", False, True
    child.ocaGroup, child.ocaType = plan.oca_group, 3
    child.orderRef, child.account = plan.child_ref, account
    return parent, child


def _plan_fields(plan: EntryPlan) -> dict:
    return {'pivot': plan.pivot, 'stop': plan.stop_price, 'stop_pct_at_limit': plan.stop_pct_at_limit,
            'risk_pct': plan.risk_pct_at_limit, 'position_pct': plan.position_pct, 'net_liq': plan.net_liq,
            'group': plan.group, 'tier': plan.tier}


def submit_plans(plans: List[EntryPlan], broker: Broker, ledger: OrderLedger, today: date,
                 client_id: int = ORDER_CLIENT_ID, ack_timeout: float = 10.0) -> List[EntryPlan]:
    """
    Send each PLANNED bracket (caller holds ledger_lock). Per plan: fresh ids (above anything the
    ledger has seen) -> both intents fsynced -> oversell guard -> parent (transmit False) -> child
    (transmit True) -> 'submitted' -> wait for the ack. A TWS failure stops the run; nothing is ever
    resent: the next reconcile settles what reached TWS.
    """
    if not broker.submit_enabled:
        raise PermissionError(SUBMIT_REFUSED)
    stopped = False
    for plan in plans:
        if not plan.ok:
            continue
        if stopped:
            plan.status, plan.reason = NOT_SENT, "an earlier order failed this run"
            continue
        key = LedgerKey(today, plan.symbol, plan.setup)
        ids = broker.next_order_ids(2)
        parent_id = max(ids[0], ledger.highest_order_id() + 1)
        plan.parent_id, plan.child_id = parent_id, parent_id + 1
        parent, child = build_bracket(plan, parent_id, broker.account)
        common = dict(client_id=client_id, account=broker.account, qty=plan.shares, plan=_plan_fields(plan))
        try:
            ledger.record_intent(key, 'entry', plan.parent_ref, parent_id, action='BUY', order_type='STP LMT',
                                 aux_price=plan.aux_price, lmt_price=plan.lmt_price, tif='DAY', **common)
        except ol.DuplicateEntry as exc:
            plan.status, plan.reason = SKIP_IN_LEDGER, str(exc)
            continue
        ledger.record_intent(key, 'stop', plan.child_ref, plan.child_id, action='SELL', order_type='STP',
                             aux_price=plan.stop_price, tif='GTC', parent_order_id=parent_id,
                             oca_group=plan.oca_group, **common)
        ident = {**key.as_fields(), 'client_id': client_id}
        try:
            assert_no_oversell(broker.state(), plan.symbol, plan.shares, broker.account,
                               pending_buy_qty=plan.shares)
        except ValueError as exc:
            for role, ref, oid in (('entry', plan.parent_ref, parent_id), ('stop', plan.child_ref, plan.child_id)):
                ledger.record({**ident, 'role': role, 'order_ref': ref, 'order_id': oid, 'state': 'cancelled',
                               'reason': 'oversell_guard'})
            plan.status, plan.reason = SKIP_OVERSELL_GUARD, str(exc)
            continue

        contract = stock_contract(plan.symbol)
        try:
            broker.place(parent_id, contract, parent)
        except Exception as exc:
            # Unknown whether TWS got it: leave 'intended'; reconcile marks it seen or never_reached_tws
            logger.error(f"{plan.symbol}: parent order failed ({exc}); stopping, no resend")
            plan.status, plan.reason, stopped = FAILED_PARENT, f"parent place failed: {exc}", True
            continue
        try:
            broker.place(plan.child_id, contract, child)
        except Exception as exc:
            # The parent went to the socket staged (transmit False) without its stop. Mark it
            # 'submitted', not 'intended': TWS may not list an untransmitted order in its open orders,
            # and reconcile must not call it never_reached_tws while it sits in the TWS window. Try to
            # cancel it now; either way a human must check TWS (a manual transmit = entry, no stop).
            ledger.record({**ident, 'role': 'entry', 'order_ref': plan.parent_ref, 'order_id': parent_id,
                           'state': 'submitted', 'reason': f"staged_untransmitted: child failed ({exc})"})
            cancelled = _cancel_quietly(broker, parent_id)
            logger.error(f"{plan.symbol}: child stop failed ({exc}); parent {parent_id} staged untransmitted, "
                         f"cancel {'requested' if cancelled else 'FAILED'}; stopping")
            notify(f"{plan.symbol}: bracket stop failed, parent order {parent_id} may sit untransmitted in TWS "
                   f"with no stop. Check TWS and cancel it by hand; never transmit it")
            plan.status, plan.reason, stopped = FAILED_CHILD, f"child stop place failed: {exc}", True
            continue
        for role, ref, oid in (('entry', plan.parent_ref, parent_id), ('stop', plan.child_ref, plan.child_id)):
            ledger.record({**ident, 'role': role, 'order_ref': ref, 'order_id': oid, 'state': 'submitted'})
        plan.status, plan.reason = SUBMITTED, "bracket transmitted"
        ack, why = broker.wait_ack(parent_id, timeout=ack_timeout)
        if ack == 'rejected':
            for role, ref, oid in (('entry', plan.parent_ref, parent_id), ('stop', plan.child_ref, plan.child_id)):
                ledger.record({**ident, 'role': role, 'order_ref': ref, 'order_id': oid, 'state': 'rejected',
                               'reason': why})
            plan.status, plan.reason = REJECTED, why
            notify(f"{plan.symbol} entry rejected: {why}")
            continue
        # The parent alone proves nothing: TWS can accept it and reject the child (bad stop tick, an
        # unsupported attribute), and the entry would then fill with no stop. Both legs must ack.
        c_ack, c_why = broker.wait_ack(plan.child_id, timeout=ack_timeout)
        if c_ack != 'working':
            if c_ack == 'rejected':
                ledger.record({**ident, 'role': 'stop', 'order_ref': plan.child_ref, 'order_id': plan.child_id,
                               'state': 'rejected', 'reason': c_why})
            cancelled = _cancel_quietly(broker, parent_id)
            ledger.record({**ident, 'role': 'entry', 'order_ref': plan.parent_ref, 'order_id': parent_id,
                           'state': 'working' if ack == 'working' else 'submitted', 'cancel_requested': cancelled,
                           'reason': f"child stop {c_ack}: {c_why}; parent cancel "
                                     f"{'requested' if cancelled else 'FAILED'}"})
            logger.error(f"{plan.symbol}: child stop {plan.child_id} {c_ack} ({c_why}); parent {parent_id} "
                         f"cancel {'requested' if cancelled else 'FAILED'}; stopping")
            notify(f"{plan.symbol}: entry stop not confirmed by TWS ({c_ack}: {c_why}); parent {parent_id} "
                   + ("cancel requested, check TWS" if cancelled else "cancel FAILED: cancel it in TWS now"))
            plan.status, stopped = FAILED_CHILD_ACK, True
            plan.reason = f"child stop {c_ack}: {c_why}; parent cancel {'requested' if cancelled else 'failed'}"
            continue
        if ack == 'working':
            for role, ref, oid in (('entry', plan.parent_ref, parent_id), ('stop', plan.child_ref, plan.child_id)):
                ledger.record({**ident, 'role': role, 'order_ref': ref, 'order_id': oid, 'state': 'working'})
            plan.status, plan.reason = WORKING, "bracket working at TWS"
        else:
            # Child acked, parent status not seen yet: reconcile settles the parent
            ledger.record({**ident, 'role': 'stop', 'order_ref': plan.child_ref, 'order_id': plan.child_id,
                           'state': 'working'})
            plan.reason = f"bracket transmitted; parent ack {ack}: {why}"
    return plans


def _cancel_quietly(broker: Broker, order_id: int) -> bool:
    """Cancel request for a bot order; False (logged) if it could not be sent."""
    try:
        broker.cancel(int(order_id))
        return True
    except Exception as exc:
        logger.error(f"cancel of order {order_id} failed: {exc}")
        return False


def cancel_incomplete_brackets(report: ol.ReconcileReport, broker: Broker, client_id: int) -> List[dict]:
    """
    Submit mode only: cancel this client's working BOT-ENTRY parents that have no child stop,
    transmitted or not. The transmit flag can't be trusted here: ibapi's openOrder decoder never sets
    Order.transmit, so it always reads as the default True and a 'staged only' rule never fired. An
    entry without its stop must not stay live either way. Another client's parent is only alerted
    (IBKR lets only the placing client cancel).
    """
    done = []
    if not broker.submit_enabled:
        return done
    for b in report.incomplete_brackets:
        if b.get('client_id') == client_id and b.get('order_id') is not None:
            logger.warning(f"Cancelling entry parent {b['order_ref']} (order {b['order_id']}): no child stop")
            broker.cancel(int(b['order_id']))
            done.append(b)
    return done


# ─── Output ───

def write_plan(plans: List[EntryPlan], path: Optional[Path] = None, run_id: str = "", mode: str = "dry-run",
               now: Optional[datetime] = None) -> Path:
    """Append this run's block to Data/orders/plan_<yyyy-mm-dd>.csv (one row per BUY row)."""
    now = now or now_et()
    path = Path(path) if path else _orders_dir() / f"plan_{now:%Y-%m-%d}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in plans:
        d = asdict(p)
        notes = d.pop('notes')
        d['reason'] = "; ".join([d['reason']] + notes) if d['reason'] else "; ".join(notes)
        d['stop_band'] = f"{p.stop_band[0]:g}-{p.stop_band[1]:g}"
        rows.append({'run_ts': now.isoformat(timespec='seconds'), 'run_id': run_id, 'mode': mode, **d})
    df = pd.DataFrame(rows, columns=PLAN_COLUMNS)
    df.to_csv(path, mode='a', header=not path.exists() or path.stat().st_size == 0, index=False)
    return path


def notify(message: str) -> None:
    """Log, plus a macOS notification (notify.py: rate-limited, off with NOTIFY_ENABLED / TRADING_NOTIFY=0)."""
    logger.warning(f"ALERT: {message}")
    notifier.notify("Order entry", message, 'warning')


def _snapshot_from_state(state: BrokerState, account: str, now: datetime) -> dict:
    """Fresh account snapshot from this run's TWS state (same rules as account_snapshot.py)."""
    import account_snapshot as acct
    try:
        group_of = acct._ibd_group_lookup()
    except Exception as exc:
        logger.warning(f"IBD group lookup unavailable ({exc}); group caps use the decisions file only")
        group_of = lambda s: None  # noqa: E731
    local_now = datetime.now()
    return acct.build_snapshot({'positions': state.positions, 'open_orders': state.open_orders,
                                'account_values': state.account_values}, account=account, group_of=group_of,
                               now=local_now, equity_history=acct.read_equity_log(account),
                               adjustments=acct.read_adjustments(account))


def _summary(plans: List[EntryPlan]) -> str:
    counts: Dict[str, int] = {}
    for p in plans:
        counts[p.status] = counts.get(p.status, 0) + 1
    return ", ".join(f"{k} {v}" for k, v in sorted(counts.items())) or "no BUY rows"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan (and, with both gates, submit) bracket entries "
                                                 "from this run's execution decisions. Paper only.")
    parser.add_argument('--decisions', required=True, help="This run's *_execution_decisions.csv (explicit path)")
    parser.add_argument('--account-snapshot', help="account_snapshot JSON (required with --plan-only)")
    parser.add_argument('--account', default=None, help="Paper account DU... (default config IBKR_ACCOUNT)")
    parser.add_argument('--port', type=int, default=int(_cfg('TWS_PORT', 7497)))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--submit', action='store_true',
                      help="Transmit (only if config AUTO_SUBMIT_ENABLED is True)")
    mode.add_argument('--plan-only', action='store_true', help="Never connect; plan from the snapshot JSON")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s", datefmt="%H:%M:%S")
    args = parse_args(argv)
    now = now_et()
    today = now.date()
    if not calendar_covers(today):
        logger.error(f"{today} is past the NYSE holiday table in market_time.py: add that year's holidays and early closes first")
        return 2
    if not is_trading_day(today):
        logger.info(f"{today} is not a trading day: nothing to do")
        return 0
    try:
        rows = load_buy_rows(Path(args.decisions).expanduser())
    except (ValueError, FileNotFoundError) as exc:
        logger.error(f"Decisions file refused: {exc}")
        return 2
    policy = ExecutionPolicy()
    account = (args.account or str(_cfg('IBKR_ACCOUNT', '') or '')).strip().upper() or None

    if args.plan_only:
        import account_snapshot as acct
        snapshot = acct.load_snapshot(args.account_snapshot, account=account) if args.account_snapshot else None
        ledger = OrderLedger(source='order_entry')
        open_orders = ([{'symbol': s, 'action': 'BUY', 'order_ref': ''}
                        for s in (snapshot or {}).get('pending_entry_orders', [])]
                       + [{'symbol': s, 'action': 'BUY', 'order_ref': ALERT_ORDER_REF}
                          for s in (snapshot or {}).get('alert_orders', [])])
        plans = build_plan(rows, snapshot, ledger, open_orders, policy, now, submit=False)
        path = write_plan(plans, run_id=ledger.run_id, mode='plan-only', now=now)
        logger.info(f"Order plan (dry run, no TWS connection): {_summary(plans)} -> {path}")
        return 2 if any(p.status in HALT_STATUSES for p in plans) else 0

    from order_broker import open_session
    submit = submission_enabled(args.submit)
    refused = args.submit and not submit
    if refused:
        logger.error(f"--submit refused: {SUBMIT_REFUSED}; writing a read-only plan instead")
    if submit:
        cutoff = dt_time.fromisoformat(str(_cfg('ENTRY_SUBMIT_CUTOFF_ET', '09:29')))
        if now.time() >= cutoff:
            logger.error(f"{now:%H:%M} ET is past the {cutoff:%H:%M} entry cutoff: nothing submitted")
            return 2
    if not account:
        logger.error("--account DU... (or config IBKR_ACCOUNT) is required to connect")
        return 2
    client_id = ORDER_CLIENT_ID if submit else ORDER_READONLY_CLIENT_ID
    try:
        with ol.ledger_lock():
            ledger = OrderLedger(source='order_entry')
            try:
                broker = open_session(args.port, account, args.submit, client_id)
            except PermissionError as exc:
                logger.error(f"Paper guard refused the session: {exc}")
                return 2
            try:
                state = broker.state()
                report = ledger.reconcile(state, today)
                for item in report.orphans + report.mismatches:
                    notify(f"ledger reconcile: {item.get('order_ref')} {item.get('issue') or item.get('reason')}")
                for b in report.incomplete_brackets:
                    notify(f"{b.get('symbol')}: entry {b.get('order_ref')} (order {b.get('order_id')}, client "
                           f"{b.get('client_id')}) is live at TWS with NO child stop"
                           + ("; cancelling" if submit and b.get('client_id') == client_id else "; cancel it in TWS"))
                if cancel_incomplete_brackets(report, broker, client_id):
                    state = broker.state()
                snapshot = _snapshot_from_state(state, account, now)
                plans = build_plan(rows, snapshot, ledger, state.open_orders, policy, now, submit=submit)
                if submit:
                    submit_plans(plans, broker, ledger, today, client_id)
                    ledger.reconcile(broker.state(), today)
                halted = [p for p in plans if p.status in HALT_STATUSES]
                if submit and halted:
                    notify(f"entries halted: {halted[0].reason}")
            finally:
                broker.close()
            mode = 'submit' if submit else ('submit-refused' if refused else 'read-only')
            path = write_plan(plans, run_id=ledger.run_id, mode=mode, now=now)
            logger.info(f"Order plan ({mode}): {_summary(plans)} -> {path}")
    except ol.LedgerLocked as exc:
        logger.error(str(exc))
        return 3
    except Exception as exc:
        logger.exception(f"order_entry failed: {exc}")
        return 1
    if refused or any(p.status in HALT_STATUSES | FAILURE_STATUSES for p in plans):
        return 2
    return 0


if __name__ == "__main__":
    # --plan-only (the daily pipeline) never sends orders: a halted plan is in the log, not a banner
    raise SystemExit(notifier.notify_exit_code(main(), "Order entry", enabled="--plan-only" not in sys.argv[1:]))
