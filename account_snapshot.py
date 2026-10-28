#!/usr/bin/env python3
"""
Account snapshot — read-only open risk from the IBKR paper account.

Reads positions, open stop orders and net liquidation, then works out the capital at
risk per position: (entry - stop) x shares, or zero once the stop is at or above the
entry (a raised stop frees risk budget). Totals are written to
Data/account/account_snapshot_YYYY-MM-DD_HHMM.json for the execution engine's
open-risk and per-group caps.

Safety:
  - read-only: data requests only, no order is ever placed, changed or cancelled
  - paper only: refuses unless the port is a paper port and every account in the
    TWS session is a paper (DU...) account
  - a position without a stop covering all its shares is flagged "unprotected";
    the engine then sends new BUYs to manual review. Its uncovered shares are also charged
    a conservative fallback risk (UNPROTECTED_FALLBACK_RISK_PCT of their cost, the widest
    stop band) so the open-risk caps never count them as zero. The account is a pure trading
    account that sits in cash, so there is no core-holdings exemption.

 - every snapshot appends (time, account, net liq) to Data/account/equity_log.csv; the
    daily / weekly / month-to-date equity change from that log feeds the loss circuit
    breakers (risk_manager.circuit_breaker). Cash moves are recorded by hand in
    Data/account/capital_adjustments.csv (deposit / withdrawal / paper reset) and the changes
    are measured net of them, so a deposit never counts as profit and a withdrawal never as a
    loss. Fallback when nothing was recorded: a rise of more than CAPITAL_CHANGE_PCT between
    two entries restarts history (deposit / reset up). Unrecorded drops are always losses.
    The equity log itself is never rewritten.
 - when TWS reports P&L, a net-liq move that P&L doesn't explain (same-day snapshots) is
    flagged as 'unexplained_capital_change' with the command to record it (warning only).
    Limits: IBKR's dailyPnL resets each session, so this only runs when a SECOND snapshot is
    taken the same day (a manual run); the once-a-day 9:20 pipeline never exercises it. A
    cross-day estimate isn't reliable (realized P&L after the last snapshot is lost at the reset).
    Across days only the 'probable paper reset' check (positions wiped + big jump) works.

Usage:
  python account_snapshot.py                 # paper TWS on config.TWS_PORT, client 13
  python account_snapshot.py --account DUX123 # pick one account if the session has several
  python account_snapshot.py --record-adjustment -5000 --account DUX123 [--at 2026-10-26T16:00] [--note ...]
  python account_snapshot.py --record-reset --account DUX123 [--at ...]
      (no TWS: only writes capital_adjustments.csv and prints the resulting figures)
"""

import argparse
import csv
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import config
import strategy_params as _sp

logger = logging.getLogger(__name__)

SNAPSHOT_DIR = Path(config.OUTPUT_DIR).expanduser().parent / "Data" / "account"
DEFAULT_CLIENT_ID = 13   # watcher = config CLIENT_ID, premarket 11, alerts 12, backtest data 98
MAX_AGE_MINUTES = 120    # an older snapshot no longer describes the account
STOP_TYPES = {'STP', 'STP LMT', 'TRAIL', 'TRAIL LIMIT'}
# Only working orders protect a position; Inactive / PendingCancel / Cancelled ones won't fire
WORKING_STATUSES = {'Submitted', 'PreSubmitted', 'PendingSubmit'}
UNSET = 1e300            # ibapi marks unset prices with sys.float_info.max
# Risk charged on shares no working stop covers: the widest (VCP pivot) stop band, % of their cost
UNPROTECTED_FALLBACK_RISK_PCT = _sp.MAX_STOP_PCT_PIVOT
EQUITY_LOG = SNAPSHOT_DIR / "equity_log.csv"
# A move this large between two snapshots is a deposit, withdrawal or paper reset, not P&L
CAPITAL_CHANGE_PCT = 25.0
ADJUSTMENTS_CSV = SNAPSHOT_DIR / "capital_adjustments.csv"
PNL_LOG = SNAPSHOT_DIR / "pnl_log.csv"     # net liq + IBKR P&L per snapshot, for implied-flow checks
# An implied cash move this large (and not recorded) is flagged: max($250, 0.5% of equity)
UNEXPLAINED_MIN_DOLLARS, UNEXPLAINED_MIN_PCT = 250.0, 0.5
ADJUSTMENT_KINDS = ('deposit', 'withdrawal', 'reset')
LINK_WAIT_SEC = 120      # wait this long (awake time) for TWS's server link before giving up


def _price(value) -> Optional[float]:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if 0 < v < UNSET else None


def stop_price(order: dict) -> Optional[float]:
    """Trigger price of a stop order: auxPrice for STP / STP LMT, trailStopPrice for trailing stops."""
    kind = str(order.get('order_type', '')).upper()
    if kind not in STOP_TYPES:
        return None
    return _price(order.get('trail_stop_price') if kind.startswith('TRAIL') else order.get('aux_price'))


def _same_contract(position: dict, order: dict) -> bool:
    """Match by IBKR contract id when both have one, else symbol + security type (an MU option
    stop must not protect MU stock)."""
    if position.get('con_id') and order.get('con_id'):
        return position['con_id'] == order['con_id']
    return (str(order.get('symbol', '')).upper() == str(position['symbol']).upper()
            and order.get('sec_type', 'STK') == position.get('sec_type', 'STK'))


def position_risks(positions: List[dict], open_orders: List[dict],
                   group_of: Callable[[str], Optional[int]] = lambda s: None,
                   fallback_risk_pct: float = UNPROTECTED_FALLBACK_RISK_PCT) -> List[dict]:
    """
    Capital at risk per stock position, from the stop orders protecting it.
    Long: SELL stops; short: BUY stops. If stops cover more shares than held, the worst
    (furthest) stops are counted. Shares without a stop are reported as unprotected and
    charged fallback_risk_pct of their cost (included in risk_dollars, also shown on its own).
    """
    # openOrder repeats on every status update: keep one copy per order. permId is 0 until TWS
    # assigns it, so fall back to the order's own fields (never id(), which keeps every repeat).
    orders = {(o.get('perm_id') or (o.get('client_id'), o.get('order_id'), o.get('symbol'),
                                    o.get('aux_price'), o.get('quantity'))): o
              for o in open_orders}.values()
    out = []
    for p in positions:
        shares = float(p['shares'])
        if shares == 0:
            continue
        symbol = str(p['symbol']).upper()
        long = shares > 0
        cost = float(p['avg_cost'])
        stops = []
        for o in orders:
            px = stop_price(o)
            status = o.get('status')
            if (px is None or (status and status not in WORKING_STATUSES)
                    or not _same_contract(p, o)
                    or o.get('account', p.get('account')) != p.get('account')
                    or str(o.get('action', '')).upper() != ('SELL' if long else 'BUY')):
                continue
            stops.append((px, float(o['quantity'])))
        # Worst stop first: lowest for a long, highest for a short
        stops.sort(key=lambda s: s[0], reverse=not long)
        remaining, risk, covered = abs(shares), 0.0, 0.0
        for px, qty in stops:
            q = min(qty, remaining)
            risk += q * max(0.0, (cost - px) if long else (px - cost))
            covered += q
            remaining -= q
            if remaining <= 0:
                break
        # Options: a stock stop never protects them, so the whole position is unprotected
        unprotected = abs(shares) - covered if p.get('sec_type', 'STK') == 'STK' else abs(shares)
        fallback = unprotected * abs(cost) * fallback_risk_pct / 100
        out.append({
            'symbol': symbol, 'sec_type': p.get('sec_type', 'STK'), 'shares': shares, 'avg_cost': cost,
            'stop': stops[0][0] if stops else None, 'protected_shares': covered,
            'unprotected_shares': unprotected,
            'fallback_risk_dollars': round(fallback, 2),
            'risk_dollars': round(risk + fallback, 2), 'ibd_group_rank': group_of(symbol),
        })
    return out


@dataclass
class CapitalAdjustment:
    """A recorded cash move. amount is signed $ (deposit +, withdrawal -); None for a reset."""
    timestamp: datetime
    kind: str
    amount: Optional[float] = None
    note: str = ''

    def as_dict(self) -> dict:
        return {'timestamp': self.timestamp.isoformat(timespec='seconds'), 'kind': self.kind,
                'amount': self.amount, 'note': self.note}


def apply_adjustments(history: Sequence[Tuple[datetime, float]], now: datetime,
                      adjustments: Sequence[CapitalAdjustment] = ()
                      ) -> Tuple[List[Tuple[datetime, float]], List[CapitalAdjustment]]:
    """
    History restated in today's capital base, and the adjustments that were used.
      reset at T:          every point before T is dropped (a paper reset wipes positions too,
                           so earlier equity isn't comparable)
      deposit/withdrawal F at T: every point before T moves by +F
    Adjustments after `now` are ignored.
    """
    used = sorted((a for a in adjustments if a.timestamp <= now), key=lambda a: a.timestamp)
    points = sorted(history)
    resets = [a for a in used if a.kind == 'reset']
    if resets:
        cut = resets[-1].timestamp
        points = [(t, eq) for t, eq in points if t >= cut]
        used = [a for a in used if a.timestamp >= cut]
    for a in used:
        if a.kind != 'reset' and a.amount:
            points = [(t, eq + a.amount) if t < a.timestamp else (t, eq) for t, eq in points]
    return points, used


def _usable_history(history: Sequence[Tuple[datetime, float]], net_liq: float, now: datetime,
                    adjustments: Sequence[CapitalAdjustment] = (), warn: bool = True
                    ) -> List[Tuple[datetime, float]]:
    """
    Earlier (time, net liq) points the breakers may compare against: restated net of recorded
    adjustments (a recorded reset drops everything before it), then cut after the last unrecorded
    jump UP of more than CAPITAL_CHANGE_PCT (deposit / paper reset). `now` itself is not included.
    """
    restated, _ = apply_adjustments(history, now, adjustments)
    points = restated + [(now, net_liq)]
    start = 0
    for i in range(1, len(points)):
        prev = points[i - 1][1]
        # Only a jump UP is read as new capital (deposit, paper reset). A big drop always stays a
        # loss: treating it as a withdrawal would switch the breakers off on the worst day.
        if prev > 0 and (points[i][1] / prev - 1) * 100 > CAPITAL_CHANGE_PCT:
            start = i
            if warn:
                logger.warning(f"Equity jumped {(points[i][1] / prev - 1) * 100:+.0f}% with no recorded "
                               f"adjustment; history restarts at {points[i][0]:%Y-%m-%d %H:%M}. Record it: "
                               "python account_snapshot.py --record-adjustment AMOUNT --account <DU...> "
                               "--at <when> (or --record-reset)")
    return points[start:-1]


def breaker_history(history: Sequence[Tuple[datetime, float]], net_liq: float, now: datetime,
                    adjustments: Sequence[CapitalAdjustment] = ()) -> Dict[str, Any]:
    """
    How much equity history the circuit breakers have:
      breaker_history_sessions  distinct dates with a snapshot after the last recorded reset and
                                after the last unrecorded >CAPITAL_CHANGE_PCT jump up, today excluded
      breaker_history_start     first such date (ISO), None if there is none
    With fewer than ExecutionPolicy.min_breaker_history_sessions (5) the daily / weekly / monthly
    breakers can't see a losing streak yet, so order_entry won't submit new entries.
    Kept apart from equity_changes() so its three breaker inputs stay exactly as tested.
    """
    days = sorted({t.date() for t, _ in _usable_history(history, net_liq, now, adjustments, warn=False)
                   if t.date() < now.date()})
    return {'breaker_history_sessions': len(days),
            'breaker_history_start': days[0].isoformat() if days else None}


def equity_changes(history: Sequence[Tuple[datetime, float]], net_liq: float,
                   now: datetime, adjustments: Sequence[CapitalAdjustment] = ()) -> Dict[str, Optional[float]]:
    """
    Equity change % for the circuit breakers, from earlier (time, net liq) entries of one account:
      daily_pnl_pct       vs the last entry before today (at the 9:20 premarket run: the
                          previous session's change, so a big losing day stands the bot down)
      weekly_pnl_pct      vs the last entry before this week's Monday
      month_drawdown_pct  vs the month's peak (entries this month, the last one before it, and now)
    None where there is no reference entry. Measured net of recorded capital adjustments
    (apply_adjustments). Fallback when a cash move wasn't recorded: history restarts after a rise
    of more than CAPITAL_CHANGE_PCT (deposit / paper reset); drops are never excused. After a
    withdrawal or a reset to a smaller balance, record it:
        python account_snapshot.py --record-adjustment -AMOUNT --account DU... --at <when it moved>
        (or --record-reset --at <when>)
    """
    past = _usable_history(history, net_liq, now, adjustments)

    def change_since(cutoff: datetime) -> Optional[float]:
        ref = [eq for t, eq in past if t < cutoff]
        return round((net_liq / ref[-1] - 1) * 100, 4) if ref and ref[-1] > 0 else None

    today = datetime(now.year, now.month, now.day)
    monday = today - timedelta(days=today.weekday())
    month_start = datetime(now.year, now.month, 1)
    before = [eq for t, eq in past if t < month_start]
    in_month = [eq for t, eq in past if t >= month_start]
    window = before[-1:] + in_month
    drawdown = round((net_liq / max(window + [net_liq]) - 1) * 100, 4) if window else None
    return {'daily_pnl_pct': change_since(today), 'weekly_pnl_pct': change_since(monday),
            'month_drawdown_pct': drawdown}


def read_adjustments(account: str, path: Optional[Path] = None) -> List[CapitalAdjustment]:
    path = Path(path or ADJUSTMENTS_CSV)
    if not path.exists():
        return []
    out = []
    with path.open() as f:
        for row in csv.DictReader(f):
            if row.get('account') != account:
                continue
            try:
                kind = row['kind'].strip().lower()
                if kind not in ADJUSTMENT_KINDS:
                    continue
                amount = float(row['amount']) if str(row.get('amount') or '').strip() else None
                out.append(CapitalAdjustment(datetime.fromisoformat(row['timestamp']), kind, amount,
                                             row.get('note') or ''))
            except (KeyError, ValueError):
                logger.warning(f"Skipping malformed capital adjustment row: {row}")
    return out


def record_adjustment(account: str, kind: str, amount: Optional[float] = None,
                      at: Optional[datetime] = None, note: str = '', path: Optional[Path] = None,
                      now: Optional[datetime] = None) -> CapitalAdjustment:
    """Append one capital adjustment (paper accounts only; never in the future)."""
    path = Path(path or ADJUSTMENTS_CSV)
    now = now or datetime.now()
    at = at or now
    account = str(account or '').strip().upper()
    if not account.startswith('DU'):
        raise ValueError(f"{account or '(none)'} is not a paper (DU...) account")
    if kind not in ADJUSTMENT_KINDS:
        raise ValueError(f"kind must be one of {ADJUSTMENT_KINDS}")
    if at > now:
        raise ValueError(f"--at {at.isoformat()} is in the future")
    if kind == 'reset':
        amount = None
    elif not amount:
        raise ValueError("a deposit / withdrawal needs a non-zero amount")
    elif (kind == 'deposit') != (amount > 0):
        raise ValueError(f"a {kind} must be {'positive' if kind == 'deposit' else 'negative'}")
    adj = CapitalAdjustment(at, kind, amount, note)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open('a', newline='') as f:
        writer = csv.writer(f)
        if new:
            writer.writerow(['timestamp', 'account', 'kind', 'amount', 'note'])
        writer.writerow([at.isoformat(timespec='seconds'), account, kind,
                         '' if amount is None else amount, note])
    return adj


def implied_capital_flow(prev: Optional[dict], net_liq: float, daily_pnl: Optional[float],
                         now: datetime, adjustments: Sequence[CapitalAdjustment] = ()) -> Optional[dict]:
    """
    Cash that moved without P&L explaining it, between the previous pnl_log entry and now:
    (net liq change) - (dailyPnL change). Only within one day (IBKR's dailyPnL resets each
    session). Returns {'checked': False, 'reason'} when it can't check, else
    {'checked': True, 'amount', 'flagged', 'prev', 'now'}; flagged when the amount is over
    max($250, 0.5% of equity) and no adjustment was recorded in between.
    """
    if not prev or daily_pnl is None or prev.get('daily_pnl') is None:
        return {'checked': False, 'reason': 'no P&L from TWS for both snapshots'}
    prev_ts = prev['timestamp']
    if prev_ts.date() != now.date():
        # dailyPnL resets each session: a cross-day comparison would misread a day's P&L as cash
        return {'checked': False, 'reason': 'previous snapshot is from another day '
                                            '(same-day check only: take a second snapshot today)'}
    amount = (net_liq - prev['net_liquidation']) - (daily_pnl - prev['daily_pnl'])
    covered = any(prev_ts < a.timestamp <= now for a in adjustments)
    limit = max(UNEXPLAINED_MIN_DOLLARS, abs(net_liq) * UNEXPLAINED_MIN_PCT / 100)
    return {'checked': True, 'amount': round(amount, 2), 'flagged': abs(amount) > limit and not covered,
            'prev': {'timestamp': prev_ts.isoformat(timespec='seconds'), 'net_liquidation': prev['net_liquidation']},
            'now': {'timestamp': now.isoformat(timespec='seconds'), 'net_liquidation': net_liq}}


def read_pnl_log(account: str, path: Optional[Path] = None) -> Optional[dict]:
    """Newest pnl_log entry for the account, or None."""
    path = Path(path or PNL_LOG)
    if not path.exists():
        return None
    last = None
    with path.open() as f:
        for row in csv.DictReader(f):
            if row.get('account') != account:
                continue
            try:
                last = {'timestamp': datetime.fromisoformat(row['timestamp']),
                        'net_liquidation': float(row['net_liquidation']),
                        'daily_pnl': float(row['daily_pnl']) if row.get('daily_pnl') else None,
                        'positions': int(row['positions']) if row.get('positions') else None}
            except (KeyError, ValueError):
                continue
    return last


def append_pnl_log(snapshot: dict, path: Optional[Path] = None) -> None:
    path = Path(path or PNL_LOG)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    pnl = snapshot.get('pnl') or {}
    with path.open('a', newline='') as f:
        writer = csv.writer(f)
        if new:
            writer.writerow(['timestamp', 'account', 'net_liquidation', 'daily_pnl', 'realized_pnl',
                             'unrealized_pnl', 'total_cash', 'positions'])
        writer.writerow([snapshot['timestamp'], snapshot['account'], snapshot['net_liquidation'],
                         '' if pnl.get('daily') is None else pnl['daily'],
                         '' if pnl.get('realized') is None else pnl['realized'],
                         '' if pnl.get('unrealized') is None else pnl['unrealized'],
                         '' if snapshot.get('total_cash') is None else snapshot['total_cash'],
                         len(snapshot.get('positions', []))])


def read_equity_log(account: str, path: Path = EQUITY_LOG) -> List[Tuple[datetime, float]]:
    if not path.exists():
        return []
    out = []
    with path.open() as f:
        for row in csv.DictReader(f):
            if row.get('account') == account:
                try:
                    out.append((datetime.fromisoformat(row['timestamp']), float(row['net_liquidation'])))
                except (KeyError, ValueError):
                    continue
    return out


def append_equity_log(snapshot: dict, path: Path = EQUITY_LOG) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open('a', newline='') as f:
        writer = csv.writer(f)
        if new:
            writer.writerow(['timestamp', 'account', 'net_liquidation'])
        writer.writerow([snapshot['timestamp'], snapshot['account'], snapshot['net_liquidation']])


def build_snapshot(state: dict, account: Optional[str] = None,
                   group_of: Callable[[str], Optional[int]] = lambda s: None,
                   now: Optional[datetime] = None,
                   equity_history: Optional[Sequence[Tuple[datetime, float]]] = None,
                   adjustments: Sequence[CapitalAdjustment] = (),
                   prev_pnl: Optional[dict] = None) -> dict:
    """Account state from IBKRConnector.fetch_account_state -> snapshot dict for one paper account."""
    accounts = sorted(state['account_values'])
    live = [a for a in accounts if not a.upper().startswith('DU')]
    if live:
        raise ValueError(f"Non-paper account(s) in this session: {live}")
    if account is None:
        if len(accounts) != 1:
            raise ValueError(f"Session has {len(accounts)} accounts {accounts}; pass --account")
        account = accounts[0]
    if account not in state['account_values']:
        raise ValueError(f"Account {account} not in this session {accounts}")
    net_liq = _price(state['account_values'][account].get('NetLiquidation'))
    if net_liq is None:
        raise ValueError(f"No NetLiquidation for {account}")

    positions = [p for p in state['positions'] if p.get('account') == account]
    orders = [o for o in state['open_orders'] if o.get('account') == account]
    risks = position_risks(positions, orders, group_of)
    by_group: Dict[str, float] = {}
    for r in risks:
        if r['ibd_group_rank'] is not None:
            key = str(int(r['ibd_group_rank']))
            by_group[key] = by_group.get(key, 0.0) + r['risk_dollars'] / net_liq * 100
    now = now or datetime.now()
    changes = (equity_changes(equity_history, net_liq, now, adjustments) if equity_history is not None
               else {'daily_pnl_pct': None, 'weekly_pnl_pct': None, 'month_drawdown_pct': None})
    # No equity log read at all = unknown history (None), which halts submit-mode entries
    changes.update(breaker_history(equity_history, net_liq, now, adjustments) if equity_history is not None
                   else {'breaker_history_sessions': None, 'breaker_history_start': None})
    used = apply_adjustments(equity_history or [], now, adjustments)[1] if equity_history is not None else []
    # Optional P&L (reqPnL) and cash; a timeout leaves them None and is not a failure
    pnl = (state.get('pnl') or {}).get(account) or {}
    total_cash = _price(state['account_values'][account].get('TotalCashValue')) \
        if state['account_values'][account].get('TotalCashValue') not in (None, '') else None
    flow = implied_capital_flow(prev_pnl, net_liq, pnl.get('daily'), now, adjustments)
    # Positions gone and net liq jumped since the last snapshot: probably a paper reset
    probable_reset = bool(prev_pnl and prev_pnl.get('positions') and not positions
                          and prev_pnl['net_liquidation'] > 0
                          and abs(net_liq / prev_pnl['net_liquidation'] - 1) * 100 > CAPITAL_CHANGE_PCT
                          and not any(prev_pnl['timestamp'] < a.timestamp <= now for a in adjustments))
    held = {r['symbol'] for r in risks}
    # BUY stop orders with no position: staged entries. Not counted (they may never fill), listed.
    # The alert setter's 1-share staged price alerts (orderRef ALERT) are listed apart: not entries.
    from ibkr_connector import ALERT_ORDER_REF
    buys = [o for o in orders
            if str(o.get('action', '')).upper() == 'BUY' and str(o['symbol']).upper() not in held]
    is_alert = lambda o: str(o.get('order_ref') or '') == ALERT_ORDER_REF
    pending = sorted({str(o['symbol']).upper() for o in buys if not is_alert(o)})
    alerts = sorted({str(o['symbol']).upper() for o in buys if is_alert(o)})
    return {
        'timestamp': now.isoformat(timespec='seconds'),
        'account': account,
        'net_liquidation': net_liq,
        'total_open_risk_pct': round(sum(r['risk_dollars'] for r in risks) / net_liq * 100, 4),
        'group_open_risk_pct': {k: round(v, 4) for k, v in by_group.items()},
        'held_symbols': sorted(held),
        'unprotected_symbols': sorted(r['symbol'] for r in risks if r['unprotected_shares'] > 0),
        'pending_entry_orders': pending,
        'alert_orders': alerts,
        **changes,
        'adjustments_applied': [a.as_dict() for a in used],
        'pnl': {k: pnl.get(k) for k in ('daily', 'unrealized', 'realized')},
        'total_cash': total_cash,
        'capital_flow_check': flow,
        'unexplained_capital_change': (
            {'amount': flow['amount'], 'prev': flow['prev'], 'now': flow['now']}
            if flow and flow.get('flagged') else
            ({'amount': None, 'reason': 'positions wiped and net liq jumped: probable paper reset',
              'prev': {'timestamp': prev_pnl['timestamp'].isoformat(timespec='seconds'),
                       'net_liquidation': prev_pnl['net_liquidation']},
              'now': {'timestamp': now.isoformat(timespec='seconds'), 'net_liquidation': net_liq}}
             if probable_reset else None)),
        'positions': risks,
    }


def save_snapshot(snapshot: dict, directory: Path = SNAPSHOT_DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.fromisoformat(snapshot['timestamp']).strftime('%Y-%m-%d_%H%M')
    path = directory / f"account_snapshot_{stamp}.json"
    path.write_text(json.dumps(snapshot, indent=2))
    return path


def load_snapshot(path: Optional[str] = None, max_age_minutes: int = MAX_AGE_MINUTES,
                  now: Optional[datetime] = None, account: Optional[str] = None) -> Optional[dict]:
    """The given snapshot, or the newest in SNAPSHOT_DIR; None if missing, older than max_age,
    or for a different account than expected."""
    if path in (None, 'latest'):
        files = sorted(SNAPSHOT_DIR.glob('account_snapshot_*.json'))
        if not files:
            return None
        path = files[-1]
    p = Path(path).expanduser()
    if not p.exists():
        return None
    snapshot = json.loads(p.read_text())
    if account and snapshot.get('account') != account:
        logger.warning(f"Account snapshot {p.name} is for {snapshot.get('account')}, expected {account}")
        return None
    age = (now or datetime.now()) - datetime.fromisoformat(snapshot['timestamp'])
    if age.total_seconds() > max_age_minutes * 60:
        logger.warning(f"Account snapshot {p.name} is {age} old; ignoring it")
        return None
    return snapshot


def _ibd_group_lookup() -> Callable[[str], Optional[int]]:
    """Symbol -> IBD industry group rank from the newest IBD export (same file the scanner uses)."""
    from ibd_universe import load_ibd_universe
    pattern = getattr(config, 'IBD_UNIVERSE_GLOB',
                      str(Path.home() / "swing-trading-bot" / "Data" / "ibd_universe_*.csv"))
    universe, _, _ = load_ibd_universe(pattern)

    def group_of(symbol: str) -> Optional[int]:
        rank = universe.get(symbol, {}).get('ibd_industry_rank')
        return int(rank) if rank is not None else None
    return group_of


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only open-risk snapshot of the IBKR paper account")
    parser.add_argument('--account', help='Paper account id, if the session has several')
    parser.add_argument('--client-id', type=int, default=DEFAULT_CLIENT_ID)
    parser.add_argument('--port', type=int, default=config.TWS_PORT)
    parser.add_argument('--out', help='Write the snapshot here (scheduled runs) instead of a stamped file')
    parser.add_argument('--record-adjustment', type=float, metavar='AMOUNT',
                        help='Record a deposit (+) or withdrawal (-) in $; no TWS connection')
    parser.add_argument('--record-reset', action='store_true',
                        help='Record a paper-account reset; no TWS connection')
    parser.add_argument('--at', help="With --record-adjustment / --record-reset (required): when the cash "
                             "actually moved, ISO time e.g. 2026-10-26T10:00, or 'now'. Every equity-log "
                             "entry before it is restated, so a late --at double-counts the move")
    parser.add_argument('--note', default='')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%H:%M:%S')

    if args.record_adjustment is not None or args.record_reset:
        return record_cli(args)

    from ibkr_connector import PAPER_PORTS, IBKRConnector, confirm_paper_session

    if args.port not in PAPER_PORTS:   # check before connecting at all
        print(f"❌ Port {args.port} is not a paper port {sorted(PAPER_PORTS)}")
        return 1

    conn = IBKRConnector()
    if not conn.connect_and_run(config.TWS_HOST, args.port, args.client_id):
        print(f"❌ Couldn't connect to TWS at {config.TWS_HOST}:{args.port} as client {args.client_id}")
        return 1
    try:
        try:
            confirm_paper_session(conn, args.port)   # paper port + DU accounts only
        except PermissionError as e:
            print(f"❌ {e}; no snapshot written")
            return 1
        # Account data needs the TWS <-> IB server link (down after a wake: 2110 / 1100);
        # no data farm is required, so required=() only checks the link
        if not conn.wait_for_data_farms(required=(), max_wait=LINK_WAIT_SEC):
            print(f"❌ TWS server link down for {LINK_WAIT_SEC}s ({conn.farm_status_summary(())}); "
                  "no snapshot written")
            return 1
        state = conn.fetch_account_state()
    finally:
        conn.disconnect_and_stop()
    if state is None:
        print("❌ Account state incomplete (timeout); no snapshot written")
        return 1
    try:
        accounts = sorted(state['account_values'])
        acct_id = args.account or (accounts[0] if len(accounts) == 1 else None)
        history = read_equity_log(acct_id) if acct_id else []
        adjustments = read_adjustments(acct_id) if acct_id else []
        snapshot = build_snapshot(state, args.account, _ibd_group_lookup(), equity_history=history,
                                  adjustments=adjustments, prev_pnl=read_pnl_log(acct_id) if acct_id else None)
    except ValueError as e:
        print(f"❌ {e}; no snapshot written")
        return 1
    if args.out:
        path = Path(args.out).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(snapshot, indent=2))
    else:
        path = save_snapshot(snapshot)
    append_equity_log(snapshot)
    append_pnl_log(snapshot)

    print(f"Account {snapshot['account']}  net liq ${snapshot['net_liquidation']:,.0f}  "
          f"open risk {snapshot['total_open_risk_pct']:.2f}%")
    fmt = lambda v: "n/a" if v is None else f"{v:+.2f}%"
    print(f"  Equity change: day {fmt(snapshot['daily_pnl_pct'])}  week {fmt(snapshot['weekly_pnl_pct'])}  "
          f"month drawdown {fmt(snapshot['month_drawdown_pct'])}")
    for r in snapshot['positions']:
        stop = f"{r['stop']:.2f}" if r['stop'] else "NONE"
        flag = (f"  ⚠ {r['unprotected_shares']:.0f} sh unprotected "
                f"(charged ${r.get('fallback_risk_dollars', 0):,.0f} at {UNPROTECTED_FALLBACK_RISK_PCT:g}%)"
                if r['unprotected_shares'] > 0 else "")
        print(f"  {r['symbol']:<6} {r['shares']:>7.0f} @ {r['avg_cost']:>8.2f}  stop {stop:>8}  "
              f"risk ${r['risk_dollars']:>8,.0f}  group {r['ibd_group_rank'] or '-'}{flag}")
    for a in snapshot['adjustments_applied']:
        print(f"  Capital adjustment applied: {a['timestamp']} {a['kind']} {a['amount'] if a['amount'] is not None else ''}")
    flag = snapshot.get('unexplained_capital_change')
    if flag:
        amt = flag.get('amount')
        print(f"  ⚠ Net liq moved {'$' + format(amt, ',.0f') if amt is not None else ''} not explained by P&L "
              f"({flag.get('reason', 'since ' + flag['prev']['timestamp'])}). If cash moved, record it:")
        print(f"    python account_snapshot.py " + (f"--record-adjustment {amt:.0f}" if amt is not None else "--record-reset")
              + f" --account {snapshot['account']} --at <when it moved>")
    if snapshot['pending_entry_orders']:
        print(f"  Pending entry orders (not counted): {', '.join(snapshot['pending_entry_orders'])}")
    print(f"Snapshot: {path}")
    notify_unprotected(snapshot)
    return 0


def notify_unprotected(snapshot: dict) -> Optional[str]:
    """macOS warning when any position lacks a full working stop. Returns the message (None if all covered)."""
    rows = [r for r in snapshot.get('positions', []) if (r.get('unprotected_shares') or 0) > 0]
    if not rows:
        return None
    parts = ", ".join(f"{r['symbol']} {r['unprotected_shares']:.0f} sh" for r in rows)
    message = f"{snapshot.get('account', '')}: {len(rows)} position(s) without a full stop: {parts}".strip()
    import notify as notifier
    notifier.notify("Unprotected positions", message, 'warning')
    return message


def record_cli(args) -> int:
    """--record-adjustment / --record-reset: write capital_adjustments.csv only (never TWS)."""
    if not args.account:
        print("❌ --account DU... is required")
        return 1
    account = args.account.strip().upper()
    history = sorted(read_equity_log(account))
    if not args.at:
        # No silent default: recorded after the fact with at=now, the move is subtracted again
        # from snapshots that already show it (review H2), which can turn a loss into a gain
        print("❌ --at is required: the time the cash actually moved (or the reset happened), "
              "e.g. --at 2026-10-26T10:00, or --at now if it moved just now; nothing recorded")
        for t_, eq in history[-2:]:
            print(f"   equity log: {t_:%Y-%m-%d %H:%M}  ${eq:,.0f}")
        return 1
    try:
        at = None if args.at.strip().lower() == 'now' else datetime.fromisoformat(args.at)
        kind = 'reset' if args.record_reset else ('deposit' if args.record_adjustment > 0 else 'withdrawal')
        adj = record_adjustment(args.account, kind, None if args.record_reset else args.record_adjustment,
                                at=at, note=args.note)
    except ValueError as e:
        print(f"❌ {e}; nothing recorded")
        return 1
    print(f"Recorded {adj.kind} {'' if adj.amount is None else f'${adj.amount:,.2f} '}for {account} "
          f"at {adj.timestamp.isoformat(timespec='seconds')} in {ADJUSTMENTS_CSV}")
    after = [(t_, eq) for t_, eq in history if t_ >= adj.timestamp]
    if after and adj.kind != 'reset':
        # Those snapshots are treated as already containing the move; if it really happened
        # before them, --at is too late and the earlier ones are restated twice
        print(f"  ⚠ {len(after)} equity-log entr{'y' if len(after) == 1 else 'ies'} at or after --at "
              f"(first {after[0][0]:%Y-%m-%d %H:%M}, ${after[0][1]:,.0f}) are taken as already including "
              f"this move. If the cash moved earlier than {adj.timestamp:%Y-%m-%d %H:%M}, fix the row in "
              f"{ADJUSTMENTS_CSV}.")
    if history:
        for t_, eq in history[-2:]:
            print(f"  equity log: {t_:%Y-%m-%d %H:%M}  ${eq:,.0f}")
        # Preview as of the newest logged snapshot; a cash move after it shows on the next snapshot
        (*past, (last_t, last_eq)) = history
        ch = equity_changes(past, last_eq, last_t, read_adjustments(account))
        fmt = lambda v: "n/a" if v is None else f"{v:+.2f}%"
        print(f"  As of the last snapshot ({last_t:%Y-%m-%d %H:%M}, ${last_eq:,.0f}): day "
              f"{fmt(ch['daily_pnl_pct'])}  week {fmt(ch['weekly_pnl_pct'])}  "
              f"month drawdown {fmt(ch['month_drawdown_pct'])}")
        if adj.timestamp > last_t:
            print("  (the adjustment is after that snapshot: the next snapshot measures net of it)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
