"""
Order ledger: the bot's own append-only record of every order it intends, sends and sees filled.

No order is placed, changed or cancelled here. order_broker is the only module that talks to
TWS; this module only writes the bot's memory of what it did, so a crash, a restart or a second
run never sends the same entry twice.

Files (under ORDERS_DIR, default ~/swing-trading-bot/Data/orders):
  ledger.jsonl                  append-only, one JSON event per line, fsynced after each write.
                                This is the only source of truth.
  ledger_view_<yyyy-mm-dd>.csv  derived view: latest state per order, rebuilt on every write.
                                Never read back.
  .lock                         ledger_lock(): one order run at a time (entry, exit, fill monitor)

Rules:
  - record_intent() is written (and fsynced) BEFORE placeOrder, so a crash after placing still
    leaves a trace that reconcile can match by orderRef.
  - One bot entry per (trade_date, symbol), whatever the setup or outcome. A cancelled or
    never-sent entry still uses the day's slot: no same-day resend.
  - Order states move forward only (ALLOWED_TRANSITIONS). An illegal transition is logged as a
    discrepancy and not applied.
  - Trade-level events (role 'trade') mark a trade manual_touched (position no longer matches
    the ledger, so the bot stops managing it) or closed.

reconcile() reads a broker state object with these attributes (order_broker.BrokerState):
  positions         [{'account','symbol','sec_type','shares',...}]
  open_orders       [{'order_id','perm_id','client_id','order_ref','symbol','action','order_type',
                      'quantity','status', optional 'filled','avg_fill_price','parent_id','tif',
                      'oca_group','transmit'}]
  executions        [{'exec_id','order_id','perm_id','client_id','order_ref','symbol','side',
                      'shares','price','account'}]
  completed_orders  same keys as open_orders, optional 'completed_status'
  complete          optional bool (default True): False if any of those reads timed out; then
                    nothing is marked never_reached_tws (absence proves nothing)

Usage:
  from order_ledger import OrderLedger, LedgerKey, ledger_lock, make_order_ref
  with ledger_lock():
      ledger = OrderLedger(run_id="entry-20261102-092405-ab12")
      report = ledger.reconcile(broker.state(), today)
      key = LedgerKey(today, "NVDA", "PIVOT_BREAKOUT")
      ledger.record_intent(key, "entry", make_order_ref("entry", today, "NVDA"), 1043, qty=120, ...)
      broker.place(...)
      ledger.record({**key.as_fields(), "role": "entry", "order_ref": ref, "order_id": 1043,
                     "state": "submitted"})
"""

import errno
import fcntl
import json
import logging
import os
import secrets
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import pandas as pd

import config
from market_time import now_et

logger = logging.getLogger(__name__)

# Same default as the config.template key; derived from OUTPUT_DIR (~/swing-trading-bot/Scans) like
# account_snapshot's Data/account, so tests (temp OUTPUT_DIR) never touch the real folder.
ORDERS_DIR = Path(getattr(config, 'ORDERS_DIR', None)
                  or Path(config.OUTPUT_DIR).expanduser().parent / "Data" / "orders").expanduser()

ORDER_STATES = ('intended', 'submitted', 'working', 'partially_filled',
                'filled', 'cancelled', 'rejected', 'expired')
TERMINAL = {'filled', 'cancelled', 'rejected', 'expired'}
ALLOWED_TRANSITIONS = {
    'intended': {'submitted', 'cancelled'},
    # 'expired' is not in the design table: a DAY order sent, never acked, gone the next day
    # would otherwise stay 'submitted' forever.
    'submitted': {'working', 'rejected', 'cancelled', 'filled', 'partially_filled', 'expired'},
    'working': {'partially_filled', 'filled', 'cancelled', 'expired'},
    'partially_filled': {'filled', 'cancelled', 'expired'},
}

ORDER_ROLES = ('entry', 'stop', 'trim', 'exit', 'flatten', 'stop_fix', 'cover', 'trim_stop')
SELL_ROLES = {'stop', 'trim', 'exit', 'flatten', 'stop_fix', 'trim_stop'}
# trim_stop: TRIM_STOP_MODE 'split_stop' only, the stop for the trim block in the trim's own OCA group
COVER_ROLE = 'cover'                       # fill_monitor's BUY back of a short the bot's SELLs caused
TRADE_ROLE = 'trade'                       # trade-level status events, not orders
TRADE_STATES = ('manual_touched', 'closed')
TRADE_TRANSITIONS = {None: {'manual_touched', 'closed'}, 'manual_touched': {'closed'}}

# orderRef scheme: BOT-<TAG>-<entry trade_date yyyymmdd>-<SYMBOL>
REF_PREFIX = "BOT-"
ROLE_TAGS = {'entry': 'ENTRY', 'stop': 'STOP', 'trim': 'TRIM', 'exit': 'EXIT',
             'flatten': 'FLAT', 'stop_fix': 'STOPFIX', COVER_ROLE: 'COVER', TRADE_ROLE: 'TRADE',
             'trim_stop': 'TSTOP'}
TAG_ROLES = {v: k for k, v in ROLE_TAGS.items()}

# TWS order status -> ledger state. Statuses not listed (PendingCancel, Inactive, '') change
# nothing, but still prove the order reached TWS.
TWS_STATE = {'PendingSubmit': 'working', 'PreSubmitted': 'working', 'Submitted': 'working',
             'ApiPending': 'working', 'Filled': 'filled',
             'Cancelled': 'cancelled', 'ApiCancelled': 'cancelled'}
DAY_TIFS = {'DAY', 'OPG'}                  # orders that die at the end of their session

KEY_FIELDS = ('trade_date', 'symbol', 'setup')
# A same-state event is written only if one of these changed. qty / prices / cancel_requested /
# reason carry a modify or cancel request (fill_monitor, exit_manager), recorded before it is sent.
CHANGE_FIELDS = ('state', 'filled_qty', 'perm_id', 'avg_fill_price', 'qty', 'aux_price', 'lmt_price',
                 'cancel_requested', 'reason')
# One order = one row in current(): the key, its role, its orderRef and its order id
ORDER_ID_FIELDS = KEY_FIELDS + ('role', 'order_ref', 'order_id')
VIEW_COLUMNS = ['ts', 'run_id', 'source', 'trade_date', 'symbol', 'setup', 'role', 'state',
                'order_ref', 'order_id', 'parent_order_id', 'perm_id', 'client_id', 'account',
                'action', 'order_type', 'qty', 'aux_price', 'lmt_price', 'tif', 'oca_group',
                'filled_qty', 'avg_fill_price', 'placed_ts', 'reason', 'orphan', 'plan']


class LedgerLocked(RuntimeError):
    """Another order run holds Data/orders/.lock (CLIs exit 3)."""


class LedgerCorrupt(RuntimeError):
    """A line other than the last one isn't valid JSON: stop, a human must look."""


class DuplicateEntry(ValueError):
    """The ledger already has an entry for this (trade_date, symbol)."""


@dataclass(frozen=True)
class LedgerKey:
    trade_date: date
    symbol: str
    setup: str

    def as_fields(self) -> Dict[str, str]:
        return {'trade_date': _iso_date(self.trade_date), 'symbol': self.symbol.upper(),
                'setup': self.setup}


@dataclass
class ReconcileReport:
    applied: List[dict] = field(default_factory=list)              # events appended
    orphans: List[dict] = field(default_factory=list)              # BOT- orders the ledger didn't know
    mismatches: List[dict] = field(default_factory=list)           # ledger vs TWS disagreements
    incomplete_brackets: List[dict] = field(default_factory=list)  # parent with no child stop

    @property
    def clean(self) -> bool:
        return not (self.orphans or self.mismatches or self.incomplete_brackets)


# ─── orderRef helpers ───

def _iso_date(d: Any) -> str:
    return d.isoformat() if isinstance(d, (date, datetime)) else str(d)[:10]


def make_order_ref(role: str, trade_date: Any, symbol: str) -> str:
    """BOT-ENTRY-20261102-NVDA. trade_date is always the ENTRY date, so one trade's orders group."""
    d = _iso_date(trade_date).replace('-', '')
    return f"{REF_PREFIX}{ROLE_TAGS[role]}-{d}-{symbol.upper()}"


def make_oca_group(trade_date: Any, symbol: str) -> str:
    """One OCA group per trade for every bot SELL (stop, trim, exits), so two can't both fill."""
    return f"{REF_PREFIX}OCA-{_iso_date(trade_date).replace('-', '')}-{symbol.upper()}"


def make_trim_oca_group(trade_date: Any, symbol: str) -> str:
    """TRIM_STOP_MODE 'split_stop': the trim block's own group (its trim_stop, the trim, its exit share)."""
    return f"{REF_PREFIX}OCAT-{_iso_date(trade_date).replace('-', '')}-{symbol.upper()}"


def parse_order_ref(ref: str) -> Optional[Tuple[str, str, str]]:
    """'BOT-STOP-20261102-NVDA' -> ('stop', '2026-11-02', 'NVDA'); None for manual / ALERT refs."""
    parts = str(ref or '').split('-', 3)
    if len(parts) != 4 or parts[0] + '-' != REF_PREFIX or parts[1] not in TAG_ROLES:
        return None
    d = parts[2]
    if len(d) != 8 or not d.isdigit():
        return None
    return TAG_ROLES[parts[1]], f"{d[:4]}-{d[4:6]}-{d[6:]}", parts[3]


# ─── Lock ───

@contextmanager
def ledger_lock(orders_dir: Optional[Path] = None, blocking: bool = False) -> Iterator[None]:
    """
    Hold an exclusive flock on <orders_dir>/.lock for the WHOLE run of any order module.
    Raises LedgerLocked if another run holds it (non-blocking by default). The kernel releases
    the lock if the process dies, so a crash never leaves it stuck.
    """
    orders_dir = Path(orders_dir or ORDERS_DIR)
    orders_dir.mkdir(parents=True, exist_ok=True)
    fh = open(orders_dir / ".lock", "a+")
    try:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except OSError as exc:
            if exc.errno in (errno.EAGAIN, errno.EACCES, errno.EWOULDBLOCK):
                raise LedgerLocked(f"another order run holds {orders_dir / '.lock'}") from exc
            raise
        # Who holds it, for a human looking at a stuck run (informational only)
        fh.seek(0)
        fh.truncate()
        fh.write(f"pid={os.getpid()} since={now_et().isoformat(timespec='seconds')}\n")
        fh.flush()
        yield
    finally:
        fh.close()   # closing the descriptor drops the flock


# ─── Ledger ───

class OrderLedger:
    """
    Append-only JSONL ledger. Callers hold ledger_lock() for the run; the ledger itself doesn't
    lock. record() merges an update into the order's previous event, so every line holds the
    order's full latest picture and current() is just "last line per order".
    """

    def __init__(self, path: Optional[Path] = None, run_id: Optional[str] = None,
                 source: str = 'order_entry', clock: Callable[[], datetime] = now_et):
        self.path = Path(path) if path else ORDERS_DIR / "ledger.jsonl"
        self.clock = clock
        self.source = source
        self.run_id = run_id or f"{source}-{clock():%Y%m%d-%H%M%S}-{secrets.token_hex(2)}"
        self._cache: Optional[Tuple[Tuple[int, int], List[dict]]] = None
        self.appended = 0          # lines this instance wrote

    # ─── Reading ───

    def events(self) -> List[dict]:
        """All events in file order. A truncated LAST line (crash mid-write) is skipped with a
        warning; any other unparseable line raises LedgerCorrupt."""
        if not self.path.exists():
            return []
        st = self.path.stat()
        stamp = (st.st_size, st.st_mtime_ns)
        if self._cache and self._cache[0] == stamp:
            return list(self._cache[1])
        lines = self.path.read_text().split('\n')
        events = []
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as exc:
                if all(not rest.strip() for rest in lines[i + 1:]):
                    logger.warning(f"Ledger {self.path}: truncated last line skipped ({line[:80]!r})")
                    continue
                raise LedgerCorrupt(f"{self.path} line {i + 1} is not valid JSON: {exc}") from exc
        self._cache = (stamp, events)
        return list(events)

    def _latest(self) -> Dict[tuple, dict]:
        latest: Dict[tuple, dict] = {}
        for ev in self.events():
            latest[_order_id(ev)] = ev
        return latest

    def current(self) -> pd.DataFrame:
        """Latest state per order (key, role, order_ref, order_id), in VIEW_COLUMNS order."""
        rows = list(self._latest().values())
        df = pd.DataFrame(rows, columns=VIEW_COLUMNS if not rows else None)
        for col in VIEW_COLUMNS:
            if col not in df.columns:
                df[col] = None
        return df[VIEW_COLUMNS + [c for c in df.columns if c not in VIEW_COLUMNS]]

    def has_entry(self, trade_date: Any, symbol: str) -> bool:
        """True if any entry (any setup, any state, cancelled included) exists for that day/symbol."""
        d, sym = _iso_date(trade_date), symbol.upper()
        return any(ev.get('role') == 'entry' and ev.get('trade_date') == d and ev.get('symbol') == sym
                   for ev in self.events())

    def entries_count(self, trade_date: Any) -> int:
        """Entry keys for that day that reached 'intended' (never_reached_tws ones count: slot used)."""
        d = _iso_date(trade_date)
        return len({(ev['symbol'], ev['setup']) for ev in self.events()
                    if ev.get('role') == 'entry' and ev.get('trade_date') == d})

    def highest_order_id(self) -> int:
        """Largest order id the ledger has seen (order ids must keep increasing per client)."""
        ids = [int(ev['order_id']) for ev in self.events() if ev.get('order_id') is not None]
        return max(ids, default=0)

    def trade_status(self, trade_date: Any, symbol: str) -> Optional[str]:
        """'manual_touched' / 'closed' from the latest trade event, else None (open or never filled)."""
        d, sym = _iso_date(trade_date), symbol.upper()
        status = None
        for ev in self.events():
            if ev.get('role') == TRADE_ROLE and ev.get('trade_date') == d and ev.get('symbol') == sym:
                status = ev.get('state')
        return status

    def open_trades(self) -> List[dict]:
        """
        Filled (or partly filled) entries without a 'closed' trade event, one dict per trade:
        the entry's latest event plus 'trade_status' ('open' / 'manual_touched'), 'sold_qty'
        (bot SELL fills), 'covered_qty' (bot BUY-to-cover fills) and 'net_qty' (entry fills - bot
        SELL fills + cover fills; negative = the bot's SELLs overfilled into a short).
        """
        latest = list(self._latest().values())
        out = []
        for ev in latest:
            if ev.get('role') != 'entry' or float(ev.get('filled_qty') or 0) <= 0:
                continue
            status = self.trade_status(ev['trade_date'], ev['symbol'])
            if status == 'closed':
                continue
            same = [s for s in latest if s.get('trade_date') == ev['trade_date'] and s.get('symbol') == ev['symbol']]
            sold = sum(float(s.get('filled_qty') or 0) for s in same if s.get('role') in SELL_ROLES)
            covered = sum(float(s.get('filled_qty') or 0) for s in same if s.get('role') == COVER_ROLE)
            out.append({**ev, 'trade_status': status or 'open', 'sold_qty': sold, 'covered_qty': covered,
                        'net_qty': float(ev['filled_qty']) - sold + covered})
        return out

    # ─── Writing ───

    def record(self, event: dict) -> Optional[dict]:
        """
        Validate and append one event (merged onto the order's previous event), fsync it and
        rebuild the view. Returns the stored event, the previous one if nothing changed, or None
        if the transition is illegal (logged as a discrepancy, not written).
        """
        ev = dict(event)
        for k in KEY_FIELDS + ('role', 'state', 'order_ref'):
            if ev.get(k) in (None, ''):
                raise ValueError(f"ledger event missing {k!r}: {ev}")
        ev['trade_date'] = _iso_date(ev['trade_date'])
        ev['symbol'] = str(ev['symbol']).upper()
        role, state = ev['role'], ev['state']
        if role == TRADE_ROLE:
            ev.setdefault('order_id', None)
            if state not in TRADE_STATES:
                raise ValueError(f"unknown trade state {state!r}")
        elif role not in ORDER_ROLES:
            raise ValueError(f"unknown role {role!r}")
        elif state not in ORDER_STATES:
            raise ValueError(f"unknown order state {state!r}")

        prev = self._latest().get(_order_id(ev))
        if not _transition_ok(prev, ev):
            logger.warning(f"Ledger discrepancy (not applied): {ev['order_ref']} order "
                           f"{ev.get('order_id')} {prev.get('state') if prev else None} -> {state}")
            return None
        merged = {**(prev or {}), **ev}
        if prev and all(merged.get(k) == prev.get(k) for k in CHANGE_FIELDS):
            return prev   # nothing new: don't bloat the ledger on every reconcile
        merged['ts'] = self.clock().isoformat(timespec='seconds')
        merged['run_id'] = ev.get('run_id') or self.run_id   # the run that wrote THIS line
        merged['source'] = ev.get('source', self.source)
        merged.setdefault('reason', '')
        if not prev:
            merged.setdefault('placed_ts', merged['ts'])
        self._append(merged)
        self._write_view()
        return merged

    def record_intent(self, key: LedgerKey, role: str, order_ref: str, order_id: int,
                      **fields) -> dict:
        """
        Write the 'intended' event BEFORE placeOrder. Raises DuplicateEntry for a second entry on
        the same (trade_date, symbol), any setup; ValueError if this order already exists.
        """
        if role == 'entry' and self.has_entry(key.trade_date, key.symbol):
            raise DuplicateEntry(f"ledger already has an entry for {key.symbol} on "
                                 f"{_iso_date(key.trade_date)}: no second entry that day")
        ev = {**fields, **key.as_fields(), 'role': role, 'order_ref': order_ref,
              'order_id': order_id, 'state': 'intended', 'filled_qty': fields.get('filled_qty', 0)}
        if _order_id(ev) in self._latest():
            raise ValueError(f"order {order_ref} id {order_id} is already in the ledger")
        stored = self.record(ev)
        assert stored is not None   # a brand-new 'intended' always passes the transition check
        return stored

    def _append(self, ev: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._repair_tail()
        line = json.dumps(ev, default=str, sort_keys=False) + '\n'
        with open(self.path, 'a') as fh:
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())
        self._cache = None
        self.appended += 1

    def _repair_tail(self) -> None:
        """A crash mid-write leaves a partial last line with no newline. Appending after it would
        glue two events into one corrupt MIDDLE line, so move the fragment to a side file first."""
        if not self.path.exists() or self.path.stat().st_size == 0:
            return
        data = self.path.read_bytes()
        if data.endswith(b'\n'):
            return
        cut = data.rfind(b'\n') + 1
        side = self.path.with_name(f"{self.path.name}.partial-{self.clock():%Y%m%d-%H%M%S}")
        side.write_bytes(data[cut:])
        with open(self.path, 'r+b') as fh:
            fh.truncate(cut)
            fh.flush()
            os.fsync(fh.fileno())
        logger.warning(f"Ledger: moved a truncated last line to {side}")

    def view_path(self) -> Path:
        return self.path.with_name(f"ledger_view_{self.clock():%Y-%m-%d}.csv")

    def _write_view(self) -> Path:
        df = self.current()
        df['plan'] = df['plan'].map(lambda p: json.dumps(p, default=str) if isinstance(p, dict) else p)
        path = self.view_path()
        tmp = path.with_suffix('.csv.tmp')
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)
        return path

    # ─── Reconcile ───

    def reconcile(self, state: Any, today: Any) -> ReconcileReport:
        """
        Bring the ledger in line with what TWS reports (read-only: never places or cancels).
          - every non-terminal order: status from open orders, else completed orders, else
            executions; filled_qty = executions summed per order (never lowered: TWS only returns
            today's executions)
          - 'intended' with no trace at TWS -> cancelled 'never_reached_tws' (slot stays used)
          - order gone everywhere: DAY/OPG placed before today -> expired, else a mismatch
          - unknown BOT- open order -> orphan event (never cancelled here)
          - BOT-ENTRY parent at TWS with no child stop -> incomplete bracket (caller cancels in
            submit mode)
          - open trade: position gone -> closed; position != ledger net -> manual_touched
        """
        today_s = _iso_date(today)
        report = ReconcileReport()
        complete = getattr(state, 'complete', True) is not False
        open_orders = list(getattr(state, 'open_orders', []) or [])
        completed = list(getattr(state, 'completed_orders', []) or [])
        executions = _dedupe_execs(getattr(state, 'executions', []) or [])
        positions = list(getattr(state, 'positions', []) or [])

        for row in list(self._latest().values()):
            if row.get('role') == TRADE_ROLE:
                continue
            o = _last_match(row, open_orders)
            c = _last_match(row, completed)
            ex = [e for e in executions if _matches(row, e)]
            if row['state'] in TERMINAL:
                continue
            self._reconcile_order(row, o, c, ex, today_s, complete, report)

        self._find_orphans(open_orders, report)
        report.incomplete_brackets = _incomplete_brackets(open_orders)
        for b in report.incomplete_brackets:
            logger.warning(f"Incomplete bracket at TWS: {b['order_ref']} (order {b['order_id']}) "
                           f"has no child stop")
        self._reconcile_positions(positions, report)
        self._expire_closed_trade_orders(open_orders, report)
        return report

    def _expire_closed_trade_orders(self, open_orders: List[dict], report: ReconcileReport) -> None:
        """
        A closed trade's orders that are still non-terminal here and no longer at TWS -> expired
        'trade_closed'. Typical case: the GTC stop filled on a day with no bot run (TWS only returns
        today's executions), so its row would otherwise stay 'working' and raise not_found_at_tws on
        every run, training the owner to ignore alerts. An order still listed at TWS is left alone
        (fill_monitor reports / cancels an orphan SELL).
        """
        expired = set()
        for row in list(self._latest().values()):
            if (row.get('role') == TRADE_ROLE or row['state'] not in ('submitted', 'working', 'partially_filled')
                    or self.trade_status(row['trade_date'], row['symbol']) != 'closed'
                    or _last_match(row, open_orders) is not None):
                continue
            ident = {k: row[k] for k in ORDER_ID_FIELDS}
            if self._apply(report, {**ident, 'state': 'expired', 'reason': 'trade_closed'}) is not None:
                expired.add(_order_id(ident))
        if expired:
            report.mismatches = [m for m in report.mismatches
                                 if not (m.get('issue') == 'not_found_at_tws' and _order_id(m) in expired)]

    def _apply(self, report: ReconcileReport, ev: dict) -> Optional[dict]:
        before = self.appended
        stored = self.record({**ev, 'source': 'reconcile'})
        if stored is None:
            report.mismatches.append({**ev, 'issue': 'illegal_transition'})
        elif self.appended > before:
            report.applied.append(stored)
        return stored

    def _reconcile_order(self, row: dict, o: Optional[dict], c: Optional[dict], ex: List[dict],
                         today_s: str, complete: bool, report: ReconcileReport) -> None:
        ident = {k: row[k] for k in ORDER_ID_FIELDS}
        qty = float(row.get('qty') or 0)
        exec_qty = sum(float(e.get('shares') or 0) for e in ex)
        reported = max([float(d.get('filled') or 0) for d in (o, c) if d] + [0.0])
        filled = max(float(row.get('filled_qty') or 0), exec_qty, reported)
        avg = row.get('avg_fill_price')
        if exec_qty > 0 and exec_qty >= filled:
            avg = round(sum(float(e.get('shares') or 0) * float(e.get('price') or 0) for e in ex) / exec_qty, 4)
        elif reported > 0:
            avg = next((d.get('avg_fill_price') for d in (o, c) if d and d.get('avg_fill_price')), avg)
        perm = row.get('perm_id') or next((d.get('perm_id') for d in [o, c] + ex
                                           if d and d.get('perm_id')), None)

        seen = o is not None or c is not None or bool(ex)
        if not seen:
            if row['state'] == 'intended':
                if complete:
                    self._apply(report, {**ident, 'state': 'cancelled', 'reason': 'never_reached_tws'})
                return
            placed = _iso_date(row.get('placed_ts') or row.get('ts') or row['trade_date'])
            # placed_ts, not trade_date: exits carry the ENTRY date in their ref and key
            if str(row.get('tif') or '').upper() in DAY_TIFS and placed < today_s:
                self._apply(report, {**ident, 'state': 'expired', 'reason': 'day_order_gone'})
            else:
                report.mismatches.append({**ident, 'state': row['state'],
                                          'issue': 'not_found_at_tws'})
                logger.warning(f"Ledger {row['order_ref']} ({row['state']}) not found at TWS")
            return

        tws = None
        if o is not None:
            tws = TWS_STATE.get(str(o.get('status') or ''))
        elif c is not None:
            tws = TWS_STATE.get(str(c.get('status') or ''))
            if tws == 'cancelled' and 'expir' in str(c.get('completed_status') or '').lower():
                tws = 'expired'
        elif ex:
            tws = 'working'   # executions but no order record: the fill count decides below
        if tws == 'filled' and filled == 0:
            filled = qty      # TWS says Filled but sent no counts: a Filled order is fully filled
        if qty > 0 and filled >= qty:
            tws = 'filled'
        elif filled > 0 and tws in (None, 'working'):
            tws = 'partially_filled'

        if row['state'] == 'intended':
            # It reached TWS although this run (or a crashed one) never recorded 'submitted'
            row = self._apply(report, {**ident, 'state': 'submitted', 'reason': 'seen_at_tws',
                                       'perm_id': perm}) or row
        target = tws or row['state']
        if target == 'working' and row['state'] == 'partially_filled':
            target = 'partially_filled'
        ev = {**ident, 'state': target, 'filled_qty': filled, 'avg_fill_price': avg, 'perm_id': perm}
        if target != row['state'] or filled != float(row.get('filled_qty') or 0) or perm != row.get('perm_id'):
            self._apply(report, ev)

    def _find_orphans(self, open_orders: List[dict], report: ReconcileReport) -> None:
        known = [r for r in self._latest().values() if r.get('role') != TRADE_ROLE]
        seen_keys = set()
        for o in open_orders:
            if any(_matches(r, o) for r in known):
                continue
            parsed = parse_order_ref(o.get('order_ref'))
            if not parsed:
                continue          # manual or ALERT order: not the bot's
            ident = (o.get('perm_id') or None, o.get('order_ref'), o.get('client_id'), o.get('order_id'))
            if ident in seen_keys:
                continue          # openOrder repeats on every status update
            seen_keys.add(ident)
            role, d, sym = parsed
            if role == TRADE_ROLE:
                continue
            setup = next((ev['setup'] for ev in self.events()
                          if ev.get('role') == 'entry' and ev['trade_date'] == d and ev['symbol'] == sym),
                         'UNKNOWN')
            orphan = {'trade_date': d, 'symbol': sym, 'setup': setup, 'role': role,
                      'order_ref': o.get('order_ref'), 'order_id': o.get('order_id'),
                      'perm_id': o.get('perm_id') or None, 'client_id': o.get('client_id'),
                      'account': o.get('account'), 'action': o.get('action'),
                      'order_type': o.get('order_type'), 'qty': o.get('quantity'),
                      'aux_price': o.get('aux_price'), 'lmt_price': o.get('lmt_price'),
                      'tif': o.get('tif'), 'oca_group': o.get('oca_group'),
                      'parent_order_id': o.get('parent_id') or None,
                      'state': TWS_STATE.get(str(o.get('status') or ''), 'working'),
                      'filled_qty': float(o.get('filled') or 0), 'orphan': True,
                      'reason': 'orphan_at_tws'}
            report.orphans.append(orphan)
            logger.warning(f"ALERT: bot order at TWS unknown to the ledger: {o.get('order_ref')} "
                           f"(order {o.get('order_id')}, client {o.get('client_id')}); not cancelled")
            self._apply(report, orphan)

    def _reconcile_positions(self, positions: List[dict], report: ReconcileReport) -> None:
        held: Dict[Tuple[Optional[str], str], float] = {}
        for p in positions:
            if p.get('sec_type', 'STK') != 'STK':
                continue
            k = (p.get('account'), str(p['symbol']).upper())
            held[k] = held.get(k, 0.0) + float(p.get('shares') or 0)

        def position_of(account: Optional[str], symbol: str) -> float:
            if account:
                return held.get((account, symbol), 0.0)
            return sum(v for (a, s), v in held.items() if s == symbol)

        for t in self.open_trades():
            pos = position_of(t.get('account'), t['symbol'])
            trade = {'trade_date': t['trade_date'], 'symbol': t['symbol'], 'setup': t['setup'],
                     'role': TRADE_ROLE, 'order_ref': make_order_ref(TRADE_ROLE, t['trade_date'], t['symbol']),
                     'order_id': None, 'account': t.get('account')}
            if pos == 0:
                reason = 'position_gone' if t['net_qty'] != 0 else 'flat'
                if reason == 'position_gone':
                    logger.warning(f"{t['symbol']}: position gone while the ledger shows "
                                   f"{t['net_qty']:g} open (stop-out or paper reset): trade closed")
                self._apply(report, {**trade, 'state': 'closed', 'reason': reason})
            elif pos != t['net_qty']:
                report.mismatches.append({**trade, 'issue': 'position_mismatch', 'position': pos,
                                          'ledger_net': t['net_qty']})
                if t['trade_status'] != 'manual_touched':
                    logger.warning(f"ALERT: {t['symbol']} position {pos:g} != ledger net "
                                   f"{t['net_qty']:g}: marked manual_touched, the bot stops managing it")
                    self._apply(report, {**trade, 'state': 'manual_touched',
                                         'reason': f"position {pos:g} != ledger net {t['net_qty']:g}"})


# ─── Helpers ───

def _order_id(ev: dict) -> tuple:
    return (_iso_date(ev.get('trade_date')), str(ev.get('symbol', '')).upper(), ev.get('setup'),
            ev.get('role'), ev.get('order_ref'), ev.get('order_id'))


def _transition_ok(prev: Optional[dict], ev: dict) -> bool:
    new = ev['state']
    if ev['role'] == TRADE_ROLE:
        old = prev.get('state') if prev else None
        return new == old or new in TRADE_TRANSITIONS.get(old, set())
    if prev is None:
        # A new order starts as 'intended'; only reconcile's orphan events may start elsewhere
        return new == 'intended' or bool(ev.get('orphan'))
    old = prev['state']
    if new == old:
        return old not in TERMINAL     # same-state update (filled_qty / perm_id) on a live order
    return new in ALLOWED_TRANSITIONS.get(old, set())


def _matches(row: dict, tws: dict) -> bool:
    """perm_id when both have one, else (orderRef, client id, order id)."""
    if row.get('perm_id') and tws.get('perm_id'):
        return int(row['perm_id']) == int(tws['perm_id'])
    return (str(tws.get('order_ref') or '') == row.get('order_ref')
            and row.get('client_id') is not None and tws.get('client_id') == row.get('client_id')
            and tws.get('order_id') == row.get('order_id'))


def _last_match(row: dict, orders: List[dict]) -> Optional[dict]:
    """Last matching dict: openOrder repeats on every status update, the newest wins."""
    found = None
    for o in orders:
        if _matches(row, o):
            found = o
    return found


def _dedupe_execs(executions: List[dict]) -> List[dict]:
    seen, out = set(), []
    for e in executions:
        k = e.get('exec_id') or id(e)
        if k in seen:
            continue
        seen.add(k)
        out.append(e)
    return out


def _incomplete_brackets(open_orders: List[dict]) -> List[dict]:
    """Working BOT-ENTRY parents at TWS with no child: neither an order whose parent_id is the
    parent's id (same client) nor a BOT-STOP ref for the same trade. 'staged' is informational only:
    ibapi never decodes Order.transmit from openOrder, so it is almost always False here."""
    out, seen = [], set()
    for p in open_orders:
        parsed = parse_order_ref(p.get('order_ref'))
        if not parsed or parsed[0] != 'entry':
            continue
        if p.get('status') and TWS_STATE.get(str(p.get('status'))) != 'working':
            continue          # filled / cancelled / inactive / pending cancel: nothing live to cancel
        ident = (p.get('client_id'), p.get('order_id'))
        if ident in seen:
            continue
        seen.add(ident)
        child_ref = make_order_ref('stop', parsed[1], parsed[2])
        has_child = any(
            (o.get('parent_id') and o.get('parent_id') == p.get('order_id')
             and o.get('client_id') == p.get('client_id'))
            or o.get('order_ref') == child_ref
            for o in open_orders)
        if not has_child:
            out.append({'order_ref': p.get('order_ref'), 'order_id': p.get('order_id'),
                        'perm_id': p.get('perm_id'), 'client_id': p.get('client_id'),
                        'symbol': p.get('symbol'), 'staged': p.get('transmit') is False,
                        'status': p.get('status')})
    return out
