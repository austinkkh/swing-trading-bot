"""
Order broker: the ONLY module that calls placeOrder / cancelOrder for the bot's own orders.

Every order-capable path goes through open_session(), which fails closed:
  1. the port must be a paper port (ibkr_connector.PAPER_PORTS); live ports 7496 / 4001 are
     refused before any connection is made
  2. submission is enabled only if config AUTO_SUBMIT_ENABLED is the bool True (not a truthy
     string) AND the CLI passed --submit; otherwise the connector is read-only, so a dry run
     physically can't place an order (IBKRConnector.placeOrder raises)
  3. connect_and_run refuses non-paper ports again; confirm_paper_session checks that every
     account in the session is a paper (DU...) account
  4. an account is required, must start with DU and must be one the session reports
IBKRBroker.place / modify / cancel re-check that submission is enabled. place / modify also refuse
any orderRef that doesn't start with the session's prefix (BOT- by default) and stamp order.account,
so an order can't land in another account. cancel takes only an id (cancelOrder carries no orderRef),
so it is NOT ref-gated: TWS lets a client cancel only its own orders, and the caller must pass an id
it placed (scripts/paper_checks.py also checks the id's orderRef in TWS's open orders first). scripts/paper_checks.py opens its sessions with ref_prefix PAPERCHECK-: such
a session can place only PAPERCHECK- orders and a bot session can never place one, so the bot's
ledger, reconcile and fill monitor never mistake a test order for a bot order.

Client IDs (IBKR only lets the client that placed an order modify or cancel it):
  ORDER_CLIENT_ID (16)           every order-writing session: entries, exits, fill-monitor --submit
  ORDER_READONLY_CLIENT_ID (17)  dry runs, bars, reconcile reads

Usage:
  from order_broker import open_session, submission_enabled
  broker = open_session(port=7497, account="DUXXXXXXX", cli_submit=args.submit,
                        client_id=ORDER_CLIENT_ID if args.submit else ORDER_READONLY_CLIENT_ID)
  try:
      state = broker.state()
  finally:
      broker.close()
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Protocol, Tuple

import pandas as pd

import config
from ibkr_connector import LIVE_PORTS, PAPER_PORTS, IBKRConnector, confirm_paper_session, is_paper_account
from market_time import bar_lag_minutes, in_regular_session, now_et, parse_ibkr_times

logger = logging.getLogger(__name__)

SUBMIT_REFUSED = "submission disabled (needs config AUTO_SUBMIT_ENABLED=True and --submit)"
ORDER_CLIENT_ID = int(getattr(config, 'ORDER_CLIENT_ID', 16))
ORDER_READONLY_CLIENT_ID = int(getattr(config, 'ORDER_READONLY_CLIENT_ID', 17))
BOT_REF_PREFIX = "BOT-"
PAPERCHECK_REF_PREFIX = "PAPERCHECK-"   # scripts/paper_checks.py test orders only
REF_PREFIXES = (BOT_REF_PREFIX, PAPERCHECK_REF_PREFIX)
BOT_OCA_PREFIX = BOT_REF_PREFIX + "OCA"   # order_ledger.make_oca_group / make_trim_oca_group
# TWS order statuses that mean the order is live at TWS (account_snapshot.WORKING_STATUSES + ApiPending)
WORKING = {'Submitted', 'PreSubmitted', 'PendingSubmit', 'ApiPending'}
REJECTED = {'Cancelled', 'ApiCancelled', 'Inactive'}
QUOTE_MAX_LAG_MIN = 3.0   # a 1-min bar that ended longer ago than this is not a live price
# Request ids (historical bars, executions) start here on an order session, far above any order
# id TWS hands out, so a request error can never be read as a message on an order and vice versa
REQ_ID_BASE = 1_000_000_000
# After a rejecting message, how long wait_ack still waits for a working orderStatus before calling
# the order rejected (TWS can send a message just ahead of the status)
REJECT_GRACE_SEC = 1.0


def submission_enabled(cli_submit: bool) -> bool:
    """True only if config AUTO_SUBMIT_ENABLED is the bool True (identity, not truthy) and --submit."""
    return getattr(config, 'AUTO_SUBMIT_ENABLED', False) is True and cli_submit is True


@dataclass
class BrokerState:
    """What TWS reports, in the shapes order_ledger.reconcile and account_snapshot read."""
    positions: List[dict] = field(default_factory=list)
    open_orders: List[dict] = field(default_factory=list)
    executions: List[dict] = field(default_factory=list)
    completed_orders: List[dict] = field(default_factory=list)
    account_values: Dict[str, dict] = field(default_factory=dict)
    fetched_at: datetime = field(default_factory=now_et)
    # False if executions or completed orders timed out: reconcile then marks nothing
    # never_reached_tws (a missing order proves nothing)
    complete: bool = True


class Broker(Protocol):
    """IBKRBroker in production, tests/helpers.FakeBroker offline."""
    account: str
    submit_enabled: bool

    def state(self) -> BrokerState: ...
    def next_order_ids(self, n: int) -> List[int]: ...
    def place(self, order_id: int, contract, order) -> None: ...
    def modify(self, order_id: int, contract, order) -> None: ...
    def cancel(self, order_id: int) -> None: ...
    def wait_ack(self, order_id: int, timeout: float = 10.0) -> Tuple[str, str]: ...
    def last_price(self, symbol: str) -> Tuple[Optional[float], bool]: ...
    def daily_bars(self, symbol: str, duration: str = "6 M", include_forming: bool = False) -> pd.DataFrame: ...


def stock_contract(symbol: str):
    from ibapi.contract import Contract
    c = Contract()
    c.symbol, c.secType, c.exchange, c.currency = symbol.upper(), "STK", "SMART", "USD"
    return c


def _check_bot_order(order, account: str, prefix: str = BOT_REF_PREFIX) -> None:
    ref = str(getattr(order, 'orderRef', '') or '')
    if not ref.startswith(prefix):
        raise PermissionError(f"refusing order with orderRef {ref!r}: this session's orders must start with {prefix}")
    order.account = account


def assert_no_oversell(state: BrokerState, symbol: str, new_sell_qty: float,
                       account: Optional[str] = None, pending_buy_qty: float = 0.0,
                       oca_group: Optional[str] = None, pool_bot_groups: bool = False) -> None:
    """
    ValueError if working SELL orders for the symbol (bot and manual) plus new_sell_qty would exceed
    the long position: two sells both filling would leave the account short. For a new bracket the
    child stop only goes live when its parent fills, so pass the parent's quantity as
    pending_buy_qty (position + parent qty is what the child may cover).

    oca_group (exit_manager): the new SELL joins this bot OCA group (ocaType 3: a fill reduces the
    other members by the filled quantity). Working SELLs in the same group then count as ONE
    exposure, the largest remaining quantity among them and the new order, instead of a sum. So does
    every other BOT OCA group (BOT-OCA* names) when pool_bot_groups (exit_manager split_stop only:
    each group is one block of one trade, every member sized to it); manual SELLs and non-bot groups
    still add up. Default (shared_oca): other groups add up, as before. Without this the trade's resting trim LMT
    could never sit beside its full-size stop. Residual risk: ocaType 3 is "no block", so two
    members filling in the same instant can still overfill; fill_monitor's SHRINK / orphan checks
    are the backstop, and OCA behaviour must be verified on paper before any --submit run.
    """
    sym = symbol.upper()
    held = sum(float(p.get('shares') or 0) for p in state.positions
               if str(p.get('symbol', '')).upper() == sym and p.get('sec_type', 'STK') == 'STK'
               and (account is None or p.get('account') == account))
    # openOrder repeats on every status update: one copy per order
    seen, working, pooled = set(), 0.0, 0.0
    groups: Dict[str, float] = {}       # other bot OCA groups: largest remaining member each
    for o in state.open_orders:
        if (str(o.get('symbol', '')).upper() != sym or str(o.get('action', '')).upper() != 'SELL'
                or (account and o.get('account') not in (None, account))
                or (o.get('status') and o.get('status') not in WORKING)):
            continue
        ident = o.get('perm_id') or (o.get('client_id'), o.get('order_id'))
        if ident in seen:
            continue
        seen.add(ident)
        remaining = float(o.get('quantity') or 0) - float(o.get('filled') or 0)
        g = o.get('oca_group') or ''
        if oca_group and g == oca_group:
            pooled = max(pooled, remaining)     # one OCA group: its members reduce each other
        elif oca_group and pool_bot_groups and g.startswith(BOT_OCA_PREFIX):
            groups[g] = max(groups.get(g, 0.0), remaining)
        else:
            working += remaining
    working += sum(groups.values())
    exposure = working + (max(pooled, new_sell_qty) if oca_group else new_sell_qty)
    if exposure > max(held, 0.0) + pending_buy_qty + 1e-9:
        raise ValueError(f"oversell guard: {sym} working SELL {working + pooled:g} + new {new_sell_qty:g} "
                         f"> long position {held:g}"
                         + (f" (OCA {oca_group} counted once: {max(pooled, new_sell_qty):g})" if oca_group else ""))


class IBKRBroker:
    """A paper TWS session opened by open_session(). Order calls need submit_enabled."""

    def __init__(self, connector: IBKRConnector, account: str, submit_enabled: bool, client_id: int,
                 ref_prefix: str = BOT_REF_PREFIX):
        if ref_prefix not in REF_PREFIXES:
            raise PermissionError(f"orderRef prefix {ref_prefix!r} is not one of {REF_PREFIXES}")
        self.connector = connector
        self.account = account
        self.submit_enabled = submit_enabled
        self.client_id = client_id
        self.ref_prefix = ref_prefix
        self._next_id: Optional[int] = None
        lock = getattr(connector, '_req_id_lock', None)
        if lock is not None:
            with lock:
                connector.req_id = max(int(getattr(connector, 'req_id', 0) or 0), REQ_ID_BASE)

    def _require_submit(self, what: str) -> None:
        if not self.submit_enabled or self.connector.read_only:
            raise PermissionError(f"{what}: {SUBMIT_REFUSED}")

    def state(self, timeout: float = 10) -> BrokerState:
        """Positions, open orders (all clients), executions, completed orders, account values."""
        acct = self.connector.fetch_account_state(timeout=timeout)
        if acct is None:
            raise RuntimeError("TWS account state timed out; nothing can be planned or sent blind")
        execs = self.connector.fetch_executions(timeout=timeout)
        completed = self.connector.fetch_completed_orders(timeout=timeout)
        opens = [dict(o) for o in acct['open_orders']]
        # orderStatus carries fill counts that openOrder lacks (this client's orders only). Another
        # client's order (client 16's stop seen from a client-17 run) takes its fills from today's
        # executions by perm id, so a partly filled SELL isn't counted at its full size.
        exec_filled: Dict[int, float] = {}
        seen_exec = set()
        for e in execs or []:
            if e.get('exec_id') in seen_exec or not e.get('perm_id'):
                continue
            seen_exec.add(e.get('exec_id'))
            exec_filled[int(e['perm_id'])] = exec_filled.get(int(e['perm_id']), 0.0) + float(e.get('shares') or 0)
        for o in opens:
            st = self.connector.order_status.get(o.get('order_id')) if o.get('client_id') == self.client_id else None
            if st:
                o.setdefault('filled', st['filled'])
                o.setdefault('avg_fill_price', st['avg_fill_price'])
            elif o.get('perm_id') and int(o['perm_id']) in exec_filled:
                o.setdefault('filled', exec_filled[int(o['perm_id'])])
        return BrokerState(positions=acct['positions'], open_orders=opens, executions=execs or [],
                           completed_orders=completed or [], account_values=acct['account_values'],
                           fetched_at=now_et(), complete=execs is not None and completed is not None)

    def next_order_ids(self, n: int) -> List[int]:
        """n fresh order ids: max of TWS's nextValidId and what this session already handed out."""
        self.connector.next_order_id = None
        self.connector.reqIds(-1)
        start = time.monotonic()
        while self.connector.next_order_id is None and time.monotonic() - start < 5:
            time.sleep(0.05)
        if self.connector.next_order_id is None:
            raise RuntimeError("TWS sent no next valid order id")
        first = max(int(self.connector.next_order_id), self._next_id or 0)
        self._next_id = first + n
        return list(range(first, first + n))

    def place(self, order_id: int, contract, order) -> None:
        self._require_submit("place")
        _check_bot_order(order, self.account, self.ref_prefix)
        logger.info(f"placeOrder {order_id} {order.orderRef} {order.action} {order.totalQuantity} "
                    f"{order.orderType} aux={order.auxPrice} lmt={order.lmtPrice} transmit={order.transmit}")
        self.connector.placeOrder(order_id, contract, order)

    def modify(self, order_id: int, contract, order) -> None:
        """Same id, same client (IBKR rejects a modify from another client id)."""
        self._require_submit("modify")
        _check_bot_order(order, self.account, self.ref_prefix)
        self.connector.placeOrder(order_id, contract, order)

    def cancel(self, order_id: int) -> None:
        self._require_submit("cancel")
        from ibapi.order_cancel import OrderCancel
        self.connector.cancelOrder(order_id, OrderCancel())

    def wait_ack(self, order_id: int, timeout: float = 10.0, poll: float = 0.1) -> Tuple[str, str]:
        """
        ('working' | 'rejected' | 'unknown', reason) from orderStatus and the connector's order
        messages. A working orderStatus wins over any message (a 399 "not placed until 09:30" is
        routine for a pre-open entry); a rejecting message (connector.order_errors, any code size)
        counts once no working status arrives within REJECT_GRACE_SEC of it, or at once with a
        Cancelled / Inactive status.
        """
        start = time.monotonic()
        err_seen: Optional[float] = None
        while True:
            st = self.connector.order_status.get(order_id)
            if st and st['status'] in WORKING:
                return 'working', st['status']
            err = getattr(self.connector, 'order_errors', {}).get(order_id)
            if st and st['status'] in REJECTED:
                why = st['status'] + (f" ({st['why_held']})" if st.get('why_held') else '')
                return 'rejected', why + (f"; TWS {err.get('code')}: {err.get('msg')}" if err else '')
            if err:
                err_seen = err_seen if err_seen is not None else time.monotonic()
                if time.monotonic() - err_seen >= REJECT_GRACE_SEC:
                    return 'rejected', f"TWS {err.get('code')}: {err.get('msg')}"
            if time.monotonic() - start >= timeout:
                if err:
                    return 'rejected', f"TWS {err.get('code')}: {err.get('msg')}"
                return 'unknown', f"no order status within {timeout:g}s"
            time.sleep(poll)

    def daily_bars(self, symbol: str, duration: str = "6 M", include_forming: bool = False,
                   timeout: float = 30) -> pd.DataFrame:
        """
        Daily TRADES bars as a DataFrame (date, open, high, low, close, volume), oldest first; a
        read-only historical request. Today's bar is dropped while the session is still open unless
        include_forming (exit_manager --close-window needs today's intraday high). Empty DataFrame on
        failure: callers then change nothing.
        """
        cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        try:
            bars, status = self.connector.fetch_historical_data_ex(symbol, duration=duration, bar_size="1 day",
                                                                   timeout=timeout, retries=1)
        except Exception as exc:
            logger.warning(f"{symbol}: daily bars failed ({exc})")
            return pd.DataFrame(columns=cols)
        if not bars:
            logger.warning(f"{symbol}: no daily bars ({status})")
            return pd.DataFrame(columns=cols)
        df = pd.DataFrame({'date': parse_ibkr_times([b.date for b in bars]).dt.normalize().values,
                           'open': [float(b.open) for b in bars], 'high': [float(b.high) for b in bars],
                           'low': [float(b.low) for b in bars], 'close': [float(b.close) for b in bars],
                           'volume': [float(b.volume) for b in bars]})
        now = now_et()
        if not include_forming and in_regular_session(now):
            df = df[df['date'].dt.date < now.date()].reset_index(drop=True)
        return df

    def last_price(self, symbol: str, timeout: float = 15) -> Tuple[Optional[float], bool]:
        """
        (last trade price, delayed?) from the newest 1-min TRADES bar (read-only historical request,
        no market-data subscription needed). delayed is True when TWS flagged delayed data (2188),
        when the newest bar ends more than QUOTE_MAX_LAG_MIN minutes before now (stale, or outside
        the session), or when there is no bar at all (price None). Callers never act on a delayed price.
        """
        try:
            bars, status = self.connector.fetch_historical_data_ex(symbol, duration="1800 S", bar_size="1 min",
                                                                   timeout=timeout)
        except Exception as exc:
            logger.warning(f"{symbol}: quote failed ({exc})")
            return None, True
        if not bars:
            logger.warning(f"{symbol}: no intraday bars for a quote ({status})")
            return None, True
        last_start = parse_ibkr_times([bars[-1].date]).iloc[-1].to_pydatetime()
        lag = bar_lag_minutes(last_start, 1, now_et())
        delayed = bool(getattr(self.connector, 'data_delayed', False)) or lag > QUOTE_MAX_LAG_MIN
        return float(bars[-1].close), delayed

    def close(self) -> None:
        try:
            self.connector.disconnect_and_stop()
        except Exception:
            pass


def open_session(port: int, account: str, cli_submit: bool, client_id: int,
                 host: Optional[str] = None, connector_factory=IBKRConnector,
                 ref_prefix: str = BOT_REF_PREFIX) -> IBKRBroker:
    """Paper guard (see module docstring), in order; raises PermissionError on any failure.
    ref_prefix: the only orderRef prefix this session may place (BOT- or PAPERCHECK-)."""
    if ref_prefix not in REF_PREFIXES:
        raise PermissionError(f"orderRef prefix {ref_prefix!r} is not one of {REF_PREFIXES}")
    if port in LIVE_PORTS:
        raise PermissionError(f"Port {port} is a LIVE port {sorted(LIVE_PORTS)}: the bot never connects to it")
    if port not in PAPER_PORTS:
        raise PermissionError(f"Port {port} is not a paper port {sorted(PAPER_PORTS)}")
    if not account or not is_paper_account(account):
        raise PermissionError(f"Account {account!r} is not a paper (DU...) account; --account DU... is required")
    account = account.strip().upper()
    submit = submission_enabled(cli_submit)
    if cli_submit and not submit:
        logger.warning(f"--submit given but config AUTO_SUBMIT_ENABLED is not True: {SUBMIT_REFUSED}")
    connector = connector_factory(read_only=not submit)
    if not connector.connect_and_run(host or getattr(config, 'TWS_HOST', '127.0.0.1'), port, client_id):
        raise ConnectionError(f"could not connect to TWS on port {port} (client {client_id})")
    try:
        accounts = confirm_paper_session(connector, port)
        if account not in accounts:
            raise PermissionError(f"Account {account} is not in this TWS session {accounts}")
    except Exception:
        try:
            connector.disconnect_and_stop()
        except Exception:
            pass
        raise
    logger.info(f"Paper session {account} on port {port}, client {client_id}, "
                f"{'SUBMIT ENABLED' if submit else 'read-only'}")
    return IBKRBroker(connector, account, submit, client_id, ref_prefix=ref_prefix)
