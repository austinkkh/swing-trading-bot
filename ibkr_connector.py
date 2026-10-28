"""
═══════════════════════════════════════════════════════════════════════════════
IBKR TWS CONNECTION MANAGER
Handles connection, disconnection, and error handling
═══════════════════════════════════════════════════════════════════════════════
"""

import time
import logging
from datetime import datetime
from threading import Lock, Thread
from typing import Dict, Iterable, Optional, Sequence, Tuple
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract

logger = logging.getLogger(__name__)

# orderRef on the staged 1-share price-alert orders (ibkr_alert_setter): not entries
ALERT_ORDER_REF = "ALERT"

# Paper-trading guard, shared by every tool that talks to TWS / IB Gateway.
PAPER_PORTS = {7497, 4002}      # TWS paper, Gateway paper
LIVE_PORTS = {7496, 4001}       # TWS live, Gateway live


def is_paper_account(account: str) -> bool:
    return str(account).strip().upper().startswith("DU")


def confirm_paper_session(connector, port: int, timeout: int = 8) -> list:
    """
    Raise PermissionError unless this session is paper-only: a paper port, at least one account,
    and every account the session reports is a paper (DU...) account. Returns the accounts.
    """
    if port not in PAPER_PORTS:
        raise PermissionError(f"Port {port} is not a paper port {sorted(PAPER_PORTS)}")
    accounts = [a.strip().upper() for a in connector.fetch_managed_accounts(timeout=timeout) if str(a).strip()]
    if not accounts:
        raise PermissionError("TWS reported no accounts; can't confirm paper trading")
    live = [a for a in accounts if not is_paper_account(a)]
    if live:
        raise PermissionError(f"Non-paper account(s) in this session: {live}")
    return accounts

# ─── TWS link / data-farm status codes (TWS sends these on change and at connect) ───
FARM_BROKEN_CODES = {2103, 2105, 2157}        # market data / HMDS / sec-def farm broken
FARM_OK_CODES = {2104, 2106, 2107, 2158}      # OK; 2107 = HMDS inactive but available on demand
LINK_DOWN_CODES = {1100, 2110}                # TWS <-> IB server link lost
LINK_UP_CODES = {1101, 1102}                  # link restored (data lost / maintained)
HMDS_FARM = 'ushmds'                          # the historical-data farm every scan depends on
# 2188: this session has no real-time API market data, so historical bars come ~15 min delayed
# (every reqHistoricalData gets one). The request still completes with bars.
DELAYED_DATA_CODE = 2188
# Messages TWS sends on an order id that do NOT reject the order (it stays live): 399 order message
# ("will not be placed at the exchange until 09:30"), 404 held for a short locate, 161 / 10147 /
# 10148 answers to a cancel request, 10311 direct-routing notice, 10349 TIF set by a preset. The
# 2100-2199 range is TWS warnings. Every other code on an order id is recorded as a rejection,
# including codes >= 1000 (e.g. 10268 unsupported order attribute).
ORDER_WARNING_CODES = {161, 399, 404, 10147, 10148, 10311, 10349}
# Gaps between retries of one historical request: >= 15s avoids IB's identical-request pacing rule
DEFAULT_BACKOFF = (15, 30, 60)

# Indirection so tests can drive the farm wait with a fake clock. time.monotonic doesn't advance
# while macOS sleeps, so a wait budget counts awake time only (a sleeping laptop can't burn it).
_monotonic = time.monotonic
_sleep = time.sleep


def farm_name(msg: str) -> str:
    """Farm id from a TWS farm message: 'HMDS data farm connection is OK:ushmds' -> 'ushmds'.
    Farm ids can contain dots (usfarm.nj), so split on ':' first; 2107 has no colon
    ('...available upon demand.ushmds'), so fall back to the last '.'."""
    msg = str(msg or '').strip()
    if ':' in msg:
        return msg.rsplit(':', 1)[1].strip()
    return msg.rsplit('.', 1)[-1].strip()


# Keep IBKR API internals quiet unless a caller explicitly opts into DEBUG.
# ibapi logs every wire message (price bars included), which flooded logs.
for _name in ("ibapi", "ibapi.client", "ibapi.connection", "ibapi.reader", "ibapi.wrapper"):
    logging.getLogger(_name).setLevel(logging.WARNING)

class IBKRConnector(EWrapper, EClient):
    """
    Synchronous wrapper for IBKR API
    Handles connection and basic requests
    """
    
    def __init__(self, read_only: bool = True, allow_live: bool = False):
        """
        read_only: order calls (placeOrder, cancelOrder, reqGlobalCancel) raise PermissionError.
                   Only code that is meant to stage orders passes read_only=False.
        allow_live: connecting to a non-paper port is refused unless this is True. Nothing in
                    the repo sets it; going live is a deliberate, explicit change.
        """
        EClient.__init__(self, self)
        self.read_only = read_only
        self.allow_live = allow_live
        
        # Connection state
        self.connected = False
        self.next_order_id = None
        
        # Data storage
        self.historical_data = {}
        self.historical_done = set()      # reqIds whose historicalDataEnd has arrived
        self.contract_details = {}
        self.fundamental_data = {}
        self.market_data = {}
        self.errors = {}
        self.managed_accounts = None
        # Read-only account state (positions, open orders, account values)
        self.positions = []
        self.positions_done = False
        self.open_orders = []
        self.open_orders_done = False
        self.account_values = {}
        self.account_summary_done = set()
        self.pnl_values = {}             # account -> {'daily', 'unrealized', 'realized'} (reqPnL)
        # Order lifecycle, read-only (order_broker / order_ledger reconcile). Nothing here sends orders.
        self.order_status: Dict[int, dict] = {}   # orderId -> latest orderStatus (this client's orders)
        # Order ids this session placed (or cancelled), and TWS messages on them. Kept apart from `errors` (request
        # ids): a code-399 warning must not read as a rejection, and request id 7 is not order 7.
        self.placed_order_ids: set = set()
        self.order_errors: Dict[int, dict] = {}    # orderId -> rejecting message (latest)
        self.order_messages: Dict[int, list] = {}  # orderId -> every message, warnings included
        self.executions: list = []
        self.executions_done = set()              # reqIds whose execDetailsEnd has arrived
        self.completed_orders: list = []
        self.completed_orders_done = False
        self._pnl_req = {}               # reqId -> account
        # Link / data-farm state from TWS status messages (see _update_link_state)
        self.farm_status: Dict[str, bool] = {}           # farm id -> usable
        self.farm_broken_since: Dict[str, datetime] = {}  # farm id -> when it went broken
        self.server_link_ok = True                       # False after 2110 / 1100 until restored
        self.link_down_since: Optional[datetime] = None
        # Timed-out reqIds whose late replies are dropped. Insertion-ordered dict used as a
        # bounded set: the watcher runs all day and only recent ids can still get a reply
        self._cancelled_hist: Dict[int, None] = {}
        self.farm_wait_sec = 600                         # farm wait between retries (callers may lower)
        # Optional shared budget (seconds) for ALL farm waits on this connector: each wait is capped
        # by what is left and deducts what it used, so one run can't wait 600s per retry (review M1)
        self.farm_wait_budget: Optional[float] = None
        self.data_delayed = False                        # True once TWS sent 2188 (delayed data)
        # Drop today's still-forming daily bar during the regular session (see _drop_forming_bar).
        # The 'afterhours' scan sets it: its rules read the last bar as a completed session.
        self.complete_daily_bars_only = False
        
        # Request tracking
        self.req_id = 0
        self._req_id_lock = Lock()   # the scanner calls this from many threads at once (review S4)

    def get_next_req_id(self):
        """Unique request ID, safe across threads (an unlocked += could hand two requests one id)."""
        with self._req_id_lock:
            self.req_id += 1
            return self.req_id
    
    def connect_and_run(self, host, port, client_id):
        """
        Connect to TWS and start message processing thread
        """
        if port not in PAPER_PORTS and not self.allow_live:
            logger.error(f"Refusing to connect to port {port}: not a paper port {sorted(PAPER_PORTS)} "
                         f"(live ports {sorted(LIVE_PORTS)} need allow_live=True)")
            return False
        try:
            self.connect(host, port, client_id)
            
            # Start message processing thread
            api_thread = Thread(target=self.run, daemon=True)
            api_thread.start()
            
            # Wait for connection
            timeout = time.time() + 10
            while not self.connected and time.time() < timeout:
                time.sleep(0.1)
            
            if not self.connected:
                logger.error("Failed to connect to TWS")
                return False
            
            # Extra delay to ensure server version is received
            time.sleep(3)
            
            # Verify server version is set
            if self.serverVersion() is None:
                logger.warning("Server version not received, waiting longer...")
                time.sleep(3)
            
            logger.info(f"✓ Connected to TWS at {host}:{port}")
            if self.serverVersion():
                logger.info(f"  Server version: {self.serverVersion()}")
            return True
            
        except Exception as e:
            logger.error(f"Connection error: {e}")
            return False
    
    # ─── Order calls are blocked on read-only connectors ───
    def placeOrder(self, *args, **kwargs):
        if self.read_only:
            raise PermissionError("placeOrder on a read-only IBKRConnector")
        if args:
            self.placed_order_ids.add(args[0])
        return EClient.placeOrder(self, *args, **kwargs)

    def cancelOrder(self, *args, **kwargs):
        if self.read_only:
            raise PermissionError("cancelOrder on a read-only IBKRConnector")
        if args:
            self.placed_order_ids.add(args[0])    # its 10147 / 202 answers are order messages
        return EClient.cancelOrder(self, *args, **kwargs)

    def reqGlobalCancel(self, *args, **kwargs):
        if self.read_only:
            raise PermissionError("reqGlobalCancel on a read-only IBKRConnector")
        return EClient.reqGlobalCancel(self, *args, **kwargs)

    def disconnect_and_stop(self):
        """Clean disconnect"""
        self.disconnect()
        logger.info("Disconnected from TWS")
    
    def nextValidId(self, orderId: int):
        """Called when connection is established"""
        self.next_order_id = orderId
        self.connected = True
        logger.debug(f"Next valid order ID: {orderId}")
    
    def error(self, *args, **kwargs):
        """Error handler - ultra flexible to handle any API version"""
        # Extract parameters based on what's provided
        reqId = args[0] if len(args) > 0 else -1
        # ibapi 10.x inserts errorTime (epoch ms) before the code: (reqId, errorTime, code, msg, ...).
        # Older versions send (reqId, code, msg). Reading args[1] as the code on 10.x made every
        # error look like "Warning <timestamp>", so real request errors were never recorded.
        if len(args) >= 4 and isinstance(args[1], int) and args[1] > 10**9:
            errorCode, errorString = args[2], args[3]
        else:
            errorCode = args[1] if len(args) > 1 else 0
            errorString = args[2] if len(args) > 2 else "Unknown error"
        
        if self._update_link_state(errorCode, errorString):
            return
        if errorCode == DELAYED_DATA_CODE:
            # One line per connector, not one per request (26 per watcher cycle before)
            if not self.data_delayed:
                logger.warning("Historical data is delayed (~15 min): no real-time API market data "
                               f"subscription on this session (TWS {errorCode}: {errorString})")
            else:
                logger.debug(f"Delayed data ({errorCode}) for reqId {reqId}")
            self.data_delayed = True
            return
        if reqId in self._cancelled_hist:
            return   # we cancelled this request after a timeout; its error is old news
        if reqId in self.placed_order_ids or reqId in self.order_status:
            self._record_order_message(reqId, errorCode, errorString)
            return
        if errorCode < 1000:
            # Codes under 1000 end the request (e.g. 162 no data, 200 no such symbol);
            # recording them lets fetches stop waiting instead of hitting the timeout.
            logger.error(f"Error {errorCode}: {errorString}")
            self.errors[reqId] = {'code': errorCode, 'msg': errorString}
        else:
            logger.warning(f"Warning {errorCode}: {errorString}")
    
    def _record_order_message(self, order_id: int, code: int, msg: str) -> None:
        """A TWS message on one of this session's orders: warnings are logged, anything else is a
        rejection that order_broker.wait_ack reads (whatever the code's size)."""
        entry = {'code': code, 'msg': msg, 'ts': time.monotonic()}
        self.order_messages.setdefault(order_id, []).append(entry)
        if code in ORDER_WARNING_CODES or 2100 <= code < 2200:
            logger.warning(f"Order {order_id} message {code}: {msg}")
        else:
            logger.error(f"Order {order_id} error {code}: {msg}")
            self.order_errors[order_id] = entry

    def _update_link_state(self, code: int, msg: str) -> bool:
        """
        Track the TWS server link and data farms from status codes. Returns True when the code
        was a link/farm status message (handled here, nothing else to do). Pure state: no TWS needed.
        """
        now = datetime.now()
        if code in FARM_BROKEN_CODES:
            farm = farm_name(msg)
            if self.farm_status.get(farm) is not False:
                self.farm_broken_since[farm] = now
            self.farm_status[farm] = False
            logger.warning(f"Data farm {farm} broken ({code}): {msg}")
            return True
        if code in FARM_OK_CODES:
            farm = farm_name(msg)
            was_broken = self.farm_status.get(farm) is False
            self.farm_status[farm] = True
            self.farm_broken_since.pop(farm, None)
            # TWS often sends no 1102 after a 2110: a farm reporting OK means the link is back
            if not self.server_link_ok:
                self._set_link(True, code)
            (logger.info if was_broken else logger.debug)(f"Data farm {farm} OK ({code})")
            return True
        if code in LINK_DOWN_CODES:
            self._set_link(False, code, msg)
            return True
        if code in LINK_UP_CODES:
            self._set_link(True, code, msg)
            return True
        return False

    def _set_link(self, ok: bool, code: int, msg: str = '') -> None:
        if ok and not self.server_link_ok:
            logger.info(f"TWS server link restored ({code})")
        elif not ok:
            if self.server_link_ok:
                self.link_down_since = datetime.now()
            logger.warning(f"TWS server link down ({code}): {msg}")
        self.server_link_ok = ok
        if ok:
            self.link_down_since = None

    def data_farms_ready(self, required: Iterable[str] = (HMDS_FARM,)) -> bool:
        """Link up and no required farm reported broken. A farm TWS never mentioned counts as OK:
        TWS only reports farms on a change or at connect."""
        return self.server_link_ok and not any(self._broken_matching(f) for f in required)

    def _broken_matching(self, required: str) -> list:
        """Broken farm ids that are `required` or a variant of it ('ushmds.nj' for 'ushmds'):
        TWS has reported dotted ids, and an exact match would read a broken variant as OK (review L3)."""
        return [f for f, ok in self.farm_status.items()
                if ok is False and (f == required or f.startswith(required + '.'))]

    def farm_status_summary(self, required: Iterable[str] = (HMDS_FARM,)) -> str:
        """One-line status for logs, e.g. 'ushmds broken since 09:26:39; server link down since 09:26:39'."""
        parts = []
        for req in required:
            for f in self._broken_matching(req):
                since = self.farm_broken_since.get(f)
                parts.append(f"{f} broken" + (f" since {since:%H:%M:%S}" if since else ""))
        if not self.server_link_ok:
            parts.append("server link down" + (f" since {self.link_down_since:%H:%M:%S}"
                                               if self.link_down_since else ""))
        return "; ".join(parts) or "OK"

    def wait_for_data_farms(self, required: Sequence[str] = (HMDS_FARM,), max_wait: float = 600,
                            poll: float = 1.0, progress_every: float = 30) -> bool:
        """
        Block until data_farms_ready(required), up to max_wait seconds of awake time (and no more
        than farm_wait_budget, when set). Returns True at once when nothing is broken. On failure
        logs one ERROR line and returns False.
        """
        if self.farm_wait_budget is not None:
            max_wait = min(max_wait, max(0.0, self.farm_wait_budget))
        start = _monotonic()
        next_progress = progress_every
        try:
            while not self.data_farms_ready(required):
                waited = _monotonic() - start
                if waited >= max_wait:
                    logger.error(f"IBKR data farm ({', '.join(required)}) not OK after {max_wait:.0f}s; "
                                 f"last status: {self.farm_status_summary(required)}")
                    return False
                if waited >= next_progress:
                    logger.info(f"Waiting for {', '.join(required)} ({self.farm_status_summary(required)}, "
                                f"{waited:.0f}s/{max_wait:.0f}s)")
                    next_progress += progress_every
                _sleep(poll)
            return True
        finally:
            if self.farm_wait_budget is not None:
                self.farm_wait_budget = max(0.0, self.farm_wait_budget - (_monotonic() - start))

    def historicalData(self, reqId, bar):
        """Receive historical bar data"""
        if reqId in self._cancelled_hist:
            return   # late reply to a request we cancelled after a timeout
        if reqId not in self.historical_data:
            self.historical_data[reqId] = []
        self.historical_data[reqId].append(bar)
    
    def historicalDataEnd(self, reqId, start, end):
        """Historical data complete"""
        if reqId in self._cancelled_hist:
            return
        self.historical_done.add(reqId)
        logger.debug(f"Historical data complete for reqId {reqId}")
    
    def contractDetails(self, reqId, contractDetails):
        """Receive contract details"""
        if reqId not in self.contract_details:
            self.contract_details[reqId] = []
        self.contract_details[reqId].append(contractDetails)
    
    def contractDetailsEnd(self, reqId):
        """Contract details complete"""
        logger.debug(f"Contract details complete for reqId {reqId}")
    
    def fundamentalData(self, reqId, data):
        """Receive fundamental data"""
        self.fundamental_data[reqId] = data

    def managedAccounts(self, accountsList: str):
        """Receive comma-delimited managed account ids."""
        accounts = [acct.strip() for acct in str(accountsList or "").split(",") if acct.strip()]
        self.managed_accounts = accounts
    
    # ─── Read-only account state ───
    def position(self, account, contract, position, avgCost):
        self.positions.append({'account': account, 'symbol': contract.symbol, 'sec_type': contract.secType,
                               'con_id': contract.conId,
                               'shares': float(position), 'avg_cost': float(avgCost)})

    def positionEnd(self):
        self.positions_done = True

    def openOrder(self, orderId, contract, order, orderState):
        self.open_orders.append({
            'order_id': orderId, 'perm_id': order.permId, 'client_id': order.clientId,
            'con_id': contract.conId, 'account': order.account, 'symbol': contract.symbol,
            'sec_type': contract.secType, 'action': order.action, 'order_type': order.orderType,
            'quantity': float(order.totalQuantity), 'aux_price': order.auxPrice,
            'lmt_price': order.lmtPrice, 'trail_stop_price': order.trailStopPrice,
            'status': getattr(orderState, 'status', ''),
            'order_ref': getattr(order, 'orderRef', '') or '',
            # Bracket / OCA fields for the order ledger (reconcile, incomplete-bracket check)
            'parent_id': getattr(order, 'parentId', 0) or 0, 'tif': getattr(order, 'tif', '') or '',
            'oca_group': getattr(order, 'ocaGroup', '') or '', 'transmit': getattr(order, 'transmit', True),
        })

    def openOrderEnd(self):
        self.open_orders_done = True

    def orderStatus(self, orderId, status, filled, remaining, avgFillPrice, permId, parentId,
                    lastFillPrice, clientId, whyHeld, mktCapPrice=0.0):
        """Status updates for orders this client can see (repeats on every change; latest wins)."""
        self.order_status[orderId] = {
            'order_id': orderId, 'status': status, 'filled': float(filled), 'remaining': float(remaining),
            'avg_fill_price': float(avgFillPrice) if avgFillPrice else None, 'perm_id': permId,
            'parent_id': parentId, 'client_id': clientId, 'why_held': whyHeld or '',
        }

    def execDetails(self, reqId, contract, execution):
        self.executions.append({
            'exec_id': execution.execId, 'order_id': execution.orderId, 'perm_id': execution.permId,
            'client_id': execution.clientId, 'order_ref': getattr(execution, 'orderRef', '') or '',
            'symbol': contract.symbol, 'sec_type': contract.secType, 'account': execution.acctNumber,
            'side': execution.side, 'shares': float(execution.shares), 'price': float(execution.price),
            'time': execution.time,
        })

    def execDetailsEnd(self, reqId):
        self.executions_done.add(reqId)

    def completedOrder(self, contract, order, orderState):
        # ibapi marks an unset filledQuantity with a huge sentinel
        filled = float(getattr(order, 'filledQuantity', 0) or 0)
        self.completed_orders.append({
            'order_id': order.orderId, 'perm_id': order.permId, 'client_id': order.clientId,
            'con_id': contract.conId, 'account': order.account, 'symbol': contract.symbol,
            'sec_type': contract.secType, 'action': order.action, 'order_type': order.orderType,
            'quantity': float(order.totalQuantity), 'aux_price': order.auxPrice, 'lmt_price': order.lmtPrice,
            'filled': filled if filled < 1e300 else 0.0,
            'status': getattr(orderState, 'status', ''),
            'completed_status': getattr(orderState, 'completedStatus', '') or '',
            'order_ref': getattr(order, 'orderRef', '') or '',
            'parent_id': getattr(order, 'parentId', 0) or 0, 'tif': getattr(order, 'tif', '') or '',
            'oca_group': getattr(order, 'ocaGroup', '') or '',
        })

    def completedOrdersEnd(self):
        self.completed_orders_done = True

    def fetch_executions(self, timeout: float = 10) -> Optional[list]:
        """Today's executions (TWS returns the current day only). Read-only; None on timeout."""
        from ibapi.execution import ExecutionFilter
        self.executions = []
        req_id = self.get_next_req_id()
        try:
            self.reqExecutions(req_id, ExecutionFilter())
        except Exception as e:
            logger.warning(f"Error requesting executions: {e}")
            return None
        if not self._wait(lambda: req_id in self.executions_done, timeout):
            logger.warning("Timeout fetching executions")
            return None
        return list(self.executions)

    def fetch_completed_orders(self, timeout: float = 10) -> Optional[list]:
        """Filled / cancelled / expired orders TWS still lists (API orders only). Read-only; None on timeout."""
        self.completed_orders, self.completed_orders_done = [], False
        try:
            self.reqCompletedOrders(True)
        except Exception as e:
            logger.warning(f"Error requesting completed orders: {e}")
            return None
        if not self._wait(lambda: self.completed_orders_done, timeout):
            logger.warning("Timeout fetching completed orders")
            return None
        return list(self.completed_orders)

    def accountSummary(self, reqId, account, tag, value, currency):
        self.account_values.setdefault(account, {})[tag] = value

    def accountSummaryEnd(self, reqId):
        self.account_summary_done.add(reqId)

    def pnl(self, reqId, dailyPnL, unrealizedPnL, realizedPnL):
        """Account P&L (reqPnL). ibapi marks unset values with sys.float_info.max."""
        account = self._pnl_req.get(reqId)
        if account is None:
            return
        clean = lambda v: None if v is None or abs(float(v)) >= 1e300 else float(v)
        self.pnl_values[account] = {'daily': clean(dailyPnL), 'unrealized': clean(unrealizedPnL),
                                    'realized': clean(realizedPnL)}

    def fetch_pnl(self, accounts, timeout: float = 3) -> dict:
        """Optional read-only P&L per account. Missing on timeout (never a failure)."""
        self.pnl_values = {}
        reqs = {}
        try:
            for account in accounts:
                rid = self.get_next_req_id()
                self._pnl_req[rid] = account
                reqs[rid] = account
                self.reqPnL(rid, account, "")
            self._wait(lambda: all(a in self.pnl_values for a in reqs.values()), timeout)
        except Exception as e:
            logger.debug(f"P&L request failed: {e}")
        finally:
            for rid in reqs:
                try:
                    self.cancelPnL(rid)
                except Exception:
                    pass
                self._pnl_req.pop(rid, None)
        return dict(self.pnl_values)

    def _wait(self, done, timeout):
        start = time.monotonic()
        while not done() and time.monotonic() - start < timeout:
            time.sleep(0.05)
        return done()

    def fetch_account_state(self, timeout=10):
        """
        Positions, all open orders (any client or TWS itself) and NetLiquidation per account.
        Read-only: only data requests, never an order call. Returns None if any part times out,
        so callers can't mistake a partial answer for "no positions".
        """
        self.positions, self.positions_done = [], False
        self.open_orders, self.open_orders_done = [], False
        self.account_values = {}
        req_id = self.get_next_req_id()
        try:
            self.reqPositions()
            self.reqAllOpenOrders()
            # TotalCashValue: lets a snapshot tell a cash move from P&L (account_snapshot)
            self.reqAccountSummary(req_id, "All", "NetLiquidation,TotalCashValue")
            ok = (self._wait(lambda: self.positions_done, timeout)
                  and self._wait(lambda: self.open_orders_done, timeout)
                  and self._wait(lambda: req_id in self.account_summary_done, timeout))
        except Exception as e:
            logger.warning(f"Error fetching account state: {e}")
            ok = False
        finally:
            for cancel in (self.cancelPositions, lambda: self.cancelAccountSummary(req_id)):
                try:
                    cancel()
                except Exception:
                    pass
        if not ok:
            logger.warning("Timeout fetching account state")
            return None
        state = {'positions': list(self.positions), 'open_orders': list(self.open_orders),
                 'account_values': dict(self.account_values)}
        state['pnl'] = self.fetch_pnl(sorted(state['account_values']))
        return state

    def get_stock_contract(self, symbol, exchange="SMART"):
        """Create stock contract object"""
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "STK"
        contract.currency = "USD"
        contract.exchange = exchange
        return contract
    
    def wait_for_data(self, req_id, data_dict, timeout=30):
        """
        Wait for specific request to complete
        Returns True if data received, False if timeout
        """
        start = time.monotonic()
        while req_id not in data_dict and (time.monotonic() - start) < timeout:
            time.sleep(0.1)
        return req_id in data_dict
    
    def fetch_historical_data(self, symbol, duration="1 Y", bar_size="1 day",
                              what_to_show="TRADES", timeout=30, retries: int = 0,
                              backoff: Sequence[float] = DEFAULT_BACKOFF):
        """
        Fetch historical data for a symbol
        Returns list of bar data or None if failed (fetch_historical_data_ex also says why)
        """
        return self.fetch_historical_data_ex(symbol, duration, bar_size, what_to_show,
                                             timeout, retries, backoff)[0]

    def fetch_historical_data_ex(self, symbol, duration="1 Y", bar_size="1 day",
                                 what_to_show="TRADES", timeout=30, retries: int = 0,
                                 backoff: Sequence[float] = DEFAULT_BACKOFF) -> Tuple[Optional[list], str]:
        """
        (bars or None, status). status: 'ok', 'no_secdef' (error 200: IBKR doesn't know the
        ticker), 'no_data' (162 / empty), 'pacing', 'session_conflict' (162: the trading
        session is connected from another IP, so no request gets data), 'error:<code>', 'timeout'.

        retries: extra attempts after a timeout or a pacing violation (default 0: the scan's
        per-stock calls keep today's speed). Critical single fetches (SPY, QQQ) pass retries=3.
        Hard request errors (162 no data, 200 no such contract) are never retried. Between
        attempts it waits backoff[i] seconds, then for the HMDS farm (returns at once if OK); if
        the farm is still down after that wait, it stops retrying.
        """
        status = 'timeout'
        for attempt in range(retries + 1):
            bars, status = self._fetch_historical_once(symbol, duration, bar_size, what_to_show, timeout)
            if bars:
                return bars, status
            if status not in ('timeout', 'pacing') or attempt == retries:
                return None, status
            delay = backoff[min(attempt, len(backoff) - 1)] if backoff else 0
            logger.warning(f"{symbol} historical fetch {status} (attempt {attempt + 1}/{retries + 1}); "
                           f"retrying in {delay:.0f}s [{self.farm_status_summary()}]")
            _sleep(delay)
            if not self.wait_for_data_farms(max_wait=self.farm_wait_sec):
                # Farm still down after the wait (or the run's wait budget is spent): more
                # attempts would only time out too
                return None, status
        return None, status

    def _fetch_historical_once(self, symbol, duration, bar_size, what_to_show,
                               timeout) -> Tuple[Optional[list], str]:
        """One reqHistoricalData. Returns (bars or None, status) with status one of
        'ok', 'no_secdef' (200), 'no_data' (162 / empty), 'pacing', 'error:<code>', 'timeout'."""
        # Extra safety check
        if self.serverVersion() is None:
            logger.error("Server version not available, cannot fetch data")
            return None, 'error:no_server_version'
        
        req_id = self.get_next_req_id()
        contract = self.get_stock_contract(symbol)
        
        try:
            # Request data
            self.reqHistoricalData(
                req_id, contract, "", duration, bar_size,
                what_to_show, 1, 1, False, []
            )
            
            # Wait for historicalDataEnd, not the first bar: bars arrive one callback at a time
            # on the reader thread, so returning early hands back a list that is still filling.
            # Monotonic clock: a wall-clock jump after a laptop sleep would end the wait early.
            start = time.monotonic()
            while (req_id not in self.historical_done and req_id not in self.errors
                   and time.monotonic() - start < timeout):
                time.sleep(0.05)
            done = req_id in self.historical_done
            error = self.errors.pop(req_id, None)
            if not done and not error:
                # Cancel so TWS stops working on it; a late reply is then dropped on arrival
                self._mark_cancelled(req_id)
                try:
                    self.cancelHistoricalData(req_id)
                except Exception:
                    pass
            bars = self.historical_data.pop(req_id, None)  # free memory: the watcher runs all day
            self.historical_done.discard(req_id)
            if done and bars and self.complete_daily_bars_only and bar_size == "1 day":
                bars = self._drop_forming_bar(bars, symbol)
            if done and bars:
                return bars, 'ok'
            if error:
                logger.debug(f"No historical data for {symbol}: {error['code']} {error['msg']}")
                return None, self._error_status(error)
            if not done:
                logger.warning(f"Timeout fetching historical data for {symbol} "
                               f"({len(bars) if bars else 0} partial bars discarded)")
                return None, 'timeout'
            return None, 'no_data'
        except Exception as e:
            logger.warning(f"Error fetching data for {symbol}: {e}")
            return None, 'error:exception'

    @staticmethod
    def _drop_forming_bar(bars: list, symbol: str = '', now: Optional[datetime] = None) -> list:
        """Without an end time, reqHistoricalData's last daily bar during the session is today's
        partial bar, so 'yesterday's low', swing lows, pivot distance and Price would be
        intraday values (audit #5). Drop it while the session is open; a no-op premarket."""
        from market_time import in_regular_session, now_et
        now = now or now_et()
        if not bars or not in_regular_session(now):
            return bars
        if str(getattr(bars[-1], 'date', ''))[:8] == now.strftime('%Y%m%d'):
            logger.debug(f"{symbol}: dropped today's forming daily bar")
            return bars[:-1]
        return bars

    CANCELLED_KEEP = 5_000      # recent cancelled reqIds remembered (late replies arrive in seconds)

    def _mark_cancelled(self, req_id: int) -> None:
        """Remember a timed-out request so its late bars / error are dropped; oldest ids age out."""
        self._cancelled_hist[req_id] = None
        while len(self._cancelled_hist) > self.CANCELLED_KEEP:
            self._cancelled_hist.pop(next(iter(self._cancelled_hist)))

    @staticmethod
    def _error_status(error: dict) -> str:
        code, msg = error.get('code'), str(error.get('msg', '')).lower()
        if code == 162 and 'pacing' in msg:
            return 'pacing'          # pacing violation: retryable after a pause
        if code == 162 and 'different ip' in msg:
            # 'Trading TWS session is connected from a different IP address': the whole session
            # gets no data. Not a per-symbol "no data" answer (2026-10-26 watcher log).
            return 'session_conflict'
        if code == 200:
            return 'no_secdef'       # IBKR doesn't know the ticker (delisted / renamed)
        if code == 162:
            return 'no_data'
        return f'error:{code}'
    
    def fetch_contract_details(self, symbol, timeout=10):
        """
        Fetch contract details (includes fundamental info)
        Returns ContractDetails or None
        """
        req_id = self.get_next_req_id()
        contract = self.get_stock_contract(symbol)
        
        try:
            self.reqContractDetails(req_id, contract)
            # Stop on the first detail OR a request error (Error 200 for a dead ticker), so the
            # weekly quarantine re-probe doesn't sit out the full timeout per symbol (review L1)
            start = time.monotonic()
            while (req_id not in self.contract_details and req_id not in self.errors
                   and time.monotonic() - start < timeout):
                time.sleep(0.05)
            details = self.contract_details.get(req_id)
            error = self.errors.get(req_id)
            if details:
                return details[0]
            if error:
                logger.debug(f"No contract details for {symbol}: {error['code']} {error['msg']}")
                return None
            logger.warning(f"Timeout fetching contract details for {symbol}")
            return None
        except Exception as e:
            logger.warning(f"Error fetching contract details for {symbol}: {e}")
            return None
        finally:
            # Free both maps: the watcher and the re-probe run for a long time
            self.errors.pop(req_id, None)
            self.contract_details.pop(req_id, None)

    def fetch_fundamental_data(self, symbol, report_type="ReportSnapshot", timeout=10):
        """
        Fetch fundamental XML data for a symbol.
        Common report types include: ReportSnapshot, ReportsFinSummary, ReportRatios.
        Returns raw XML string or None.
        """
        req_id = self.get_next_req_id()
        contract = self.get_stock_contract(symbol)

        try:
            self.reqFundamentalData(req_id, contract, report_type, [])

            if self.wait_for_data(req_id, self.fundamental_data, timeout):
                return self.fundamental_data.get(req_id)

            logger.warning(f"Timeout fetching {report_type} fundamental data for {symbol}")
            return None
        except Exception as e:
            logger.warning(f"Error fetching {report_type} fundamental data for {symbol}: {e}")
            return None

    def fetch_managed_accounts(self, timeout=10):
        """
        Fetch managed accounts visible to this API session.
        Returns list of account ids or empty list on timeout/error.
        """
        self.managed_accounts = None
        try:
            self.reqManagedAccts()
        except Exception as e:
            logger.warning(f"Error requesting managed accounts: {e}")
            return []

        start = time.monotonic()
        while self.managed_accounts is None and (time.monotonic() - start) < timeout:
            time.sleep(0.1)

        if self.managed_accounts is None:
            logger.warning("Timeout fetching managed accounts")
            return []
        return self.managed_accounts
