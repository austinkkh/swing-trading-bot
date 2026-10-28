"""Builders for synthetic stocks and price paths used across tests."""

import copy

import numpy as np
import pandas as pd


def price_path(legs, start=50.0, wiggle=0.004, volume=1_000_000):
    """
    Daily OHLCV from straight-line legs: [(bars, target_close), ...].
    High/low sit `wiggle` above/below the close, so swing points land on leg ends.
    """
    closes = [start]
    for bars, target in legs:
        closes.extend(np.linspace(closes[-1], target, bars + 1)[1:])
    closes = np.array(closes)
    return pd.DataFrame({
        'date': pd.bdate_range("2025-01-02", periods=len(closes)),
        'open': closes,
        'high': closes * (1 + wiggle),
        'low': closes * (1 - wiggle),
        'close': closes,
        'volume': np.full(len(closes), float(volume)),
    })


def price_action(stop_low=97.0, price=100.0):
    """Price-action dict as _analyze_price_action would produce, with every swing low at stop_low."""
    return {
        'current_price': price, 'yesterday_low': stop_low, 'swing_low_3d': stop_low,
        'swing_low_5d': stop_low, 'swing_low_7d': stop_low, 'swing_low_10d': stop_low,
        'is_contracting': True, 'base_high': price * 1.03, 'base_low': price * 0.9,
        'base_depth_pct': 10.0, 'period1_volatility': 2.0, 'period2_volatility': 1.5,
        'period3_volatility': 1.0, 'contraction_1_to_3_pct': 50.0,
    }


BASE_STOCK = {
    'current_price': 100.0, 'adr': 3.0, 'ytd_return': 30.0, 'eps_yoy': None,
    'rs_rating': 92.0, 'vcp_score': 0.0, 'has_real_vcp': False, 'vcp_num_contractions': 0,
    'vcp_pivot_distance_pct': 0.0, 'ma_50': 98.0, 'ma_200': 80.0, 'distance_from_50ma': 2.0,
    'distance_from_200ma': 25.0, 'distance_from_10ema': 0.5, 'distance_from_21ema': 1.0,
    'volume_ratio': 1.3, 'ibd_rs': 92.0, 'ibd_eps_rating': 90.0, 'ibd_acc_dist': 'B',
}


def make_stock(symbol, stop_low=97.0, price_action_override=..., **fields):
    """A stock dict ready for _apply_tier_filters. Defaults pass BUY in a healthy market."""
    stock = copy.deepcopy(BASE_STOCK)
    stock.update(symbol=symbol, **fields)
    stock['price_action'] = price_action(stop_low) if price_action_override is ... else price_action_override
    return stock


# ─── Offline TWS fakes for the order path (order_broker / order_entry tests) ───
# Nothing here opens a socket. FakeConnector subclasses the real IBKRConnector so its read-only
# order guard is the code under test; FakeBroker implements order_broker.Broker.

class FakeConnector:
    """Factory for a connector that never connects. accounts: what TWS 'reports'."""

    def __new__(cls, accounts=("DUX1",), connect_ok=True):
        from ibkr_connector import IBKRConnector

        class _Fake(IBKRConnector):
            instances = []

            def __init__(self, read_only=True, allow_live=False):
                super().__init__(read_only=read_only, allow_live=allow_live)
                self.connect_calls = []
                _Fake.instances.append(self)

            def connect_and_run(self, host, port, client_id):
                self.connect_calls.append((host, port, client_id))
                return connect_ok

            def fetch_managed_accounts(self, timeout=10):
                return list(accounts)

            def disconnect_and_stop(self):
                pass
        return _Fake


class FakeBroker:
    """
    In-memory broker. place() records the order and shows it in open_orders like TWS would
    (status PreSubmitted, transmit flag kept). fail_on_place=n raises ConnectionError on the n-th
    place call (1-based), simulating TWS dropping mid-bracket.
    """

    def __init__(self, account="DUX1", submit_enabled=True, positions=(), open_orders=(),
                 net_liq=100_000.0, first_id=1000, fail_on_place=None, ack=("working", "PreSubmitted"),
                 quotes=None, client_id=16, bars=None, ref_prefix="BOT-"):
        from order_broker import BrokerState
        self.account = account
        self.ref_prefix = ref_prefix       # the only orderRef prefix this session may place (order_broker)
        self.submit_enabled = submit_enabled
        self._state = BrokerState(positions=list(positions), open_orders=list(open_orders),
                                  account_values={account: {"NetLiquidation": str(net_liq)}})
        self.next_id = first_id
        self.placed = []          # (order_id, contract, order)
        self.cancelled = []
        self.modified = []
        self.place_calls = 0
        self.fail_on_place = fail_on_place
        self.ack = ack
        self.quotes = dict(quotes or {})   # symbol -> (price, delayed)
        self.client_id = client_id
        self.quote_calls = []
        self.bars = dict(bars or {})       # symbol -> daily bars DataFrame (exit_manager)
        self.bar_calls = []

    def state(self):
        return self._state

    def next_order_ids(self, n):
        ids = list(range(self.next_id, self.next_id + n))
        self.next_id += n
        return ids

    def _guard(self, what, order=None):
        if not self.submit_enabled:
            raise PermissionError(f"{what}: submission disabled")
        if order is not None:
            if not str(order.orderRef or "").startswith(self.ref_prefix):
                raise PermissionError(f"orderRef must start with {self.ref_prefix}")
            order.account = self.account

    def place(self, order_id, contract, order):
        self._guard("place", order)
        self.place_calls += 1
        if self.fail_on_place and self.place_calls == self.fail_on_place:
            raise ConnectionError("TWS connection dropped")
        self.placed.append((order_id, contract, order))
        self._state.open_orders.append({
            "order_id": order_id, "perm_id": 0, "client_id": self.client_id, "account": self.account,
            "symbol": contract.symbol, "sec_type": "STK", "action": order.action,
            "order_type": order.orderType, "quantity": float(order.totalQuantity),
            "aux_price": order.auxPrice, "lmt_price": order.lmtPrice, "status": "PreSubmitted",
            "order_ref": order.orderRef, "parent_id": order.parentId, "tif": order.tif,
            "oca_group": order.ocaGroup, "transmit": order.transmit})

    def modify(self, order_id, contract, order):
        self._guard("modify", order)
        self.modified.append((order_id, contract, order))
        for o in self._state.open_orders:       # TWS shows the modified quantity / trigger
            if o.get("order_id") == order_id and o.get("client_id") == self.client_id:
                o["quantity"], o["aux_price"] = float(order.totalQuantity), order.auxPrice

    def last_price(self, symbol):
        self.quote_calls.append(symbol)
        return self.quotes.get(symbol, (None, True))

    def daily_bars(self, symbol, duration="6 M", include_forming=False):
        self.bar_calls.append((symbol, include_forming))
        df = self.bars.get(symbol)
        return df.copy() if df is not None else pd.DataFrame(columns=['date', 'open', 'high', 'low', 'close', 'volume'])

    def cancel(self, order_id):
        self._guard("cancel")
        self.cancelled.append(order_id)

    def wait_ack(self, order_id, timeout=10.0):
        return self.ack

    def close(self):
        self.closed = True
