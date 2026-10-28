#!/usr/bin/env python3
"""
Backtest — replay the scanner's price-based entry setups on cached daily bars.

Setups (price/volume only; IBD ratings have no history, so earnings and A/D rules
can't be tested here):
  VCP_BREAKOUT  trend template + valid VCP (detected on data through the prior day);
                buy when today's high crosses the pivot, at max(open, pivot).
                Stop: low of the final contraction.
  EMA_PULLBACK  trend template; today's low tags the 10-EMA (or 21-EMA) and the
                close holds above it, green day; buy next open. Stop: today's low.

Exits (execution defaults): initial stop; trim TRIM_FRACTION at +1R; after the trim the stop
moves to breakeven; any close below the 21-EMA exits at the next open (live
invalidation rule); time stop: exit at the close 8 trading days after the fill if
+1R was never reached. Sensitivity variants rerun the same signals with a
pessimistic intrabar order and with the 21-EMA exit applied only after the trim.

No lookahead: a signal on day t uses bars through t only; fills happen at t+1's
open, or intraday on day t at a pivot known before t. Stops are gap-aware.

Costs: IBKR fixed commission ($0.005/share, $1 min) and 0.05% slippage per side,
on a notional position risking RISK_DOLLARS.

Biases to keep in mind (printed with every report):
  - selection: the universe is today's IBD/TradingView leaders, chosen on how they traded
    in the back half of this window (worse than plain survivorship); see BY ENTRY YEAR
  - IBD gates (EPS Rating, A/D, IBD RS) not applied: historical ratings unavailable
  - one position per symbol at a time; no portfolio caps (2 entries/day, 2% open risk)

Usage:
  python backtest.py                    # all cached symbols, both setups
  python backtest.py --setup VCP_BREAKOUT --out trades.csv
"""

import argparse
import logging
import re
from bisect import bisect_right
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

import strategy_params as sp
from vcp_detector import VCPDetector

logger = logging.getLogger(__name__)

# ─── Rules ───
# Example values from strategy_params.py
MIN_STOP_PCT = sp.MIN_STOP_PCT
MAX_STOP_PCT = sp.MAX_STOP_PCT_PIVOT   # results are bucketed by stop width
TIME_STOP_DAYS = sp.TIME_STOP_SESSIONS
TRIM_FRACTION = sp.TRIM_FRACTION
PULLBACK_TAG_PCT = 1.0        # low within 1% of the EMA counts as a tag
VCP_MAX_PIVOT_EXTENSION = 5.0  # don't fill more than 5% above the pivot (gap-ups)
EXTENDED_NO_BASE_200MA_PCT = 50.0

# ─── Costs ───
RISK_DOLLARS = 500.0          # risk per trade used to size costs (example: 0.5% of a $100k account)
COMMISSION_PER_SHARE = 0.005
COMMISSION_MIN = 1.0
SLIPPAGE_PCT = 0.05


# ═══════════════════════════════════════════════════════════════════════════
# INDICATORS (each value on row t uses rows <= t only)
# ═══════════════════════════════════════════════════════════════════════════
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['ma50'] = df['close'].rolling(50).mean()
    df['ma150'] = df['close'].rolling(150).mean()
    df['ma200'] = df['close'].rolling(200).mean()
    df['ema10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['high52'] = df['high'].rolling(252, min_periods=200).max()
    df['low52'] = df['low'].rolling(252, min_periods=200).min()
    df['adr_pct'] = ((df['high'] / df['low'] - 1) * 100).rolling(20).mean()
    df['dist200_pct'] = (df['close'] / df['ma200'] - 1) * 100
    # Minervini trend template (simplified): stacked MAs, rising 200, near highs, well off lows
    df['trend_ok'] = (
        (df['close'] > df['ma50']) & (df['ma50'] > df['ma150']) & (df['ma150'] > df['ma200'])
        & (df['ma200'] > df['ma200'].shift(20))
        & (df['close'] >= df['high52'] * 0.75) & (df['close'] >= df['low52'] * 1.30)
    )
    return df


def market_regime(spy: pd.DataFrame) -> pd.Series:
    """'UP' when SPY closes above its 50- and 200-day averages, else 'DOWN' ('UNKNOWN' during
    the 200-day warm-up). Indexed by date; shift by a day before labeling entries."""
    s = spy.copy()
    ma50, ma200 = s['close'].rolling(50).mean(), s['close'].rolling(200).mean()
    up = (s['close'] > ma50) & (s['close'] > ma200)
    label = np.where(ma200.isna(), 'UNKNOWN', np.where(up, 'UP', 'DOWN'))
    return pd.Series(label, index=s['date'])


# ═══════════════════════════════════════════════════════════════════════════
# SIGNALS
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class Signal:
    setup: str
    fill_index: int          # bar index of the entry fill
    entry: float
    stop: float
    context: dict = field(default_factory=dict)


def pullback_signals(df: pd.DataFrame) -> List[Signal]:
    """Tag of the 10/21-EMA on day t (held above, green close); fill at t+1 open, stop = low of t."""
    signals = []
    for t in range(200, len(df) - 1):
        row = df.iloc[t]
        if not row['trend_ok'] or row['close'] <= row['open']:
            continue
        tagged = None
        for ema in ('ema10', 'ema21'):
            if row['low'] <= row[ema] * (1 + PULLBACK_TAG_PCT / 100) and row['close'] > row[ema]:
                tagged = ema
                break
        if not tagged:
            continue
        nxt = df.iloc[t + 1]
        signals.append(Signal('EMA_PULLBACK', t + 1, float(nxt['open']), float(row['low']),
                              {'ema': tagged, 'dist200_pct': row['dist200_pct'], 'has_vcp': False}))
    return signals


def vcp_signals(df: pd.DataFrame, detector: Optional[VCPDetector] = None) -> List[Signal]:
    """VCP on bars through t-1; fill on day t when the high crosses the pivot."""
    detector = detector or VCPDetector()
    signals = []
    for t in range(201, len(df)):
        prev = df.iloc[t - 1]
        if not prev['trend_ok']:
            continue
        result = detector.detect(df.iloc[:t][['date', 'open', 'high', 'low', 'close', 'volume']], 3)
        if not result.has_vcp or result.pivot_price <= 0:
            continue
        pivot = result.pivot_price
        day = df.iloc[t]
        if day['high'] < pivot or prev['close'] >= pivot:
            continue  # no cross today (or already above the pivot yesterday)
        entry = max(float(day['open']), pivot)
        if entry > pivot * (1 + VCP_MAX_PIVOT_EXTENSION / 100):
            continue  # gapped too far past the buy point
        stop = result.pullbacks[-1].swing_low.price
        signals.append(Signal('VCP_BREAKOUT', t, entry, float(stop),
                              {'contractions': result.num_contractions, 'vcp_score': result.score,
                               'dist200_pct': prev['dist200_pct'], 'has_vcp': True}))
    return signals


# ═══════════════════════════════════════════════════════════════════════════
# TRADE SIMULATION
# ═══════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class Rules:
    """Exit assumptions for the simple daily-bar simulator (partial at +trim_r, 21-EMA trail). The bot's
    own exit plan is exit_rules.py; exit_rules.simulate_r scores that plan on the same bars."""
    name: str = "live"
    # Daily bars can't say whether a bar's +1R high came before or after a dip back to entry.
    # 'realistic': the remainder exits at breakeven only when the trim bar CLOSES below entry
    #   (that crossing is certain). 'pessimistic': any trim bar whose low reaches entry exits it.
    intrabar: str = "realistic"
    # Live invalidation (execution_engine): a close below the 21-EMA exits at any time,
    # not only after the trim.
    exit_below_21ema_before_trim: bool = True
    # Research knobs (defaults = live rules)
    trim_r: float = 1.0                 # trim at +N R
    trim_fraction: float = TRIM_FRACTION
    time_stop_days: int = TIME_STOP_DAYS
    trail_ema: str = 'ema21'            # close below this exits (ema10 = tighter trail)
    min_stop_pct: float = MIN_STOP_PCT
    max_stop_pct: float = MAX_STOP_PCT
    stop_cap_pct: Optional[float] = None  # tighten: stop no further than N% below entry


LIVE = Rules()
VARIANTS = [
    LIVE,
    Rules(name="pessimistic intrabar", intrabar="pessimistic"),
    Rules(name="21EMA trail after trim only", exit_below_21ema_before_trim=False),
]


@dataclass
class Trade:
    symbol: str
    setup: str
    entry_date: pd.Timestamp
    entry: float
    stop: float
    stop_pct: float
    exit_date: pd.Timestamp = None
    exit_reason: str = ""
    r_gross: float = 0.0     # result in R before costs
    r_net: float = 0.0       # after commission + slippage
    days_held: int = 0
    trimmed: bool = False
    trim_date: pd.Timestamp = None   # when +1R was hit (risk drops to 0: stop at breakeven)
    context: dict = field(default_factory=dict)


def _costs_in_r(entry: float, stop: float, orders: int = 2) -> float:
    """Commission (per order, with IBKR's $1 minimum) + slippage per side, as a fraction of R."""
    risk_per_share = entry - stop
    shares = max(1, int(RISK_DOLLARS / risk_per_share))
    commission = max(COMMISSION_MIN, shares * COMMISSION_PER_SHARE) * 2
    if orders > 2:  # the trim is a third order; its shares are already in the per-share total
        commission += COMMISSION_MIN
    slippage = 2 * shares * entry * SLIPPAGE_PCT / 100
    return (commission + slippage) / (shares * risk_per_share)


def simulate(df: pd.DataFrame, sig: Signal, symbol: str, rules: Rules = LIVE) -> Optional[Trade]:
    """Walk forward from the fill. Returns None when the stop is outside the allowed band."""
    entry, stop = sig.entry, sig.stop
    if rules.stop_cap_pct is not None:
        stop = max(stop, entry * (1 - rules.stop_cap_pct / 100))
    if stop >= entry:
        return None
    stop_pct = (entry - stop) / entry * 100
    if not rules.min_stop_pct <= stop_pct <= rules.max_stop_pct:
        return None
    r = entry - stop
    trade = Trade(symbol, sig.setup, df['date'].iloc[sig.fill_index], entry, stop, round(stop_pct, 2),
                  context=sig.context)

    pnl_r = 0.0          # realized R, weighted by the fraction of the position closed
    open_frac = 1.0
    cur_stop = stop
    pending_exit = None  # set on a close below the 21-EMA: exit at the next open

    def close_out(i, price, reason):
        nonlocal pnl_r, open_frac
        pnl_r += open_frac * (price - entry) / r
        open_frac = 0.0
        trade.exit_date, trade.exit_reason = df['date'].iloc[i], reason
        trade.days_held = i - sig.fill_index

    for i in range(sig.fill_index, len(df)):
        bar = df.iloc[i]
        if pending_exit is not None:
            close_out(i, float(bar['open']), pending_exit)
            break
        # Stop first (worst case). On the fill day a low under the stop counts as hit.
        if bar['low'] <= cur_stop:
            fill = min(float(bar['open']), cur_stop) if i > sig.fill_index else cur_stop
            close_out(i, fill, "stop" if not trade.trimmed else "breakeven stop")
            break
        if not trade.trimmed and bar['high'] >= entry + rules.trim_r * r:
            # a bar that opens above the target fills the trim at the open
            target = entry + rules.trim_r * r
            pnl_r += rules.trim_fraction * (max(float(bar['open']), target) - entry) / r
            open_frac -= rules.trim_fraction
            trade.trimmed = True
            trade.trim_date = df['date'].iloc[i]
            cur_stop = max(cur_stop, entry)  # protect the rest at breakeven
            # Same bar: did price come back to entry after the +1R print?
            back_to_entry = (bar['close'] < entry if rules.intrabar == "realistic"
                             else bar['low'] <= entry)
            if back_to_entry:
                close_out(i, entry, "breakeven stop (same bar)")
                break
        held = i - sig.fill_index  # trading days after the fill (fill day = 0)
        if not trade.trimmed and held >= rules.time_stop_days:
            close_out(i, float(bar['close']), "time stop")
            break
        if bar['close'] < bar[rules.trail_ema] and (trade.trimmed or rules.exit_below_21ema_before_trim):
            pending_exit = "close < 21EMA"
    else:
        # Still open at the end of the data: mark to the last close
        close_out(len(df) - 1, float(df['close'].iloc[-1]), "open (marked)")

    trade.r_gross = round(pnl_r, 3)
    trade.r_net = round(pnl_r - _costs_in_r(entry, stop, orders=3 if trade.trimmed else 2), 3)
    return trade


def symbol_signals(bars: pd.DataFrame, setups: List[str]):
    """Indicator frame and all signals for one symbol (computed once, simulated per variant)."""
    df = add_indicators(bars)
    signals = []
    if 'EMA_PULLBACK' in setups:
        signals += pullback_signals(df)
    if 'VCP_BREAKOUT' in setups:
        signals += vcp_signals(df)
    return df, sorted(signals, key=lambda s: s.fill_index)


def trades_for(df: pd.DataFrame, signals: List[Signal], symbol: str, rules: Rules = LIVE) -> List[Trade]:
    """
    One position per symbol per setup at a time: a signal during an open trade of the same
    setup is skipped. Setups don't block each other here (a pullback trade used to hide most
    VCP breakouts); simulate_portfolio stops the account holding one symbol twice.
    """
    trades, busy_until = [], {}
    for sig in signals:
        if sig.fill_index <= busy_until.get(sig.setup, -1):
            continue
        trade = simulate(df, sig, symbol, rules)
        if trade is None:
            continue
        trades.append(trade)
        busy_until[sig.setup] = int(df.index[df['date'] == trade.exit_date][0])
    return trades


def backtest_symbol(symbol: str, bars: pd.DataFrame, setups: List[str], rules: Rules = LIVE) -> List[Trade]:
    df, signals = symbol_signals(bars, setups)
    return trades_for(df, signals, symbol, rules)


# ═══════════════════════════════════════════════════════════════════════════
# REPORTING
# ═══════════════════════════════════════════════════════════════════════════
def max_drawdown(r: pd.Series) -> float:
    """Largest peak-to-trough drop of cumulative R, counting from the 0R starting point."""
    equity = np.r_[0.0, r.cumsum().values]
    return float((np.maximum.accumulate(equity) - equity).max())


def summarize(trades: pd.DataFrame) -> pd.Series:
    """Expectancy and friends, in R after costs. Expects trades sorted by exit date."""
    r = trades['r_net']
    wins, losses = r[r > 0], r[r <= 0]
    return pd.Series({
        'trades': len(r),
        'win_rate_%': round(100 * len(wins) / len(r), 1) if len(r) else 0,
        'avg_win_R': round(wins.mean(), 2) if len(wins) else 0,
        'avg_loss_R': round(losses.mean(), 2) if len(losses) else 0,
        'expectancy_R': round(r.mean(), 3) if len(r) else 0,
        'total_R': round(r.sum(), 1),
        'max_drawdown_R': round(max_drawdown(r), 1) if len(r) else 0,
        'avg_days': round(trades['days_held'].mean(), 1) if len(r) else 0,
    })


def _table(t: pd.DataFrame, by) -> str:
    return t.groupby(by, observed=True)[['r_net', 'days_held']].apply(summarize).to_string()


def report(trades: pd.DataFrame, variants: Optional[dict] = None, universe_note: Optional[str] = None) -> str:
    t = trades.sort_values('exit_date').copy()
    t['stop_bucket'] = pd.cut(t['stop_pct'], [MIN_STOP_PCT, 3, 5, MAX_STOP_PCT + 1e-4], right=False,
                             labels=[f'{MIN_STOP_PCT:g}-3%', '3-5%', f'5-{MAX_STOP_PCT:g}%'])
    t['entry_year'] = pd.to_datetime(t['entry_date']).dt.year
    ext = t['dist200_pct'].ge(EXTENDED_NO_BASE_200MA_PCT) & ~t['has_vcp'].astype(bool)
    t['extended_no_base'] = np.where(ext, 'yes', 'no')
    lines = [
        "LIVE RULES — BY SETUP\n" + _table(t, 'setup'),
        "BY ENTRY YEAR (the universe is today's leaders, so later years are flattered)\n"
        + _table(t, ['setup', 'entry_year']),
        "BY STOP SIZE\n" + _table(t, ['setup', 'stop_bucket']),
        "BY MARKET REGIME (SPY vs 50/200-day, prior close)\n" + _table(t, ['setup', 'regime']),
        "EXTENDED WITHOUT A BASE (50%+ over 200-day, no VCP) — pullbacks only\n"
        + _table(t[t['setup'] == 'EMA_PULLBACK'], 'extended_no_base'),
        "EXIT REASONS\n" + t.groupby(['setup', 'exit_reason']).size().unstack(fill_value=0).to_string(),
    ]
    if variants:
        rows = []
        for name, vt in variants.items():
            vt = vt.sort_values('exit_date')
            for setup, g in vt.groupby('setup'):
                rows.append(pd.concat([pd.Series({'rules': name, 'setup': setup}),
                                       summarize(g[['r_net', 'days_held']])]))
        lines.append("SENSITIVITY (same signals, different exit assumptions)\n"
                     + pd.DataFrame(rows).set_index(['rules', 'setup']).to_string())
    universe_caveat = (f"universe: point-in-time scan lists ({universe_note})" if universe_note else
                       "universe selected on recent performance (today's IBD/TradingView leaders), "
                       "not just survivorship")
    lines.append(
        f"CAVEATS: {universe_caveat}; IBD EPS/A-D/RS gates not applied (no history); per-trade "
        "stats above ignore portfolio caps (see PORTFOLIO); daily bars can't order intrabar "
        f"moves (see sensitivity); costs = IBKR fixed commission + {SLIPPAGE_PCT}% slippage per "
        "side. Provisional.")
    return "\n\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# POINT-IN-TIME UNIVERSE (dated scan lists)
# ═══════════════════════════════════════════════════════════════════════════
DATE_IN_NAME = re.compile(r"(\d{4}-\d{2}-\d{2})")
SYMBOL_COLUMNS = ('symbol', 'symbols', 'ticker', 'tickers')
SKIP_LIST_FILES = ('backtest_',)  # outcome-selected outputs, never a scan list


def _symbol_column(df: pd.DataFrame) -> Optional[str]:
    cols = {str(c).strip().lower(): c for c in df.columns}
    for name in SYMBOL_COLUMNS:
        if name in cols:
            return cols[name]
    return df.columns[0] if len(df.columns) == 1 else None


def _to_day(values) -> pd.Series:
    """Parse dates of any common format/timezone to tz-naive midnight (one value per day)."""
    parsed = pd.to_datetime(pd.Series(values), errors='coerce', utc=True, format='mixed')
    return parsed.dt.tz_convert(None).dt.normalize()


def load_universe_lists(path: str) -> pd.DataFrame:
    """
    Dated scan lists -> DataFrame(date, symbol, list).
    Accepts a directory of CSVs with the date in the file name (stage2_2024-03-15.csv,
    universe_2026-10-24.csv), or one CSV with date and symbol columns (plus an optional list
    column). The list name is the file name without its date, or the 'list' column.
    Dates are normalized to the day (times and timezones dropped); unparseable dates are
    dropped with a warning. backtest_* files are ignored.
    """
    p = Path(path).expanduser()
    frames = []
    files = sorted(p.glob('*.csv')) if p.is_dir() else [p]
    for f in files:
        if f.name.startswith(SKIP_LIST_FILES):
            continue
        df = pd.read_csv(f, dtype=str)
        cols = {str(c).strip().lower(): c for c in df.columns}
        sym_col = _symbol_column(df)
        if sym_col is None:
            logger.warning(f"Skipping {f.name}: no symbol column")
            continue
        if 'date' in cols:
            out = pd.DataFrame({'date': _to_day(df[cols['date']]).values, 'symbol': df[sym_col].values})
            # No list column: name it after the file without a trailing year (stage2_2025 -> stage2)
            out['list'] = (df[cols['list']].values if 'list' in cols
                           else re.sub(r'[_-]?\d{4}$', '', f.stem).lower() or 'default')
            bad = out['date'].isna().sum()
            if bad:
                logger.warning(f"{f.name}: dropped {bad} row(s) with unparseable dates")
        else:
            m = DATE_IN_NAME.search(f.name)
            if not m:
                logger.warning(f"Skipping {f.name}: no date column and no YYYY-MM-DD in the name")
                continue
            out = pd.DataFrame({'date': pd.Timestamp(m.group(1)), 'symbol': df[sym_col].values,
                                'list': DATE_IN_NAME.sub('', f.stem).strip('_- ').lower() or f.stem})
        frames.append(out)
    if not frames:
        return pd.DataFrame(columns=['date', 'symbol', 'list'])
    lists = pd.concat(frames, ignore_index=True)
    # IBKR spells share classes with a space (BRK B); data vendors use '-' or '.'
    lists['symbol'] = lists['symbol'].astype(str).str.strip().str.upper().str.replace(r'[-.]', ' ', regex=True)
    lists = lists[lists['symbol'].ne('') & lists['symbol'].ne('NAN') & lists['date'].notna()]
    return lists.drop_duplicates(['date', 'list', 'symbol']).reset_index(drop=True)


class UniverseHistory:
    """
    Was a symbol on any scan list as of a date? Each list (stage2, new_highs, ...) is tracked on
    its own: its most recent version dated on or before the date counts (never a later one:
    that would be lookahead), if it's at most max_age_days old. A new file of one list doesn't
    replace a still-valid version of another.
    """

    def __init__(self, lists: pd.DataFrame, max_age_days: int = 10):
        self.max_age = pd.Timedelta(days=max_age_days)
        self.by_list = {}
        for name, g in lists.groupby('list'):
            dates = sorted(pd.to_datetime(g['date'].unique()))
            members = {pd.Timestamp(d): set(gg['symbol']) for d, gg in g.groupby('date')}
            self.by_list[name] = (dates, members)
        all_dates = sorted({d for dates, _ in self.by_list.values() for d in dates})
        self.dates = all_dates

    @property
    def symbols(self) -> List[str]:
        return sorted({s for _, members in self.by_list.values() for m in members.values() for s in m})

    def coverage(self) -> str:
        if not self.dates:
            return "no lists"
        return (f"{len(self.by_list)} list(s), {len(self.dates)} dates, "
                f"{self.dates[0]:%Y-%m-%d} to {self.dates[-1]:%Y-%m-%d}")

    def on_list(self, symbol: str, date) -> bool:
        date = pd.Timestamp(date).normalize()
        symbol = symbol.upper()
        for dates, members in self.by_list.values():
            i = bisect_right(dates, date) - 1
            if i >= 0 and date - dates[i] <= self.max_age and symbol in members[dates[i]]:
                return True
        return False


# ═══════════════════════════════════════════════════════════════════════════
# PORTFOLIO SIMULATION
# ═══════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class Account:
    start_equity: float = 100_000.0
    risk_pct: float = sp.RISK_PER_TRADE_PCT            # per trade, % of equity at entry
    max_position_pct: float = sp.MAX_POSITION_PCT      # caps size, so tight stops risk less than risk_pct
    max_new_per_day: int = sp.MAX_NEW_ENTRIES_PER_DAY
    max_open_risk_pct: float = sp.MAX_TOTAL_OPEN_RISK_PCT  # capital at risk; a trimmed trade (stop at breakeven) holds none
    max_positions: int = sp.MAX_OPEN_POSITIONS          # execution_engine.ExecutionPolicy.max_open_positions
    max_gross_pct: float = 100.0     # cash account: total position value <= equity (no margin)


SETUP_PRIORITY = {'VCP_BREAKOUT': 0, 'EMA_PULLBACK': 1}


def simulate_portfolio(trades: pd.DataFrame, account: Account = Account(),
                       closes: Optional[dict] = None, calendar=None) -> dict:
    """
    Replay candidate trades through one account in time order.
    Each day: take entries (VCP first, then tighter stops) while the daily cap, position cap,
    open-risk cap and gross-exposure cap allow; then trims free their risk (and the trimmed fraction of the
    position); then exits realize P&L (r_net x dollars at risk). Entries are processed before
    same-day trims/exits, so freed room is reusable only the next day (conservative).
    With closes ({symbol: close Series}) and a calendar (all trading days), equity is also
    marked to market daily, and drawdown uses that curve; otherwise it's realized-only.
    """
    t = trades.copy()
    for col in ('entry_date', 'exit_date', 'trim_date'):
        t[col] = pd.to_datetime(t[col])
    t['_prio'] = t['setup'].map(SETUP_PRIORITY).fillna(9)
    t = t.sort_values(['entry_date', '_prio', 'stop_pct']).reset_index(drop=True)

    event_days = set(t['entry_date']) | set(t['exit_date']) | set(t['trim_date'].dropna())
    days = sorted(event_days | set(pd.to_datetime(calendar))) if calendar is not None else sorted(event_days)
    entries_by_day = {d: g for d, g in t.groupby('entry_date')}
    equity = account.start_equity
    pos = {}       # trade index -> {'dollars', 'risk', 'notional', 'frac', 'trimmed'}
    taken, curve = [], []
    skipped = {'already held': 0, 'daily cap': 0, 'position cap': 0, 'open-risk cap': 0, 'gross exposure cap': 0}

    for d in days:
        new_today = 0
        for idx, tr in (entries_by_day[d].iterrows() if d in entries_by_day else []):
            risk_frac = min(account.risk_pct, account.max_position_pct * tr['stop_pct'] / 100)
            dollars = equity * risk_frac / 100
            notional = dollars / (tr['stop_pct'] / 100)
            gross = sum(p['notional'] * p['frac'] for p in pos.values())
            if 'symbol' in t.columns and any(t.at[i, 'symbol'] == tr['symbol'] for i in pos):
                skipped['already held'] += 1
            elif new_today >= account.max_new_per_day:
                skipped['daily cap'] += 1
            elif len(pos) >= account.max_positions:
                skipped['position cap'] += 1
            elif (sum(p['risk'] for p in pos.values()) + dollars) / equity * 100 > account.max_open_risk_pct + 1e-9:
                skipped['open-risk cap'] += 1
            elif (gross + notional) / equity * 100 > account.max_gross_pct + 1e-9:
                skipped['gross exposure cap'] += 1
            else:
                pos[idx] = {'dollars': dollars, 'risk': dollars, 'notional': notional, 'frac': 1.0, 'trimmed': False}
                new_today += 1
                taken.append({'idx': idx, 'dollars_at_risk': dollars, 'risk_pct': risk_frac})
        for idx in [i for i in pos if t.at[i, 'trim_date'] == d]:
            pos[idx].update(risk=0.0, frac=1 - TRIM_FRACTION, trimmed=True)
        for idx in [i for i in pos if t.at[i, 'exit_date'] == d]:
            equity += t.at[idx, 'r_net'] * pos[idx]['dollars']
            del pos[idx]

        unrealized = 0.0
        if closes is not None:
            for idx, p in pos.items():
                series = closes.get(t.at[idx, 'symbol'])
                px = series.asof(d) if series is not None and len(series) else np.nan
                if pd.isna(px):
                    continue
                r = t.at[idx, 'entry'] - t.at[idx, 'stop']
                open_r = p['frac'] * (px - t.at[idx, 'entry']) / r + (TRIM_FRACTION if p['trimmed'] else 0.0)
                unrealized += open_r * p['dollars']
        curve.append({'date': d, 'equity': equity, 'equity_mtm': equity + unrealized,
                      'open_positions': len(pos),
                      'gross_pct': sum(p['notional'] * p['frac'] for p in pos.values()) / equity * 100})

    curve = pd.DataFrame(curve)
    taken_df = t.loc[[x['idx'] for x in taken]].assign(
        dollars_at_risk=[x['dollars_at_risk'] for x in taken], risk_pct=[x['risk_pct'] for x in taken])
    dd_col = 'equity_mtm' if closes is not None else 'equity'
    peak = curve[dd_col].cummax().clip(lower=account.start_equity) if len(curve) else None
    years = (curve['date'].iloc[-1] - curve['date'].iloc[0]).days / 365.25 if len(curve) > 1 else 0
    summary = {
        'candidates': len(t), 'taken': len(taken), **{f'skipped ({k})': v for k, v in skipped.items()},
        'final_equity': round(equity, 0),
        'total_return_%': round((equity / account.start_equity - 1) * 100, 1),
        # annualized over the whole simulated window, not just the span between trades
        'cagr_%': round(((equity / account.start_equity) ** (1 / years) - 1) * 100, 1) if years >= 1 else None,
        'max_drawdown_%': round(((peak - curve[dd_col]) / peak).max() * 100, 1) if len(curve) else 0,
        'drawdown_basis': 'daily mark-to-market' if closes is not None else 'realized only (understates)',
        'avg_open_positions': round(curve['open_positions'].mean(), 1) if len(curve) else 0,
        'peak_gross_exposure_%': round(curve['gross_pct'].max(), 1) if len(curve) else 0,
        'expectancy_R_taken': round(taken_df['r_net'].mean(), 3) if len(taken_df) else 0,
    }
    return {'summary': summary, 'curve': curve, 'taken': taken_df}


def portfolio_report(trades: pd.DataFrame, account: Account = Account(),
                     closes: Optional[dict] = None, calendar=None) -> str:
    rows = {}
    for label, setups in (("VCP only (auto rules)", ['VCP_BREAKOUT']),
                          ("VCP + EMA pullbacks", ['VCP_BREAKOUT', 'EMA_PULLBACK'])):
        sub = trades[trades['setup'].isin(setups)]
        if len(sub):
            rows[label] = simulate_portfolio(sub, account, closes, calendar)['summary']
    return (f"PORTFOLIO (${account.start_equity:,.0f} cash account, {account.risk_pct}% risk/trade, "
            f"{account.max_position_pct}% max position, {account.max_positions} max positions, "
            f"{account.max_new_per_day} entries/day, {account.max_open_risk_pct}% max open risk, "
            f"positions <= {account.max_gross_pct:.0f}% of equity)\n"
            + pd.DataFrame(rows).to_string()
            + "\nNotes: the daily cap picks among setups that actually triggered (live picks 2 orders before"
              " knowing which trigger), so trade count and compounding lean optimistic; no group cap"
              " (no IBD group history); trades still open at the end are marked at the last close.")


def _rows(trades: List[Trade], regime: Optional[pd.Series]) -> List[dict]:
    rows = []
    for tr in trades:
        row = asdict(tr)
        ctx = row.pop('context')
        row.update(dist200_pct=ctx.get('dist200_pct'), has_vcp=ctx.get('has_vcp', False))
        row['regime'] = regime.get(tr.entry_date, 'UNKNOWN') if regime is not None else 'UNKNOWN'
        rows.append(row)
    return rows


def run(symbols: List[str], setups: List[str], variants: Optional[List[Rules]] = None,
        universe: Optional['UniverseHistory'] = None) -> dict:
    """
    {rules name: trades DataFrame}. Signals are computed once per symbol. With a universe,
    a signal only counts if the symbol was on the lists as of the signal day (the day before
    the fill), using lists dated on or before it.
    """
    from backtest_data import BENCHMARK, load_bars
    variants = variants or [LIVE]
    spy = load_bars(BENCHMARK)
    # Label each entry with the regime known BEFORE it: the prior day's close
    regime = market_regime(spy).shift(1) if spy is not None else None
    rows = {v.name: [] for v in variants}
    for i, symbol in enumerate(symbols, 1):
        if symbol == BENCHMARK:
            continue
        bars = load_bars(symbol)
        if bars is None or len(bars) < 260:
            continue
        df, signals = symbol_signals(bars, setups)
        if universe is not None:
            signals = [sg for sg in signals
                       if universe.on_list(symbol, df['date'].iloc[max(sg.fill_index - 1, 0)])]
        for v in variants:
            rows[v.name] += _rows(trades_for(df, signals, symbol, v), regime)
        if i % 50 == 0:
            logger.info(f"  {i}/{len(symbols)} symbols, {len(rows[LIVE.name])} live-rule trades")
    return {name: pd.DataFrame(r) for name, r in rows.items()}


def main():
    parser = argparse.ArgumentParser(description="Backtest the scanner's price-based setups")
    parser.add_argument('--setup', choices=['EMA_PULLBACK', 'VCP_BREAKOUT', 'ALL'], default='ALL')
    parser.add_argument('--symbols', nargs='*', help='Default: every cached symbol')
    parser.add_argument('--out', help='Write all trades to this CSV')
    parser.add_argument('--universe-lists', help='Dated scan lists (dir of *_YYYY-MM-DD.csv or one '
                        'CSV with date,symbol): only trade names on the lists at the time')
    parser.add_argument('--max-list-age-days', type=int, default=10)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%H:%M:%S')

    from backtest_data import BARS_DIR
    universe = None
    if args.universe_lists:
        universe = UniverseHistory(load_universe_lists(args.universe_lists), args.max_list_age_days)
        logger.info(f"Point-in-time universe: {len(universe.dates)} list dates, {len(universe.symbols)} symbols")
    cached = sorted(p.stem for p in BARS_DIR.glob('*.csv'))
    symbols = args.symbols or (sorted(set(universe.symbols) & set(cached)) if universe else cached)
    if universe:
        missing = sorted(set(universe.symbols) - set(cached))
        if missing:
            logger.warning(f"{len(missing)} list symbols have no cached bars (run backtest_data.py "
                           f"--symbols-from {args.universe_lists}): {', '.join(missing[:10])}")
    setups = ['EMA_PULLBACK', 'VCP_BREAKOUT'] if args.setup == 'ALL' else [args.setup]
    results = run(symbols, setups, VARIANTS, universe)
    trades = results[LIVE.name]
    if trades.empty:
        print("No trades.")
        return
    if args.out:
        trades.to_csv(args.out, index=False)
    universe_note = None
    if universe:
        missing = sorted(set(universe.symbols) - set(cached))
        universe_note = f"{universe.coverage()}; {len(missing)} listed symbols had no bars"
    print(report(trades, variants=results, universe_note=universe_note))
    print()
    from backtest_data import BENCHMARK, load_bars
    closes = {}
    for sym in trades['symbol'].unique():
        bars = load_bars(sym)
        if bars is not None:
            closes[sym] = bars.set_index('date')['close']
    spy = load_bars(BENCHMARK)
    calendar = None
    if spy is not None:
        first = pd.to_datetime(trades['entry_date']).min()
        calendar = spy.loc[spy['date'] >= first, 'date']
    print(portfolio_report(trades, closes=closes, calendar=calendar))


if __name__ == "__main__":
    main()
