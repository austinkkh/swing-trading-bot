#!/usr/bin/env python3
"""
Forward-test report: the weekly scorecard of what the system did on data it had never seen. From
now on the system is judged by this forward evidence, not by backtests. READ-ONLY: no IBKR
connection, no orders, no ledger / exit_state / log is written. The only files written are the
report itself.

Sources (each may be missing or empty: the report shows zeros, it never crashes on that):
  signal log   Data/paper_tracking/signal_log.csv (signal_log.py). Every scanner VCP signal and every
               pocket pivot, with simulated fills and the adopted exit's R (net of costs). Bucketed by
               signal_date. adopted_status closed = finished trade; open = marked to the last close;
               no adopted_R yet = pending (waiting for a fill or bars): counted, left out of R stats.
  conviction   Data/paper_tracking/conviction_log.csv (conviction_log.py). The owner's ideas, logged
               before the outcome. R is whatever the last `conviction_log.py score` stored (this report
               does not re-score). late=True ideas are excluded from every statistic (counted only).
               Overlap = owner idea and bot signal on the same symbol in the same ISO week.
  paper orders Data/orders/ledger.jsonl (order_ledger.py) plus Data/orders/exit_state.json
               (exit_manager.new_trade). Bucketed by the entry trade_date. PAPERCHECK- orderRefs
               (scripts/paper_checks.py test orders) are excluded. Realised R of a closed trade =
               (exit fills - entry fill) / (entry shares x (entry fill - initial stop)), the initial
               stop being the bracket child's first recorded stop (as exit_manager measures it).
               GROSS of commissions: the ledger has no commission field. A closed trade whose exit
               fills don't add up to the entry shares (e.g. a GTC stop filled on a day with no bot
               run) has no honest R: it is counted as closed, R unknown.

Progress to a verdict: TARGET_N closed trades are needed to detect a mean of TARGET_EFFECT_R per trade
(std ~TARGET_STD_R, 80% power, two-sided 5%; example values). Shown: closed-trade count vs that target, mean R and
a 95% interval (normal approximation, mean +/- 1.96 x std / sqrt(n)), and a plain status line. Below
the target the status never claims an edge, whatever the interval says.

Weeks are ISO weeks (Monday start). --as-of drops rows dated after it; the R values themselves are
as stored now (the logs keep no history), so a past --as-of is not a point-in-time replay.

Output: a text table on stdout, Data/reports/forward_report_<as_of>.csv (one row per week plus an
ALL row, cumulative to as_of) and Data/reports/forward_report_latest.md.

Usage:
  python forward_report.py                       # last 8 weeks + cumulative, as of today
  python forward_report.py --weeks 12 --as-of 2026-10-30 --out /tmp/reports
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import order_ledger as ol
from backtest_data import BARS_DIR

logger = logging.getLogger(__name__)

DATA_DIR = BARS_DIR.parent
SIGNAL_LOG = DATA_DIR / "paper_tracking" / "signal_log.csv"
CONVICTION_LOG = DATA_DIR / "paper_tracking" / "conviction_log.csv"
REPORTS_DIR = DATA_DIR / "reports"

TARGET_EFFECT_R = 0.25    # example effect size to detect, R per trade
TARGET_STD_R = 1.5
TARGET_N = round((2.8 * TARGET_STD_R / TARGET_EFFECT_R) ** 2)   # n = (z_a/2 + z_b)^2 x (std / effect)^2
INTERIM_CHECKS = (50, 100, 150)
HARM_R = -0.3             # protocol interim bar: mean R not below -0.3R
Z95 = 1.96
# Student t 97.5% quantiles for df 1..29 (no scipy): a z interval on a handful of trades looks far too tight
T975 = (12.71, 4.30, 3.18, 2.78, 2.57, 2.45, 2.36, 2.31, 2.26, 2.23, 2.20, 2.18, 2.16, 2.14, 2.13, 2.12,
        2.11, 2.10, 2.09, 2.09, 2.08, 2.07, 2.07, 2.06, 2.06, 2.06, 2.05, 2.05, 2.05)
BOOT_N = 5000            # protocol interim check: 90% bootstrap CI (fixed seed: same data, same answer)
DEFAULT_WEEKS = 8
PAPERCHECK_PREFIX = "PAPERCHECK-"   # = order_broker.PAPERCHECK_REF_PREFIX (not imported: read-only module)
FILL_STATUSES = ('filled', 'waiting', 'no_entry', 'not_triggered')
NO_TRADE = {'no_entry', 'not_triggered'}


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS (pure)
# ═══════════════════════════════════════════════════════════════════════════
def iso_week(d) -> str:
    """'2026-W44' for any date-like (weekends fall in their Monday-start ISO week)."""
    y, w, _ = pd.Timestamp(d).isocalendar()
    return f"{y}-W{w:02d}"


def week_start(d) -> date:
    t = pd.Timestamp(d).date()
    return t - timedelta(days=t.weekday())


def week_list(as_of: date, n: int) -> List[str]:
    """The n ISO weeks ending with as_of's week, oldest first."""
    start = week_start(as_of)
    return [iso_week(start - timedelta(weeks=k)) for k in range(n - 1, -1, -1)]


def r_stats(r: pd.Series) -> dict:
    """n, mean, median, win rate (R > 0), average win / loss, expectancy, std for one R sample.
    Expectancy = win% x avg win - loss% x avg loss (equal to the mean, shown as the trader's form)."""
    r = pd.to_numeric(r, errors='coerce').dropna()
    n = int(len(r))
    if n == 0:
        return {'n': 0, 'mean': np.nan, 'median': np.nan, 'win_rate': np.nan, 'avg_win': np.nan,
                'avg_loss': np.nan, 'expectancy': np.nan, 'std': np.nan, 'total': 0.0}
    wins, losses = r[r > 0], r[r <= 0]
    wr = len(wins) / n
    avg_w = float(wins.mean()) if len(wins) else 0.0
    avg_l = float(-losses.mean()) if len(losses) else 0.0
    return {'n': n, 'mean': float(r.mean()), 'median': float(r.median()), 'win_rate': wr,
            'avg_win': avg_w if len(wins) else np.nan, 'avg_loss': avg_l if len(losses) else np.nan,
            'expectancy': wr * avg_w - (1 - wr) * avg_l,
            'std': float(r.std(ddof=1)) if n > 1 else np.nan, 'total': float(r.sum())}


def mean_ci(r: pd.Series, z: Optional[float] = None) -> Tuple[int, float, float, float]:
    """(n, mean, lo, hi): 95% interval mean +/- q x std / sqrt(n), q = Student t below 30 trades, else
    z (1.96, or the z passed); NaN bounds if n < 2."""
    r = pd.to_numeric(r, errors='coerce').dropna()
    n = int(len(r))
    if n == 0:
        return 0, np.nan, np.nan, np.nan
    m = float(r.mean())
    if n < 2:
        return n, m, np.nan, np.nan
    q = z if z is not None else (T975[n - 2] if n - 1 < 30 else Z95)
    half = q * float(r.std(ddof=1)) / math.sqrt(n)
    return n, m, m - half, m + half


def bootstrap_ci(r: pd.Series, level: float = 0.90, n_boot: int = BOOT_N, seed: int = 0) -> Tuple[float, float]:
    """Percentile bootstrap interval of the mean (the protocol's interim harm check)."""
    x = pd.to_numeric(r, errors='coerce').dropna().to_numpy(dtype=float)
    if len(x) < 2:
        return np.nan, np.nan
    means = np.random.default_rng(seed).choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    a = (1 - level) / 2
    return float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


def verdict_status(r: pd.Series, target: int = TARGET_N, unscored: int = 0, shadow: bool = False) -> str:
    """
    One plain-English line. Never implies significance below the target sample.
      unscored  closed trades with no R (exit fills missing): left out of the mean, named in the line;
                no verdict while any are missing (dropping them biases the mean)
      shadow    a simulated sample that is not the protocol's (the signal log): never a verdict
    """
    n, m, lo, hi = mean_ci(r)
    gap = (f"; {unscored} closed trade(s) have no R (missing exit fills): left out, so the mean may be "
           f"biased; backfill them" if unscored else "")
    if n == 0:
        return f"n=0 of ~{target}: no closed trades with R yet, nothing to judge{gap}"
    ci = f", 95% CI {lo:+.2f} to {hi:+.2f}R" if n >= 2 else ""
    head = f"n={n} of ~{target}: mean {m:+.2f}R{ci}"
    if shadow:
        return (f"{head}: shadow sample (simulated fills, pocket pivots and VCP pooled, clustered by date): "
                f"descriptive only, never a verdict")
    b_lo, b_hi = bootstrap_ci(r)
    harm = n >= 2 and m < HARM_R and b_hi < 0
    if n < INTERIM_CHECKS[0]:
        return (f"{head}: too few to judge; an edge can neither be confirmed nor ruled out "
                f"(first interim check at {INTERIM_CHECKS[0]})")
    if n < target:
        boot = f"90% bootstrap CI {b_lo:+.2f} to {b_hi:+.2f}R"
        if harm:
            return (f"{head}: FAILS the interim harm check (mean below -0.3R, {boot}, entirely below 0): "
                    f"halt new submissions and review (protocol section 4){gap}")
        return (f"{head}: interim harm check only, passes ({boot}); not enough trades to confirm an "
                f"edge{gap}")
    sd = float(pd.to_numeric(r, errors='coerce').dropna().std(ddof=1))
    t = m / (sd / math.sqrt(n)) if sd > 0 else np.nan
    rr = pd.to_numeric(r, errors='coerce').dropna().sort_values()
    ex5 = float(rr.iloc[:-5].mean()) if n > 5 else np.nan
    ok = bool(m > 0 and t >= 2 and ex5 > 0)
    if unscored:
        return f"{head}, t={t:.2f}, ex-top-5 {ex5:+.2f}R: target reached, but no verdict yet{gap}"
    return (f"{head}, t={t:.2f}, ex-top-5 {ex5:+.2f}R: target reached; "
            f"{'meets' if ok else 'does NOT meet'} the protocol's verdict bar (mean > 0, t >= 2, "
            f"positive ex-top-5)")


def _r(x) -> str:
    """'+0.25R', or '-' when there is no value."""
    v = _fmt(x)
    return v if v == '-' else v + 'R'


def _fmt(x, nd: int = 2, signed: bool = True) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return f"{x:+.{nd}f}" if signed else f"{x:.{nd}f}"


# ═══════════════════════════════════════════════════════════════════════════
# SIGNAL LOG
# ═══════════════════════════════════════════════════════════════════════════
def load_signals(path: Path, as_of: Optional[date] = None) -> pd.DataFrame:
    """signal_log.csv with 'week' and 'r_class' (closed / open / pending / no_trade); empty if missing."""
    cols = ['signal_date', 'symbol', 'setup', 'fill_status', 'adopted_status', 'adopted_R', 'mfe3_R', 'mfe5_R']
    if Path(path).exists():
        df = pd.read_csv(path)
    else:
        logger.info(f"No signal log at {path}")
        df = pd.DataFrame(columns=cols)
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    df = df[pd.to_datetime(df['signal_date'], errors='coerce').notna()].copy()
    if as_of is not None:
        df = df[pd.to_datetime(df['signal_date']).dt.date <= as_of].copy()
    for c in ('adopted_R', 'mfe3_R', 'mfe5_R'):
        df[c] = pd.to_numeric(df[c], errors='coerce')
    # MFE over k sessions exists early only for trades that already closed (mostly stop-outs): count a
    # trade only once k sessions after its fill (the session after the signal) have passed for all
    end = np.datetime64(as_of or date.today())
    days = pd.to_datetime(df['signal_date']).values.astype('datetime64[D]')
    elapsed = np.busday_count(days, end) if len(df) else np.array([], dtype=int)
    for c, k in (('mfe3_R', 3), ('mfe5_R', 5)):
        df.loc[elapsed < k + 1, c] = np.nan
    df['symbol'] = df['symbol'].astype(str).str.upper()
    df['week'] = df['signal_date'].map(iso_week) if len(df) else pd.Series(dtype=object)
    df['r_class'] = pd.Series([_signal_class(f, s, r) for f, s, r in
                               zip(df['fill_status'], df['adopted_status'], df['adopted_R'])],
                              index=df.index, dtype=object)
    return df


def _signal_class(fill_status, adopted_status, r) -> str:
    if str(fill_status) in NO_TRADE:
        return 'no_trade'
    if pd.isna(r):
        return 'pending'
    return 'closed' if str(adopted_status) == 'closed' else 'open'


def signal_summary(df: pd.DataFrame, setups: List[str]) -> dict:
    out: Dict[str, object] = {'sig_new': int(len(df))}
    for s in setups:
        out[f"sig_setup_{s}"] = int((df['setup'] == s).sum())
    for f in FILL_STATUSES:
        out[f"sig_fill_{f}"] = int((df['fill_status'] == f).sum())
    for k in ('closed', 'open', 'pending', 'no_trade'):
        out[f"sig_{k}"] = int((df['r_class'] == k).sum())
    closed = r_stats(df.loc[df['r_class'] == 'closed', 'adopted_R'])
    for k in ('mean', 'median', 'win_rate', 'avg_win', 'avg_loss', 'expectancy'):
        out[f"sig_closed_{k}_R" if k not in ('win_rate',) else 'sig_closed_win_rate'] = closed[k]
    out['sig_open_mean_R'] = r_stats(df.loc[df['r_class'] == 'open', 'adopted_R'])['mean']
    scored = df.loc[df['r_class'].isin(['closed', 'open']), 'adopted_R']
    out['sig_scored_mean_R'] = r_stats(scored)['mean']
    out['sig_scored_win_rate'] = r_stats(scored)['win_rate']
    out['sig_mfe3_mean_R'] = float(df['mfe3_R'].mean()) if df['mfe3_R'].notna().any() else np.nan
    out['sig_mfe5_mean_R'] = float(df['mfe5_R'].mean()) if df['mfe5_R'].notna().any() else np.nan
    return out


# ═══════════════════════════════════════════════════════════════════════════
# CONVICTION LOG
# ═══════════════════════════════════════════════════════════════════════════
def load_conviction(path: Path, as_of: Optional[date] = None) -> pd.DataFrame:
    """conviction_log.csv with 'week', 'late' (bool), 'R' (NaN if unscored); empty if missing."""
    cols = ['date', 'symbol', 'conviction', 'setup', 'late', 'fill_status', 'status', 'R']
    if Path(path).exists():
        df = pd.read_csv(path)
    else:
        logger.info(f"No conviction log at {path}")
        df = pd.DataFrame(columns=cols)
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    df = df[pd.to_datetime(df['date'], errors='coerce').notna()].copy()
    if as_of is not None:
        df = df[pd.to_datetime(df['date']).dt.date <= as_of].copy()
    df['late'] = (df['late'].astype(str).str.lower() == 'true').astype(bool)
    df['R'] = pd.to_numeric(df['R'], errors='coerce')
    # an unfilled idea has no trade: its R (if any) is not a result
    df.loc[df['fill_status'].astype(str) != 'filled', 'R'] = np.nan
    df['conviction'] = pd.to_numeric(df['conviction'], errors='coerce')
    df['symbol'] = df['symbol'].astype(str).str.upper()
    df['week'] = df['date'].map(iso_week) if len(df) else pd.Series(dtype=object)
    return df


def conviction_summary(conv: pd.DataFrame, sig: pd.DataFrame) -> dict:
    """Week-row fields: idea counts, mean R (on-time, scored), overlap with bot signals."""
    ontime = conv.loc[~conv['late'].astype(bool)]
    bot_keys = set(zip(sig['symbol'], sig['week']))
    overlap = [k in bot_keys for k in zip(ontime['symbol'], ontime['week'])]
    s = r_stats(ontime['R'])
    return {'conv_ideas': int(len(ontime)), 'conv_late_excluded': int(conv['late'].sum()),
            'conv_scored': s['n'], 'conv_closed': int((ontime['status'] == 'closed').sum()),
            'conv_mean_R': s['mean'], 'conv_overlap_bot': int(sum(overlap))}


def conviction_by_level(conv: pd.DataFrame) -> pd.DataFrame:
    """Levels 1-5 (late excluded): ideas, scored, closed, mean R (open trades marked to the last close)."""
    ontime = conv.loc[~conv['late'].astype(bool)]
    rows = []
    for lvl in range(1, 6):
        g = ontime[ontime['conviction'] == lvl]
        s = r_stats(g['R'])
        rows.append({'conviction': lvl, 'ideas': int(len(g)), 'scored': s['n'],
                     'closed': int((g['status'] == 'closed').sum()), 'mean_R': s['mean'],
                     'win_rate': s['win_rate']})
    return pd.DataFrame(rows)


def owner_vs_bot(conv: pd.DataFrame, sig: pd.DataFrame) -> dict:
    """Owner ideas vs bot signals keyed by (symbol, ISO week): overlap count and the mean R of each
    side's overlapping / exclusive picks (bot R = adopted_R of scored signals, open marked to market)."""
    ontime = conv.loc[~conv['late'].astype(bool)]
    o_keys = set(zip(ontime['symbol'], ontime['week']))
    b_keys = set(zip(sig['symbol'], sig['week']))
    both = o_keys & b_keys
    o_in = [k in both for k in zip(ontime['symbol'], ontime['week'])]
    b_in = [k in both for k in zip(sig['symbol'], sig['week'])]
    o_in_s = pd.Series(o_in, index=ontime.index, dtype=bool)
    b_in_s = pd.Series(b_in, index=sig.index, dtype=bool)
    bot_r = sig['adopted_R'].where(sig['r_class'].isin(['closed', 'open']))
    return {'overlap': len(both),
            'owner_only_n': r_stats(ontime.loc[~o_in_s, 'R'])['n'],
            'owner_only_mean_R': r_stats(ontime.loc[~o_in_s, 'R'])['mean'],
            'owner_overlap_mean_R': r_stats(ontime.loc[o_in_s, 'R'])['mean'],
            'bot_only_n': r_stats(bot_r[~b_in_s])['n'], 'bot_only_mean_R': r_stats(bot_r[~b_in_s])['mean'],
            'bot_overlap_mean_R': r_stats(bot_r[b_in_s])['mean']}


# ═══════════════════════════════════════════════════════════════════════════
# PAPER ORDERS (ledger + exit_state)
# ═══════════════════════════════════════════════════════════════════════════
TRADE_COLUMNS = ['trade_date', 'symbol', 'setup', 'week', 'entry_state', 'placed', 'entry_qty',
                 'entry_price', 'initial_stop', 'status', 'sold_qty', 'R', 'r_note']


def load_ledger_events(orders_dir: Path) -> Tuple[List[dict], Optional[str]]:
    """(events without PAPERCHECK- refs, error text or None). Missing ledger -> ([], None)."""
    path = Path(orders_dir) / "ledger.jsonl"
    try:
        events = ol.OrderLedger(path=path).events()
    except (ol.LedgerCorrupt, OSError) as exc:
        logger.error(f"Ledger unreadable: {exc}")
        return [], f"ledger unreadable: {exc}"
    return [e for e in events if not str(e.get('order_ref') or '').startswith(PAPERCHECK_PREFIX)], None


def load_exit_trades(orders_dir: Path) -> Dict[Tuple[str, str], dict]:
    """exit_state.json trades keyed by (entry_date, SYMBOL); {} if missing or unreadable."""
    path = Path(orders_dir) / "exit_state.json"
    if not path.exists():
        return {}
    try:
        trades = json.loads(path.read_text()).get('trades', {}) or {}
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(f"exit_state {path} unreadable ({exc}): ignored")
        return {}
    return {(str(t.get('entry_date')), str(t.get('symbol', '')).upper()): t for t in trades.values()}


def _order_key(ev: dict) -> tuple:
    return (str(ev.get('trade_date')), str(ev.get('symbol', '')).upper(), ev.get('setup'),
            ev.get('role'), ev.get('order_ref'), ev.get('order_id'))


def _f(x) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if math.isnan(v) else v


def ledger_trades(events: List[dict], exit_trades: Dict[Tuple[str, str], dict],
                  as_of: Optional[date] = None) -> pd.DataFrame:
    """One row per bot entry (trade_date, symbol): placed / filled / status and realised R if derivable."""
    latest: Dict[tuple, dict] = {}
    for ev in events:
        latest[_order_key(ev)] = ev
    by_trade: Dict[Tuple[str, str], List[dict]] = {}
    for ev in latest.values():
        by_trade.setdefault((str(ev.get('trade_date')), str(ev.get('symbol', '')).upper()), []).append(ev)
    first_stop: Dict[Tuple[str, str], float] = {}
    trade_state: Dict[Tuple[str, str], str] = {}
    for ev in events:                                   # file order: first stop intent, last trade event
        k = (str(ev.get('trade_date')), str(ev.get('symbol', '')).upper())
        if ev.get('role') == 'stop' and k not in first_stop and _f(ev.get('aux_price')) > 0:
            first_stop[k] = _f(ev.get('aux_price'))
        if ev.get('role') == ol.TRADE_ROLE:
            trade_state[k] = str(ev.get('state'))

    rows = []
    for k, evs in by_trade.items():
        entry = next((e for e in evs if e.get('role') == 'entry'), None)
        if entry is None:
            continue                                     # orphan exits with no entry: not a trade
        d, sym = k
        if as_of is not None and pd.Timestamp(d).date() > as_of:
            continue
        xt = exit_trades.get(k, {})
        qty, px = _f(entry.get('filled_qty')), _f(entry.get('avg_fill_price'))
        stop = first_stop.get(k) or _f(xt.get('initial_stop')) or _f((entry.get('plan') or {}).get('stop'))
        state = str(entry.get('state'))
        placed = not (state == 'intended' or (state == 'cancelled' and entry.get('reason') == 'never_reached_tws'))
        sells = [e for e in evs if e.get('role') in ol.SELL_ROLES and _f(e.get('filled_qty')) > 0]
        covers = [e for e in evs if e.get('role') == ol.COVER_ROLE and _f(e.get('filled_qty')) > 0]
        sold = sum(_f(e.get('filled_qty')) for e in sells) - sum(_f(e.get('filled_qty')) for e in covers)
        if qty <= 0:
            status = 'no_fill'
        else:
            status = trade_state.get(k) or ('closed' if xt.get('status') == 'closed' else 'open')
        r, note = np.nan, ''
        if status == 'closed':
            r, note = _realised_r(qty, px, stop, sells, covers)
        elif status == 'manual_touched':
            note = 'manual_touched: the bot stopped managing it'
        rows.append({'trade_date': d, 'symbol': sym, 'setup': entry.get('setup'), 'week': iso_week(d),
                     'entry_state': state, 'placed': placed, 'entry_qty': qty, 'entry_price': px,
                     'initial_stop': stop, 'status': status, 'sold_qty': sold, 'R': r, 'r_note': note})
    df = pd.DataFrame(rows, columns=TRADE_COLUMNS)
    return df.sort_values(['trade_date', 'symbol']).reset_index(drop=True) if len(df) else df


def _realised_r(qty: float, px: float, stop: float, sells: List[dict], covers: List[dict]) -> Tuple[float, str]:
    """Gross R of a closed trade from its fills, or (NaN, why not)."""
    if px <= 0 or stop <= 0 or stop >= px:
        return np.nan, f"no usable entry fill / initial stop (fill {px:g}, stop {stop:g})"
    fills = sells + covers
    if any(_f(e.get('avg_fill_price')) <= 0 for e in fills):
        return np.nan, "an exit fill has no price"
    sold = sum(_f(e.get('filled_qty')) for e in sells) - sum(_f(e.get('filled_qty')) for e in covers)
    if abs(sold - qty) > 1e-6:
        return np.nan, f"exit fills {sold:g} of {qty:g} shares in the ledger"
    pnl = (sum(_f(e['filled_qty']) * _f(e['avg_fill_price']) for e in sells)
           - sum(_f(e['filled_qty']) * _f(e['avg_fill_price']) for e in covers) - qty * px)
    return round(pnl / (qty * (px - stop)), 4), 'gross of commissions'


def order_summary(trades: pd.DataFrame) -> dict:
    closed = trades.loc[trades['status'] == 'closed']
    s = r_stats(closed['R'])
    return {'ord_entries': int(len(trades)), 'ord_placed': int(trades['placed'].sum()),
            'ord_filled': int((trades['entry_qty'] > 0).sum()), 'ord_closed': int(len(closed)),
            'ord_closed_with_R': s['n'], 'ord_open': int((trades['status'] == 'open').sum()),
            'ord_manual_touched': int((trades['status'] == 'manual_touched').sum()),
            'ord_mean_R': s['mean'], 'ord_win_rate': s['win_rate'], 'ord_total_R': s['total']}


# ═══════════════════════════════════════════════════════════════════════════
# REPORT
# ═══════════════════════════════════════════════════════════════════════════
def build(as_of: date, weeks: int, signal_path: Optional[Path] = None, conviction_path: Optional[Path] = None,
          orders_dir: Optional[Path] = None) -> dict:
    """Everything the report shows: week rows (+ ALL), per-level conviction, owner vs bot, trades, notes."""
    signal_path = Path(signal_path or SIGNAL_LOG)
    conviction_path = Path(conviction_path or CONVICTION_LOG)
    orders_dir = Path(orders_dir or ol.ORDERS_DIR)
    sig = load_signals(signal_path, as_of)
    conv = load_conviction(conviction_path, as_of)
    events, ledger_err = load_ledger_events(orders_dir)
    trades = ledger_trades(events, load_exit_trades(orders_dir), as_of)
    setups = sorted(sig['setup'].dropna().astype(str).unique()) if len(sig) else []

    def row(label: str, start: Optional[date], s: pd.DataFrame, c: pd.DataFrame, t: pd.DataFrame) -> dict:
        return {'week': label, 'week_start': start.isoformat() if start else '',
                **signal_summary(s, setups), **conviction_summary(c, s), **order_summary(t)}

    wk = week_list(as_of, weeks)
    rows = [row(w, date.fromisocalendar(int(w[:4]), int(w[-2:]), 1), sig.loc[sig['week'] == w],
                conv.loc[conv['week'] == w], trades.loc[trades['week'] == w]) for w in wk]
    rows.append(row('ALL', None, sig, conv, trades))
    older = int((sig['week'] < wk[0]).sum()) if len(sig) else 0
    return {'as_of': as_of, 'weeks': pd.DataFrame(rows), 'by_level': conviction_by_level(conv),
            'owner_vs_bot': owner_vs_bot(conv, sig), 'trades': trades, 'ledger_error': ledger_err,
            'paper_status': verdict_status(trades.loc[trades['status'] == 'closed', 'R'],
                                           unscored=int(((trades['status'] == 'closed') & trades['R'].isna()).sum())),
            'paper_ci': mean_ci(trades.loc[trades['status'] == 'closed', 'R']),
            'shadow_status': verdict_status(sig.loc[sig['r_class'] == 'closed', 'adopted_R'], shadow=True),
            'shadow_ci': mean_ci(sig.loc[sig['r_class'] == 'closed', 'adopted_R']),
            'conviction_present': Path(conviction_path).exists(),
            'ledger_present': (orders_dir / "ledger.jsonl").exists(), 'signals_before_window': older,
            'closed_no_r': int(((trades['status'] == 'closed') & trades['R'].isna()).sum())}


def week_table(weeks: pd.DataFrame) -> str:
    """Concise fixed-width table: one line per week + ALL."""
    hdr = (f"{'week':<9} {'sig':>4} {'fill':>4} {'pend':>4} {'clsd':>4} {'meanR':>6} {'win%':>5} "
           f"{'openR':>6} {'mfe5':>5} | {'ideas':>5} {'ovl':>3} {'cnvR':>6} | {'ent':>3} {'fill':>4} "
           f"{'clsd':>4} {'open':>4} {'ordR':>6}")
    lines = [hdr, '-' * len(hdr)]
    for _, r in weeks.iterrows():
        wr = r['sig_closed_win_rate']
        lines.append(
            f"{r['week']:<9} {r['sig_new']:>4} {r['sig_fill_filled']:>4} {r['sig_pending']:>4} "
            f"{r['sig_closed']:>4} {_fmt(r['sig_closed_mean_R']):>6} "
            f"{('-' if pd.isna(wr) else f'{wr * 100:.0f}'):>5} {_fmt(r['sig_open_mean_R']):>6} "
            f"{_fmt(r['sig_mfe5_mean_R']):>5} | {r['conv_ideas']:>5} {r['conv_overlap_bot']:>3} "
            f"{_fmt(r['conv_mean_R']):>6} | {r['ord_entries']:>3} {r['ord_filled']:>4} {r['ord_closed']:>4} "
            f"{r['ord_open']:>4} {_fmt(r['ord_mean_R']):>6}")
    return '\n'.join(lines)


def notes(rep: dict) -> List[str]:
    out = []
    if rep['ledger_error']:
        out.append(f"LEDGER UNREADABLE ({rep['ledger_error']}): paper-order figures below are NOT real zeros.")
    elif not rep['ledger_present']:
        out.append("No order ledger yet (Data/orders/ledger.jsonl): paper-order figures are zeros.")
    if not rep['conviction_present']:
        out.append("No conviction log yet (Data/paper_tracking/conviction_log.csv).")
    if rep['closed_no_r']:
        out.append(f"{rep['closed_no_r']} closed paper trade(s) without a derivable R (exit fills incomplete "
                   f"in the ledger): left out of mean R.")
    if rep['signals_before_window']:
        out.append(f"{rep['signals_before_window']} signal(s) dated before the shown weeks are in ALL only.")
    out.append("Signal-log R is simulated (bars, adopted exit, net of costs); 'open' R is marked to the last "
               "close and moves. Paper-order R is from real paper fills, gross of commissions.")
    out.append("The signal log mixes pocket pivots with VCP signals: it is NOT the protocol's frozen VCP sample.")
    return out


def render_text(rep: dict) -> str:
    w = rep['weeks']
    a = w.iloc[-1]
    pv = rep['owner_vs_bot']
    setups = [c[len('sig_setup_'):] + '=' + str(a[c]) for c in w.columns if c.startswith('sig_setup_')]
    lines = [f"FORWARD REPORT as of {rep['as_of']} (read-only)", '',
             "Columns: signal log (sig new, fill filled, pend unscored, clsd closed, meanR/win% of closed, openR "
             "mark-to-market, mfe5 mean) | conviction (ideas on-time, ovl overlap with bot, cnvR mean) | "
             "paper orders (ent entries, fill, clsd, open, ordR mean realised)", '',
             week_table(w), '',
             f"Signals by setup (ALL): {', '.join(setups) or 'none'}",
             "Fill status (ALL): " + ', '.join(f"{f}={a['sig_fill_' + f]}" for f in FILL_STATUSES),
             f"Closed signals: median {_r(a['sig_closed_median_R'])}, avg win {_r(a['sig_closed_avg_win_R'])}, "
             f"avg loss {_r(a['sig_closed_avg_loss_R'])}, expectancy {_r(a['sig_closed_expectancy_R'])}; "
             f"all scored (closed + open) mean {_r(a['sig_scored_mean_R'])}; mfe3 {_r(a['sig_mfe3_mean_R'])}",
             '', "Conviction by level (late excluded):"]
    bl = rep['by_level']
    lines += [f"  {int(r.conviction)}: ideas {r.ideas}, scored {r.scored}, closed {r.closed}, mean "
              f"{_r(r.mean_R)}" for r in bl.itertuples()]
    lines += [f"Owner vs bot (same symbol + week): overlap {pv['overlap']}; owner-only n={pv['owner_only_n']} "
              f"mean {_r(pv['owner_only_mean_R'])}; bot-only n={pv['bot_only_n']} mean "
              f"{_r(pv['bot_only_mean_R'])} (descriptive only, protocol section 6)", '',
              f"Paper orders (ALL): entries {a['ord_entries']}, placed {a['ord_placed']}, filled {a['ord_filled']}, "
              f"closed {a['ord_closed']} (with R {a['ord_closed_with_R']}), open {a['ord_open']}, "
              f"manual_touched {a['ord_manual_touched']}", '',
              f"PROGRESS TO A VERDICT (target ~{TARGET_N} closed trades)",
              f"  Paper trades (primary): {rep['paper_status']}",
              f"  Signal log closed (shadow, simulated): {rep['shadow_status']}", '']
    lines += [f"Note: {n}" for n in notes(rep)]
    return '\n'.join(lines)


def render_md(rep: dict) -> str:
    w = rep['weeks']
    show = w[['week', 'sig_new', 'sig_fill_filled', 'sig_pending', 'sig_closed', 'sig_closed_mean_R',
              'sig_closed_win_rate', 'sig_open_mean_R', 'conv_ideas', 'conv_overlap_bot', 'conv_mean_R',
              'ord_entries', 'ord_filled', 'ord_closed', 'ord_open', 'ord_mean_R']].copy()
    show.columns = ['Week', 'Signals', 'Filled', 'Pending', 'Closed', 'Closed mean R', 'Win rate',
                    'Open R (MTM)', 'Owner ideas', 'Overlap', 'Owner mean R', 'Entries', 'Entry fills',
                    'Trades closed', 'Trades open', 'Trade mean R']
    md = [f"# Forward report, as of {rep['as_of']}", '',
          "Read-only summary of forward (unseen-data) evidence. R = P&L in multiples of the risk set at entry.",
          '', "## Progress to a verdict", '',
          f"Target: about {TARGET_N} closed trades to detect +{TARGET_EFFECT_R:g}R per trade.", '',
          f"- **Paper trades (primary):** {rep['paper_status']}",
          f"- **Signal log, closed (shadow, simulated fills):** {rep['shadow_status']}", '',
          "## Weekly", '', _md_table(show), '', "## Conviction by level (late ideas excluded)", '',
          _md_table(rep['by_level']), '']
    pv = rep['owner_vs_bot']
    md += ["## Owner vs bot", '',
           f"Same symbol in the same ISO week: {pv['overlap']} overlap(s). Owner-only: n={pv['owner_only_n']}, "
           f"mean {_r(pv['owner_only_mean_R'])}. Bot-only: n={pv['bot_only_n']}, mean "
           f"{_r(pv['bot_only_mean_R'])}. Descriptive only until each side has its own ~200 trades.", '']
    t = rep['trades']
    md += ["## Paper trades", '', _md_table(t.drop(columns=['week'])) if len(t) else "No bot entries in the ledger.",
           '', "## Notes", ''] + [f"- {n}" for n in notes(rep)]
    return '\n'.join(md) + '\n'


def _md_table(df: pd.DataFrame) -> str:
    def cell(v) -> str:
        if isinstance(v, float):
            return '-' if math.isnan(v) else f"{v:.3g}" if abs(v) >= 100 else f"{v:.2f}"
        return str(v)
    head = '| ' + ' | '.join(map(str, df.columns)) + ' |'
    sep = '|' + '|'.join(['---'] * len(df.columns)) + '|'
    body = ['| ' + ' | '.join(cell(v) for v in r) + ' |' for r in df.itertuples(index=False)]
    return '\n'.join([head, sep] + body)


def write_outputs(rep: dict, out_dir: Path) -> Tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"forward_report_{rep['as_of'].isoformat()}.csv"
    rep['weeks'].round(4).to_csv(csv_path, index=False)
    md_path = out_dir / "forward_report_latest.md"
    md_path.write_text(render_md(rep))
    return csv_path, md_path


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Weekly forward-test report (read-only)")
    ap.add_argument('--weeks', type=int, default=DEFAULT_WEEKS, help="weeks shown (default 8)")
    ap.add_argument('--out', type=Path, default=None, help="output folder (default Data/reports)")
    ap.add_argument('--as-of', type=date.fromisoformat, default=None, help="YYYY-MM-DD (default today)")
    args = ap.parse_args(argv)
    if args.weeks < 1:
        ap.error("--weeks must be >= 1")
    rep = build(args.as_of or date.today(), args.weeks)
    print(render_text(rep))
    csv_path, md_path = write_outputs(rep, args.out or REPORTS_DIR)
    print(f"\nWrote {csv_path} and {md_path}")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
    raise SystemExit(main())
