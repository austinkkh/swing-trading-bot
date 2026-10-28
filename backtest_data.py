#!/usr/bin/env python3
"""
Backtest data — fetch and cache daily bars from IBKR for the scanner's universe.

Bars are cached as Data/bars/<SYMBOL>.csv (git-ignored) and refreshed when the
cache is older than today, so backtests can run repeatedly without TWS.

Usage:
  python backtest_data.py                 # scanner universe + SPY, 3 years, client 98
  python backtest_data.py --years 2 --client-id 98

Survivorship bias: the universe is today's list, so stocks that were delisted or
fell out of the lists during the test period are missing. Results lean optimistic.
"""

import argparse
import logging
import time
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

import config

logger = logging.getLogger(__name__)

BARS_DIR = Path(config.OUTPUT_DIR).expanduser().parent / "Data" / "bars"
BENCHMARK = "SPY"
DEFAULT_CLIENT_ID = 98  # distinct from the watcher (config CLIENT_ID), premarket (11), alerts (12)


def universe_symbols() -> List[str]:
    """The scanner's universe (IBD/TradingView lists + manual watchlist), plus the benchmark."""
    from momentum_scanner import MomentumScanner
    symbols = MomentumScanner().get_stock_universe() or []
    return sorted(set(symbols) | {BENCHMARK})


def cache_path(symbol: str) -> Path:
    return BARS_DIR / f"{symbol.upper()}.csv"


def is_fresh(symbol: str) -> bool:
    path = cache_path(symbol)
    return path.exists() and datetime.fromtimestamp(path.stat().st_mtime).date() == date.today()


def fetch_and_cache(symbols: List[str], years: int = 3, client_id: int = DEFAULT_CLIENT_ID,
                    host: str = config.TWS_HOST, port: int = config.TWS_PORT) -> Dict[str, int]:
    """Fetch daily bars for symbols not cached today. Returns {symbol: bar count} for fetched ones."""
    from ibkr_connector import IBKRConnector

    BARS_DIR.mkdir(parents=True, exist_ok=True)
    todo = [s for s in symbols if not is_fresh(s)]
    logger.info(f"{len(symbols) - len(todo)} cached today, fetching {len(todo)}")
    if not todo:
        return {}

    conn = IBKRConnector()
    if not conn.connect_and_run(host, port, client_id):
        raise ConnectionError(f"Couldn't connect to TWS at {host}:{port} as client {client_id}")
    fetched = {}

    def fetch_one(symbol: str) -> bool:
        bars = conn.fetch_historical_data(symbol, duration=f"{years} Y", bar_size="1 day")
        if bars:
            pd.DataFrame({
                'date': pd.to_datetime([str(b.date) for b in bars]),
                'open': [b.open for b in bars], 'high': [b.high for b in bars],
                'low': [b.low for b in bars], 'close': [b.close for b in bars],
                'volume': [float(b.volume) for b in bars],
            }).to_csv(cache_path(symbol), index=False)
            fetched[symbol] = len(bars)
        time.sleep(0.1)  # stay well inside IBKR's historical-data pacing
        return bool(bars)

    try:
        failed = []
        for i, symbol in enumerate(todo, 1):
            if not fetch_one(symbol):
                failed.append(symbol)
            if i % 50 == 0:
                logger.info(f"  {i}/{len(todo)} fetched")
        # The first requests after connecting sometimes fail transiently (seen: AAPL, AA, AAOI);
        # one retry pass separates those from symbols IBKR genuinely has no data for.
        if failed:
            logger.info(f"Retrying {len(failed)} that returned no data")
            for symbol in failed:
                fetch_one(symbol)
    finally:
        conn.disconnect_and_stop()
    return fetched


def load_bars(symbol: str) -> Optional[pd.DataFrame]:
    """Cached daily bars, oldest first, or None if not cached."""
    path = cache_path(symbol)
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=['date'])
    return df.sort_values('date').reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Fetch and cache daily bars for backtesting")
    parser.add_argument('--years', type=int, default=3)
    parser.add_argument('--client-id', type=int, default=DEFAULT_CLIENT_ID)
    parser.add_argument('--symbols-from', help='Also fetch every symbol in these dated scan lists '
                        '(same formats as backtest.py --universe-lists)')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger('ibapi').setLevel(logging.WARNING)

    symbols = universe_symbols()
    if args.symbols_from:
        from backtest import load_universe_lists
        listed = sorted(set(load_universe_lists(args.symbols_from)['symbol']))
        logger.info(f"{len(listed)} symbols in {args.symbols_from}")
        symbols = sorted(set(symbols) | set(listed))
    fetched = fetch_and_cache(symbols, years=args.years, client_id=args.client_id)
    missing = [s for s in symbols if load_bars(s) is None]
    logger.info(f"Done: {len(fetched)} fetched, {len(symbols) - len(missing)}/{len(symbols)} cached")
    if missing:
        logger.info(f"No data for {len(missing)}: {', '.join(missing[:15])}{' ...' if len(missing) > 15 else ''}")


if __name__ == "__main__":
    main()
