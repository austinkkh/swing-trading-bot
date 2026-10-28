"""
═══════════════════════════════════════════════════════════════════════════════
WATCH MODE v2 - Variable Refresh + ORB Detection
─────────────────────────────────────────────────────────────────────────────
Two alert types during market hours:
  1. ORB BREAKOUT — opening range broken with volume (range end to 10:30)
  2. MIGRATION — ELITE_HOLD stock pulled back to EMA, became actionable

Variable refresh schedule:
  9:30 - 9:40am   →  IDLE (opening range forming; follows orb_detector.ORB_WINDOW_MINUTES)
  9:40 - 10:00am  →  30 sec (ORB hunt - fast)
  10:00 - 10:30am →  30 sec (ORB hunt - winding down)
  10:30 - 3:25pm  →  5 min  (quiet midday)
  3:25 - 4:00pm   →  1 min  (closing window)

Usage:
  python watch_mode.py
═══════════════════════════════════════════════════════════════════════════════
"""

import time
import logging
import pandas as pd

from market_time import is_trading_day, market_close_time, now_et
from datetime import datetime, time as dt_time
from pathlib import Path

from config import *
from momentum_scanner import MomentumScanner
from orb_detector import ORBDetector, ORB_FORMATION_END
from symbol_hygiene import apply_symbol_hygiene
import notify as notifier

# ═══════════════════════════════════════════════════════════════════════════
# WATCH SCHEDULE CONFIG
# ═══════════════════════════════════════════════════════════════════════════
MARKET_OPEN_TIME       = dt_time(9, 30)
ORB_FORMATION_DONE     = ORB_FORMATION_END  # follows the detector's ORB window (9:40 for 10 min)
ORB_FAST_END           = dt_time(10, 0)
ORB_HUNT_DONE          = dt_time(10, 30)
CLOSING_WINDOW_START   = dt_time(15, 25)
MARKET_CLOSE_TIME      = dt_time(16, 0)

# Refresh intervals (seconds)
INTERVAL_ORB_FAST      = 30   # range end (9:40) - 10:00am
INTERVAL_ORB_MEDIUM    = 30   # 10:00-10:30am
INTERVAL_MIDDAY        = 300  # 10:30-3:25pm
INTERVAL_CLOSING       = 60   # 3:25-4:00pm

WATCH_SOUND_ON_ALERT   = True
STALE_DATA_BANNER_MIN  = 2    # intraday bars trailing the clock by more than this get a banner
WATCH_FARM_WAIT_SEC    = 60   # farm wait per retry inside the loop (the 9:20 scan uses 600)
LINK_PROBE_AFTER_SEC   = 300  # link marked down this long: probe IB once to see if it's back

logger = logging.getLogger(__name__)


# Watcher init-failure banner: only from ALERT_FROM (ET) to the close on trading days, once per 2 h
INIT_ALERT_FROM = dt_time(9, 0)
INIT_ALERT_DEDUP_SECONDS = 2 * 3600


def init_failure_should_alert(now: datetime) -> bool:
    """True when a failed scanner init is worth a banner: a trading day between 9:00 ET and the close.

    Outside that (nights, weekends, holidays, IBKR's nightly restart) TWS being down is normal and
    the failure only goes to the log.
    """
    return is_trading_day(now.date()) and INIT_ALERT_FROM <= now.time() < market_close_time(now.date())


class WatchModeMonitor:
    """Intraday monitor: ORB breakouts + ELITE_HOLD migrations."""
    
    def __init__(self):
        self.watchlist_symbols = []
        self.previous_state = {}
        self.migration_log = []
        self.orb_alerts_today = set()
        self.scan_count = 0
        self.loaded_scan = None          # (path, mtime) of the scan the watchlist came from
        self.last_data_lag = None        # worst intraday bar lag (min) seen in the last ORB scan
        self.data_down_since = None      # monotonic time data_ready first went False
        self.last_link_probe = None      # monotonic time of the last link probe
    
    def get_current_phase(self):
        """Returns (phase_name, sleep_seconds, should_scan)"""
        today = now_et()
        if not is_trading_day(today.date()):
            return ("MARKET_CLOSED", 1800, False)   # weekends/holidays: no IBKR requests
        now = today.time()
        
        if now < MARKET_OPEN_TIME:
            return ("PRE_MARKET", 600, False)
        if MARKET_OPEN_TIME <= now < ORB_FORMATION_DONE:
            return ("ORB_FORMING", 30, False)
        if ORB_FORMATION_DONE <= now < ORB_FAST_END:
            return ("ORB_HUNT_FAST", INTERVAL_ORB_FAST, True)
        if ORB_FAST_END <= now < ORB_HUNT_DONE:
            return ("ORB_HUNT_SLOW", INTERVAL_ORB_MEDIUM, True)
        if ORB_HUNT_DONE <= now < CLOSING_WINDOW_START:
            return ("MIDDAY_QUIET", INTERVAL_MIDDAY, True)
        if CLOSING_WINDOW_START <= now < MARKET_CLOSE_TIME:
            return ("CLOSING_WINDOW", INTERVAL_CLOSING, True)
        return ("AFTER_HOURS", 600, False)
    
    @staticmethod
    def data_ready(connector) -> bool:
        """False while TWS reports the historical-data farm or its server link down."""
        ready = getattr(connector, 'data_farms_ready', None)
        return True if ready is None else bool(ready())

    def probe_stale_link(self, connector, now_mono: float) -> bool:
        """
        TWS marks the server link up again only on 1101/1102 or a farm-OK message. If it restores
        the link without sending one, the watcher would skip every scan until restart (review L5).
        After LINK_PROBE_AFTER_SEC of 'down', ask IB for SPY's contract details (cheap, read-only)
        at most once per interval; an answer proves the link is up, so clear the stale flag.
        Only the link flag: a farm TWS explicitly reported broken stays broken. True if cleared.
        """
        if getattr(connector, 'server_link_ok', True) or self.data_down_since is None:
            return False
        if now_mono - self.data_down_since < LINK_PROBE_AFTER_SEC:
            return False
        if self.last_link_probe is not None and now_mono - self.last_link_probe < LINK_PROBE_AFTER_SEC:
            return False
        self.last_link_probe = now_mono
        if connector.fetch_contract_details('SPY', timeout=5) is None:
            return False
        logger.info("TWS answered a contract-details probe: marking the server link up")
        connector._set_link(True, 0, 'probe')
        return True

    def is_orb_window(self) -> bool:
        now = now_et().time()
        return ORB_FORMATION_DONE <= now < ORB_HUNT_DONE
    
    @staticmethod
    def latest_scan_csv():
        """Newest full-scan COMPLETE csv, or None. Old *_watch_COMPLETE.csv files (written by
        earlier versions of the watcher) only hold the watchlist, so they're never a source."""
        scans_dir = Path(OUTPUT_DIR)
        if not scans_dir.exists():
            return None
        csvs = sorted((p for p in scans_dir.glob("scan_*_COMPLETE.csv") if "_watch_" not in p.name),
                      reverse=True)
        return csvs[0] if csvs else None

    def reload_if_new_scan(self) -> bool:
        """Reload the watchlist when a newer scan exists (the 9:20 premarket scan lands after the
        watcher has started for the day). Review LT-01. Returns True if it reloaded."""
        latest = self.latest_scan_csv()
        if latest is None:
            return False
        key = (latest, latest.stat().st_mtime)
        if key == self.loaded_scan:
            return False
        logger.info(f"🆕 New scan {latest.name}: reloading the watchlist")
        return self.load_latest_elite_hold()

    def load_latest_elite_hold(self) -> bool:
        """Load watchlist from latest COMPLETE scan (Elite + Buy stocks)"""
        latest = self.latest_scan_csv()
        if latest is None:
            logger.error(f"❌ No COMPLETE scan CSVs in {OUTPUT_DIR}: run python momentum_scanner.py first")
            return False

        df = pd.read_csv(latest)
        self.loaded_scan = (latest, latest.stat().st_mtime)

        relevant = df[df['Tier'].isin(['TIER_ELITE_BUY', 'TIER_ELITE_HOLD', 'TIER_BUY', 'TIER_RS_LEADER'])]
        symbols_set = set(relevant['Symbol'].tolist())

        # Tiers are compared against the new scan from here on; yesterday's states would turn
        # into spurious migrations
        self.previous_state = {}
        for _, row in relevant.iterrows():
            self.previous_state[row['Symbol']] = row['Tier']
        
        self.watchlist_symbols = sorted(list(symbols_set))
        # Same dead/renamed filter as the scanner (read-only here: the watcher never writes state)
        try:
            self.watchlist_symbols, dropped = apply_symbol_hygiene(self.watchlist_symbols)
            if dropped:
                logger.info(f"   Skipped {len(dropped)} dead/renamed: {', '.join(s for s, _ in dropped)}")
        except Exception as e:                                   # noqa: BLE001
            logger.warning(f"Symbol hygiene skipped ({e}); watching the full list")
        
        logger.info(f"✓ Loaded {len(self.watchlist_symbols)} stocks from {latest.name}")
        logger.info(f"   ELITE_BUY:  {len(df[df['Tier']=='TIER_ELITE_BUY'])}")
        logger.info(f"   ELITE_HOLD: {len(df[df['Tier']=='TIER_ELITE_HOLD'])}")
        logger.info(f"   TIER_BUY:   {len(df[df['Tier']=='TIER_BUY'])}")
        logger.info(f"   RS_LEADER:  {len(df[df['Tier']=='TIER_RS_LEADER'])}")
        
        return len(self.watchlist_symbols) > 0
    
    @staticmethod
    def stale_data_banner(lag_min, data_delayed: bool = False):
        """One line per cycle when intraday data trails the clock, else None. Prices, ORH distance
        and volume in this cycle's alerts are that stale."""
        if lag_min is not None and lag_min > STALE_DATA_BANNER_MIN:
            return (f"⚠️ DATA DELAYED ~{lag_min:.0f}m: intraday bars trail the clock "
                    "(no real-time API market data on this session?); alerts below use stale prices")
        if lag_min is None and data_delayed:
            return "⚠️ DATA DELAYED (TWS 2188: no real-time API market data); prices ~15 min stale"
        return None

    def _lag_note(self) -> str:
        return f" (data ~{self.last_data_lag:.0f} min delayed)" if (
            self.last_data_lag is not None and self.last_data_lag > STALE_DATA_BANNER_MIN) else ""

    def run_orb_scan(self, orb_detector: ORBDetector):
        """Check ORB breakouts on watchlist"""
        new_breakouts = []
        lags = []
        
        for symbol in self.watchlist_symbols:
            if symbol in self.orb_alerts_today:
                continue
            
            try:
                signal = orb_detector.check_orb_breakout(symbol)
                if signal.data_lag_min is not None:
                    lags.append(signal.data_lag_min)
                if signal.has_breakout:
                    new_breakouts.append(signal)
                    self.orb_alerts_today.add(symbol)
            except Exception as e:
                logger.warning(f"ORB error {symbol}: {e}")
        
        self.last_data_lag = max(lags) if lags else None
        if new_breakouts:
            self._alert_orb_breakouts(new_breakouts)
        
        return len(new_breakouts)
    
    def run_migration_scan(self, scanner: MomentumScanner):
        """Run scanner on watchlist, detect HOLD→BUY migrations"""
        original_universe = scanner.get_stock_universe
        scanner.get_stock_universe = lambda: self.watchlist_symbols
        
        try:
            scanner.run_full_scan(scan_type="watch")
        except Exception as e:
            logger.error(f"❌ Migration scan error: {e}")
            return 0
        finally:
            scanner.get_stock_universe = original_universe
        
        migrations = self._detect_migrations(scanner)
        return len(migrations)
    
    def _detect_migrations(self, scanner: MomentumScanner):
        new_state = {}
        
        for stock in scanner.elite_buy_results:
            new_state[stock['symbol']] = 'TIER_ELITE_BUY'
        for stock in scanner.elite_hold_results:
            new_state[stock['symbol']] = 'TIER_ELITE_HOLD'
        for stock in scanner.buy_results:
            new_state[stock['symbol']] = 'TIER_BUY'
        for stock in getattr(scanner, 'rs_leader_results', []):
            new_state[stock['symbol']] = 'TIER_RS_LEADER'
        
        migrations = []
        for symbol in self.watchlist_symbols:
            prev = self.previous_state.get(symbol, 'UNKNOWN')
            curr = new_state.get(symbol, 'DROPPED')
            
            if prev != curr and curr == 'TIER_ELITE_BUY':
                matching = next(
                    (s for s in scanner.elite_buy_results if s['symbol'] == symbol), 
                    None
                )
                if matching:
                    migrations.append((prev, matching))
        
        if migrations:
            self._alert_migrations(migrations)
        
        for sym, state in new_state.items():
            self.previous_state[sym] = state
        
        return migrations
    
    def _alert_orb_breakouts(self, breakouts):
        print("\n" + "🚨"*40)
        print(f"\n  ⚡⚡⚡  ORB BREAKOUT ALERT: {len(breakouts)} STOCK(S)  ⚡⚡⚡")
        print("\n" + "🚨"*40)
        
        for sig in breakouts:
            print(f"\n  ➡️  {sig.symbol}")
            print(f"      Opening Range:  ${sig.orl:.2f} - ${sig.orh:.2f}")
            print(f"      Range Width:    {sig.or_range_pct:.2f}%")
            print(f"      Current:        ${sig.current_price:.2f} (+{sig.distance_above_orh_pct:.2f}% above ORH)")
            print(f"      Volume:         {sig.volume_ratio:.1f}x average")
            print(f"      Stop (ORL):     ${sig.stop_price:.2f} (-{sig.stop_pct:.2f}% risk)")
            if sig.data_lag_min is not None and sig.data_lag_min > STALE_DATA_BANNER_MIN:
                print(f"      ⚠ Data lag:     ~{sig.data_lag_min:.0f} min (price is that stale)")
        
        print("\n" + "🚨"*40 + "\n")
        
        if WATCH_SOUND_ON_ALERT:
            try:
                import subprocess
                symbols_str = ", ".join([s.symbol for s in breakouts])
                subprocess.run(
                    ["say", f"O R B breakout. {symbols_str}"],
                    timeout=5, check=False
                )
            except Exception:
                pass
    
    def _alert_migrations(self, migrations):
        print("\n" + "🔔"*40)
        print(f"\n  📌  MIGRATION ALERT: {len(migrations)} STOCK(S) NOW BUYABLE{self._lag_note()}  📌")
        print("\n" + "🔔"*40)
        
        for prev_tier, stock in migrations:
            print(f"\n  ➡️  {stock['symbol']}")
            print(f"      Migrated:    {prev_tier} → TIER_ELITE_BUY")
            print(f"      Setup:       {stock.get('elite_entry_setup', '-')}")
            print(f"      Price:       ${stock['current_price']:.2f}")
            print(f"      Stop:        ${stock.get('stop_loss', 0):.2f} ({stock.get('stop_loss_pct', 0):.1f}%)")
            print(f"      ATR×50MA:    {stock.get('atr_multiple_from_50ma', 0):.1f}")
            print(f"      Priority:    {stock.get('priority_rating', 0):.1f}")
            
            self.migration_log.append({
                'time': now_et(),
                'symbol': stock['symbol'],
                'from_tier': prev_tier,
                'setup': stock.get('elite_entry_setup'),
                'stop_pct': stock.get('stop_loss_pct')
            })
        
        print("\n" + "🔔"*40 + "\n")
        
        if WATCH_SOUND_ON_ALERT:
            try:
                import subprocess
                symbols_str = ", ".join([m[1]['symbol'] for m in migrations])
                subprocess.run(
                    ["say", f"Migration alert. {symbols_str} now actionable"],
                    timeout=5, check=False
                )
            except Exception:
                pass
    
    def run_forever(self):
        if not self.load_latest_elite_hold():
            return
        
        print("\n" + "═"*70)
        print("  🔎  WATCH MODE v2 - VARIABLE REFRESH + ORB DETECTION  🔎")
        print("═"*70)
        print(f"  Watchlist:       {len(self.watchlist_symbols)} stocks")
        print(f"  Schedule:")
        orb_end = ORB_FORMATION_DONE.strftime("%H:%M")
        print(f"    9:30 - {orb_end}am   →  IDLE (opening range forming)")
        print(f"    {orb_end} - 10:00am  →  30 sec (ORB hunt - fast)")
        print(f"    10:00 - 10:30am →  30 sec (ORB hunt - winding down)")
        print(f"    10:30 - 3:25pm  →  5 min (quiet midday)")
        print(f"    3:25 - 4:00pm   →  1 min (closing window)")
        print(f"  Alerts:          ORB breakouts + HOLD→BUY migrations")
        print(f"  Stop:            Press Ctrl+C")
        print("═"*70 + "\n")
        
        scanner = MomentumScanner()
        if not scanner.initialize():
            # Non-zero exit so the failure shows in launchd's log (it relaunches the job either way)
            logger.error("❌ Scanner initialization failed")
            # launchd relaunches every 60s around the clock, and TWS is down every night and weekend:
            # only banner during the watcher's day (trading day, 9:00 ET to the close), at most once per 2 h
            if init_failure_should_alert(now_et()):
                notifier.notify("Watcher", "Scanner initialization failed (TWS down or logged out?); "
                                "launchd keeps retrying", 'error', dedup_seconds=INIT_ALERT_DEDUP_SECONDS)
            raise SystemExit(1)
        
        # All-day process: no run-wide farm budget, and short waits so a mid-day regime refetch
        # can't block the loop for 3 x 10 min while the farm is down (review M1)
        scanner.connector.farm_wait_budget = None
        scanner.connector.farm_wait_sec = WATCH_FARM_WAIT_SEC
        
        scanner_logger = logging.getLogger('momentum_scanner')
        scanner_logger.setLevel(logging.WARNING)
        
        orb_detector = ORBDetector(scanner.connector)
        
        last_orb_date = None
        
        try:
            while True:
                phase, sleep_secs, should_scan = self.get_current_phase()
                now = now_et()
                now_str = now.strftime('%H:%M:%S')
                
                # New day (ET): reset ORB alerts and the day's migration log
                if last_orb_date != now.date():
                    self.orb_alerts_today.clear()
                    self.migration_log = []
                    last_orb_date = now.date()
                # A newer scan (e.g. today's 9:20 premarket run) replaces the watchlist
                self.reload_if_new_scan()
                
                if not should_scan:
                    if phase in ("PRE_MARKET", "AFTER_HOURS"):
                        logger.info(f"💤 [{now_str}] {phase} — sleeping {sleep_secs}s")
                    elif phase == "ORB_FORMING":
                        logger.info(f"⏳ [{now_str}] Opening range forming — wait until {ORB_FORMATION_DONE.strftime('%H:%M')}")
                    time.sleep(sleep_secs)
                    continue
                
                # After a TWS restart every fetch returns nothing and the loop would log
                # "no breakouts" all day. Exit instead; launchd starts a fresh watcher.
                if not scanner.connector.isConnected() or scanner.connector.serverVersion() is None:
                    logger.error("❌ Lost the TWS connection; exiting so launchd restarts the watcher")
                    raise SystemExit(1)

                # HMDS farm down: every fetch would come back empty and be logged as "no breakouts"
                if not self.data_ready(scanner.connector):
                    if self.data_down_since is None:
                        self.data_down_since = time.monotonic()
                    if self.probe_stale_link(scanner.connector, time.monotonic()) \
                            and self.data_ready(scanner.connector):
                        self.data_down_since = None
                if not self.data_ready(scanner.connector):
                    logger.warning(f"⚠️ [{now_str}] IBKR data farm down "
                                   f"({scanner.connector.farm_status_summary()}); skipping scan")
                    time.sleep(sleep_secs)
                    continue
                self.data_down_since = None

                self.scan_count += 1
                logger.info(f"\n🔄 SCAN #{self.scan_count}  [{now_str}]  Phase: {phase}")
                
                if self.is_orb_window():
                    orb_count = self.run_orb_scan(orb_detector)
                    if orb_count == 0:
                        logger.info(f"   No ORB breakouts ({len(self.watchlist_symbols)} checked)")
                else:
                    self.last_data_lag = None
                banner = self.stale_data_banner(self.last_data_lag,
                                                getattr(scanner.connector, 'data_delayed', False))
                if banner:
                    logger.warning(f"   {banner}")
                
                migration_count = self.run_migration_scan(scanner)
                if migration_count == 0:
                    logger.info(f"   No migrations this scan")
                
                logger.info(f"💤 Sleeping {sleep_secs}s until next scan...")
                time.sleep(sleep_secs)
        
        except KeyboardInterrupt:
            print("\n\n⚠️  Watch mode stopped by user")
            self._print_summary()
        finally:
            scanner.cleanup()
    
    def _print_summary(self):
        print("\n" + "="*70)
        print("  WATCH MODE SESSION SUMMARY")
        print("="*70)
        print(f"  Total scans:           {self.scan_count}")
        print(f"  ORB breakouts today:   {len(self.orb_alerts_today)}")
        print(f"  Migrations today:      {len(self.migration_log)}")
        
        if self.orb_alerts_today:
            print(f"\n  ORB breakouts:")
            for sym in sorted(self.orb_alerts_today):
                print(f"    • {sym}")
        
        if self.migration_log:
            print(f"\n  Migrations:")
            for m in self.migration_log:
                print(f"    {m['time'].strftime('%H:%M:%S')}  {m['symbol']:<6}  "
                      f"{m['from_tier']} → BUY  {m['setup']:<7}  stop {m['stop_pct']:.1f}%")
        
        print("="*70 + "\n")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    # ibapi logs every wire message (price bars included); in the launchd job that
    # grew watcher-launchd.err.log past 125 MB. Keep it to warnings.
    logging.getLogger('ibapi').setLevel(logging.WARNING)
    
    monitor = WatchModeMonitor()
    monitor.run_forever()


if __name__ == "__main__":
    main()
