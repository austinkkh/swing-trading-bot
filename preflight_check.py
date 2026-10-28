#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
PRE-FLIGHT VALIDATOR
Run this BEFORE market open to check that everything is wired correctly.
─────────────────────────────────────────────────────────────────────────────
What it checks:
  1. All required Python modules can be imported
  2. config.py settings are valid
  3. Data directory exists and contains expected files
  4. IBD xlsx files parse correctly
  5. Universe builder produces non-empty output
  6. TWS connection works (optional - requires TWS running)
  7. Scanner can initialize without errors

Usage:
  python preflight_check.py            # Full check
  python preflight_check.py --no-tws   # Skip TWS connection test (faster)
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import argparse
from pathlib import Path

# ANSI colors
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'

def ok(msg):
    print(f"  {GREEN}✓{RESET} {msg}")

def warn(msg):
    print(f"  {YELLOW}⚠{RESET}  {msg}")

def fail(msg):
    print(f"  {RED}✗{RESET} {msg}")

def section(title):
    print(f"\n{BOLD}{BLUE}━━━ {title} ━━━{RESET}")

results = {'pass': 0, 'warn': 0, 'fail': 0}

def check_imports():
    section("1. PYTHON IMPORTS")
    
    required = [
        'pandas', 'numpy', 'ibapi.client', 'ibapi.wrapper',
    ]
    project_modules = [
        'config', 'ibkr_connector', 'momentum_scanner', 
        'vcp_detector', 'orb_detector', 'watch_mode',
        'universe_builder', 'market_regime', 'risk_manager',
        'relative_strength', 'ibkr_alert_setter'
    ]
    
    for mod in required:
        try:
            __import__(mod)
            ok(f"External: {mod}")
            results['pass'] += 1
        except ImportError as e:
            fail(f"External: {mod} — {e}")
            results['fail'] += 1
    
    for mod in project_modules:
        try:
            __import__(mod)
            ok(f"Project:  {mod}")
            results['pass'] += 1
        except ImportError as e:
            fail(f"Project:  {mod} — {e}")
            results['fail'] += 1


def check_config():
    section("2. CONFIG.PY SETTINGS")
    
    try:
        import config
    except Exception as e:
        fail(f"Cannot import config: {e}")
        results['fail'] += 1
        return
    
    required_settings = [
        ('TWS_HOST', str),
        ('TWS_PORT', int),
        ('CLIENT_ID', int),
        ('OUTPUT_DIR', str),
        ('LOG_DIR', str),
        ('SCAN_UNIVERSE', str),
        ('MIN_PRICE', (int, float)),
        ('MAX_PRICE', (int, float)),
        ('MIN_AVG_VOLUME', (int, float)),
    ]
    
    for setting, expected_type in required_settings:
        if hasattr(config, setting):
            value = getattr(config, setting)
            if isinstance(value, expected_type):
                ok(f"{setting} = {value}")
                results['pass'] += 1
            else:
                fail(f"{setting} wrong type: {type(value).__name__} (expected {expected_type})")
                results['fail'] += 1
        else:
            fail(f"{setting} missing from config.py")
            results['fail'] += 1
    
    # Check SCAN_UNIVERSE makes sense
    if hasattr(config, 'SCAN_UNIVERSE'):
        if config.SCAN_UNIVERSE in ('DYNAMIC', 'IBD_UNIVERSE'):
            ok(f"SCAN_UNIVERSE='{config.SCAN_UNIVERSE}' will use universe_builder")
        elif config.SCAN_UNIVERSE == 'STAGE2_WATCHLIST':
            warn(f"SCAN_UNIVERSE='STAGE2_WATCHLIST' uses hardcoded list "
                 f"({len(getattr(config, 'STAGE2_WATCHLIST', []))} stocks). "
                 f"Set to 'DYNAMIC' to use universe_builder.")
            results['warn'] += 1
        else:
            warn(f"SCAN_UNIVERSE='{config.SCAN_UNIVERSE}' — make sure this is what you want")
            results['warn'] += 1


def check_data_directory():
    section("3. DATA DIRECTORY (IBD files)")
    
    data_dir = Path(os.path.expanduser('~/swing-trading-bot/Data'))
    
    if not data_dir.exists():
        warn(f"Data directory does not exist: {data_dir}")
        warn("  Universe builder will fall back to STAGE2_WATCHLIST")
        results['warn'] += 1
        return
    
    ok(f"Data directory exists: {data_dir}")
    results['pass'] += 1
    
    # Look for IBD xlsx files
    xlsx_files = sorted(data_dir.glob('*.xlsx'))
    txt_files = sorted(data_dir.glob('tv_*.txt'))
    
    if xlsx_files:
        ok(f"Found {len(xlsx_files)} IBD xlsx file(s):")
        for f in xlsx_files:
            print(f"      • {f.name}")
        results['pass'] += 1
    else:
        warn("No IBD xlsx files found in data directory")
        warn("  Export from IBD and save as ~/swing-trading-bot/Data/IBD_50.xlsx etc.")
        results['warn'] += 1
    
    if txt_files:
        ok(f"Found {len(txt_files)} TradingView text file(s):")
        for f in txt_files:
            print(f"      • {f.name}")
        results['pass'] += 1
    else:
        warn("No tv_*.txt files found (optional — only needed if you maintain TV symbol lists)")
        results['warn'] += 1


def check_universe_builder():
    section("4. UNIVERSE BUILDER")
    
    try:
        from universe_builder import UniverseBuilder
    except ImportError as e:
        fail(f"Cannot import UniverseBuilder: {e}")
        results['fail'] += 1
        return
    
    try:
        builder = UniverseBuilder(data_dir='~/swing-trading-bot/Data')
        universe = builder.build()
        
        if not universe.symbols:
            warn("Universe builder returned 0 symbols")
            warn("  Will fall back to STAGE2_WATCHLIST when scanner runs")
            results['warn'] += 1
        else:
            ok(f"Universe built: {len(universe.symbols)} symbols")
            ok(f"Sources loaded: {len(universe.sources_loaded)}")
            results['pass'] += 2
            
            # Show a few top stocks
            top = sorted(universe.symbols, 
                        key=lambda s: universe.metadata.get(s, {}).get('universe_score', 0),
                        reverse=True)[:5]
            print(f"      Top 5 by score: {', '.join(top)}")
    except Exception as e:
        fail(f"Universe builder error: {e}")
        results['fail'] += 1


def check_tws_connection():
    section("5. TWS CONNECTION (live test)")
    
    try:
        import config
        from ibkr_connector import IBKRConnector
    except Exception as e:
        fail(f"Cannot import: {e}")
        results['fail'] += 1
        return
    
    print(f"  Attempting connection to {config.TWS_HOST}:{config.TWS_PORT} (client {config.CLIENT_ID})...")
    
    connector = IBKRConnector()
    try:
        if connector.connect_and_run(config.TWS_HOST, config.TWS_PORT, config.CLIENT_ID):
            ok(f"Connected to TWS at {config.TWS_HOST}:{config.TWS_PORT}")
            ok(f"Server version: {connector.serverVersion()}")
            results['pass'] += 2
            # Farm status arrives as TWS messages right after connect (2104/2106 OK, 2103/2105 broken)
            summary = connector.farm_status_summary()
            farms = ", ".join(f"{f} {'OK' if up else 'BROKEN'}" for f, up in sorted(connector.farm_status.items()))
            if connector.data_farms_ready():
                ok(f"Data farms: {farms or 'no status reported (assumed OK)'}")
                results['pass'] += 1
            else:
                warn(f"Data farms not ready: {summary} ({farms})")
            
            # Try fetching one bar to verify market data subscription
            print("  Testing historical data fetch (SPY)...")
            bars = connector.fetch_historical_data("SPY", duration="5 D", bar_size="1 day")
            if bars and len(bars) > 0:
                ok(f"Historical data working ({len(bars)} bars)")
                results['pass'] += 1
            else:
                fail("Historical data fetch returned empty")
                results['fail'] += 1
        else:
            fail("Connection failed")
            fail("  Check that TWS is running and API is enabled:")
            fail("  TWS → File → Global Configuration → API → Settings → Enable ActiveX and Socket Clients")
            results['fail'] += 1
    except Exception as e:
        fail(f"TWS connection error: {e}")
        results['fail'] += 1
    finally:
        try:
            connector.disconnect_and_stop()
        except Exception:
            pass


def check_scanner_init():
    section("6. SCANNER INITIALIZATION (no scan, just init)")
    
    try:
        from momentum_scanner import MomentumScanner
        scanner = MomentumScanner()
        ok("MomentumScanner instantiated")
        results['pass'] += 1
        
        # Check that all attributes exist
        required_attrs = ['elite_buy_results', 'elite_hold_results', 'buy_results',
                         'watch_results', 'extended_results', 'universe_obj']
        for attr in required_attrs:
            if hasattr(scanner, attr):
                ok(f"  Attribute: {attr}")
                results['pass'] += 1
            else:
                fail(f"  Missing attribute: {attr}")
                results['fail'] += 1
    except Exception as e:
        fail(f"Scanner instantiation error: {e}")
        results['fail'] += 1


def check_directories():
    section("7. OUTPUT DIRECTORIES")
    
    try:
        import config
    except Exception:
        return
    
    for dir_name in ['OUTPUT_DIR', 'LOG_DIR']:
        if hasattr(config, dir_name):
            path = Path(os.path.expanduser(getattr(config, dir_name)))
            if not path.exists():
                warn(f"{dir_name} does not exist: {path}")
                warn("  (Will be created automatically on first run)")
                results['warn'] += 1
            else:
                ok(f"{dir_name}: {path}")
                results['pass'] += 1


def main():
    parser = argparse.ArgumentParser(description="Pre-flight validator")
    parser.add_argument('--no-tws', action='store_true', 
                        help="Skip TWS connection test (faster)")
    args = parser.parse_args()
    
    print(f"\n{BOLD}{'═'*70}{RESET}")
    print(f"{BOLD}  PRE-FLIGHT VALIDATOR{RESET}")
    print(f"{BOLD}{'═'*70}{RESET}")
    
    check_imports()
    check_config()
    check_data_directory()
    check_universe_builder()
    check_directories()
    
    if not args.no_tws:
        check_tws_connection()
    else:
        section("5. TWS CONNECTION")
        print("  Skipped (--no-tws)")
    
    check_scanner_init()
    
    # Summary
    print(f"\n{BOLD}{'═'*70}{RESET}")
    print(f"{BOLD}  SUMMARY{RESET}")
    print(f"{BOLD}{'═'*70}{RESET}")
    print(f"  {GREEN}Passed:   {results['pass']}{RESET}")
    print(f"  {YELLOW}Warnings: {results['warn']}{RESET}")
    print(f"  {RED}Failed:   {results['fail']}{RESET}")
    
    if results['fail'] == 0 and results['warn'] == 0:
        print(f"\n  {GREEN}{BOLD}✓ ALL CHECKS PASSED — Ready to scan{RESET}\n")
        sys.exit(0)
    elif results['fail'] == 0:
        print(f"\n  {YELLOW}{BOLD}⚠ Warnings present but no failures — should still work{RESET}\n")
        sys.exit(0)
    else:
        print(f"\n  {RED}{BOLD}✗ FAILURES DETECTED — fix before running scanner{RESET}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
