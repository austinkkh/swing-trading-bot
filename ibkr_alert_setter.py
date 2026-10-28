#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
IBKR PRICE ALERT SETTER
Reads the latest scan CSV and creates IBKR price alerts above each VCP pivot.
─────────────────────────────────────────────────────────────────────────────
What this does:
  1. Reads the execution engine's newest scan_*_execution_decisions.csv (never a raw scan:
     only rows that passed every engine risk rule, BUY or MANUAL_REVIEW, get an alert)
  2. Refuses decisions from a scan older than the engine's max_scan_age_hours (18h)
  3. Filters to stocks with real VCPs (Has_Real_VCP=True), still below their pivot
  4. For each, places an IBKR price alert at pivot * (1 + buffer%)
  5. Logs each alert ID for tracking

How IBKR alerts work:
  - Alerts are server-side conditions IBKR monitors for you
  - When triggered, IBKR sends a notification (email/popup if configured in TWS)
  - You can ALSO have alerts trigger automatic order placement (advanced)
  - This script just SETS the alerts — you respond manually when fired

Safety: each alert is a real 1-share BUY order staged in TWS (transmit=False).
It won't send by itself, but TWS's Transmit button would send it. So:
  - preview is the default; --place is required to stage anything
  - paper only: the port must be a paper port and every account the TWS
    session reports must be a paper (DU...) account, or nothing is staged

Usage:
  python ibkr_alert_setter.py                       # Preview from the latest engine decisions
  python ibkr_alert_setter.py --tier TIER_BUY       # Preview one tier
  python ibkr_alert_setter.py --place               # Stage alerts (paper accounts only)
═══════════════════════════════════════════════════════════════════════════════
"""

import sys
import logging
from typing import Optional
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime

from config import *
from execution_engine import BUY, MANUAL_REVIEW, ExecutionPolicy
from ibkr_connector import ALERT_ORDER_REF, IBKRConnector, confirm_paper_session as shared_confirm_paper_session

# IBKR API imports for orders/conditions
from ibapi.contract import Contract
from ibapi.order import Order
from ibapi.order_condition import PriceCondition, OrderCondition

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════
ALERT_BUFFER_PCT = 0.1                  # Set alert at pivot * 1.001 (just above)
ALERT_PRICE_TYPE = 1                    # 1 = LAST price, 2 = BID, 3 = ASK
DEFAULT_TIERS = ['TIER_ELITE_BUY', 'TIER_BUY', 'TIER_RS_LEADER']  # Which tiers to alert on
MIN_PIVOT_PRICE = 1.0                   # Skip alerts under $1
DEFAULT_CLIENT_ID = 12                  # The watcher holds config CLIENT_ID; use a separate one


def confirm_paper_session(connector: IBKRConnector, port: int) -> None:
    """Exit unless this TWS session is paper-only (shared guard in ibkr_connector)."""
    try:
        accounts = shared_confirm_paper_session(connector, port)
    except PermissionError as e:
        sys.exit(f"❌ {e}; refusing to stage orders")
    logger.info(f"✓ Paper session confirmed: {accounts}")


# ═══════════════════════════════════════════════════════════════════════════
# ALERT MANAGER
# ═══════════════════════════════════════════════════════════════════════════
class AlertManager:
    """Set price alerts on IBKR using order conditions"""
    
    def __init__(self, connector: IBKRConnector):
        self.connector = connector
        self.alerts_placed = []
    
    def find_latest_decisions_csv(self) -> Path:
        """The execution engine's newest decisions file (run_daily_pipeline.sh writes it)."""
        scans_dir = Path(OUTPUT_DIR)
        if not scans_dir.exists():
            raise FileNotFoundError(f"Scans directory not found: {scans_dir}")
        csvs = sorted(scans_dir.glob("scan_*_execution_decisions.csv"), key=lambda p: p.stat().st_mtime)
        if not csvs:
            raise FileNotFoundError("No execution decisions found: run run_daily_pipeline.sh "
                                    "(or execution_engine.py) first")
        return csvs[-1]

    def load_alert_candidates(self, csv_path: Path, tiers: list, now: Optional[datetime] = None,
                              max_age_hours: float = ExecutionPolicy().max_scan_age_hours) -> pd.DataFrame:
        """Engine-approved rows (BUY / MANUAL_REVIEW) with real VCPs, from a fresh scan only."""
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} decisions from {csv_path.name}")

        required = {'Decision', 'Tier', 'Scan_Timestamp', 'VCP_Pivot', 'Has_Real_VCP', 'VCP_Dist_From_Pivot_%'}
        if not required <= set(df.columns):
            logger.error(f"❌ {csv_path.name} lacks {sorted(required - set(df.columns))}: "
                         "not an engine decisions file from this version; no alerts")
            return df.iloc[0:0]
        # Scan age: Scan_Timestamp is the scanning Mac's local clock, like the engine's check
        stamps = pd.to_datetime(df['Scan_Timestamp'], errors='coerce')
        age_h = ((now or datetime.now()) - stamps).dt.total_seconds() / 3600
        # Every row must be dated, not stale and not from the future (one bad row refuses the file)
        if stamps.isna().any() or age_h.max() > max_age_hours or age_h.min() < -1:
            logger.error(f"❌ Scan behind {csv_path.name} is undated, older than {max_age_hours:g}h "
                         f"or future-dated (ages {age_h.min():.0f}h to {age_h.max():.0f}h); no alerts")
            return df.iloc[0:0]

        df = df[df['Decision'].isin([BUY, MANUAL_REVIEW])]
        logger.info(f"  → {len(df)} passed the engine's rules (BUY / MANUAL_REVIEW)")

        # Filter to specified tiers
        df = df[df['Tier'].isin(tiers)]
        logger.info(f"  → {len(df)} stocks in tiers: {tiers}")
        
        # Filter to stocks with real VCPs
        if 'Has_Real_VCP' in df.columns:
            df = df[df['Has_Real_VCP'] == True]
            logger.info(f"  → {len(df)} have real VCP patterns")
        
        # Filter pivots above minimum price
        if 'VCP_Pivot' in df.columns:
            df = df[df['VCP_Pivot'] >= MIN_PIVOT_PRICE]
            logger.info(f"  → {len(df)} have valid pivots ≥ ${MIN_PIVOT_PRICE}")
        
        # Skip stocks already above their pivot (already broke out)
        if 'VCP_Dist_From_Pivot_%' in df.columns:
            df = df[df['VCP_Dist_From_Pivot_%'] < 0]
            logger.info(f"  → {len(df)} are still below their pivot (not yet broken out)")
        
        return df
    
    def create_alert(self, symbol: str, trigger_price: float, dry_run: bool = False) -> dict:
        """
        Create a price alert at trigger_price.
        
        IBKR doesn't have a pure 'alert' API — you have to use a conditional 
        order. The trick: place a tiny BUY order with a price condition,
        but with TIF=DAY+ParentID so it acts as a notification trigger.
        
        Simpler alternative (used here): place order with `whatIf=True` flag
        which simulates without executing. We use `transmit=False` so order
        sits unsent in TWS waiting for manual review.
        """
        if dry_run:
            return {
                'symbol': symbol,
                'trigger_price': trigger_price,
                'dry_run': True,
                'order_id': None,
                'status': 'PREVIEW'
            }
        
        try:
            # The price condition needs IBKR's contract ID; conId=0 isn't a valid contract
            details = self.connector.fetch_contract_details(symbol)
            if details is None or not getattr(details.contract, 'conId', 0):
                raise ValueError("couldn't resolve IBKR contract ID")
            con_id = details.contract.conId

            order_id = self.connector.next_order_id
            self.connector.next_order_id += 1
            
            # Build contract
            contract = Contract()
            contract.symbol = symbol
            contract.secType = "STK"
            contract.exchange = "SMART"
            contract.currency = "USD"
            
            # Build a NON-TRANSMITTING limit order with price condition
            # This sits in TWS unfilled — you'll see it as a pending order
            order = Order()
            order.action = "BUY"
            order.orderType = "LMT"
            order.totalQuantity = 1  # Tiny size — this is just an alert
            order.lmtPrice = trigger_price
            order.transmit = False  # KEY: Don't actually send, just stage it
            order.tif = "DAY"
            # Tag it: an alert is not an entry. Lets the snapshot (and a future order ledger) tell
            # these 1-share staged alerts apart from real entry orders (audit #11)
            order.orderRef = ALERT_ORDER_REF
            
            # Price condition: trigger when last price >= pivot
            # ibapi 10.x signature: (triggerMethod, conId, exch, isMore, price); the old
            # conjunction=/exchange= keywords raised TypeError, so no alert was ever staged
            condition = PriceCondition(
                triggerMethod=PriceCondition.TriggerMethodEnum.Default,
                conId=con_id,
                exch="SMART",
                isMore=True,
                price=trigger_price,
            ).And()
            order.conditions.append(condition)
            order.conditionsCancelOrder = False
            
            # Place the conditional order
            self.connector.placeOrder(order_id, contract, order)
            
            return {
                'symbol': symbol,
                'trigger_price': trigger_price,
                'order_id': order_id,
                'status': 'PLACED'
            }
        
        except Exception as e:
            logger.error(f"❌ Failed to create alert for {symbol}: {e}")
            return {
                'symbol': symbol,
                'trigger_price': trigger_price,
                'order_id': None,
                'status': f'ERROR: {e}'
            }
    
    def set_alerts_from_scan(self, tiers: list = None, dry_run: bool = False):
        """Main entry point"""
        if tiers is None:
            tiers = DEFAULT_TIERS
        
        csv_path = self.find_latest_decisions_csv()
        df = self.load_alert_candidates(csv_path, tiers)
        
        if df.empty:
            logger.warning("⚠️ No alert candidates found")
            return
        
        print("\n" + "="*70)
        print(f"  PIVOT ALERTS  ({'PREVIEW' if dry_run else 'STAGING IN TWS (paper, not transmitted)'})")
        print("="*70)
        print(f"  Source: {csv_path.name}")
        print(f"  Candidates: {len(df)}")
        print("="*70 + "\n")
        
        for _, row in df.iterrows():
            symbol = row['Symbol']
            pivot = row['VCP_Pivot']
            current = row.get('Entry_Price', 0)
            distance = row.get('VCP_Dist_From_Pivot_%', 0)
            tier = row['Tier']
            contractions = row.get('VCP_Contractions', 0)
            final_pb = row.get('VCP_Final_PB_%', 0)
            if row['Decision'] == MANUAL_REVIEW:
                print(f"  ⚠ MANUAL REVIEW: {row.get('Reason', '')[:150]}")
            
            trigger = round(pivot * (1 + ALERT_BUFFER_PCT / 100), 2)
            
            print(f"  {symbol:<8} {tier:<16}")
            print(f"    Pivot:        ${pivot:.2f}")
            print(f"    Trigger:      ${trigger:.2f}  (+{ALERT_BUFFER_PCT}% buffer)")
            print(f"    Current:      ${current:.2f}  ({distance:+.1f}% from pivot)")
            print(f"    VCP:          {contractions} contractions, final pullback -{final_pb:.1f}%")
            
            result = self.create_alert(symbol, trigger, dry_run=dry_run)
            self.alerts_placed.append(result)
            
            if result['status'] == 'PLACED':
                print(f"    ✅ Alert placed (order ID: {result['order_id']})")
            elif result['status'] == 'PREVIEW':
                print(f"    📋 PREVIEW (would place order)")
            else:
                print(f"    ❌ {result['status']}")
            print()
        
        print("="*70)
        print(f"  {'PLACED' if not dry_run else 'PREVIEWED'}: {len(self.alerts_placed)} alerts")
        print("="*70 + "\n")
        
        if not dry_run:
            print("⚠️  IMPORTANT NOTES:")
            print("   1. Alerts are NON-TRANSMITTING orders in TWS — you'll see them in")
            print("      the Orders tab with status 'PreSubmitted' or similar.")
            print("   2. They will NOT execute automatically.")
            print("   3. When triggered, you'll see them flash in TWS — that's your signal.")
            print("   4. Cancel any alerts manually in TWS when no longer needed.")
            print("   5. To get email/sound alerts, configure TWS notifications:")
            print("      File → Global Configuration → Messages → Notifications")


def main():
    parser = argparse.ArgumentParser(description="Set IBKR price alerts at VCP pivots")
    parser.add_argument('--tier', default='ALL',
                        help='Which tier(s) to alert on (ALL, TIER_ELITE_BUY, TIER_BUY, TIER_RS_LEADER)')
    parser.add_argument('--place', action='store_true',
                        help='Stage the alert orders in TWS (paper accounts only). Default: preview')
    parser.add_argument('--dry-run', action='store_true',
                        help='Preview without placing alerts (the default; kept for old commands)')
    parser.add_argument('--client-id', type=int, default=DEFAULT_CLIENT_ID,
                        help=f'IBKR client ID (default {DEFAULT_CLIENT_ID}; the watcher uses config CLIENT_ID)')
    args = parser.parse_args()
    dry_run = not args.place or args.dry_run

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    logging.getLogger('ibapi').setLevel(logging.WARNING)

    if args.tier == 'ALL':
        tiers = DEFAULT_TIERS
    else:
        tiers = [args.tier]

    print("\n" + "="*70)
    print("  IBKR PIVOT ALERT SETTER")
    print("="*70)

    if dry_run:
        print("  🔍 PREVIEW — nothing will be staged (add --place to stage alerts)")

    # Connect to TWS
    # Order calls are only enabled with --place (the one tool that stages untransmitted orders)
    connector = IBKRConnector(read_only=dry_run)
    if not dry_run:
        if not connector.connect_and_run(TWS_HOST, TWS_PORT, args.client_id):
            print("❌ Failed to connect to TWS")
            sys.exit(1)

    try:
        if not dry_run:
            confirm_paper_session(connector, TWS_PORT)
        manager = AlertManager(connector)
        manager.set_alerts_from_scan(tiers=tiers, dry_run=dry_run)
    finally:
        if not dry_run:
            connector.disconnect_and_stop()


if __name__ == "__main__":
    main()
