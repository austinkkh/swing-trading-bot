#!/usr/bin/env python3
"""
Symbol hygiene: stop requesting tickers IBKR no longer knows (delisted, acquired, renamed).

Universe hygiene only: it never changes a selection rule, and it never adds a symbol to the scan.
Two sources, applied in this order by apply_symbol_hygiene():

  1. symbol_overrides.csv (repo root, tracked, hand-maintained)
       symbol,action,new_symbol,reason,evidence,added,review_after
       action 'skip'   -> the symbol is dropped
       action 'rename' -> the old ticker is dropped; the new one is NOT added (that is a universe
                          change for the owner); if it is already scanned nothing else changes
  2. auto-quarantine, state in Data/symbol_health.json
       Only IBKR Error 200 ("no security definition") counts. Timeouts and Error 162 never do:
       valid names (MPC, VLO, P) time out in a busy watcher. A symbol failing on
       SYMBOL_QUARANTINE_DAYS distinct scan days in a row (no success in between) is quarantined.
       Lifted when a newer IBD export (dated after the first failure) lists it again, or when the
       weekly re-probe (one contract-details request per quarantined name) finds it.
       Only full (afterhours / premarket) scans write the state; the watcher only reads it.

Never dropped: symbols with an open position or a working order in the newest account snapshot.
If that can't be checked, quarantine is not applied for the run (overrides still are).
Everything fails open: a missing or malformed file means the full universe is scanned.

  python symbol_hygiene.py --report   # universe symbols missing from the newest IBD export (no TWS)
"""

import argparse
import csv
import json
import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import config

logger = logging.getLogger(__name__)

REPO_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(config.OUTPUT_DIR).expanduser().parent / "Data"
OVERRIDES_CSV = REPO_DIR / "symbol_overrides.csv"
HEALTH_JSON = DATA_DIR / "symbol_health.json"
SNAPSHOT_DIR = DATA_DIR / "account"
# The scheduled 9:20 pipeline writes its snapshot here (account_snapshot.py --out), not to SNAPSHOT_DIR
PIPELINE_SNAPSHOT_DIR = Path(config.OUTPUT_DIR).expanduser() / "paper_runs"
QUARANTINE_DAYS = int(getattr(config, 'SYMBOL_QUARANTINE_DAYS', 3))
SNAPSHOT_MAX_AGE_DAYS = 4        # a weekend plus a holiday; older snapshots can't vouch for positions
SECDEF_STATUS = 'no_secdef'      # ibkr_connector status for Error 200
_UNCHECKED = object()            # "look up protected symbols yourself"


@dataclass
class Override:
    symbol: str
    action: str                  # skip | rename
    new_symbol: str = ''
    reason: str = ''
    evidence: str = ''
    added: str = ''
    review_after: str = ''

    def review_due(self, today: date) -> bool:
        try:
            return bool(self.review_after) and date.fromisoformat(self.review_after) <= today
        except ValueError:
            return False


def load_overrides(path: Optional[Path] = None) -> Dict[str, Override]:
    """symbol -> Override. Missing file: {}. Malformed file: warning and {} (scan everything)."""
    path = Path(path or OVERRIDES_CSV)
    if not path.exists():
        return {}
    out: Dict[str, Override] = {}
    try:
        with path.open(newline='') as f:
            for row in csv.DictReader(f):
                sym = str(row.get('symbol') or '').strip().upper()
                action = str(row.get('action') or '').strip().lower()
                if not sym:
                    continue
                if action not in ('skip', 'rename'):
                    raise ValueError(f"unknown action {action!r} for {sym}")
                out[sym] = Override(sym, action, str(row.get('new_symbol') or '').strip().upper(),
                                    *(str(row.get(k) or '').strip()
                                      for k in ('reason', 'evidence', 'added', 'review_after')))
    except Exception as e:                                       # noqa: BLE001
        logger.warning(f"⚠️ {path.name} unreadable ({e}); scanning the full universe")
        return {}
    return out


def load_health(path: Optional[Path] = None) -> dict:
    """{'symbols': {sym: {...}}, 'last_reprobe': iso date}. Missing/malformed: empty (fail open)."""
    path = Path(path or HEALTH_JSON)
    empty = {'symbols': {}, 'last_reprobe': None}
    if not path.exists():
        return empty
    try:
        state = json.loads(path.read_text())
        if not isinstance(state.get('symbols'), dict):
            raise ValueError("no 'symbols' map")
        state.setdefault('last_reprobe', None)
        return state
    except Exception as e:                                       # noqa: BLE001
        logger.warning(f"⚠️ {path.name} unreadable ({e}); no symbol is quarantined this run")
        return empty


def save_health(state: dict, path: Optional[Path] = None) -> None:
    path = Path(path or HEALTH_JSON)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True))
    tmp.replace(path)            # atomic: a reader never sees half a file


def protected_symbols(snapshot_dir: Optional[Path] = None, now: Optional[datetime] = None,
                      max_age_days: int = SNAPSHOT_MAX_AGE_DAYS,
                      extra_dirs: Optional[Sequence[Path]] = None) -> Optional[Set[str]]:
    """Held symbols + symbols with working entry orders from the newest account snapshot (by its
    own timestamp) in snapshot_dir or the pipeline's paper_runs folder, or None when there's no
    recent readable snapshot (then nothing is quarantined)."""
    dirs = [Path(snapshot_dir or SNAPSHOT_DIR)]
    dirs += [Path(d) for d in (extra_dirs if extra_dirs is not None
                               else ([] if snapshot_dir else [PIPELINE_SNAPSHOT_DIR]))]
    newest = None
    for d in dirs:
        for f in (d.glob('account_snapshot_*.json') if d.exists() else []):
            try:
                snap = json.loads(f.read_text())
                ts = datetime.fromisoformat(snap['timestamp'])
            except Exception:                                    # noqa: BLE001
                continue          # an unreadable file can't vouch for positions; try the others
            if newest is None or ts > newest[0]:
                newest = (ts, snap)
    if newest is None:
        return None
    ts, snap = newest
    if (now or datetime.now()) - ts > timedelta(days=max_age_days):
        return None
    try:
        return {str(s).upper() for s in list(snap.get('held_symbols', [])) + list(snap.get('pending_entry_orders', []))
                + list(snap.get('alert_orders', []))}
    except Exception:                                            # noqa: BLE001
        return None


def quarantined(state: dict) -> Dict[str, dict]:
    return {s: e for s, e in state.get('symbols', {}).items() if e.get('quarantined_on')}


def apply_symbol_hygiene(symbols: Sequence[str], overrides: Optional[Dict[str, Override]] = None,
                         state: Optional[dict] = None, protected=_UNCHECKED
                         ) -> Tuple[List[str], List[Tuple[str, str]]]:
    """(kept symbols in input order, [(dropped symbol, reason)]). Overrides first, then quarantine."""
    overrides = load_overrides() if overrides is None else overrides
    state = load_health() if state is None else state
    if protected is _UNCHECKED:
        protected = protected_symbols()
    held = protected or set()
    in_scan = {str(s).upper() for s in symbols}
    q = quarantined(state)
    kept, dropped = [], []
    for sym in symbols:
        key = str(sym).upper()
        o = overrides.get(key)
        if o and key not in held:
            if o.action == 'rename':
                note = "already scanned" if o.new_symbol in in_scan else "not added: owner to approve"
                dropped.append((sym, f"renamed to {o.new_symbol} ({note})"))
            else:
                dropped.append((sym, f"skip: {o.reason or 'symbol_overrides.csv'}"))
            continue
        # Quarantine only when positions could be checked (None = unknown -> fail open)
        if key in q and protected is not None and key not in held:
            dropped.append((sym, f"auto-quarantined {q[key]['quarantined_on']} "
                                 f"(no security definition on {q[key].get('consecutive_secdef_days')} scan days)"))
            continue
        kept.append(sym)
    return kept, dropped


def record_scan_results(state: dict, statuses: Dict[str, str], day: date,
                        ibd_symbols: Iterable[str] = (), ibd_date: Optional[date] = None,
                        protected: Optional[Set[str]] = None, days: int = QUARANTINE_DAYS
                        ) -> Tuple[List[str], List[str]]:
    """
    Update failure counts from one full scan's fetch statuses (symbol -> ibkr status).
    Returns (newly quarantined, lifted). Counts distinct scan days, so rescans on the same day
    can't fill the count. Any successful fetch resets a symbol. Other failures don't count.
    """
    syms = state.setdefault('symbols', {})
    today = day.isoformat()
    ibd = {str(s).upper() for s in ibd_symbols}
    new_q, lifted = [], []
    for sym, status in statuses.items():
        sym = str(sym).upper()
        if status == 'ok':
            entry = syms.pop(sym, None)
            if entry and entry.get('quarantined_on'):
                lifted.append(sym)
            continue
        if status != SECDEF_STATUS:
            continue                       # timeouts, 162, pacing: never evidence of a dead ticker
        entry = syms.setdefault(sym, {'consecutive_secdef_days': 0, 'first_fail': today,
                                      'last_fail': None, 'quarantined_on': None})
        if entry.get('last_fail') != today:
            entry['consecutive_secdef_days'] = int(entry.get('consecutive_secdef_days') or 0) + 1
            entry['last_fail'] = today
        can_check = protected is not None and sym not in protected
        if (entry['consecutive_secdef_days'] >= days and not entry.get('quarantined_on') and can_check):
            entry['quarantined_on'] = today
            new_q.append(sym)
    # A newer IBD export listing the symbol again means it trades: forget the failures
    if ibd_date is not None:
        for sym in list(syms):
            entry = syms[sym]
            try:
                first = date.fromisoformat(entry.get('first_fail') or today)
            except ValueError:
                continue
            if sym in ibd and ibd_date > first:
                if entry.get('quarantined_on'):
                    lifted.append(sym)
                del syms[sym]
    return new_q, lifted


def weekly_reprobe(state: dict, connector, today: date) -> List[str]:
    """First full scan of each ISO week: one contract-details request per quarantined symbol;
    a hit lifts the quarantine. Returns the lifted symbols."""
    last = state.get('last_reprobe')
    if last and date.fromisoformat(last).isocalendar()[:2] == today.isocalendar()[:2]:
        return []
    lifted = []
    for sym in list(quarantined(state)):
        try:
            found = connector.fetch_contract_details(sym, timeout=5)
        except Exception:                                        # noqa: BLE001
            found = None
        if found:
            del state['symbols'][sym]
            lifted.append(sym)
    state['last_reprobe'] = today.isoformat()
    return lifted


def secdef_days(state: dict) -> Dict[str, int]:
    """symbol -> consecutive Error-200 scan days so far (for the scan display)."""
    return {s: int(e.get('consecutive_secdef_days') or 0) for s, e in state.get('symbols', {}).items()}


def review_due(overrides: Dict[str, Override], today: Optional[date] = None) -> List[Override]:
    today = today or date.today()
    return [o for o in overrides.values() if o.review_due(today)]


def report() -> str:
    """Universe symbols missing from the newest IBD export (no TWS). IBD lists ~every US listing,
    so a missing name has probably stopped trading. Advisory only: a fresh corporate action
    can make a valid name look missing, so nothing is skipped automatically from this."""
    import momentum_scanner as ms
    from ibd_universe import load_ibd_universe
    scanner = ms.MomentumScanner()
    symbols = scanner._raw_stock_universe()
    ibd, ibd_date, path = load_ibd_universe(ms.IBD_UNIVERSE_GLOB)
    if not ibd:
        return "No IBD universe export found; nothing to compare."
    overrides = load_overrides()
    missing = sorted(s for s in set(symbols) if s.upper() not in ibd)
    lines = [f"{len(missing)} of {len(set(symbols))} universe symbols are not in {Path(path).name} ({ibd_date}):"]
    for s in missing:
        src = ','.join(scanner.universe_obj.source_lists.get(s, [])) if scanner.universe_obj else ''
        manual = 'STAGE2_WATCHLIST' if s in scanner.manual_watchlist_symbols else ''
        o = overrides.get(s.upper())
        lines.append(f"  {s:<7} {', '.join(x for x in (src, manual) if x) or '-':<30} "
                     f"{('override: ' + o.action) if o else ''}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Symbol hygiene (dead / renamed tickers)")
    ap.add_argument('--report', action='store_true', help='universe symbols missing from the newest IBD export')
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format='%(message)s')
    if args.report:
        print(report())
        return
    state = load_health()
    q = quarantined(state)
    print(f"Overrides: {len(load_overrides())}  quarantined: {len(q)}  tracked failures: {len(state['symbols'])}")
    for s, e in sorted(state['symbols'].items()):
        print(f"  {s:<7} days {e.get('consecutive_secdef_days')}  first {e.get('first_fail')}  "
              f"last {e.get('last_fail')}  quarantined {e.get('quarantined_on') or '-'}")


if __name__ == "__main__":
    main()
