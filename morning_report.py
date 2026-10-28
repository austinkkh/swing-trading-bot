#!/usr/bin/env python3
"""
Morning report: one read-only health check of the whole setup, any time of day. Not scheduled.

Rows (OK / WARN / FAIL):
  tws          TWS reachable on the paper port (client 18, read-only connector) and data-farm status
  account      paper account, net liquidation, positions, unprotected shares (account_snapshot rules)
  breakers     equity-log sessions since the last reset, and whether submit-mode entries are allowed
               (>= 5 sessions and no loss circuit breaker tripped)
  pipeline     result of the newest Logs/scheduler/premarket_pipeline_*.log
  scan         age of the newest afterhours (pipeline) scan and of the watcher's newest scan
  ibd          age of the newest IBD Data Tables export (Data/ibd_universe_<date>.csv)
  ledger       bot order ledger: working orders, open trades, orders not yet settled
  launchd      the repo's launchd jobs loaded (launchctl list), running / last exit status
  disk / logs  free disk, Logs/ size, per-run pipeline logs (count, total), launchd logs over 50 MB

Read-only: no order call is possible (IBKRConnector read_only), live ports are refused before
connecting, and it writes nothing but notify.py's banner dedupe state (Data/notify_state.json): the
equity log is NOT appended (only account_snapshot.py does that, so this report never changes the
breaker history), no ledger or order file is touched. Deletes nothing.
Posts one notify.py banner summarizing WARN / FAIL rows (none when all OK; off with --no-notify).

Usage:
  python morning_report.py              # full report, TWS on config TWS_PORT (paper only)
  python morning_report.py --no-tws     # files only; breakers from the equity log
  python morning_report.py --account DUXXXXXXX
Exit codes: 0 no FAIL · 1 at least one FAIL.
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import plistlib
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import config
import notify as notifier

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
OK, WARN, FAIL = "OK", "WARN", "FAIL"
REPORT_CLIENT_ID = 18            # this report's own read-only client (99 stays free for ad-hoc scans)
FARM_SETTLE_SEC = 2.0            # TWS sends farm status messages right after connecting
IBD_MAX_AGE_DAYS = 7             # the scanner warns after 7 days too
LAUNCHD_LOG_MAX_MB = 50.0        # scripts/rotate_launchd_logs.py's default threshold
LOGS_WARN_MB = 1024.0
DISK_WARN_GB, DISK_FAIL_GB = 10.0, 2.0
PIPELINE_LOG_GLOB = "premarket_pipeline_*.log"


@dataclass
class Row:
    name: str
    status: str
    detail: str


@dataclass
class Paths:
    """Everything the report reads; tests point these at a temp folder."""
    root: Path = ROOT
    scans: Path = field(default_factory=lambda: Path(config.OUTPUT_DIR).expanduser())
    logs: Path = field(default_factory=lambda: ROOT / "Logs")
    launchd: Path = field(default_factory=lambda: ROOT / "launchd")
    ibd_glob: str = field(default_factory=lambda: str(getattr(config, 'IBD_UNIVERSE_GLOB', None)
                                                     or Path.home() / "swing-trading-bot" / "Data" / "ibd_universe_*.csv"))
    orders: Optional[Path] = None              # default order_ledger.ORDERS_DIR
    equity_log: Optional[Path] = None          # default account_snapshot.EQUITY_LOG
    adjustments: Optional[Path] = None         # default account_snapshot.ADJUSTMENTS_CSV


def _mb(n_bytes: float) -> float:
    return n_bytes / 1024 / 1024


def _age_text(delta: timedelta) -> str:
    hours = delta.total_seconds() / 3600
    return f"{hours:.0f}h" if hours < 48 else f"{hours / 24:.1f}d"


def previous_trading_day(d: date) -> date:
    from market_time import is_trading_day
    d -= timedelta(days=1)
    while not is_trading_day(d):
        d -= timedelta(days=1)
    return d


# ─── TWS + account (read-only) ───

def check_tws(port: int, account: Optional[str], connector_factory=None, sleep: Callable[[float], None] = time.sleep,
              now: Optional[datetime] = None, paths: Optional[Paths] = None) -> Tuple[List[Row], Optional[dict]]:
    """[tws, account] rows and the snapshot dict (None if TWS wasn't readable)."""
    from ibkr_connector import LIVE_PORTS, PAPER_PORTS, IBKRConnector, confirm_paper_session
    if port in LIVE_PORTS:
        return [Row("tws", FAIL, f"port {port} is a LIVE port: refused, nothing connected")], None
    if port not in PAPER_PORTS:
        return [Row("tws", FAIL, f"port {port} is not a paper port {sorted(PAPER_PORTS)}")], None
    factory = connector_factory or IBKRConnector
    conn = factory(read_only=True)
    host = getattr(config, 'TWS_HOST', '127.0.0.1')
    if not conn.connect_and_run(host, port, REPORT_CLIENT_ID):
        return [Row("tws", FAIL, f"can't connect to {host}:{port} as client {REPORT_CLIENT_ID} (TWS down or "
                                 "API off?)")], None
    rows: List[Row] = []
    snapshot = None
    try:
        sleep(FARM_SETTLE_SEC)
        farms = getattr(conn, 'farm_status', {}) or {}
        broken = sorted(f for f, up in farms.items() if not up)
        link_ok = getattr(conn, 'server_link_ok', True)
        farm_text = ", ".join(f"{f} {'OK' if up else 'BROKEN'}" for f, up in sorted(farms.items())) or "no farm messages"
        if not link_ok:
            rows.append(Row("tws", FAIL, f"connected on {port}, but TWS's link to IB servers is down; {farm_text}"))
        elif broken:
            hmds = any(f.startswith('ushmds') for f in broken)
            rows.append(Row("tws", FAIL if hmds else WARN, f"connected on {port}; broken farms: {', '.join(broken)}"
                            + (" (ushmds: scans can't fetch bars)" if hmds else "")))
        else:
            rows.append(Row("tws", OK, f"connected on {port} (client {REPORT_CLIENT_ID}); {farm_text}"))
        try:
            confirm_paper_session(conn, port)
        except PermissionError as exc:
            rows.append(Row("account", FAIL, f"{exc}"))
            return rows, None
        state = conn.fetch_account_state()
        if state is None:
            rows.append(Row("account", FAIL, "account state timed out"))
            return rows, None
        snapshot = build_snapshot_readonly(state, account, now, paths)
    except ValueError as exc:
        rows.append(Row("account", FAIL, str(exc)))
        return rows, None
    finally:
        try:
            conn.disconnect_and_stop()
        except Exception:
            pass
    rows.append(account_row(snapshot))
    return rows, snapshot


def build_snapshot_readonly(state: dict, account: Optional[str], now: Optional[datetime],
                            paths: Optional[Paths]) -> dict:
    """account_snapshot.build_snapshot with the equity history, WITHOUT appending to any log."""
    import account_snapshot as acct
    accounts = sorted(state['account_values'])
    acct_id = account or (accounts[0] if len(accounts) == 1 else None)
    paths = paths or Paths()
    history = acct.read_equity_log(acct_id, paths.equity_log or acct.EQUITY_LOG) if acct_id else []
    adjustments = acct.read_adjustments(acct_id, paths.adjustments) if acct_id else []
    return acct.build_snapshot(state, account, equity_history=history, adjustments=adjustments,
                               now=now or datetime.now())


def account_row(snap: dict) -> Row:
    positions = snap.get('positions') or []
    unprot = [r for r in positions if (r.get('unprotected_shares') or 0) > 0]
    text = (f"{snap['account']} net liq ${snap['net_liquidation']:,.0f}, {len(positions)} position(s), "
            f"open risk {snap.get('total_open_risk_pct', 0):.2f}%")
    if snap.get('pending_entry_orders'):
        text += f", pending entries {', '.join(snap['pending_entry_orders'])}"
    if unprot:
        parts = ", ".join(f"{r['symbol']} {r['unprotected_shares']:.0f} sh" for r in unprot)
        return Row("account", WARN, f"{text}; UNPROTECTED: {parts} (order_entry --submit halts)")
    return Row("account", OK, text)


def breakers_row(snap: Optional[dict], account: Optional[str], now: datetime, paths: Paths) -> Row:
    """Sessions since the last reset and whether the breakers let a submit run enter."""
    import account_snapshot as acct
    import risk_manager
    from execution_engine import ExecutionPolicy
    policy = ExecutionPolicy()
    source = "TWS"
    if snap is None:
        source = "equity log"
        log_path = paths.equity_log or acct.EQUITY_LOG
        acct_id = account or _single_account(log_path)
        if not acct_id:
            return Row("breakers", WARN, f"no account to read in {log_path} (pass --account)")
        history = sorted(acct.read_equity_log(acct_id, log_path))
        if not history:
            return Row("breakers", WARN, f"no equity history for {acct_id}: submit entries halt")
        adjustments = acct.read_adjustments(acct_id, paths.adjustments)
        (*past, (last_t, last_eq)) = history
        snap = {**acct.equity_changes(past, last_eq, last_t, adjustments),
                **acct.breaker_history(history, last_eq, now, adjustments)}
        source = f"equity log, last {last_t:%Y-%m-%d %H:%M}"
    sessions = snap.get('breaker_history_sessions')
    halt, mult, note = risk_manager.circuit_breaker(snap.get('daily_pnl_pct'), snap.get('weekly_pnl_pct'),
                                                    snap.get('month_drawdown_pct'), policy.circuit_breakers)
    need = policy.min_breaker_history_sessions
    fmt = lambda v: "n/a" if v is None else f"{v:+.2f}%"  # noqa: E731
    changes = (f"day {fmt(snap.get('daily_pnl_pct'))} week {fmt(snap.get('weekly_pnl_pct'))} "
               f"month dd {fmt(snap.get('month_drawdown_pct'))}")
    if halt:
        return Row("breakers", FAIL, f"{halt} ({changes}; {source})")
    if sessions is None or int(sessions) < need:
        since = snap.get('breaker_history_start') or 'n/a'
        return Row("breakers", WARN, f"{sessions if sessions is not None else 'unknown'} of {need} sessions since "
                                     f"the last reset (first {since}): submit entries halt; {changes} ({source})")
    return Row("breakers", OK, f"{sessions} sessions; entries allowed{' (' + note + ')' if note else ''}; "
                               f"{changes} ({source})")


def _single_account(log_path: Path) -> Optional[str]:
    import csv
    if not log_path.exists():
        return None
    with log_path.open() as f:
        accounts = {r.get('account') for r in csv.DictReader(f) if r.get('account')}
    configured = str(getattr(config, 'IBKR_ACCOUNT', '') or '').strip().upper()
    if configured in accounts:
        return configured
    return accounts.pop() if len(accounts) == 1 else None


# ─── files ───

def pipeline_logs(paths: Paths) -> List[Path]:
    return sorted((paths.logs / "scheduler").glob(PIPELINE_LOG_GLOB))


def pipeline_row(paths: Paths, now: datetime) -> Row:
    """Result of the newest premarket pipeline log (FAIL lines win over WARN lines)."""
    from market_time import is_trading_day
    logs = pipeline_logs(paths)
    if not logs:
        return Row("pipeline", WARN, f"no {PIPELINE_LOG_GLOB} in {paths.logs / 'scheduler'}")
    newest = logs[-1]
    m = re.search(r"(\d{4}-\d{2}-\d{2})_(\d{4})", newest.name)
    stamp = datetime.strptime(f"{m.group(1)} {m.group(2)}", "%Y-%m-%d %H%M") if m else None
    try:
        text = newest.read_text(errors='replace')
    except OSError as exc:
        return Row("pipeline", WARN, f"{newest.name} unreadable ({exc})")
    lines = text.splitlines()
    errors = [l.strip() for l in lines if l.startswith("ERROR:") or "SCAN FAILED" in l]
    warns = [l.strip() for l in lines if l.startswith("WARN:")]
    done = any(l.startswith("==> Done") for l in lines)
    when = f"{newest.name}"
    stale = ""
    if stamp and is_trading_day(now.date()) and stamp.date() < now.date() and now.hour >= 10:
        stale = f"; no run today yet (last {stamp:%a %m-%d})"
    if errors:
        return Row("pipeline", FAIL, f"{when}: {errors[-1][:120]}{stale}")
    if not done:
        return Row("pipeline", WARN, f"{when}: no '==> Done' line (still running or killed){stale}")
    if warns:
        return Row("pipeline", WARN, f"{when}: done with {len(warns)} warning(s): {warns[-1][:100]}{stale}")
    return Row("pipeline", WARN if stale else OK, f"{when}: done{stale}")


def _scan_stamp(path: Path) -> Optional[datetime]:
    m = re.match(r"scan_(\d{4}-\d{2}-\d{2})_(\d{4})_", path.name)
    return datetime.strptime(f"{m.group(1)} {m.group(2)}", "%Y-%m-%d %H%M") if m else None


def scan_row(paths: Paths, now: datetime) -> Row:
    """Newest pipeline (afterhours) scan should be from today or the last trading day."""
    from market_time import is_trading_day
    after = [(t, p) for p in paths.scans.glob("scan_*_afterhours_COMPLETE.csv") if (t := _scan_stamp(p))]
    watch = [(t, p) for p in paths.scans.glob("scan_*_watch_COMPLETE.csv") if (t := _scan_stamp(p))]
    if not after:
        return Row("scan", FAIL, f"no afterhours COMPLETE scan in {paths.scans}")
    t, p = max(after)
    # The 9:20 pipeline should have scanned today by 10:00; before that (or on a holiday) the last
    # trading day's scan is the newest one expected
    expected = now.date() if is_trading_day(now.date()) and now.hour >= 10 else previous_trading_day(now.date())
    text = f"pipeline scan {t:%Y-%m-%d %H:%M} ({_age_text(now - t)} old)"
    if watch:
        wt, _ = max(watch)
        text += f"; watcher scan {wt:%m-%d %H:%M} ({_age_text(now - wt)} old)"
    if t.date() < expected:
        return Row("scan", WARN, text + f"; expected one from {expected}")
    return Row("scan", OK, text)


def ibd_row(paths: Paths, now: datetime) -> Row:
    files = []
    for p in Path(paths.ibd_glob).parent.glob(Path(paths.ibd_glob).name):
        m = re.search(r"(\d{4}-\d{2}-\d{2})", p.name)
        if m:
            files.append((datetime.strptime(m.group(1), "%Y-%m-%d").date(), p))
    if not files:
        return Row("ibd", FAIL, f"no IBD export matching {paths.ibd_glob}")
    d, p = max(files)
    age = (now.date() - d).days
    status = WARN if age > IBD_MAX_AGE_DAYS else OK
    return Row("ibd", status, f"{p.name}: {age} day(s) old" + (f" (> {IBD_MAX_AGE_DAYS}: re-export from IBD)"
                                                               if status == WARN else ""))


def ledger_row(paths: Paths) -> Row:
    import order_ledger as ol
    path = Path(paths.orders or ol.ORDERS_DIR) / "ledger.jsonl"
    if not path.exists():
        return Row("ledger", OK, "no bot orders yet (ledger.jsonl absent)")
    try:
        ledger = ol.OrderLedger(path=path, source='morning_report')
        cur = ledger.current()
        trades = ledger.open_trades()
    except Exception as exc:
        return Row("ledger", FAIL, f"{path.name} unreadable: {exc}")
    if not cur.empty:
        cur = cur[cur['role'] != ol.TRADE_ROLE]     # trade status events ('closed', ...) are not orders
    live = cur[~cur['state'].isin(ol.TERMINAL)] if not cur.empty else cur
    unsettled = live[live['state'].isin(['intended', 'submitted'])] if not live.empty else live
    shorts = [t for t in trades if float(t.get('net_qty') or 0) < 0]
    text = (f"{len(live)} live bot order(s), {len(trades)} open trade(s)"
            + (f" ({', '.join(sorted({str(t['symbol']) for t in trades}))})" if trades else ""))
    if shorts:
        return Row("ledger", FAIL, text + f"; SHORT from bot sells: {', '.join(t['symbol'] for t in shorts)}")
    if len(unsettled):
        return Row("ledger", WARN, text + f"; {len(unsettled)} not yet confirmed by TWS (intended/submitted): "
                                          "run a read-only fill_monitor to reconcile")
    return Row("ledger", OK, text)


def expected_labels(launchd_dir: Path) -> List[str]:
    labels = []
    for plist in sorted(launchd_dir.glob("*.plist")):
        try:
            label = plistlib.loads(plist.read_bytes()).get('Label')
        except Exception:
            continue
        if label:
            labels.append(str(label))
    return labels


def launchd_row(paths: Paths, runner: Optional[Callable] = None) -> Row:
    """`launchctl list` (read-only): each repo job loaded; a stopped job's last exit status."""
    labels = expected_labels(paths.launchd)
    if not labels:
        return Row("launchd", WARN, f"no plists in {paths.launchd}")
    run = runner or subprocess.run
    try:
        res = run(["launchctl", "list"], capture_output=True, text=True, timeout=10, check=False)
        listing = res.stdout or ""
    except Exception as exc:
        return Row("launchd", WARN, f"launchctl list failed ({exc})")
    jobs: Dict[str, Tuple[str, str]] = {}
    for line in listing.splitlines():
        parts = line.split("\t") if "\t" in line else line.split()
        if len(parts) >= 3:
            jobs[parts[2].strip()] = (parts[0].strip(), parts[1].strip())
    missing = [l for l in labels if l not in jobs]
    bad = []
    for l in labels:
        if l in jobs:
            pid, status = jobs[l]
            if pid in ('-', '') and status not in ('0', '-'):
                bad.append(f"{l.replace('com.example.', '')} last exit {status}")
    loaded = [l.replace('com.example.', '') + (" running" if jobs[l][0] not in ('-', '') else "")
              for l in labels if l in jobs]
    text = f"loaded: {', '.join(loaded) or 'none'}"
    if missing:
        return Row("launchd", WARN, text + f"; NOT loaded: {', '.join(m.replace('com.example.', '') for m in missing)}")
    if bad:
        return Row("launchd", WARN, text + f"; {'; '.join(bad)}")
    return Row("launchd", OK, text)


def _launchd_log_paths(launchd_dir: Path) -> List[Path]:
    """StandardOutPath / StandardErrorPath of the repo plists (as scripts/rotate_launchd_logs.py reads them)."""
    spec = importlib.util.spec_from_file_location("rotate_launchd_logs", ROOT / "scripts" / "rotate_launchd_logs.py")
    rll = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rll)
    return rll.plist_log_paths(launchd_dir)


def _tree_bytes(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file() and not p.is_symlink():
                total += p.stat().st_size
        except OSError:
            continue
    return total


def disk_rows(paths: Paths, disk_usage: Callable = shutil.disk_usage) -> List[Row]:
    rows = []
    try:
        free_gb = disk_usage(str(paths.root)).free / 1024 ** 3
        status = FAIL if free_gb < DISK_FAIL_GB else WARN if free_gb < DISK_WARN_GB else OK
        rows.append(Row("disk", status, f"{free_gb:.0f} GB free"))
    except OSError as exc:
        rows.append(Row("disk", WARN, f"disk usage unavailable ({exc})"))
    logs_mb = _mb(_tree_bytes(paths.logs)) if paths.logs.exists() else 0.0
    plogs = pipeline_logs(paths)
    p_mb = _mb(sum(p.stat().st_size for p in plogs))
    try:
        big = [(p, _mb(p.stat().st_size)) for p in _launchd_log_paths(paths.launchd)
               if p.exists() and _mb(p.stat().st_size) > LAUNCHD_LOG_MAX_MB]
    except Exception:
        big = []
    text = f"Logs/ {logs_mb:,.0f} MB; {len(plogs)} pipeline run log(s) {p_mb:,.1f} MB"
    if big:
        text += "; over 50 MB: " + ", ".join(f"{p.name} {mb:,.0f} MB" for p, mb in big) + \
                " (python scripts/rotate_launchd_logs.py)"
    rows.append(Row("logs", WARN if big or logs_mb > LOGS_WARN_MB else OK, text))
    return rows


# ─── report ───

def build_report(port: int, account: Optional[str], use_tws: bool = True, paths: Optional[Paths] = None,
                 now: Optional[datetime] = None, connector_factory=None, launchctl_runner=None,
                 disk_usage: Callable = shutil.disk_usage, sleep: Callable[[float], None] = time.sleep) -> List[Row]:
    paths = paths or Paths()
    now = now or datetime.now()
    rows: List[Row] = []
    snapshot = None
    if use_tws:
        try:
            tws_rows, snapshot = check_tws(port, account, connector_factory, sleep, now, paths)
        except Exception as exc:
            tws_rows = [Row("tws", FAIL, f"TWS check failed: {exc}")]
        rows += tws_rows
    else:
        rows.append(Row("tws", WARN, "skipped (--no-tws)"))
    steps: Sequence[Callable[[], object]] = (
        lambda: breakers_row(snapshot, account, now, paths),
        lambda: pipeline_row(paths, now),
        lambda: scan_row(paths, now),
        lambda: ibd_row(paths, now),
        lambda: ledger_row(paths),
        lambda: launchd_row(paths, launchctl_runner),
        lambda: disk_rows(paths, disk_usage),
    )
    names = ("breakers", "pipeline", "scan", "ibd", "ledger", "launchd", "disk")
    for name, step in zip(names, steps):
        try:
            out = step()
        except Exception as exc:   # one broken check never hides the others
            out = Row(name, FAIL, f"check failed: {exc}")
        rows += out if isinstance(out, list) else [out]
    return rows


def format_table(rows: Sequence[Row]) -> str:
    width = max(len(r.name) for r in rows)
    return "\n".join(f"{r.status:<4}  {r.name:<{width}}  {r.detail}" for r in rows)


def summary_message(rows: Sequence[Row]) -> Optional[Tuple[str, str]]:
    """(level, message) for one banner, or None if every row is OK."""
    bad = [r for r in rows if r.status in (WARN, FAIL)]
    if not bad:
        return None
    fails = [r for r in bad if r.status == FAIL]
    head = f"{len(fails)} FAIL, {len(bad) - len(fails)} WARN: "
    return ('error' if fails else 'warning'), head + "; ".join(f"{r.name} {r.status}" for r in bad)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only morning health report (paper only)")
    parser.add_argument('--port', type=int, default=int(getattr(config, 'TWS_PORT', 7497)))
    parser.add_argument('--account', default=None, help="Paper account DU... (default config IBKR_ACCOUNT)")
    parser.add_argument('--no-tws', action='store_true', help="Skip TWS; files only")
    parser.add_argument('--no-notify', action='store_true', help="Print only, no macOS banner")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="[%(asctime)s] %(levelname)s - %(message)s", datefmt="%H:%M:%S")
    account = (args.account or str(getattr(config, 'IBKR_ACCOUNT', '') or '')).strip().upper() or None
    rows = build_report(args.port, account, use_tws=not args.no_tws)
    print(f"Morning report {datetime.now():%Y-%m-%d %H:%M}")
    print(format_table(rows))
    msg = summary_message(rows)
    if msg and not args.no_notify:
        notifier.notify("Morning report", msg[1], msg[0])
    return 1 if any(r.status == FAIL for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
