#!/usr/bin/env python3
"""
rotate_launchd_logs.py: move oversized launchd stdout/stderr logs aside (never deletes anything).

launchd itself writes these files (StandardOutPath / StandardErrorPath in the plists), so Python's
log rotation can't cap them. This script renames each log over --max-mb to a timestamped name next
to it, e.g. watcher-launchd.err.log -> watcher-launchd.err.20261026-132100.log, and prints the
total size of the live logs and every rotated copy so you can see what is piling up. Old copies
are kept: delete or compress them yourself when you decide to.

A running job keeps writing to the renamed file (launchd holds it open) until the job restarts.
The scheduled jobs (premarket, pp-tracker) reopen the path on their next run. The watcher runs all
day, so restart it once after rotating, when no scan is in progress:
  launchctl kickstart -k gui/$(id -u)/com.example.trading-watcher

Not scheduled. Usage:
  python scripts/rotate_launchd_logs.py              # rotate logs over 50 MB
  python scripts/rotate_launchd_logs.py --dry-run    # only report
  python scripts/rotate_launchd_logs.py --max-mb 20
"""
import argparse
import plistlib
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / 'Logs'
LAUNCHD_DIR = ROOT / 'launchd'

# Fallback when launchd/*.plist can't be read: the plists' StandardOutPath / StandardErrorPath
# files relative to LOGS. (The watcher and tv-convert plists write to .../swing-trading-bot/logs/,
# lowercase; on the default case-insensitive APFS that is the same folder as Logs/.)
LAUNCHD_LOGS = (
    'watcher-launchd.out.log', 'watcher-launchd.err.log',
    'tv-convert.out.log', 'tv-convert.err.log',
    'scheduler/launchd.out.log', 'scheduler/launchd.err.log',
    'scheduler/pp_tracker.out.log', 'scheduler/pp_tracker.err.log',
)
STAMP_RE = re.compile(r'\.\d{8}-\d{6}(-\d+)?\.log$')


def rotated_name(path: Path, now: datetime) -> Path:
    """watcher-launchd.err.log -> watcher-launchd.err.<yyyymmdd-HHMMSS>.log, never an existing name."""
    base = path.name[:-len('.log')] if path.name.endswith('.log') else path.name
    target = path.with_name(f"{base}.{now:%Y%m%d-%H%M%S}.log")
    n = 1
    while target.exists():   # two rotations in one second: add a counter, never overwrite
        target = path.with_name(f"{base}.{now:%Y%m%d-%H%M%S}-{n}.log")
        n += 1
    return target


def rotated_copies(path: Path) -> List[Path]:
    """Earlier rotations of this log (timestamped siblings)."""
    base = path.name[:-len('.log')] if path.name.endswith('.log') else path.name
    return sorted(p for p in path.parent.glob(f"{base}.*.log") if STAMP_RE.search(p.name))


def plist_log_paths(launchd_dir: Path = LAUNCHD_DIR) -> List[Path]:
    """StandardOutPath / StandardErrorPath of every launchd/*.plist, exactly as the plist spells them."""
    paths: List[Path] = []
    for plist in sorted(launchd_dir.glob('*.plist')):
        try:
            data = plistlib.loads(plist.read_bytes())
        except Exception:
            continue   # unreadable plist: skip it (the others still rotate)
        for key in ('StandardOutPath', 'StandardErrorPath'):
            value = data.get(key)
            if isinstance(value, str) and value and Path(value) not in paths:
                paths.append(Path(value))
    return paths


def default_paths(logs_dir: Optional[Path] = None) -> List[Path]:
    """--logs-dir given: the known names under it. Otherwise the plists' own paths (fallback: LOGS)."""
    if logs_dir is None:
        found = plist_log_paths()
        if found:
            return found
        logs_dir = LOGS
    return [logs_dir / name for name in LAUNCHD_LOGS]


def rotate(logs_dir: Optional[Path] = None, max_mb: float = 50, dry_run: bool = False,
           now: Optional[datetime] = None, paths: Optional[List[Path]] = None) -> Tuple[List[Tuple[Path, Path]], int]:
    """Rename each log over max_mb. Returns ([(old, new)], total bytes of live + rotated logs).

    paths default: default_paths(logs_dir).
    """
    now = now or datetime.now()
    limit = max_mb * 1024 * 1024
    moved: List[Tuple[Path, Path]] = []
    total = 0
    for path in (default_paths(logs_dir) if paths is None else paths):
        for old in rotated_copies(path):
            total += old.stat().st_size
        if not path.exists():
            continue
        size = path.stat().st_size
        total += size
        if size > limit:
            target = rotated_name(path, now)
            if not dry_run:
                path.rename(target)
            moved.append((path, target))
    return moved, total


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Rename oversized launchd logs to timestamped names (no deletion)")
    parser.add_argument('--max-mb', type=float, default=50, help='rotate logs bigger than this (default 50)')
    parser.add_argument('--logs-dir', default=None,
                        help='folder holding the logs (default: the paths in launchd/*.plist)')
    parser.add_argument('--dry-run', action='store_true', help='report only, rename nothing')
    args = parser.parse_args(argv)
    logs_dir = Path(args.logs_dir).expanduser() if args.logs_dir else None
    moved, total = rotate(logs_dir, args.max_mb, args.dry_run)
    verb = 'would rename' if args.dry_run else 'renamed'
    for old, new in moved:
        print(f"{verb}: {old.name} -> {new.name}")
    if not moved:
        print(f"no launchd log over {args.max_mb:g} MB")
    print(f"launchd logs incl. rotated copies: {total / 1024 / 1024:,.1f} MB (nothing deleted)")
    if moved and not args.dry_run and any('watcher' in o.name for o, _ in moved):
        print("The watcher keeps writing to the renamed file until it restarts:\n"
              "  launchctl kickstart -k gui/$(id -u)/com.example.trading-watcher")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
