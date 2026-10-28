#!/usr/bin/env python3
"""
macOS notifications for things the owner should look at (scan failed, unprotected shares, ...).

notify(title, message, level) posts a Notification Center banner via osascript; warnings and
errors also play a sound. It never raises: a failed post is logged and the caller carries on.

Silent when:
  - config NOTIFY_ENABLED is False (default True), or env TRADING_NOTIFY=0
  - not on macOS, or running under pytest
  - the same title + message was posted in the last 30 minutes (state: Data/notify_state.json,
    locked so concurrent jobs don't race), so a launchd job that relaunches every 60s alerts once,
    not every minute. Only a successful post counts: a failed/denied one doesn't start the window.

CLI (for run_daily_pipeline.sh):
  python notify.py --title "Premarket scan FAILED" --level error "momentum scanner exited 2"
Always exits 0.
"""
import argparse
import fcntl
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator, List, Optional

logger = logging.getLogger(__name__)

STATE_PATH = Path(__file__).resolve().parent / 'Data' / 'notify_state.json'
DEDUP_SECONDS = 30 * 60
STATE_KEEP_SECONDS = 24 * 3600     # older entries are pruned from the state file
SOUNDS = {'warning': 'Funk', 'error': 'Basso'}   # 'info' is silent
LEVELS = ('info', 'warning', 'error')

# osascript reads title/message/sound as arguments, so quotes in a message can't break the script
_SCRIPT_SOUND = ['-e', 'on run argv',
                 '-e', 'display notification (item 2 of argv) with title (item 1 of argv) '
                       'sound name (item 3 of argv)',
                 '-e', 'end run']
_SCRIPT_QUIET = ['-e', 'on run argv',
                 '-e', 'display notification (item 2 of argv) with title (item 1 of argv)',
                 '-e', 'end run']


def _under_pytest() -> bool:
    return 'pytest' in sys.modules or 'PYTEST_CURRENT_TEST' in os.environ


def enabled() -> bool:
    """False if config NOTIFY_ENABLED is False, TRADING_NOTIFY=0, not macOS, or under pytest."""
    try:
        import config
        if getattr(config, 'NOTIFY_ENABLED', True) is False:
            return False
    except Exception:
        pass   # no config.py: notifications stay on (the default)
    if os.environ.get('TRADING_NOTIFY', '1') == '0':
        return False
    return sys.platform == 'darwin' and not _under_pytest()


def osascript_args(title: str, message: str, level: str = 'info') -> List[str]:
    sound = SOUNDS.get(level)
    if sound:
        return ['osascript', *_SCRIPT_SOUND, title, message, sound]
    return ['osascript', *_SCRIPT_QUIET, title, message]


def _key(title: str, message: str) -> str:
    return hashlib.sha1(f"{title}\n{message}".encode('utf-8')).hexdigest()


def _load_state(path: Path) -> dict:
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}   # missing or corrupt: start over (worst case one duplicate banner)


def _save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(state))
    os.replace(tmp, path)   # atomic: two jobs posting at once can't leave half a file


@contextmanager
def _state_lock(path: Path) -> Iterator[None]:
    """Hold an exclusive lock (<state>.lock) so two jobs alerting at once can't drop each other's key.

    Best effort: if the lock file can't be opened we carry on unlocked (worst case one duplicate banner).
    """
    handle = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(path.with_name(f"{path.name}.lock"), 'a')
        fcntl.flock(handle, fcntl.LOCK_EX)
    except Exception:
        if handle is not None:
            handle.close()
        handle = None
    try:
        yield
    finally:
        if handle is not None:
            handle.close()   # closing releases the flock


def recently_posted(title: str, message: str, now: float, path: Path = STATE_PATH,
                    dedup_seconds: float = DEDUP_SECONDS) -> bool:
    """True if the same title + message was posted successfully within dedup_seconds."""
    last = _load_state(path).get(_key(title, message))
    return isinstance(last, (int, float)) and now - last < dedup_seconds


def record_post(title: str, message: str, now: float, path: Path = STATE_PATH) -> None:
    """Remember a successful post (and prune entries older than a day)."""
    state = {k: v for k, v in _load_state(path).items()
             if isinstance(v, (int, float)) and now - v < STATE_KEEP_SECONDS}
    state[_key(title, message)] = now
    _save_state(path, state)


def notify(title: str, message: str, level: str = 'info', *, dedup_seconds: float = DEDUP_SECONDS,
           state_path: Optional[Path] = None, now: Optional[float] = None,
           runner: Optional[Callable] = None, force_enabled: bool = False) -> bool:
    """Post a macOS notification. Returns True if one was sent. Never raises.

    dedup_seconds: how long the same title + message stays suppressed after a successful post
    (default 30 min; the watcher's init failure uses longer). Only a successful post starts the
    window, so a failed or denied osascript call doesn't silence the next attempt.
    force_enabled / runner / state_path / now exist for tests only.
    """
    try:
        if not (force_enabled or enabled()):
            return False
        level = level if level in LEVELS else 'info'
        title, message = str(title)[:120], str(message)[:400]
        now = time.time() if now is None else now
        path = state_path or STATE_PATH
        # The lock covers check -> post -> record, so concurrent jobs post the same message once
        with _state_lock(path):
            if recently_posted(title, message, now, path, dedup_seconds):
                logger.debug(f"notification suppressed (same one within {dedup_seconds / 60:.0f} min): "
                             f"{title}: {message}")
                return False
            run = runner or subprocess.run
            result = run(osascript_args(title, message, level), timeout=5, check=False, capture_output=True)
            rc = getattr(result, 'returncode', 0)
            if rc:
                logger.warning(f"macOS notification failed (osascript exit {rc}): {title}: {message}")
                return False
            record_post(title, message, now, path)
        return True
    except Exception as exc:
        logger.warning(f"macOS notification failed ({exc}): {title}: {message}")
        return False


def notify_batch(title: str, messages: List[str], level: str = 'warning', **kw) -> bool:
    """One banner for a run's alerts ("3 alerts: a; b; c"), not one sounded banner per row.

    An unchanged set of alerts on the next run is the same message, so it dedups like any other.
    """
    messages = [str(m) for m in messages if m]
    if not messages:
        return False
    if len(messages) == 1:
        return notify(title, messages[0], level, **kw)
    return notify(title, f"{len(messages)} alerts: " + "; ".join(messages), level, **kw)


def notify_exit_code(rc: int, title: str, enabled: bool = True) -> int:
    """For `raise SystemExit(...)`: an exit code 2 (refused by a guard / halted) posts a warning.

    enabled=False (e.g. order_entry --plan-only, which never sends orders) just passes rc through.
    """
    if rc == 2 and enabled:
        notify(title, "exited 2 (refused by a guard or halted): check the log and Data/orders/", 'warning')
    return rc


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Post a macOS notification (rate-limited; never fails)")
    parser.add_argument('message')
    parser.add_argument('--title', default='swing-trading-bot')
    parser.add_argument('--level', choices=LEVELS, default='info')
    args = parser.parse_args(argv)
    notify(args.title, args.message, args.level)
    return 0


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
    try:
        main()
    except SystemExit:
        pass   # argparse errors exit 2; the pipeline must never fail because of a notification
    raise SystemExit(0)
