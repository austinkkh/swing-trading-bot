"""Log size caps: the scanner's rotating file log and the launchd log rename script."""

import importlib.util
import logging.handlers
from datetime import datetime
from pathlib import Path

import momentum_scanner as ms

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("rotate_launchd_logs", ROOT / "scripts" / "rotate_launchd_logs.py")
rll = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rll)


def test_scan_log_handler_is_size_capped_and_daily(tmp_path, monkeypatch):
    monkeypatch.setattr(ms, "LOG_MAX_MB", 50, raising=False)
    monkeypatch.setattr(ms, "LOG_BACKUP_COUNT", 5, raising=False)
    h = ms.scan_log_handler(tmp_path, day=datetime(2026, 10, 26))
    try:
        assert isinstance(h, logging.handlers.RotatingFileHandler)
        assert h.maxBytes == 50 * 1024 * 1024 and h.backupCount == 5
        assert Path(h.baseFilename).name == "momentum_scan_20261026.log"
    finally:
        h.close()


def test_scan_log_rolls_over_at_the_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(ms, "LOG_MAX_MB", 0.001, raising=False)   # ~1 KB
    monkeypatch.setattr(ms, "LOG_BACKUP_COUNT", 2, raising=False)
    h = ms.scan_log_handler(tmp_path, day=datetime(2026, 10, 26))
    log = logging.getLogger("test_scan_log_rolls")
    log.propagate = False
    log.addHandler(h)
    try:
        for i in range(200):
            log.warning("x" * 50)
    finally:
        log.removeHandler(h)
        h.close()
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == ["momentum_scan_20261026.log", "momentum_scan_20261026.log.1", "momentum_scan_20261026.log.2"]
    assert all(p.stat().st_size <= 1100 for p in tmp_path.iterdir())


def test_rotate_renames_only_oversized_logs_and_deletes_nothing(tmp_path):
    (tmp_path / "scheduler").mkdir()
    big = tmp_path / "watcher-launchd.err.log"
    big.write_bytes(b"x" * 3000)
    small = tmp_path / "watcher-launchd.out.log"
    small.write_bytes(b"x" * 10)
    other = tmp_path / "momentum_scan_20261026.log"   # not a launchd log: never touched
    other.write_bytes(b"x" * 5000)
    old = tmp_path / "watcher-launchd.err.20261001-090000.log"   # an earlier rotation
    old.write_bytes(b"x" * 100)
    now = datetime(2026, 10, 26, 13, 21, 0)

    moved, total = rll.rotate(tmp_path, max_mb=0.001, now=now)

    target = tmp_path / "watcher-launchd.err.20261026-132100.log"
    assert moved == [(big, target)]
    assert target.read_bytes() == b"x" * 3000 and not big.exists()
    assert small.exists() and other.exists() and old.exists()
    assert total == 3000 + 10 + 100


def test_rotate_never_overwrites_an_existing_copy(tmp_path):
    now = datetime(2026, 10, 26, 13, 21, 0)
    taken = tmp_path / "watcher-launchd.err.20261026-132100.log"
    taken.write_bytes(b"first")
    (tmp_path / "watcher-launchd.err.log").write_bytes(b"x" * 3000)
    moved, _ = rll.rotate(tmp_path, max_mb=0.001, now=now)
    assert moved[0][1].name == "watcher-launchd.err.20261026-132100-1.log"
    assert taken.read_bytes() == b"first"


def test_dry_run_renames_nothing(tmp_path, capsys):
    log = tmp_path / "watcher-launchd.err.log"
    log.write_bytes(b"x" * 3000)
    assert rll.main(["--logs-dir", str(tmp_path), "--max-mb", "0.001", "--dry-run"]) == 0
    assert log.exists()
    assert "would rename" in capsys.readouterr().out


def test_default_paths_come_from_the_plists_with_their_own_case():
    paths = rll.plist_log_paths(ROOT / "launchd")
    assert Path("/opt/swing-trading-bot/logs/watcher-launchd.err.log") in paths     # lowercase logs/
    assert Path("/opt/swing-trading-bot/Logs/scheduler/launchd.out.log") in paths
    # every plist log is one the fallback list also knows about
    assert {p.name for p in paths} <= {Path(n).name for n in rll.LAUNCHD_LOGS}


def test_rotate_explicit_paths(tmp_path):
    lower = tmp_path / "logs"
    lower.mkdir()
    log = lower / "watcher-launchd.err.log"
    log.write_bytes(b"x" * 3000)
    moved, total = rll.rotate(paths=[log, tmp_path / "missing.log"], max_mb=0.001,
                              now=datetime(2026, 10, 26, 13, 21, 0))
    assert moved == [(log, lower / "watcher-launchd.err.20261026-132100.log")] and total == 3000
