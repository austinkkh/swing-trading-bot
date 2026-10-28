"""macOS notifications (notify.py) and where they are wired in. Nothing here ever runs osascript."""

import subprocess
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

import notify


class Runner:
    """Stands in for subprocess.run; records the argv."""

    def __init__(self, returncode=0, exc=None):
        self.calls, self.returncode, self.exc = [], returncode, exc

    def __call__(self, args, **kw):
        if self.exc:
            raise self.exc
        self.calls.append(args)
        return SimpleNamespace(returncode=self.returncode)


@pytest.fixture
def state(tmp_path):
    return tmp_path / "notify_state.json"


@pytest.fixture
def no_osascript(monkeypatch):
    """Fail the test if anything reaches the real subprocess.run."""
    def boom(*a, **k):
        raise AssertionError("osascript must not run under pytest")
    monkeypatch.setattr(subprocess, "run", boom)


# ─── notify() ───
def test_posts_with_title_message_and_a_sound_for_warnings(state):
    run = Runner()
    assert notify.notify("Fill monitor", 'AAPL "REPORT": 50 sh', "warning", state_path=state, now=1000,
                         runner=run, force_enabled=True)
    args = run.calls[0]
    assert args[0] == "osascript" and args[-3:] == ["Fill monitor", 'AAPL "REPORT": 50 sh', "Funk"]
    assert any("sound name" in a for a in args)


def test_info_is_silent_and_error_uses_its_own_sound(state):
    run = Runner()
    notify.notify("Engine", "1 BUY", "info", state_path=state, now=1000, runner=run, force_enabled=True)
    notify.notify("Scan", "failed", "error", state_path=state, now=1000, runner=run, force_enabled=True)
    assert not any("sound name" in a for a in run.calls[0]) and run.calls[0][-2:] == ["Engine", "1 BUY"]
    assert run.calls[1][-1] == "Basso"


def test_same_title_and_message_within_30_min_posts_once(state):
    run = Runner()
    kw = dict(state_path=state, runner=run, force_enabled=True)
    assert notify.notify("Watcher", "init failed", "error", now=1000, **kw)
    assert not notify.notify("Watcher", "init failed", "error", now=1000 + 29 * 60, **kw)
    assert notify.notify("Watcher", "other message", "error", now=1000 + 29 * 60, **kw)
    assert notify.notify("Watcher", "init failed", "error", now=1000 + 31 * 60, **kw)
    assert len(run.calls) == 3


def test_failed_post_does_not_start_the_dedup_window(state):
    # A denied/failed osascript call (e.g. the first-run permission prompt) must not silence the retry
    kw = dict(state_path=state, force_enabled=True)
    assert not notify.notify("Scan", "failed", "error", now=1000, runner=Runner(returncode=1), **kw)
    assert not notify.notify("Scan", "failed", "error", now=1001, runner=Runner(exc=OSError("x")), **kw)
    run = Runner()
    assert notify.notify("Scan", "failed", "error", now=1002, runner=run, **kw)
    assert len(run.calls) == 1
    assert not notify.notify("Scan", "failed", "error", now=1003, runner=run, **kw)   # now it dedups


def test_per_call_dedup_window(state):
    run = Runner()
    kw = dict(state_path=state, runner=run, force_enabled=True, dedup_seconds=6 * 3600)
    assert notify.notify("Watcher", "init failed", "error", now=0, **kw)
    assert not notify.notify("Watcher", "init failed", "error", now=5 * 3600, **kw)
    assert notify.notify("Watcher", "init failed", "error", now=6 * 3600 + 1, **kw)
    assert len(run.calls) == 2


def test_concurrent_posters_keep_each_others_keys(state):
    import threading
    run = Runner()
    msgs = [f"m{i}" for i in range(8)]
    threads = [threading.Thread(target=notify.notify, args=("T", m),
                                kwargs=dict(state_path=state, now=1000, runner=run, force_enabled=True))
               for m in msgs]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(run.calls) == 8
    assert all(notify.recently_posted("T", m, 1001, state) for m in msgs)   # no key lost to a race


def test_corrupt_state_file_still_posts(state):
    state.write_text("{not json")
    run = Runner()
    assert notify.notify("T", "m", state_path=state, now=1, runner=run, force_enabled=True)
    assert len(run.calls) == 1


def test_never_raises_and_logs_failures(state, caplog):
    assert not notify.notify("T", "m", state_path=state, now=1, runner=Runner(exc=OSError("no osascript")),
                             force_enabled=True)
    assert not notify.notify("T", "m2", state_path=state, now=1, runner=Runner(returncode=1), force_enabled=True)
    assert "notification failed" in caplog.text


def test_disabled_under_pytest_by_config_env_or_platform(monkeypatch, no_osascript, state):
    assert notify.enabled() is False                       # running under pytest
    assert notify.notify("T", "m", "error", state_path=state) is False
    monkeypatch.setattr(notify, "_under_pytest", lambda: False)
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setenv("TRADING_NOTIFY", "1")
    import config
    monkeypatch.setattr(config, "NOTIFY_ENABLED", True, raising=False)
    assert notify.enabled() is True
    monkeypatch.setattr(config, "NOTIFY_ENABLED", False)
    assert notify.enabled() is False
    monkeypatch.setattr(config, "NOTIFY_ENABLED", True)
    monkeypatch.setenv("TRADING_NOTIFY", "0")
    assert notify.enabled() is False
    monkeypatch.setenv("TRADING_NOTIFY", "1")
    monkeypatch.setattr(sys, "platform", "linux")
    assert notify.enabled() is False


def test_template_defaults_notifications_on():
    import config
    assert config.NOTIFY_ENABLED is True


def test_cli_always_exits_zero():
    import os
    env = dict(os.environ, TRADING_NOTIFY="0")   # the child isn't under pytest: keep it silent
    for argv in (["--title", "Premarket scan FAILED", "--level", "error", "scanner exited 2"],
                 ["--level", "bogus"]):            # bad arguments must not fail the pipeline either
        proc = subprocess.run([sys.executable, notify.__file__, *argv], env=env, capture_output=True, timeout=30)
        assert proc.returncode == 0


def test_exit_code_2_notifies_others_pass_through(monkeypatch):
    sent = []
    monkeypatch.setattr(notify, "notify", lambda *a, **k: sent.append(a))
    assert notify.notify_exit_code(0, "Order entry") == 0
    assert notify.notify_exit_code(1, "Order entry") == 1
    assert notify.notify_exit_code(2, "Order entry") == 2
    assert len(sent) == 1 and sent[0][0] == "Order entry" and sent[0][2] == "warning"


# ─── wiring ───
@pytest.mark.parametrize("module,title", [("order_entry", "Order entry"), ("fill_monitor", "Fill monitor"),
                                          ("exit_manager", "Exit manager")])
def test_order_module_alerts_go_through_notify(monkeypatch, caplog, module, title):
    mod = __import__(module)
    sent = []
    monkeypatch.setattr(notify, "notify", lambda *a, **k: sent.append(a))
    mod.notify("AAPL ALERT_COVER_SHORT: short 10 sh")
    assert sent == [(title, "AAPL ALERT_COVER_SHORT: short 10 sh", "warning")]
    assert "ALERT: AAPL ALERT_COVER_SHORT" in caplog.text   # still in the log


@pytest.mark.parametrize("module", ["order_entry", "fill_monitor", "exit_manager"])
def test_order_module_cli_exit_goes_through_notify_exit_code(module):
    from pathlib import Path
    src = Path(__import__(module).__file__).read_text()
    assert "raise SystemExit(notifier.notify_exit_code(main()," in src


def test_engine_sends_one_summary_with_buy_and_review_counts(monkeypatch):
    import execution_engine as ee
    sent = []
    monkeypatch.setattr(notify, "notify", lambda *a, **k: sent.append(a))
    df = pd.DataFrame({"Decision": [ee.BUY, ee.MANUAL_REVIEW, ee.MANUAL_REVIEW, "SKIP"]})
    ee.notify_decisions(df, "premarket")
    assert sent == [("Execution engine", "1 BUY, 2 MANUAL_REVIEW (premarket)", "info")]
    assert ee.decision_summary(pd.DataFrame({"Decision": ["SKIP"]}), "rth") == "0 BUY, 0 MANUAL_REVIEW (rth)"


def test_snapshot_warns_on_unprotected_positions_only(monkeypatch):
    import account_snapshot as acct
    sent = []
    monkeypatch.setattr(notify, "notify", lambda *a, **k: sent.append(a))
    covered = {"account": "DUX1", "positions": [{"symbol": "MU", "unprotected_shares": 0}]}
    assert acct.notify_unprotected(covered) is None and sent == []
    naked = {"account": "DUX1", "positions": [{"symbol": "MU", "unprotected_shares": 0},
                                             {"symbol": "AAPL", "unprotected_shares": 50.0}]}
    msg = acct.notify_unprotected(naked)
    assert msg == "DUX1: 1 position(s) without a full stop: AAPL 50 sh"
    assert sent == [("Unprotected positions", msg, "warning")]


@pytest.mark.parametrize("now,alerts", [
    ("2026-10-26 09:20", True),     # Monday, before the open: the watcher should be up
    ("2026-10-26 15:59", True),
    ("2026-10-26 08:59", False),    # overnight / IBKR restart
    ("2026-10-26 16:00", False),
    ("2026-10-24 11:00", False),    # Saturday
    ("2026-11-26 11:00", False),    # Thanksgiving
    ("2026-11-27 13:30", False),    # early close at 13:00
])
def test_watcher_init_failure_notifies_only_during_the_session(monkeypatch, now, alerts):
    import watch_mode
    sent = []
    monkeypatch.setattr(notify, "notify", lambda *a, **k: sent.append((a, k)))
    monkeypatch.setattr(watch_mode, "now_et", lambda: pd.Timestamp(now).to_pydatetime())
    monkeypatch.setattr(watch_mode.WatchModeMonitor, "load_latest_elite_hold", lambda self: True)
    monkeypatch.setattr(watch_mode.MomentumScanner, "initialize", lambda self: False)
    with pytest.raises(SystemExit) as exc:
        watch_mode.WatchModeMonitor().run_forever()
    assert exc.value.code == 1                      # exit code unchanged either way
    if alerts:
        assert len(sent) == 1 and sent[0][0][0] == "Watcher" and sent[0][0][2] == "error"
        assert sent[0][1]["dedup_seconds"] == 2 * 3600
    else:
        assert sent == []


def test_batch_posts_one_banner(monkeypatch):
    sent = []
    monkeypatch.setattr(notify, "notify", lambda *a, **k: sent.append(a))
    assert notify.notify_batch("Fill monitor", []) is False and sent == []
    notify.notify_batch("Fill monitor", ["AAPL REPORT: 50 sh"])
    notify.notify_batch("Fill monitor", ["AAPL REPORT: 50 sh", "TSLA REPORT: 10 sh"])
    assert sent == [("Fill monitor", "AAPL REPORT: 50 sh", "warning"),
                    ("Fill monitor", "2 alerts: AAPL REPORT: 50 sh; TSLA REPORT: 10 sh", "warning")]


def test_exit_code_banner_can_be_turned_off(monkeypatch):
    sent = []
    monkeypatch.setattr(notify, "notify", lambda *a, **k: sent.append(a))
    assert notify.notify_exit_code(2, "Order entry", enabled=False) == 2 and sent == []


def test_order_entry_plan_only_exit_2_posts_no_banner():
    # The daily pipeline runs order_entry --plan-only (never sends orders): exit 2 stays in the log
    from pathlib import Path
    import order_entry
    src = Path(order_entry.__file__).read_text()
    assert 'enabled="--plan-only" not in sys.argv[1:]' in src


def test_pipeline_keeps_notify_failures_in_its_log():
    from pathlib import Path
    src = (Path(notify.__file__).parent / "run_daily_pipeline.sh").read_text()
    line = next(l for l in src.splitlines() if l.strip().startswith('"$PYTHON" notify.py'))
    assert "/dev/null" not in line and line.rstrip().endswith("|| true")
