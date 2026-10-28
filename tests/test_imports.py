"""Every tracked top-level module imports under config.template (review TS-01: a fresh clone broke)."""

import importlib
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SKIP = {"config", "config.template"}


def tracked_modules():
    files = subprocess.run(["git", "ls-files", "*.py"], cwd=ROOT, capture_output=True, text=True).stdout.split()
    return sorted(Path(f).stem for f in files if "/" not in f and Path(f).stem not in SKIP)


@pytest.mark.parametrize("name", tracked_modules())
def test_module_imports(name):
    importlib.import_module(name)


ORDER_MODULES = ("order_broker", "order_ledger", "order_entry", "fill_monitor", "exit_manager")


def test_order_modules_are_tracked():
    """The bot order layer is in the import sweep above (an untracked module would be skipped)."""
    assert set(ORDER_MODULES) <= set(tracked_modules())


@pytest.mark.parametrize("name", ORDER_MODULES)
def test_order_module_import_has_no_side_effects(name, monkeypatch):
    """Importing an order module never connects to TWS: a fresh import with IBKRConnector rigged to
    raise must still succeed, and submission stays off under config.template."""
    import sys

    import ibkr_connector

    def _no_connect(*args, **kwargs):
        raise AssertionError(f"importing {name} constructed an IBKRConnector")

    monkeypatch.setattr(ibkr_connector, "IBKRConnector", _no_connect)
    monkeypatch.delitem(sys.modules, name, raising=False)
    importlib.import_module(name)
    import order_broker
    assert order_broker.submission_enabled(True) is False
