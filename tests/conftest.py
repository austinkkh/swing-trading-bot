"""
Test setup: load config.template.py as the `config` module, so tests never
depend on a personal config.py, and outputs go to a temp folder.
"""

import importlib.util
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location("config", ROOT / "config.template.py")
config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(config)
config.OUTPUT_DIR = config.LOG_DIR = tempfile.mkdtemp(prefix="scanner_tests_")
sys.modules["config"] = config

# notify.py pops macOS notifications via osascript; tests must never do that (it is also a no-op under pytest)
import os  # noqa: E402
os.environ["TRADING_NOTIFY"] = "0"
