"""
The daily pipeline's order-plan step: dry run only, gated by ORDER_PLAN_ENABLED, never fatal.

The step's block is cut out of run_daily_pipeline.sh and run under bash with `set -euo pipefail`
(same as the pipeline) and a stub $PYTHON: `-c` gate checks go to the real interpreter, anything
else (order_entry.py) just records its argv and exits with STUB_RC. Nothing connects to TWS.
"""

import os
import re
import stat
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "run_daily_pipeline.sh"


def _script() -> str:
    return PIPELINE.read_text()


def _order_plan_block() -> str:
    """From the ORDER_PLAN_ON gate to the `fi` before '==> Done'."""
    text = _script()
    start = text.index('ORDER_PLAN_ON=')
    end = text.index('echo "==> Done"')
    return text[start:end]


def _order_module_lines(text: str):
    """Logical lines (backslash continuations joined) that invoke an order module."""
    joined = re.sub(r'\\\n\s*', ' ', text)
    return [ln for ln in joined.splitlines()
            if re.search(r'\b(order_entry|fill_monitor|exit_manager|order_broker)\.py\b', ln)]


def test_order_entry_line_is_plan_only_and_never_submit():
    lines = _order_module_lines(_script())
    assert lines, "pipeline should run order_entry.py --plan-only"
    for ln in lines:
        assert 'order_entry.py' in ln, f"unexpected order module in the pipeline: {ln}"
        assert '--plan-only' in ln
        assert '--submit' not in ln
        assert '--decisions "$DECISIONS_CSV"' in ln      # this run's file, never 'latest'


def test_pipeline_never_passes_submit_to_any_order_module():
    # --submit-paper belongs to ibkr_paper_executor (contract validation only); a bare --submit must not appear
    code = "\n".join(ln for ln in _script().splitlines() if not ln.lstrip().startswith("#"))
    assert not re.search(r'--submit(?![-\w])', code)


def _run_block(tmp_path: Path, config_body, stub_rc: int = 0):
    """Run the order-plan block with a config.py in tmp_path (None = no config module at all)."""
    if config_body is not None:
        (tmp_path / "config.py").write_text(config_body)
    argv_log = tmp_path / "argv.txt"
    stub = tmp_path / "python_stub.sh"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        f'if [[ "${{1:-}}" == "-c" ]]; then exec "{sys.executable}" "$@"; fi\n'
        f'printf "%s\\n" "$@" > "{argv_log}"\n'
        'exit "${STUB_RC:-0}"\n')
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    script = ("set -euo pipefail\n"
              f'PYTHON="{stub}"\nDECISIONS_CSV=/tmp/d.csv\nSNAPSHOT_JSON=/tmp/s.json\nACCOUNT=DUX123\n'
              + _order_plan_block()
              + 'echo "REACHED_DONE"\n')
    env = {**os.environ, "STUB_RC": str(stub_rc), "PYTHONPATH": ""}
    proc = subprocess.run(["bash", "-c", script], cwd=tmp_path, env=env, capture_output=True, text=True)
    argv = argv_log.read_text().split("\n") if argv_log.exists() else None
    return proc, argv


def test_enabled_runs_plan_only_with_account(tmp_path):
    proc, argv = _run_block(tmp_path, "ORDER_PLAN_ENABLED = True\n")
    assert proc.returncode == 0 and "REACHED_DONE" in proc.stdout
    assert argv[:3] == ["order_entry.py", "--decisions", "/tmp/d.csv"]
    assert "--plan-only" in argv and "--submit" not in argv
    assert argv[argv.index("--account") + 1] == "DUX123"


def test_missing_key_defaults_to_on(tmp_path):
    _, argv = _run_block(tmp_path, "OTHER = 1\n")
    assert argv is not None and "--plan-only" in argv


@pytest.mark.parametrize("body", ["ORDER_PLAN_ENABLED = False\n", 'ORDER_PLAN_ENABLED = "True"\n',
                                  "ORDER_PLAN_ENABLED = 1\n", "raise RuntimeError('broken config')\n", None])
def test_gate_off_unless_bool_true(tmp_path, body):
    proc, argv = _run_block(tmp_path, body)
    assert argv is None, "order_entry.py must not run"
    assert proc.returncode == 0 and "REACHED_DONE" in proc.stdout
    assert "Order plan step disabled" in proc.stdout


@pytest.mark.parametrize("rc,msg", [(2, "halted or refused"), (1, "order plan failed (exit 1)"),
                                    (3, "order plan failed (exit 3)")])
def test_failure_is_non_fatal(tmp_path, rc, msg):
    proc, argv = _run_block(tmp_path, "ORDER_PLAN_ENABLED = True\n", stub_rc=rc)
    assert argv is not None
    assert proc.returncode == 0 and "REACHED_DONE" in proc.stdout
    assert msg in proc.stdout


def test_template_default_keeps_submission_off():
    import config
    assert config.AUTO_SUBMIT_ENABLED is False
    assert config.ORDER_PLAN_ENABLED is True
    assert config.ORDER_CLIENT_ID == 16 and config.ORDER_READONLY_CLIENT_ID == 17
    assert config.CLOSE_WINDOW_ET == ("15:35", "15:48")
