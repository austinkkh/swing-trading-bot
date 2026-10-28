"""Point-in-time scan lists and the portfolio simulation."""

from dataclasses import replace

import pandas as pd
import pytest

import backtest as bt
import strategy_params as sp

# The live sizing caps (strategy_params), not Account's own defaults
ACCT = bt.Account(risk_pct=sp.RISK_PER_TRADE_PCT, max_position_pct=sp.MAX_POSITION_PCT,
                  max_new_per_day=sp.MAX_NEW_ENTRIES_PER_DAY, max_open_risk_pct=sp.MAX_TOTAL_OPEN_RISK_PCT,
                  max_positions=sp.MAX_OPEN_POSITIONS)
R = ACCT.risk_pct
WIDE = R / ACCT.max_position_pct * 100 * 2      # stop % at which full risk uses half the position cap


# ─── Point-in-time lists ───
def test_loads_dated_files_and_long_format(tmp_path):
    d = tmp_path / "lists"
    d.mkdir()
    pd.DataFrame({"Ticker": ["mu", "SNDK"]}).to_csv(d / "stage2_2024-03-15.csv", index=False)
    pd.DataFrame({"Symbol": ["WDC"]}).to_csv(d / "new_highs_2024-03-22.csv", index=False)
    lists = bt.load_universe_lists(str(d))
    assert set(lists['symbol']) == {"MU", "SNDK", "WDC"}
    assert set(lists['date']) == {pd.Timestamp("2024-03-15"), pd.Timestamp("2024-03-22")}

    long = tmp_path / "scan_list.csv"
    pd.DataFrame({"date": ["2024-01-05", "2024-01-05"], "symbol": ["NVDA", "AMD"],
                  "list": ["stage2", "new_highs"]}).to_csv(long, index=False)
    assert len(bt.load_universe_lists(str(long))) == 2


def history(max_age=10):
    lists = pd.DataFrame({
        "date": pd.to_datetime(["2024-03-15", "2024-03-15", "2024-03-22"]),
        "symbol": ["MU", "SNDK", "MU"],
        "list": "stage2",
    })
    return bt.UniverseHistory(lists, max_age_days=max_age)


def test_uses_latest_list_on_or_before_the_date():
    h = history()
    assert h.on_list("SNDK", "2024-03-18")        # on the 3/15 list
    assert not h.on_list("SNDK", "2024-03-25")    # dropped from the 3/22 list
    assert h.on_list("MU", "2024-03-25")


def test_never_uses_a_future_list():
    h = history()
    assert not h.on_list("MU", "2024-03-14")      # first list is dated 3/15


def test_stale_list_stops_counting():
    h = history(max_age=10)
    assert h.on_list("MU", "2024-04-01")          # 10 days after 3/22
    assert not h.on_list("MU", "2024-04-02")      # 11 days: too old


# ─── Portfolio ───
def trades(rows):
    """rows: (entry, exit, trim or None, stop_pct, r_net, setup)."""
    return pd.DataFrame([{"symbol": f"S{i}", "entry_date": e, "exit_date": x, "trim_date": tr,
                          "stop_pct": stop, "r_net": r, "setup": st}
                         for i, (e, x, tr, stop, r, st) in enumerate(rows)])


def test_daily_entry_cap():
    t = trades([("2025-01-02", "2025-01-10", None, WIDE, 1.0, "VCP_BREAKOUT")] * (ACCT.max_new_per_day + 1))
    res = bt.simulate_portfolio(t, ACCT)
    assert res['summary']['taken'] == ACCT.max_new_per_day
    assert res['summary']['skipped (daily cap)'] == 1


def test_open_risk_cap_until_risk_is_freed_by_a_trim():
    # Fill the open-risk cap one entry per day; the last two get trimmed the next day. An entry that
    # day is blocked (trims free risk after entries); one the day after fits (two trades hold no risk).
    fit = int(ACCT.max_open_risk_pct / R + 1e-9)
    assert fit + 1 <= ACCT.max_positions
    days = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2025-01-02", periods=fit + 2)]
    rows = [(days[i], "2025-02-28", days[fit] if i >= fit - 2 else None, WIDE, 0.0, "VCP_BREAKOUT")
            for i in range(fit)]
    rows += [(days[fit], "2025-02-28", None, WIDE, 0.0, "VCP_BREAKOUT"),       # blocked
             (days[fit + 1], "2025-02-28", None, WIDE, 0.0, "VCP_BREAKOUT")]   # allowed
    res = bt.simulate_portfolio(trades(rows), ACCT)
    taken = set(res['taken']['entry_date'].dt.strftime("%Y-%m-%d"))
    assert res['summary']['skipped (open-risk cap)'] == 1
    assert days[fit] not in taken and days[fit + 1] in taken


def test_tight_stop_is_capped_by_max_position():
    # full risk at this stop would need twice the position cap; the cap halves the risk
    stop = WIDE / 4
    res = bt.simulate_portfolio(trades([("2025-01-02", "2025-01-10", None, stop, 1.0, "VCP_BREAKOUT")]), ACCT)
    assert res['taken']['risk_pct'].iloc[0] == pytest.approx(R / 2)
    assert res['summary']['final_equity'] == pytest.approx(100_000 * (1 + R / 2 / 100), abs=1)


def test_pnl_compounds_from_realized_equity():
    rows = [("2025-01-02", "2025-01-03", None, WIDE, 2.0, "VCP_BREAKOUT"),
            ("2025-01-06", "2025-01-07", None, WIDE, -1.0, "VCP_BREAKOUT")]
    res = bt.simulate_portfolio(trades(rows), ACCT)
    after_first = 100_000 * (1 + R / 100 * 2)
    assert res['summary']['final_equity'] == pytest.approx(after_first * (1 - R / 100), abs=1)


def test_vcp_gets_priority_over_pullbacks_on_a_full_day():
    rows = [("2025-01-02", "2025-01-10", None, WIDE / 2, 0.0, "EMA_PULLBACK")] * ACCT.max_new_per_day + \
           [("2025-01-02", "2025-01-10", None, WIDE, 0.0, "VCP_BREAKOUT")]
    res = bt.simulate_portfolio(trades(rows), ACCT)
    assert "VCP_BREAKOUT" in set(res['taken']['setup'])


# ─── Audit fixes ───
def test_lists_are_tracked_separately():
    # A new new_highs file must not hide a still-valid stage2 list
    lists = pd.DataFrame({"date": pd.to_datetime(["2024-03-15", "2024-03-18"]),
                          "symbol": ["SNDK", "MU"], "list": ["stage2", "new_highs"]})
    h = bt.UniverseHistory(lists)
    assert h.on_list("SNDK", "2024-03-19")
    assert h.on_list("MU", "2024-03-19")


def test_dates_with_times_and_timezones_normalize_to_the_day(tmp_path):
    f = tmp_path / "scan_list.csv"
    pd.DataFrame({"date": ["2024-01-05 16:00", "2024-01-05T21:00:00Z", "garbage"],
                  "symbol": ["NVDA", "AMD", "TSLA"]}).to_csv(f, index=False)
    lists = bt.load_universe_lists(str(f))
    assert set(lists['date']) == {pd.Timestamp("2024-01-05")}
    assert "TSLA" not in set(lists['symbol'])


def test_backtest_output_files_are_ignored(tmp_path):
    pd.DataFrame({"Symbol": ["MU"]}).to_csv(tmp_path / "stage2_2024-03-15.csv", index=False)
    pd.DataFrame({"Symbol": ["WINNER"]}).to_csv(tmp_path / "backtest_trades_2024-03-15.csv", index=False)
    assert set(bt.load_universe_lists(str(tmp_path))['symbol']) == {"MU"}


def test_gross_exposure_cap_blocks_piling_up_trimmed_positions():
    # Positions at half the cap; trimmed ones hold no risk but still use cash.
    rows = []
    for day in pd.bdate_range("2025-01-02", periods=20):
        d = day.strftime("%Y-%m-%d")
        rows += [(d, "2025-06-30", d, WIDE, 0.0, "VCP_BREAKOUT")] * 2  # trimmed same day
    res = bt.simulate_portfolio(trades(rows), replace(ACCT, max_positions=100))
    assert res['summary']['peak_gross_exposure_%'] <= 100.0 + 1e-6
    assert res['summary']['skipped (gross exposure cap)'] > 0


def test_max_positions_cap():
    rows = []
    for day in pd.bdate_range("2025-01-02", periods=ACCT.max_positions):
        d = day.strftime("%Y-%m-%d")
        rows += [(d, "2025-06-30", d, WIDE * 4, 0.0, "VCP_BREAKOUT")] * 2  # small positions, no risk after trim
    res = bt.simulate_portfolio(trades(rows), ACCT)
    assert res['curve']['open_positions'].max() == ACCT.max_positions
    assert res['summary']['skipped (position cap)'] > 0


def test_mark_to_market_drawdown_sees_open_losses():
    t = trades([("2025-01-02", "2025-01-08", None, 5, 2.0, "VCP_BREAKOUT")])
    t['entry'], t['stop'] = 100.0, 95.0
    days = pd.bdate_range("2025-01-02", "2025-01-08")
    closes = {"S0": pd.Series([100, 90, 97, 105, 110], index=days)}
    realized = bt.simulate_portfolio(t, ACCT)['summary']
    mtm = bt.simulate_portfolio(t, ACCT, closes=closes, calendar=days)['summary']
    assert realized['max_drawdown_%'] == 0
    # down 2R at the 90 close (5% stop: full risk, under the cap)
    assert mtm['max_drawdown_%'] == pytest.approx(2 * R, abs=0.05)


def test_cagr_needs_at_least_a_year_and_uses_full_calendar():
    t = trades([("2025-01-02", "2025-01-03", None, 5, 1.0, "VCP_BREAKOUT")])
    assert bt.simulate_portfolio(t, ACCT)['summary']['cagr_%'] is None
    cal = pd.bdate_range("2025-01-02", "2027-01-04")
    s = bt.simulate_portfolio(t, ACCT, closes={}, calendar=cal)['summary']
    assert s['cagr_%'] == pytest.approx(((1 + R / 100) ** (1 / 2.0) - 1) * 100, abs=0.05)


def test_account_never_holds_one_symbol_twice():
    t = trades([("2025-01-02", "2025-01-10", None, 5, 1.0, "EMA_PULLBACK"),
                ("2025-01-03", "2025-01-12", None, 5, 1.0, "VCP_BREAKOUT")])
    t['symbol'] = "MU"
    res = bt.simulate_portfolio(t, ACCT)
    assert res['summary']['taken'] == 1 and res['summary']['skipped (already held)'] == 1


def test_yearly_files_without_a_list_column_are_named_by_file(tmp_path):
    for name, sym in (("stage2_2025.csv", "MU"), ("newhighs_2025.csv", "BRK-B")):
        pd.DataFrame({"date": ["2025-08-12"], "ticker": [sym]}).to_csv(tmp_path / name, index=False)
    lists = bt.load_universe_lists(str(tmp_path))
    assert set(lists['list']) == {"stage2", "newhighs"}
    assert "BRK B" in set(lists['symbol'])
