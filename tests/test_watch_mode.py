"""Watcher watchlist: which tiers it monitors for ORB breakouts and migrations."""

import pandas as pd

import watch_mode


def test_watchlist_includes_rs_leaders(tmp_path, monkeypatch):
    pd.DataFrame({
        "Symbol": ["ELB", "ELH", "BUY", "RSL", "WCH"],
        "Tier": ["TIER_ELITE_BUY", "TIER_ELITE_HOLD", "TIER_BUY", "TIER_RS_LEADER", "TIER_WATCH"],
    }).to_csv(tmp_path / "scan_2026-10-23_0920_afterhours_COMPLETE.csv", index=False)
    monkeypatch.setattr(watch_mode, "OUTPUT_DIR", str(tmp_path))

    monitor = watch_mode.WatchModeMonitor()
    assert monitor.load_latest_elite_hold()
    assert monitor.watchlist_symbols == ["BUY", "ELB", "ELH", "RSL"]


def test_watcher_ignores_its_own_intraday_files(tmp_path, monkeypatch):
    pd.DataFrame({"Symbol": ["MU", "WDC"], "Tier": ["TIER_BUY", "TIER_BUY"]}).to_csv(
        tmp_path / "scan_2026-10-23_0920_afterhours_COMPLETE.csv", index=False)
    pd.DataFrame({"Symbol": ["MU"], "Tier": ["TIER_BUY"]}).to_csv(
        tmp_path / "scan_2026-10-23_1530_watch_COMPLETE.csv", index=False)
    monkeypatch.setattr(watch_mode, "OUTPUT_DIR", str(tmp_path))
    monitor = watch_mode.WatchModeMonitor()
    assert monitor.load_latest_elite_hold()
    assert monitor.watchlist_symbols == ["MU", "WDC"]


def test_no_scanning_on_weekends_or_holidays(monkeypatch):
    from datetime import datetime
    monitor = watch_mode.WatchModeMonitor()
    monkeypatch.setattr(watch_mode, "now_et", lambda: datetime(2026, 10, 24, 11, 0))   # Saturday
    assert monitor.get_current_phase() == ("MARKET_CLOSED", 1800, False)
    monkeypatch.setattr(watch_mode, "now_et", lambda: datetime(2026, 10, 23, 11, 0))   # Friday
    assert monitor.get_current_phase()[2] is True


def test_watcher_reloads_when_a_newer_scan_lands(tmp_path, monkeypatch):
    """LT-01: the 9:20 scan lands after the watcher started; it used to wait until midnight."""
    monkeypatch.setattr(watch_mode, "OUTPUT_DIR", str(tmp_path))
    pd.DataFrame({"Symbol": ["OLD"], "Tier": ["TIER_BUY"]}).to_csv(
        tmp_path / "scan_2026-10-23_0920_afterhours_COMPLETE.csv", index=False)
    monitor = watch_mode.WatchModeMonitor()
    assert monitor.load_latest_elite_hold()
    assert monitor.reload_if_new_scan() is False          # nothing new
    pd.DataFrame({"Symbol": ["NEW"], "Tier": ["TIER_ELITE_HOLD"]}).to_csv(
        tmp_path / "scan_2026-10-26_0920_afterhours_COMPLETE.csv", index=False)
    assert monitor.reload_if_new_scan() is True
    assert monitor.watchlist_symbols == ["NEW"]
    assert monitor.previous_state == {"NEW": "TIER_ELITE_HOLD"}   # yesterday's states cleared


class _Regime:
    status, should_take_new_positions = "CONFIRMED_UPTREND", True

    def __init__(self, ts):
        self.timestamp = ts


def test_watch_scans_write_no_files_and_pin_the_regime(tmp_path, monkeypatch):
    """LT-02: watch rescans skip CSV export/dashboard and reuse today's regime."""
    from datetime import datetime
    import momentum_scanner as ms
    scanner = ms.MomentumScanner()
    scanner.market_regime = _Regime(datetime.now())
    calls = []
    monkeypatch.setattr(scanner, "_set_regime_gating", lambda: calls.append("regime"))
    monkeypatch.setattr(scanner, "get_stock_universe", lambda: ["MU"])
    monkeypatch.setattr(scanner, "_parallel_filter_stocks", lambda symbols: [{"symbol": "MU"}])
    for name in ("_calculate_rs_for_stocks", "_detect_vcp_for_stocks", "_analyze_price_action_for_stocks"):
        monkeypatch.setattr(scanner, name, lambda stocks: stocks)
    for name in ("_apply_ibd_universe", "_apply_tier_filters"):
        monkeypatch.setattr(scanner, name, lambda stocks: None)
    for name in ("_calculate_priority_ratings_for_all_tiers", "_apply_stop_filter_to_tiers"):
        monkeypatch.setattr(scanner, name, lambda: None)
    monkeypatch.setattr(scanner, "_export_results", lambda t: calls.append("export"))
    monkeypatch.setattr(scanner, "_render_results", lambda t: calls.append("render"))
    assert scanner.run_full_scan(scan_type="watch") is True
    assert calls == []
    scanner.market_regime = _Regime(datetime(2026, 1, 2))      # yesterday's regime: re-detect
    scanner.run_full_scan(scan_type="watch")
    assert calls == ["regime"]
