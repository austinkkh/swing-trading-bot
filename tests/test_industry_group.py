"""Industry_Group comes from Data/fundamentals/industry.csv (the IBD export has only the group rank).
Display / reporting only: tiers, priority, sizing and engine decisions must not change."""

from datetime import datetime
from types import SimpleNamespace

import pandas as pd
import pytest

import execution_engine as ee
import momentum_scanner as ms
from helpers import make_stock


@pytest.fixture
def files(tmp_path, monkeypatch):
    ind = tmp_path / "industry.csv"
    pd.DataFrame({"symbol": ["VCPX", "MANU", "NOGRP"], "industry": ["Technology"] * 3,
                  "category": ["Computers"] * 3,
                  "subcategory": ["Computers-Memory Devices", "Medical Information Sys", None],
                  "long_name": ["x", "y", "z"]}).to_csv(ind, index=False)
    monkeypatch.setattr(ms, "INDUSTRY_CSV", ind)
    monkeypatch.setattr(ms, "THEME_DASHBOARD_CSV", tmp_path / "no_dashboard.csv")
    monkeypatch.setattr(ms, "OUTPUT_DIR", str(tmp_path / "scans"))
    (tmp_path / "scans").mkdir()
    return tmp_path


def export(stocks, out_dir):
    scanner = ms.MomentumScanner()
    scanner.scan_timestamp = datetime.now()
    scanner._apply_tier_filters(stocks)
    scanner._calculate_priority_ratings_for_all_tiers()
    scanner._export_results("afterhours")
    return pd.read_csv(next(out_dir.glob("*_COMPLETE.csv"))).set_index("Symbol"), out_dir


def vcp(symbol, **kw):
    return make_stock(symbol, setup_type="VCP", has_real_vcp=True, vcp_score=8.0, vcp_num_contractions=3,
                      vcp_pivot_distance_pct=-1.0, ibd_industry_rank=12, trail_ma="sma10", **kw)


def test_industry_group_filled_from_industry_csv(files):
    df, _ = export([vcp("VCPX")], files / "scans")
    assert df.loc["VCPX", "Industry_Group"] == "Computers-Memory Devices"


def test_metadata_name_wins_over_csv(files):
    scanner = ms.MomentumScanner()
    scanner.universe_obj = SimpleNamespace(metadata={"VCPX": {"Industry_Group": "From Metadata"}})
    assert scanner._normalize_industry_from_meta("VCPX") == "From Metadata"


def test_symbol_without_universe_metadata_still_gets_a_name(files):
    scanner = ms.MomentumScanner()
    scanner.universe_obj = SimpleNamespace(metadata={})          # manual-watchlist symbol
    assert scanner._normalize_industry_from_meta("MANU") == "Medical Information Sys"
    assert scanner._normalize_industry_from_meta("NOGRP") == ""   # subcategory missing


def test_missing_industry_csv_is_blank_not_a_crash(files, monkeypatch, caplog):
    import logging
    monkeypatch.setattr(ms, "INDUSTRY_CSV", files / "nope.csv")
    with caplog.at_level(logging.WARNING):
        df, _ = export([vcp("VCPX"), vcp("VCPY"), vcp("VCPZ")], files / "scans")
    assert pd.isna(df.loc["VCPX", "Industry_Group"])
    # Review M4: one warning per scan, not two per exported stock
    assert sum("Industry_Group will be blank" in r.getMessage() for r in caplog.records) == 1


def test_theme_columns_are_appended_and_informational(files, monkeypatch):
    base, _ = export([vcp("VCPX")], files / "scans")
    for f in (files / "scans").glob("*"):
        f.unlink()
    dash = files / "dash.csv"
    pd.DataFrame({"as_of_date": ["2026-10-23"], "rank": [2], "group": ["Computers-Memory Devices"],
                  "emerging": [True]}).to_csv(dash, index=False)
    monkeypatch.setattr(ms, "THEME_DASHBOARD_CSV", dash)
    df, _ = export([vcp("VCPX")], files / "scans")
    assert list(df.columns[-3:]) == ["Theme_Group_Rank", "Theme_Emerging", "Theme_As_Of"]
    row = df.loc["VCPX"]
    assert row["Theme_Group_Rank"] == 2 and bool(row["Theme_Emerging"]) and row["Theme_As_Of"] == "2026-10-23"
    for col in ("Tier", "Priority_Rating", "Size_Multiplier"):
        assert row[col] == base.loc["VCPX", col]


def test_sector_summary_has_real_groups_and_keeps_unknowns(files):
    _, out = export([vcp("VCPX"), vcp("NOGRP"), vcp("MANU")], files / "scans")
    summary = pd.read_csv(next(out.glob("*_SECTOR_SUMMARY.csv")))
    assert set(summary["Industry_Group"]) == {"Computers-Memory Devices", "Medical Information Sys", "Unknown"}
    assert summary["symbol_count"].sum() == 3


def test_engine_sector_passthrough_and_decision_unchanged(files):
    df, _ = export([vcp("VCPX")], files / "scans")
    df = df.reset_index()
    out = ee.build_execution_decisions(df, mode="afterhours", orb_window=10).set_index("Symbol").loc["VCPX"]
    blank = ee.build_execution_decisions(df.assign(Industry_Group=""), mode="afterhours",
                                         orb_window=10).set_index("Symbol").loc["VCPX"]
    assert out["Sector"] == "Computers-Memory Devices"
    assert out["Decision"] == blank["Decision"] and out["Reason"] == blank["Reason"]
