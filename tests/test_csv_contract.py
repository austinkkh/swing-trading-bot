"""Scanner CSV -> execution engine contract (review TS-09): the real export, not a hand-built frame.

The engine is tolerant of missing columns (unknown setup = tight band, no Trail_MA = ema21,
no Scan_Timestamp = not stale), so a renamed export column would silently degrade decisions.
"""

from datetime import datetime

import pandas as pd
import pytest

import execution_engine as ee
import momentum_scanner as ms
from helpers import make_stock

ENGINE_COLUMNS = ["Symbol", "Tier", "Stop_Loss_%", "Price", "Entry_Setup", "Trail_MA", "Scan_Timestamp",
                  "Regime_Blocked", "Size_Multiplier", "IBD_Group_Rank", "Dollar_Vol_Month_M",
                  "Has_Real_VCP", "VCP_Pivot", "VCP_Dist_From_Pivot_%", "Priority_Rating"]


@pytest.fixture
def exported(tmp_path, monkeypatch):
    monkeypatch.setattr(ms, "OUTPUT_DIR", str(tmp_path))
    scanner = ms.MomentumScanner()
    scanner.scan_timestamp = datetime.now()
    stock = make_stock("VCPX", setup_type="VCP", has_real_vcp=True, vcp_score=8.0, vcp_num_contractions=3,
                       vcp_pivot_distance_pct=-1.0, ibd_industry_rank=12, trail_ma="sma10")
    scanner._apply_tier_filters([stock])
    scanner._calculate_priority_ratings_for_all_tiers()
    scanner._export_results("afterhours")
    files = list(tmp_path.glob("*_COMPLETE.csv"))
    assert len(files) == 1
    return pd.read_csv(files[0])


def test_export_has_every_column_the_engine_reads(exported):
    missing = [c for c in ENGINE_COLUMNS if c not in exported.columns]
    assert not missing, f"scanner export lost engine columns: {missing}"


def test_exported_vcp_buy_flows_through_the_engine(exported):
    row = exported.set_index("Symbol").loc["VCPX"]
    assert row["Tier"] in ("TIER_BUY", "TIER_ELITE_BUY")
    out = ee.build_execution_decisions(exported, mode="afterhours", orb_window=10).set_index("Symbol").loc["VCPX"]
    assert out["Entry_Type"] == "PIVOT_BREAKOUT"
    assert out["Decision"] == ee.BUY, out["Reason"]
    assert out["Stop_Pct"] == pytest.approx(row["Stop_Loss_%"])
