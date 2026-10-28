"""Paper executor: risk-based sizing metadata (no TWS)."""

import pandas as pd
import pytest

import ibkr_paper_executor as ex
import strategy_params as sp

RISK, CAP = sp.RISK_PER_TRADE_PCT, sp.MAX_POSITION_PCT
WIDE = RISK / CAP * 100 * 2           # risk-sized at half the cap


def intent(**row):
    base = {"Symbol": "MU", "Decision": "BUY", "Entry_Price": 100.0, "Stop_Pct": WIDE,
            "Risk_Pct_Target": RISK, "Position_Cap_Pct": CAP}
    base.update(row)
    out, _ = ex._build_buy_intents(pd.DataFrame([base]))
    return out.iloc[0]


def test_wide_stop_is_sized_by_risk_not_the_position_cap():
    r = intent()
    assert r["Sizing_Status"] == "RISK_SIZED"
    assert r["Position_Pct_Of_Equity"] == pytest.approx(RISK / WIDE * 100)


def test_tight_stop_is_capped_by_max_position():
    assert intent(Stop_Pct=WIDE / 4)["Position_Pct_Of_Equity"] == pytest.approx(CAP)   # uncapped: twice the cap


def test_half_size_rs_leader():
    half = intent(Risk_Pct_Target=RISK / 2, Position_Cap_Pct=CAP / 2)
    assert half["Position_Pct_Of_Equity"] == pytest.approx(RISK / 2 / WIDE * 100)
