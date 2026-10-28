"""Execution decisions: stops, regime blocks, and size multipliers."""

import pandas as pd
import pytest

import execution_engine as ee
import strategy_params as sp

P = ee.ExecutionPolicy()
RISK, CAP = P.risk_per_trade_pct, P.max_position_pct
CAP_STOP = RISK / CAP * 100                          # full risk at this stop exactly fills the position cap
OPEN_STOP = (CAP_STOP + P.max_stop_pct) / 2          # inside the tight band, cap not binding
CAPPED_STOP = (P.min_stop_pct + CAP_STOP) / 2        # inside the band, cap binds
WIDE_STOP = (P.max_stop_pct + P.max_stop_pct_pivot) / 2   # pivot band only
MIN_HIST = sp.MIN_BREAKER_HISTORY_SESSIONS


def decide(**row):
    base = {"Symbol": "TEST", "Tier": "TIER_BUY", "Stop_Loss_%": 2.0}
    base.update(row)
    out = ee.build_execution_decisions(pd.DataFrame([base]), mode="afterhours", orb_window=10)
    return out.iloc[0]


def test_buy_with_valid_stop():
    assert decide()["Decision"] == ee.BUY


def test_missing_stop_is_skipped():
    row = decide(**{"Stop_Loss_%": None})
    assert row["Decision"] == ee.SKIP
    assert "no valid stop" in row["Reason"]


def test_regime_blocked_row_is_skipped():
    row = decide(Regime_Blocked=True)
    assert row["Decision"] == ee.SKIP
    assert "market regime" in row["Reason"]


def test_rs_leader_trades_at_half_size():
    policy = ee.ExecutionPolicy()
    row = decide(Tier="TIER_RS_LEADER", Size_Multiplier=0.5, **{"Stop_Loss_%": OPEN_STOP})
    assert row["Decision"] == ee.BUY
    assert row["Risk_Pct_Target"] == pytest.approx(policy.risk_per_trade_pct * 0.5)
    assert row["Position_Cap_Pct"] == pytest.approx(policy.max_position_pct * 0.5)


def test_size_multiplier_never_exceeds_full_size():
    policy = ee.ExecutionPolicy()
    assert decide(Size_Multiplier=3.0, **{"Stop_Loss_%": OPEN_STOP})["Risk_Pct_Target"] == pytest.approx(policy.risk_per_trade_pct)


def test_watch_tier_is_not_tradeable():
    assert decide(Tier="TIER_WATCH")["Decision"] == ee.SKIP


def test_blank_tier_is_not_tradeable():
    assert decide(Tier="")["Decision"] == ee.SKIP


def test_zero_size_multiplier_means_no_trade():
    assert decide(Size_Multiplier=0)["Decision"] == ee.SKIP


# ─── Per-setup stop bands ───
def test_vcp_entry_auto_trades_a_stop_wider_than_the_tight_band():
    row = decide(**{"Setup_Type": "VCP", "Entry_Setup": "VCP_PIVOT", "Stop_Loss_%": WIDE_STOP})
    assert row["Entry_Type"] == "PIVOT_BREAKOUT"
    assert row["Decision"] == ee.BUY


def test_vcp_stop_over_the_pivot_band_needs_review():
    stop = P.max_stop_pct_pivot + 1
    assert decide(**{"Entry_Setup": "VCP_PIVOT", "Stop_Loss_%": stop})["Decision"] == ee.MANUAL_REVIEW


def test_rs_leader_vcp_uses_the_pivot_band():
    row = decide(**{"Tier": "TIER_RS_LEADER", "Setup_Type": "VCP", "Entry_Setup": "VCP_PIVOT",
                    "Size_Multiplier": 0.5, "Stop_Loss_%": WIDE_STOP})
    assert row["Decision"] == ee.BUY


def test_elite_ema_entry_is_a_pullback_not_a_pivot():
    """Elites leave Setup_Type blank; their entry (10EMA) must still be read as a pullback."""
    row = decide(**{"Tier": "TIER_ELITE_BUY", "Setup_Type": "", "Entry_Setup": "10EMA",
                    "Stop_Loss_%": P.max_stop_pct + 0.5})
    assert row["Entry_Type"] == "PULLBACK_BOUNCE"
    assert row["Decision"] == ee.MANUAL_REVIEW  # over the pullback band


def test_pullback_inside_band_on_a_bounce_day_trades_only_when_policy_allows():
    base = {"Symbol": "TEST", "Tier": "TIER_BUY", "Setup_Type": "10EMA", "Stop_Loss_%": 2.0, "Bounce_Day": True}
    allowed = ee.build_execution_decisions(pd.DataFrame([base]), mode="afterhours", orb_window=10,
                                           policy=ee.ExecutionPolicy(auto_trade_pullbacks=True)).iloc[0]
    assert allowed["Decision"] == ee.BUY


def test_pullbacks_stay_manual_by_default_even_with_a_bounce_flag():
    """Audit #9: the owner's 'pullbacks are manual' decision must not hinge on a missing column."""
    for setup in ("10EMA", "21EMA", "50MA", "pullback"):
        row = decide(**{"Setup_Type": setup, "Stop_Loss_%": 2.0, "Bounce_Day": True})
        assert row["Decision"] == ee.MANUAL_REVIEW, setup
        assert "auto_trade_pullbacks" in row["Reason"]


def test_unknown_setup_gets_the_tight_band():
    row = decide(**{"Setup_Type": "", "Stop_Loss_%": WIDE_STOP})
    assert row["Entry_Type"] == "UNKNOWN"
    assert row["Decision"] == ee.MANUAL_REVIEW


@pytest.mark.parametrize("labels", [
    {"Setup_Type": "VCP"},                           # bare VCP label: not proven to be at a pivot
    {"Setup_Type": "VCP_LOOSE"},                     # loose base, no contraction required
    {"Setup_Type": "VCP", "Entry_Setup": "BASE"},    # scanner says: VCP-ish, but not near the pivot
])
def test_only_an_exact_vcp_pivot_gets_the_wide_band(labels):
    row = decide(**{**labels, "Stop_Loss_%": WIDE_STOP})
    assert row["Entry_Type"] == "UNKNOWN"
    assert row["Decision"] == ee.MANUAL_REVIEW


# ─── Group risk cap (per IBD industry group) ───
def decide_many(rows):
    return ee.build_execution_decisions(pd.DataFrame(rows), mode="afterhours", orb_window=10)


def test_name_past_the_group_cap_needs_permission():
    policy = ee.ExecutionPolicy(max_new_entries_per_day=10)  # isolate the group cap from the daily cap
    fit = int(P.max_group_open_risk_pct / RISK + 1e-9)        # full-risk names that fit in one group
    rows = [{"Symbol": f"S{i}", "Tier": "TIER_BUY", "Stop_Loss_%": OPEN_STOP, "IBD_Group_Rank": 3,
             "Priority_Rating": 90 - i} for i in range(fit + 1)]
    out = ee.build_execution_decisions(pd.DataFrame(rows), mode="afterhours", orb_window=10, policy=policy)
    by = out.set_index("Symbol")["Decision"]
    assert list(by.iloc[:fit]) == [ee.BUY] * fit
    assert by[f"S{fit}"] == ee.MANUAL_REVIEW                 # one more is over the group cap: owner decides
    assert "Group risk cap" in out.set_index("Symbol").at[f"S{fit}", "Reason"]


def test_existing_group_risk_counts():
    held = P.max_group_open_risk_pct - RISK / 2
    out = decide_many([{"Symbol": "NEW", "Tier": "TIER_BUY", "Stop_Loss_%": OPEN_STOP,
                        "IBD_Group_Rank": 7, "Current_Group_Open_Risk_Pct": held}])
    assert out.iloc[0]["Decision"] == ee.MANUAL_REVIEW  # held + one full risk > group cap


def test_half_size_rs_leader_uses_less_group_budget():
    policy = ee.ExecutionPolicy(max_new_entries_per_day=10)
    fit_full = int(P.max_group_open_risk_pct / RISK + 1e-9)
    fit_half = int(P.max_group_open_risk_pct / (RISK * 0.5) + 1e-9)
    rows = [{"Symbol": f"S{i}", "Tier": "TIER_RS_LEADER", "Entry_Setup": "VCP_PIVOT", "Size_Multiplier": 0.5,
             "Stop_Loss_%": WIDE_STOP, "IBD_Group_Rank": 5, "Priority_Rating": 100 - i} for i in range(fit_half + 1)]
    out = ee.build_execution_decisions(pd.DataFrame(rows), mode="afterhours", orb_window=10, policy=policy)
    assert (out["Decision"] == ee.BUY).sum() == fit_half > fit_full


def test_capped_name_does_not_use_a_daily_slot():
    # A's group is already full; every daily slot must still go to the other names
    rows = [{"Symbol": "A", "Tier": "TIER_BUY", "Stop_Loss_%": 2.0, "IBD_Group_Rank": 7,
             "Current_Group_Open_Risk_Pct": P.max_group_open_risk_pct, "Priority_Rating": 99}]
    rows += [{"Symbol": f"B{i}", "Tier": "TIER_BUY", "Stop_Loss_%": 2.0, "IBD_Group_Rank": 8 + i,
              "Priority_Rating": 90 - i} for i in range(P.max_new_entries_per_day)]
    by = decide_many(rows).set_index("Symbol")["Decision"]
    assert by["A"] == ee.MANUAL_REVIEW and (by.drop("A") == ee.BUY).all()


def test_no_group_means_no_cap():
    policy = ee.ExecutionPolicy(max_new_entries_per_day=10)
    rows = [{"Symbol": f"N{i}", "Tier": "TIER_BUY", "Stop_Loss_%": 2.0} for i in range(5)]
    out = ee.build_execution_decisions(pd.DataFrame(rows), mode="afterhours", orb_window=10, policy=policy)
    assert (out["Decision"] == ee.BUY).all()


def test_premarket_plan_is_valid_before_the_open_only(monkeypatch):
    from datetime import datetime

    def at(hh, mm, day=23):  # 2026-10-23 is a Friday; times are US/Eastern
        monkeypatch.setattr(ee, "now_et", lambda: datetime(2026, 10, day, hh, mm))
        df = pd.DataFrame([{"Symbol": "TEST", "Tier": "TIER_BUY", "Stop_Loss_%": 2.0}])
        return ee.build_execution_decisions(df, mode="premarket", orb_window=10).iloc[0]["Decision"]

    assert at(9, 20) == ee.BUY
    assert at(10, 0) == ee.SKIP
    assert at(9, 20, day=24) == ee.SKIP   # Saturday


def test_rth_mode_is_closed_on_weekends_and_holidays():
    from datetime import datetime
    assert ee._is_rth_open(datetime(2026, 10, 23, 11, 0))
    assert not ee._is_rth_open(datetime(2026, 10, 24, 11, 0))    # Saturday
    assert not ee._is_rth_open(datetime(2026, 11, 26, 11, 0))    # Thanksgiving


def test_stale_scan_is_not_traded():
    from datetime import datetime, timedelta
    old = (datetime.now() - timedelta(hours=30)).strftime("%Y-%m-%d %H:%M:%S")
    fresh = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    assert decide(Scan_Timestamp=old)["Decision"] == ee.SKIP
    assert "old" in decide(Scan_Timestamp=old)["Reason"]
    assert decide(Scan_Timestamp=fresh)["Decision"] == ee.BUY


def test_unchecked_rules_are_named_in_the_reason():
    row = decide()
    assert row["Decision"] == ee.BUY
    assert "Not checked (no data): earnings blackout, re-entry/stop-out cooldown" in row["Reason"]
    assert "earnings" not in decide(Earnings_Date="2099-01-01", Recent_Stop_Outs=0)["Reason"]


def test_decision_states_the_adopted_exit_with_the_stocks_trail_line():
    row = decide(Trail_MA="ema5")
    assert f"+{sp.TRIM_R:g}R" in row["Exit_Trim_Rule"] and "5-day EMA" in row["Exit_Trail_Rule"]
    assert decide()["Trail_MA"] == "ema21"


def test_risk_counted_is_what_the_capped_position_carries():
    """One sizing rule: a tight stop hits the position cap, so the trade risks less than the budget."""
    row = decide(**{"Stop_Loss_%": CAPPED_STOP})
    assert row["Position_Pct"] == pytest.approx(CAP)
    assert row["Risk_Pct_Target"] == pytest.approx(CAP * CAPPED_STOP / 100) and row["Risk_Pct_Target"] < RISK


@pytest.mark.parametrize("stop", [-2.0, 0.0])
def test_stop_at_or_above_price_is_no_stop(stop):
    """ES-12: a negative Stop_Loss_% must not become a valid stop via abs()."""
    row = decide(**{"Stop_Loss_%": stop})
    assert row["Decision"] == ee.SKIP and "no valid stop" in row["Reason"]


# ─── Owner decision 2026-10-26 (c): unlabeled setups with a pullback stop are manual review ───
@pytest.mark.parametrize("ctx", ["21EMA_PULLBACK", "10EMA_PULLBACK", "50MA_BOUNCE"])
def test_blank_setup_with_pullback_stop_is_manual_review(ctx):
    row = decide(Stop_Context=ctx)
    assert row["Entry_Type"] == "UNKNOWN"
    assert row["Decision"] == ee.MANUAL_REVIEW and row["Decision"] != ee.BUY
    assert f"Unlabeled setup with a pullback stop ({ctx})" in row["Reason"]
    assert row["Stop_Context"] == ctx


def test_base_label_with_pullback_stop_is_manual_review():
    row = decide(Setup_Type="VCP", Entry_Setup="BASE", Stop_Context="21EMA_PULLBACK")
    assert row["Entry_Type"] == "UNKNOWN" and row["Decision"] == ee.MANUAL_REVIEW


def test_vcp_pivot_with_pullback_context_is_unchanged():
    row = decide(Setup_Type="VCP", Entry_Setup="VCP_PIVOT", Stop_Context="21EMA_PULLBACK",
                 **{"Stop_Loss_%": WIDE_STOP})
    assert row["Entry_Type"] == "PIVOT_BREAKOUT" and row["Decision"] == ee.BUY


def test_blank_setup_with_breakout_stop_context_still_buys():
    assert decide(Stop_Context="DEFAULT")["Decision"] == ee.BUY


def test_skip_from_another_rule_wins_over_pullback_review():
    row = decide(Stop_Context="21EMA_PULLBACK", Tier="TIER_WATCH")
    assert row["Decision"] == ee.SKIP


# ─── Owner decision 2026-10-26 (b): breaker history flag (decision column unchanged) ───
def test_breaker_history_short_is_flagged_not_decided():
    row = decide(Account_Breaker_History_Sessions=MIN_HIST - 1)
    assert row["Decision"] == ee.BUY              # the engine never knows if this is a submit run
    assert not row["Breaker_History_OK"]
    assert row["Breaker_History_Sessions"] == MIN_HIST - 1
    assert (f"Circuit breakers have {MIN_HIST - 1} of {MIN_HIST} sessions since the last reset: "
            "auto entries halted") in row["Reason"]


def test_breaker_history_unknown_is_not_ok():
    row = decide()
    assert not row["Breaker_History_OK"] and pd.isna(row["Breaker_History_Sessions"])
    assert f"Circuit breakers have unknown of {MIN_HIST} sessions" in row["Reason"]


def test_breaker_history_enough_is_ok():
    row = decide(Account_Breaker_History_Sessions=MIN_HIST)
    assert row["Breaker_History_OK"] and "auto entries halted" not in row["Reason"]


def test_snapshot_breaker_history_reaches_the_row():
    snap = {"account": "DUX1", "total_open_risk_pct": 0.0, "group_open_risk_pct": {}, "held_symbols": [],
            "unprotected_symbols": [], "breaker_history_sessions": MIN_HIST + 1}
    df = ee.apply_account_snapshot(pd.DataFrame([{"Symbol": "A", "Tier": "TIER_BUY", "Stop_Loss_%": 3.0}]), snap)
    out = ee.build_execution_decisions(df, mode="afterhours", orb_window=10).iloc[0]
    assert out["Breaker_History_OK"] and out["Breaker_History_Sessions"] == MIN_HIST + 1
    old = {k: v for k, v in snap.items() if k != "breaker_history_sessions"}   # older snapshot file
    df = ee.apply_account_snapshot(pd.DataFrame([{"Symbol": "A", "Tier": "TIER_BUY", "Stop_Loss_%": 3.0}]), old)
    assert not ee.build_execution_decisions(df, mode="afterhours", orb_window=10).iloc[0]["Breaker_History_OK"]


# ─── New bracket inputs: Stop_Price and Risk_Pct_Budget ───
def test_stop_price_from_scanner_or_from_stop_pct():
    assert decide(Price=100.0, Stop_Loss_Price=96.5, **{"Stop_Loss_%": 3.5})["Stop_Price"] == pytest.approx(96.5)
    assert decide(Price=100.0, **{"Stop_Loss_%": 3.0})["Stop_Price"] == pytest.approx(97.0)
    assert pd.isna(decide(Price=100.0, **{"Stop_Loss_%": None})["Stop_Price"])


def test_risk_budget_is_before_the_position_cap():
    row = decide(**{"Stop_Loss_%": CAPPED_STOP})  # cap binds: actual risk below the budget
    assert row["Risk_Pct_Budget"] == pytest.approx(RISK)
    assert row["Risk_Pct_Target"] == pytest.approx(CAP * CAPPED_STOP / 100)
    half = decide(Size_Multiplier=0.5, **{"Stop_Loss_%": OPEN_STOP})
    assert half["Risk_Pct_Budget"] == pytest.approx(RISK * 0.5)


def test_risk_budget_includes_the_weekly_breaker_cut():
    row = decide(Account_Weekly_PnL_Pct=-sp.WEEKLY_LOSS_LIMIT_PCT - 0.5, **{"Stop_Loss_%": OPEN_STOP})
    assert row["Risk_Pct_Budget"] == pytest.approx(RISK * sp.WEEKLY_LOSS_RISK_MULT)
