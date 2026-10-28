#!/usr/bin/env python3
"""
Execution decision engine for scanner exports.

Consumes scanner CSV output (COMPLETE or BUY_LIST) and emits explicit
per-symbol decisions: BUY / SKIP / MANUAL_REVIEW.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime, time as dt_time
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import pandas as pd

import exit_rules
import risk_manager
import strategy_params as sp
from market_time import is_trading_day, now_et
from orb_detector import ORB_OPENING_TIME


RTH_CLOSE_TIME = dt_time(16, 0)
BUY = "BUY"
SKIP = "SKIP"
MANUAL_REVIEW = "MANUAL_REVIEW"


@dataclass(frozen=True)
class ExecutionPolicy:
    # Defaults are example values from strategy_params.py
    risk_per_trade_pct: float = sp.RISK_PER_TRADE_PCT
    max_position_pct: float = sp.MAX_POSITION_PCT
    max_new_entries_per_day: int = sp.MAX_NEW_ENTRIES_PER_DAY
    max_sector_concentration_pct: float = sp.MAX_SECTOR_CONCENTRATION_PCT
    max_total_open_risk_pct: float = sp.MAX_TOTAL_OPEN_RISK_PCT
    max_group_open_risk_pct: float = sp.MAX_GROUP_OPEN_RISK_PCT  # per industry group; more needs manual permission
    min_stop_pct: float = sp.MIN_STOP_PCT
    max_stop_pct: float = sp.MAX_STOP_PCT             # ORB, pullback and unknown entries
    max_stop_pct_pivot: float = sp.MAX_STOP_PCT_PIVOT # VCP/pivot breakouts
    earnings_blackout_days: int = sp.EARNINGS_BLACKOUT_DAYS
    time_stop_days: int = sp.TIME_STOP_SESSIONS
    max_scan_age_hours: float = 18.0    # an older scan doesn't describe today's setups
    max_pct_of_daily_dollar_volume: float = sp.MAX_PCT_OF_DAILY_DOLLAR_VOLUME  # position <= N% of avg daily $ volume
    max_open_positions: int = sp.MAX_OPEN_POSITIONS   # held positions + today's BUYs
    # Design decision: EMA pullbacks stay on manual review.
    # Before this switch the only guard was that the scanner
    # exports no Bounce_Day column; one added column would have made them BUY (audit #9).
    auto_trade_pullbacks: bool = False
    # Design decision: the loss breakers need this many sessions of equity history
    # since the last reset before the bot may submit entries (order_entry halts in submit mode).
    # The engine only flags it (Breaker_History_OK); the Decision column is unchanged.
    min_breaker_history_sessions: int = sp.MIN_BREAKER_HISTORY_SESSIONS
    # Design decision: a row with no recognised setup label but a pullback-style stop
    # (scanner Stop_Context) is a pullback in disguise -> MANUAL_REVIEW, never an auto BUY.
    pullback_stop_contexts: Tuple[str, ...] = ("10EMA_PULLBACK", "21EMA_PULLBACK", "50MA_BOUNCE")
    circuit_breakers: risk_manager.CircuitBreakers = risk_manager.CircuitBreakers()


def _to_float(value) -> Optional[float]:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value) -> Optional[int]:
    num = _to_float(value)
    if num is None:
        return None
    return int(num)


def _to_bool(value) -> Optional[bool]:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n"}:
        return False
    return None


def _first_present(row: pd.Series, candidates: Iterable[str]):
    for col in candidates:
        if col in row.index:
            value = row[col]
            if value is not None and not pd.isna(value):
                return value
    return None


def _parse_earnings_date(row: pd.Series) -> Optional[date]:
    earnings_value = _first_present(
        row,
        (
            "Earnings_Date",
            "EarningsDate",
            "Next_Earnings_Date",
            "NextEarningsDate",
            "Earnings",
        ),
    )
    if earnings_value is None:
        return None
    parsed = pd.to_datetime(earnings_value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.date()


def _resolve_stop_pct(row: pd.Series) -> Optional[float]:
    stop_pct = _to_float(_first_present(row, ("Stop_Loss_%", "Stop_Pct", "stop_pct")))
    if stop_pct is not None:
        # A zero or negative stop % means the stop is at/above the price: no valid stop (ES-12;
        # abs() used to turn -2% into a valid 2% stop)
        return stop_pct if stop_pct > 0 else None

    price = _to_float(_first_present(row, ("Price", "Entry_Price", "entry_price")))
    stop_price = _to_float(_first_present(row, ("Stop_Loss_Price", "Stop_Price", "stop_price")))
    if price is None or stop_price is None or price <= 0:
        return None
    return max(0.0, ((price - stop_price) / price) * 100.0)


def _is_rth_open(now: datetime) -> bool:
    """now is US/Eastern (market_time.now_et). Weekends and NYSE holidays are closed."""
    return is_trading_day(now.date()) and ORB_OPENING_TIME <= now.time() <= RTH_CLOSE_TIME


def _resolve_entry_type(row: pd.Series, orb_window: int) -> str:
    """
    Entry type from the most specific setup label present (Entry_Setup, then Setup_Type).
    Only an exact VCP_PIVOT (scanner: real VCP, price within the pivot buy range) is a
    pivot breakout, because that type gets the wide stop band. Bare VCP / VCP_LOOSE / BASE
    and anything unrecognized are UNKNOWN, which gets the tight band.
    """
    label = ""
    for col in ("Entry_Setup", "Setup_Type", "Setup"):
        value = _first_present(row, (col,))
        if value is not None and str(value).strip():
            label = str(value).strip().lower()
            break
    if "orb" in label:
        return f"ORB_{orb_window}M_BREAKOUT"
    if any(key in label for key in ("ema", "50ma", "pullback", "bounce")):
        return "PULLBACK_BOUNCE"
    if label == "vcp_pivot":
        return "PIVOT_BREAKOUT"
    return "UNKNOWN"


def _evaluate_row(
    row: pd.Series,
    mode: str,
    orb_window: int,
    policy: ExecutionPolicy,
    now: datetime,
) -> dict:
    symbol = str(_first_present(row, ("Symbol", "symbol", "Ticker", "ticker")) or "").strip().upper()
    tier = str(_first_present(row, ("Tier", "tier")) or "").strip().upper()
    sector = str(_first_present(row, ("Industry_Group", "Sector", "sector")) or "").strip()
    entry_type = _resolve_entry_type(row, orb_window)
    stop_pct = _resolve_stop_pct(row)
    earnings_date = _parse_earnings_date(row)
    # Scanner's size multiplier (e.g. 0.5 for RS leaders); missing = full size, never above it.
    # Zero or negative means "don't size this" and is skipped below, never turned into full size.
    size_mult = _to_float(_first_present(row, ("Size_Multiplier", "size_multiplier")))
    size_mult = 1.0 if size_mult is None else min(size_mult, 1.0)
    risk_pct = policy.risk_per_trade_pct * max(size_mult, 0.0)
    # Stop band by entry type: pivot breakouts may run wider; position size shrinks to keep risk fixed
    max_stop_pct = policy.max_stop_pct_pivot if entry_type == "PIVOT_BREAKOUT" else policy.max_stop_pct

    reasons: list[str] = []
    skip = False
    manual = False

    if not symbol:
        skip = True
        reasons.append("Missing symbol")

    if tier not in {"TIER_ELITE_BUY", "TIER_BUY", "TIER_RS_LEADER"}:
        skip = True
        reasons.append(f"Tier {tier or '(missing)'} not eligible for new entry")

    if size_mult <= 0:
        skip = True
        reasons.append(f"Size multiplier {size_mult:g}: no position")

    # The scanner marks BUY / RS leader rows it blocked for market regime (e.g. CORRECTION)
    if _to_bool(_first_present(row, ("Regime_Blocked", "regime_blocked"))):
        skip = True
        reasons.append("Blocked by market regime")

    scan_age_h = _to_float(_first_present(row, ("__scan_age_hours",)))
    if scan_age_h is not None and scan_age_h > policy.max_scan_age_hours:
        skip = True
        reasons.append(f"Scan is {scan_age_h:.0f}h old (max {policy.max_scan_age_hours:g}h)")
    elif scan_age_h is not None and scan_age_h < -1:
        skip = True
        reasons.append(f"Scan timestamp is {-scan_age_h:.0f}h in the future; clock or file is wrong")
    elif scan_age_h is None and _to_bool(_first_present(row, ("__scan_dated",))) is False:
        skip = True
        reasons.append("Scan timestamp missing on this row; its age can't be checked")

    # Account state from account_snapshot.py (apply_account_snapshot). Unknown positions mean
    # the risk caps can't be checked, so nothing is bought blind.
    if str(_first_present(row, ("Account_State",)) or "").lower() == "missing":
        skip = True
        reasons.append("Open positions unknown (no fresh account snapshot); risk caps can't be checked")
    if _to_bool(_first_present(row, ("Already_Held",))):
        skip = True
        reasons.append("Already holding this symbol")
    unprotected = str(_first_present(row, ("Account_Unprotected",)) or "").strip()
    if unprotected:
        manual = True
        reasons.append(f"Open position(s) without a full stop ({unprotected}); counted at the 8% fallback risk")

    # Loss circuit breakers from the snapshot's equity history (risk_manager.circuit_breaker)
    daily_pnl = _to_float(_first_present(row, ("Account_Daily_PnL_Pct",)))
    weekly_pnl = _to_float(_first_present(row, ("Account_Weekly_PnL_Pct",)))
    month_dd = _to_float(_first_present(row, ("Account_Month_DD_Pct",)))
    halt, breaker_mult, breaker_note = risk_manager.circuit_breaker(
        daily_pnl, weekly_pnl, month_dd, policy.circuit_breakers)
    if halt:
        skip = True
        reasons.append(halt)
    elif breaker_mult < 1.0:
        risk_pct *= breaker_mult
        reasons.append(breaker_note)
    # Decision (b): breakers are blind without enough history since the last reset. Flag only;
    # order_entry refuses to submit on it (the dry-run plan and paper intents still show the row).
    history_sessions = _to_int(_first_present(row, ("Account_Breaker_History_Sessions",)))
    breaker_history_ok = (history_sessions is not None
                          and history_sessions >= policy.min_breaker_history_sessions)

    if mode == "rth" and not _is_rth_open(now):
        skip = True
        reasons.append("Outside regular trading hours")
    # premarket: a plan for today's open, so it's only valid before the open on a weekday
    if mode == "premarket" and (not is_trading_day(now.date()) or now.time() >= ORB_OPENING_TIME):
        skip = True
        reasons.append("Premarket plan run after the open (ET), or not a trading day")

    earnings_blackout = False
    if earnings_date is not None:
        days_to_earnings = (earnings_date - now.date()).days
        if 0 <= days_to_earnings <= policy.earnings_blackout_days:
            earnings_blackout = True
            skip = True
            reasons.append(
                f"Earnings blackout ({days_to_earnings} day(s) to earnings, limit {policy.earnings_blackout_days})"
            )

    if stop_pct is None:
        # No stop, no entry: never route a position without a defined exit
        skip = True
        reasons.append("Stop missing; no valid stop means no entry")
    elif stop_pct > max_stop_pct:
        manual = True
        reasons.append(f"Stop {stop_pct:.2f}% > {max_stop_pct:.2f}% max for {entry_type}")
    elif stop_pct < policy.min_stop_pct:
        skip = True
        reasons.append(f"Stop {stop_pct:.2f}% < {policy.min_stop_pct:.2f}% minimum")

    # Decision (c): unlabeled setup (blank, unrecognised or BASE) with a pullback stop context
    stop_context = str(_first_present(row, ("Stop_Context", "stop_context")) or "").strip().upper()
    if entry_type == "UNKNOWN" and stop_context in policy.pullback_stop_contexts:
        manual = True
        reasons.append(f"Unlabeled setup with a pullback stop ({stop_context}): manual review")

    if entry_type == "PULLBACK_BOUNCE":
        bounce_day = _to_bool(_first_present(row, ("Bounce_Day", "Is_Bounce_Day", "bounce_day")))
        if not policy.auto_trade_pullbacks:
            manual = True
            reasons.append("Pullback entries are manual review (policy auto_trade_pullbacks=False)")
        if bounce_day is False:
            skip = True
            reasons.append("Pullback bounce not on bounce day")
        elif bounce_day is None:
            manual = True
            reasons.append("Bounce-day flag unavailable")

    same_day_reentries = _to_int(_first_present(row, ("Same_Day_Reentries", "same_day_reentries")))
    if same_day_reentries is not None and same_day_reentries >= 1:
        skip = True
        reasons.append("Same-day re-entry already used")

    recent_stop_outs = _to_int(_first_present(row, ("Recent_Stop_Outs", "stop_out_count")))
    if recent_stop_outs is not None and recent_stop_outs >= 2:
        manual = True
        reasons.append("Two+ stop-outs detected; 1-day cooldown + manual permission required")

    current_sector_exposure_pct = _to_float(
        _first_present(row, ("Current_Sector_Exposure_Pct", "sector_exposure_pct"))
    )
    if current_sector_exposure_pct is not None and current_sector_exposure_pct >= policy.max_sector_concentration_pct:
        skip = True
        reasons.append(
            f"Sector cap reached: {current_sector_exposure_pct:.2f}% >= {policy.max_sector_concentration_pct:.2f}%"
        )

    # Liquidity: a position may not exceed 1% of average daily dollar volume. Needs the account
    # size (snapshot) and the scan's Dollar_Vol_Month_M; a binding cap shrinks size (and risk).
    position_cap_pct = policy.max_position_pct * max(size_mult, 0.0)
    dvol_m = _to_float(_first_present(row, ("Dollar_Vol_Month_M",)))
    net_liq = _to_float(_first_present(row, ("Account_Net_Liq",)))
    if dvol_m and net_liq and net_liq > 0:
        liq_cap_pct = policy.max_pct_of_daily_dollar_volume / 100 * dvol_m * 1e6 / 21 / net_liq * 100
        if liq_cap_pct < position_cap_pct:
            position_cap_pct = liq_cap_pct
            reasons.append(f"Liquidity caps position at {liq_cap_pct:.1f}% of account "
                           f"(1% of ${dvol_m * 1e6 / 21 / 1e6:.1f}M daily volume)")
    # The one sizing rule (risk_manager.size_position): risk / stop, capped; Risk_Pct_Target is the
    # risk the capped position actually carries, which is what the open-risk caps count
    # Risk budget before the position cap (order_entry re-sizes at its worst-case fill from it)
    risk_pct_budget = risk_pct
    position_pct, sized_risk_pct = risk_manager.size_position(risk_pct, stop_pct, position_cap_pct)
    if stop_pct:
        risk_pct = sized_risk_pct

    if skip:
        decision = SKIP
        manual = False
    elif manual:
        decision = MANUAL_REVIEW
    else:
        decision = BUY
        reasons.append("Meets execution rules")

    # Rules the engine has but the scanner doesn't feed yet: say so instead of implying they passed
    if decision != SKIP:
        unchecked = []
        if earnings_date is None:
            unchecked.append("earnings blackout")
        if same_day_reentries is None and recent_stop_outs is None:
            unchecked.append("re-entry/stop-out cooldown")
        if current_sector_exposure_pct is None:
            unchecked.append("sector cap")
        if daily_pnl is None and weekly_pnl is None and month_dd is None:
            unchecked.append("loss circuit breakers")
        if unchecked:
            reasons.append(f"Not checked (no data): {', '.join(unchecked)}")
        if not breaker_history_ok:
            have = "unknown" if history_sessions is None else str(history_sessions)
            reasons.append(f"Circuit breakers have {have} of {policy.min_breaker_history_sessions} "
                           "sessions since the last reset: auto entries halted")

    entry_price = _to_float(_first_present(row, ("Price", "Entry_Price", "entry_price")))
    # +TRIM_R level for the exit plan's trim sale (exit_rules.TRIM_R)
    take_profit = (round(entry_price * (1 + exit_rules.TRIM_R * stop_pct / 100), 2)
                   if entry_price and stop_pct else None)
    # Absolute stop for the bracket's child order: the scanner's own price, else from Stop_Pct
    stop_abs = _to_float(_first_present(row, ("Stop_Loss_Price", "Stop_Price", "stop_price")))
    if (stop_abs is None or stop_abs <= 0) and entry_price and stop_pct:
        stop_abs = entry_price * (1 - stop_pct / 100)
    if stop_pct is None:
        stop_abs = None     # no valid stop % (e.g. at/above price): no stop to place either
    return {
        "Symbol": symbol,
        "Decision": decision,
        # Carried through so the executor can size shares (it has no scan file)
        "Entry_Price": entry_price,
        "Take_Profit_Price": take_profit,
        "Reason": "; ".join(reasons),
        "Entry_Type": entry_type,
        "Stop_Pct": round(stop_pct, 2) if stop_pct is not None else None,
        "Needs_Manual_Permission": bool(manual and not skip),
        "Earnings_Blackout": earnings_blackout,
        "Time_Stop_Days": policy.time_stop_days,
        "Size_Multiplier": size_mult,
        "Risk_Pct_Target": round(risk_pct, 4),
        "Risk_Pct_Budget": round(risk_pct_budget, 4),
        "Stop_Price": round(stop_abs, 2) if stop_abs else None,
        "Stop_Context": stop_context,
        "Breaker_History_OK": breaker_history_ok,
        "Breaker_History_Sessions": history_sessions,
        "Position_Pct": round(position_pct, 4),
        "Position_Cap_Pct": round(position_cap_pct, 4),
        "Sector_Cap_Pct": policy.max_sector_concentration_pct,
        "Max_New_Entries_Day": policy.max_new_entries_per_day,
        "Max_Open_Risk_Pct": policy.max_total_open_risk_pct,
        "Mode": mode,
        "ORB_Window_Min": orb_window,
        "Sector": sector,
        # Passed through so order_entry can re-check the per-group risk cap from a fresh snapshot
        "IBD_Group_Rank": _to_int(_first_present(row, ("IBD_Group_Rank", "ibd_group_rank"))),
        # Adopted exit plan (exit_rules.py): trim at +TRIM_R unless +TRIM_R comes fast, breakeven at +TRIM_R,
        # then trail the MA the stock respected before entry (scanner's Trail_MA)
        **exit_rules.describe(str(_first_present(row, ("Trail_MA",)) or "ema21")),
        "Reentry_Policy": "One same-day re-entry; after 2 stop-outs: 1-day cooldown + manual permission",
        # Passed through for the alert setter, which reads only this file (never a raw scan)
        "Tier": tier,
        "Scan_Timestamp": _first_present(row, ("Scan_Timestamp",)),
        "Has_Real_VCP": _to_bool(_first_present(row, ("Has_Real_VCP",))),
        "VCP_Pivot": _to_float(_first_present(row, ("VCP_Pivot",))),
        "VCP_Dist_From_Pivot_%": _to_float(_first_present(row, ("VCP_Dist_From_Pivot_%",))),
        "VCP_Contractions": _to_int(_first_present(row, ("VCP_Contractions",))),
        "VCP_Final_PB_%": _to_float(_first_present(row, ("VCP_Final_PB_%",))),
    }


def _apply_group_risk_cap(out: pd.DataFrame, df: pd.DataFrame, policy: ExecutionPolicy) -> None:
    """
    Combined risk per IBD industry group <= policy.max_group_open_risk_pct, counted in priority
    order (out is already sorted) on top of any existing open risk in that group
    (Current_Group_Open_Risk_Pct). A BUY that would exceed it becomes MANUAL_REVIEW: more size in
    one group is allowed, but only with the owner's permission. Rows without a group aren't capped.
    Runs before the daily entry cap, so a capped name doesn't use one of the day's slots.
    """
    planned: Dict[int, float] = {}
    for idx in out.index:
        group = _to_float(_first_present(df.loc[idx], ("IBD_Group_Rank", "ibd_group_rank")))
        if group is None:
            continue
        group = int(group)
        if group not in planned:
            existing = _to_float(_first_present(df.loc[idx], ("Current_Group_Open_Risk_Pct",)))
            planned[group] = existing or 0.0
        if out.at[idx, "Decision"] != BUY:
            continue
        risk = float(out.at[idx, "Risk_Pct_Target"])
        projected = planned[group] + risk
        if projected > policy.max_group_open_risk_pct + 1e-9:
            reason = (f"Group risk cap: industry group #{group} would carry {projected:.2f}% "
                      f"> {policy.max_group_open_risk_pct:.2f}%; manual permission required")
            out.at[idx, "Decision"] = MANUAL_REVIEW
            out.at[idx, "Needs_Manual_Permission"] = True
            existing_reason = out.at[idx, "Reason"]
            if existing_reason == "Meets execution rules":
                existing_reason = ""
            out.at[idx, "Reason"] = f"{existing_reason}; {reason}" if existing_reason else reason
        else:
            planned[group] = projected


def _apply_open_risk_cap(out: pd.DataFrame, df: pd.DataFrame, policy: ExecutionPolicy) -> None:
    """
    Total capital at risk <= policy.max_total_open_risk_pct, counting today's BUYs in priority
    order on top of the account's existing open risk (Current_Open_Risk_Pct, the same on every
    row). Checked cumulatively: two names that each fit alone may not both fit. MANUAL_REVIEW
    rows are checked and counted too (approving one must not breach the cap; counting them is
    conservative). The cap is hard: a breach is a SKIP, not something to approve.
    """
    existing = None
    if len(df):
        values = pd.to_numeric(df.get("Current_Open_Risk_Pct", pd.Series(dtype=float)), errors="coerce").dropna()
        existing = float(values.max()) if len(values) else None
    if existing is None:
        return
    planned = existing
    for idx in out.index:
        if out.at[idx, "Decision"] not in (BUY, MANUAL_REVIEW):
            continue
        risk = float(out.at[idx, "Risk_Pct_Target"])
        if planned + risk > policy.max_total_open_risk_pct + 1e-9:
            reason = (f"Open risk cap: {planned:.2f}% + {risk:.2f}% > {policy.max_total_open_risk_pct:.2f}%")
            existing_reason = out.at[idx, "Reason"]
            if existing_reason == "Meets execution rules":
                existing_reason = ""
            out.at[idx, "Decision"] = SKIP
            out.at[idx, "Needs_Manual_Permission"] = False
            out.at[idx, "Reason"] = f"{existing_reason}; {reason}" if existing_reason else reason
        else:
            planned += risk


def apply_account_snapshot(df: pd.DataFrame, snapshot: Optional[dict]) -> pd.DataFrame:
    """
    Add the account's open risk to every scan row (see account_snapshot.py):
    Current_Open_Risk_Pct, Current_Group_Open_Risk_Pct (by the row's IBD_Group_Rank),
    Already_Held and Account_Unprotected. snapshot=None marks every row Account_State=missing,
    which blocks new entries: the caps can't be checked without knowing what's held.
    """
    df = df.copy()
    if snapshot is None:
        df["Account_State"] = "missing"
        return df
    held = set(snapshot.get("held_symbols", []))
    groups = snapshot.get("group_open_risk_pct", {})
    symbols = df.get("Symbol", pd.Series([""] * len(df))).astype(str).str.strip().str.upper()
    ranks = pd.to_numeric(df.get("IBD_Group_Rank", pd.Series([None] * len(df))), errors="coerce")
    df["Account_State"] = "ok"
    df["Current_Open_Risk_Pct"] = float(snapshot["total_open_risk_pct"])
    df["Current_Group_Open_Risk_Pct"] = [groups.get(str(int(r)), 0.0) if pd.notna(r) else None for r in ranks]
    df["Already_Held"] = symbols.isin(held)
    df["Account_Unprotected"] = ", ".join(snapshot.get("unprotected_symbols", []))
    df["Account_Net_Liq"] = float(snapshot.get("net_liquidation") or 0) or None
    df["Account_Open_Positions"] = len(held)
    # Equity change % for the loss circuit breakers; absent in older snapshots = not checked
    for col, key in (("Account_Daily_PnL_Pct", "daily_pnl_pct"), ("Account_Weekly_PnL_Pct", "weekly_pnl_pct"),
                     ("Account_Month_DD_Pct", "month_drawdown_pct")):
        df[col] = snapshot.get(key)
    # Sessions of equity history since the last reset (decision b); older snapshots lack it = None
    df["Account_Breaker_History_Sessions"] = snapshot.get("breaker_history_sessions")
    return df


def build_execution_decisions(
    df: pd.DataFrame, mode: str, orb_window: int, policy: Optional[ExecutionPolicy] = None
) -> pd.DataFrame:
    policy = policy or ExecutionPolicy()
    now = now_et()

    df = df.reset_index(drop=True)  # keep df and out aligned by position
    if "Scan_Timestamp" in df.columns:
        # Scan_Timestamp is the scanning Mac's local clock, so age is measured on the local clock
        stamps = pd.to_datetime(df["Scan_Timestamp"], errors="coerce")
        df = df.assign(__scan_age_hours=(datetime.now() - stamps).dt.total_seconds() / 3600,
                       __scan_dated=stamps.notna())
    evaluated = [
        _evaluate_row(row=row, mode=mode, orb_window=orb_window, policy=policy, now=now)
        for _, row in df.iterrows()
    ]
    out = pd.DataFrame(evaluated)

    rank_columns_ascending = []
    rank_columns_descending = []
    if "Buy_Priority" in df.columns:
        out["__rank_buy_priority"] = pd.to_numeric(df["Buy_Priority"], errors="coerce").fillna(10_000)
        rank_columns_ascending.append("__rank_buy_priority")
    if "Priority_Rating" in df.columns:
        out["__rank_priority_rating"] = pd.to_numeric(df["Priority_Rating"], errors="coerce").fillna(-1)
        rank_columns_descending.append("__rank_priority_rating")

    if rank_columns_ascending or rank_columns_descending:
        sort_cols = []
        ascending = []
        for c in rank_columns_ascending:
            sort_cols.append(c)
            ascending.append(True)
        for c in rank_columns_descending:
            sort_cols.append(c)
            ascending.append(False)
        out = out.sort_values(sort_cols, ascending=ascending, kind="stable")

    _apply_group_risk_cap(out, df, policy)
    _apply_open_risk_cap(out, df, policy)

    # Max open positions: held + today's BUY and MANUAL_REVIEW rows (approving a review row
    # opens a position too), counted in priority order; rows past the free slots are skipped
    held_count = None
    if "Account_Open_Positions" in df.columns:
        values = pd.to_numeric(df["Account_Open_Positions"], errors="coerce").dropna()
        held_count = int(values.max()) if len(values) else None
    if held_count is not None:
        free = max(0, policy.max_open_positions - held_count)
        candidates = out.index[out["Decision"].isin([BUY, MANUAL_REVIEW])].tolist()
        for idx in candidates[free:]:
            reason = f"Max open positions reached ({held_count} held, max {policy.max_open_positions})"
            existing_reason = out.at[idx, "Reason"]
            if existing_reason == "Meets execution rules":
                existing_reason = ""
            out.at[idx, "Decision"] = SKIP
            out.at[idx, "Needs_Manual_Permission"] = False
            out.at[idx, "Reason"] = f"{existing_reason}; {reason}" if existing_reason else reason

    slots = policy.max_new_entries_per_day
    cap_reason = f"Daily new-entry cap reached ({policy.max_new_entries_per_day})"
    buy_idxs = out.index[out["Decision"] == BUY].tolist()
    if len(buy_idxs) > slots:
        for idx in buy_idxs[slots:]:
            existing_reason = out.at[idx, "Reason"]
            out.at[idx, "Decision"] = SKIP
            out.at[idx, "Needs_Manual_Permission"] = False
            out.at[idx, "Reason"] = f"{existing_reason}; {cap_reason}" if existing_reason else cap_reason

    helper_cols = [c for c in out.columns if c.startswith("__rank_")]
    if helper_cols:
        out = out.drop(columns=helper_cols)
    return out.reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate per-symbol execution decisions from scanner CSV output."
    )
    parser.add_argument("--input", required=True, help="Path to scanner CSV (COMPLETE or BUY_LIST).")
    parser.add_argument("--output", required=True, help="Path to write decision CSV.")
    parser.add_argument("--mode", default="rth", choices=["rth", "premarket"],
                        help="rth: entries during regular hours; premarket: plan for today's open (run before 9:30)")
    parser.add_argument("--orb-window", type=int, default=10, help="ORB window in minutes (default: 10).")
    parser.add_argument("--account-snapshot", help="account_snapshot.py JSON, or 'latest'. Adds open "
                        "positions to the open-risk and group caps.")
    parser.add_argument("--require-account-snapshot", action="store_true",
                        help="Block new entries if no fresh snapshot is found (scheduled runs).")
    parser.add_argument("--expect-account", help="Ignore a snapshot for any other account.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")
    if args.orb_window <= 0:
        raise ValueError("--orb-window must be a positive integer")

    df = pd.read_csv(input_path)
    if df.empty:
        raise ValueError(f"Input CSV has no rows: {input_path}")

    if args.account_snapshot or args.require_account_snapshot:
        from account_snapshot import load_snapshot
        snapshot = load_snapshot(args.account_snapshot or "latest", account=args.expect_account or None)
        if snapshot is None and not args.require_account_snapshot:
            print("⚠ No fresh account snapshot; caps ignore existing positions")
        else:
            df = apply_account_snapshot(df, snapshot)
            if snapshot:
                print(f"Account {snapshot['account']}: open risk {snapshot['total_open_risk_pct']:.2f}%, "
                      f"{len(snapshot['held_symbols'])} position(s)")
            else:
                print("❌ No fresh account snapshot: new entries blocked")

    decisions = build_execution_decisions(df=df, mode=args.mode, orb_window=args.orb_window)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    decisions.to_csv(output_path, index=False)

    counts = decisions["Decision"].value_counts().to_dict()
    print(f"Execution decisions written: {output_path}")
    print(f"Decision counts: {counts}")
    notify_decisions(decisions, args.mode)
    return 0


def decision_summary(decisions: pd.DataFrame, mode: str) -> str:
    """'2 BUY, 1 MANUAL_REVIEW (premarket)': the one line the owner gets per engine run."""
    counts = decisions["Decision"].value_counts() if "Decision" in decisions else pd.Series(dtype=int)
    return f"{int(counts.get(BUY, 0))} BUY, {int(counts.get(MANUAL_REVIEW, 0))} MANUAL_REVIEW ({mode})"


def notify_decisions(decisions: pd.DataFrame, mode: str) -> None:
    """One macOS notification per run with today's BUY / MANUAL_REVIEW counts (notify.py never raises)."""
    import notify as notifier
    notifier.notify("Execution engine", decision_summary(decisions, mode), 'info')


if __name__ == "__main__":
    raise SystemExit(main())
