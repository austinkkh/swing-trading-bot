"""
Position sizing and loss circuit breakers: the ONE sizing policy.

Every path that turns a decision into a size uses size_position():
    position % of equity = risk % / stop %, capped at the position cap
    (a binding cap lowers the actual risk below the target, never raises it)
execution_engine computes Position_Pct / Risk_Pct_Target with it, and ibkr_paper_executor sizes
its paper intents with it. The old PositionSizer class (a second, unused policy with its
own copies of every limit) was removed; its circuit breakers live on here and are applied by the
engine from the account snapshot's equity history. The exit plan is exit_rules.describe(), which
the engine writes on every decision row.

Circuit breakers (from realized + unrealized equity change, account_snapshot.equity_changes):
    daily loss     >= DAILY_LOSS_LIMIT_PCT       -> no new entries today
    month drawdown >= MONTHLY_DRAWDOWN_LIMIT_PCT -> no new entries; review the system before resuming
    weekly loss    >= WEEKLY_LOSS_LIMIT_PCT      -> new entries at reduced risk
All limits are example values from strategy_params.py.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import strategy_params as sp


@dataclass(frozen=True)
class CircuitBreakers:
    daily_loss_limit_pct: float = sp.DAILY_LOSS_LIMIT_PCT              # stop trading today
    weekly_loss_limit_pct: float = sp.WEEKLY_LOSS_LIMIT_PCT            # reduce risk for the rest of the week
    monthly_drawdown_limit_pct: float = sp.MONTHLY_DRAWDOWN_LIMIT_PCT  # stop, review, rebuild
    weekly_loss_risk_mult: float = sp.WEEKLY_LOSS_RISK_MULT            # risk multiplier after the weekly limit


def size_position(risk_pct: float, stop_pct: Optional[float], position_cap_pct: float) -> Tuple[float, float]:
    """
    (position % of equity, actual risk % of equity) for a target risk and stop distance.
    No usable stop means no risk-based size: (0, 0). The cap only ever shrinks the position.
    """
    if not stop_pct or stop_pct <= 0 or risk_pct <= 0 or position_cap_pct <= 0:
        return 0.0, 0.0
    position_pct = min(risk_pct / stop_pct * 100, position_cap_pct)
    return position_pct, position_pct * stop_pct / 100


def circuit_breaker(daily_pct: Optional[float], weekly_pct: Optional[float],
                    month_drawdown_pct: Optional[float],
                    limits: CircuitBreakers = CircuitBreakers()) -> Tuple[Optional[str], float, str]:
    """
    (halt reason or None, risk multiplier, note) from equity change % (negative = loss).
    None inputs are unknown and never trigger a breaker; the caller reports them as not checked.
    """
    if daily_pct is not None and daily_pct <= -limits.daily_loss_limit_pct:
        return (f"Daily loss limit hit: {daily_pct:.2f}% (limit {limits.daily_loss_limit_pct:g}%); "
                "no new entries today"), 0.0, ""
    if month_drawdown_pct is not None and month_drawdown_pct <= -limits.monthly_drawdown_limit_pct:
        return (f"Monthly drawdown limit hit: {month_drawdown_pct:.2f}% "
                f"(limit {limits.monthly_drawdown_limit_pct:g}%); review the system before resuming"), 0.0, ""
    if weekly_pct is not None and weekly_pct <= -limits.weekly_loss_limit_pct:
        return None, limits.weekly_loss_risk_mult, (
            f"Weekly loss {weekly_pct:.2f}% (limit {limits.weekly_loss_limit_pct:g}%): "
            f"risk x{limits.weekly_loss_risk_mult:g}")
    return None, 1.0, ""


def is_elite_leader(
    eps_yoy: Optional[float] = None,
    rs_rating: Optional[float] = None,
    composite_rating: Optional[float] = None,
    ytd_return_pct: Optional[float] = None
) -> bool:
    """
    Elite leader ("generals" rule): the only names tradeable in a CORRECTION (at half size).

    Path 1, fundamental elite: EPS YoY, RS and Composite (if available) at or above their minimums.
    Path 2, RS elite: very high RS plus a strong YTD return.
    Thresholds are example values from strategy_params.py.
    """
    if eps_yoy is not None and rs_rating is not None:
        if eps_yoy >= sp.ELITE_EPS_YOY_MIN and rs_rating >= sp.ELITE_RS_MIN:
            if composite_rating is None or composite_rating >= sp.ELITE_COMPOSITE_MIN:
                return True
    if rs_rating is not None and ytd_return_pct is not None:
        if rs_rating >= sp.ELITE_RS_ONLY_MIN and ytd_return_pct >= sp.ELITE_RS_ONLY_YTD_MIN:
            return True
    return False
