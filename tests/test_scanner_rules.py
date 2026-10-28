"""Tier assignment, stops, and scoring rules in momentum_scanner."""

from datetime import datetime

import pytest

import momentum_scanner as ms
import strategy_params as sp
from helpers import make_stock, price_action
from momentum_scanner import MomentumScanner


@pytest.fixture
def scanner():
    s = MomentumScanner()  # no initialize(): no TWS needed
    s.scan_timestamp = datetime.now()
    return s


def tier_of(scanner, symbol):
    for name in ('elite_buy_results', 'elite_hold_results', 'buy_results', 'rs_leader_results',
                 'watch_results', 'extended_results'):
        if any(s['symbol'] == symbol for s in getattr(scanner, name)):
            return name.replace('_results', '')
    return None


def run(scanner, *stocks):
    scanner._apply_tier_filters(list(stocks))
    scanner._calculate_priority_ratings_for_all_tiers()
    return {s['symbol']: s for s in stocks}


# ─── Stops ───
def test_stop_from_swing_low(scanner):
    stock = make_stock("OK", stop_low=97.0)
    scanner._calculate_context_based_stop(stock)
    assert stock['stop_loss_pct'] == pytest.approx(3.0)
    assert stock['stop_status'] != 'NO_STOP'


def test_missing_price_action_is_no_stop(scanner):
    stock = make_stock("NOPA", price_action_override=None)
    scanner._calculate_context_based_stop(stock)
    assert stock['stop_status'] == 'NO_STOP' and stock['stop_loss_pct'] is None


def test_stop_above_price_is_no_stop(scanner):
    stock = make_stock("ABOVE", stop_low=101.0)
    scanner._calculate_context_based_stop(stock)
    assert stock['stop_status'] == 'NO_STOP'


# ─── Scoring ───
def test_priority_rating_within_0_to_110(scanner):
    stock = make_stock("SCORE", eps_yoy=600, industry_rank=1, rs_rating=99, vcp_score=10, ytd_return=80)
    score = scanner.calculate_priority_rating(stock)
    assert 0 <= score <= 110


@pytest.mark.parametrize("stop_low, label", [(None, "no stop"), (99.95, "too tight")])
def test_bad_stops_earn_no_stop_quality_points(scanner, stop_low, label):
    pa_override = None if stop_low is None else ...
    stock = make_stock("BAD", stop_low=stop_low or 97.0, price_action_override=pa_override)
    scanner.calculate_priority_rating(stock)
    assert stock['priority_details']['stop_quality_score'] == 0, label


# ─── Tiers ───
def test_elite_reaches_clean_entry(scanner):
    """Regression: the stop must exist before tiering, or every elite lands in HOLD."""
    stocks = run(scanner, make_stock("ELITE", rs_rating=99, ibd_rs=99, ytd_return=80))
    assert tier_of(scanner, "ELITE") == 'elite_buy'
    assert stocks["ELITE"]['stop_loss_pct'] is not None


def test_elite_blocked_by_low_eps_rating(scanner):
    stocks = run(scanner, make_stock("LOWEPS", rs_rating=99, ibd_rs=99, ytd_return=80, ibd_eps_rating=40))
    assert tier_of(scanner, "LOWEPS") not in ('elite_buy', 'elite_hold')
    assert "EPS Rtg 40" in stocks["LOWEPS"]['elite_blocked_reason']


def test_elite_blocked_by_distribution(scanner):
    stocks = run(scanner, make_stock("ADSELL", rs_rating=99, ibd_rs=99, ytd_return=80, ibd_acc_dist='D'))
    assert tier_of(scanner, "ADSELL") not in ('elite_buy', 'elite_hold')
    assert "A/D D" in stocks["ADSELL"]['elite_blocked_reason']


def test_buy_setup(scanner):
    run(scanner, make_stock("BUY"))
    assert tier_of(scanner, "BUY") == 'buy'


def test_buy_needs_clean_acc_dist(scanner):
    stocks = run(scanner, make_stock("SELLING", ibd_acc_dist='D-'))
    assert tier_of(scanner, "SELLING") == 'watch'
    assert "A/D D-" in stocks["SELLING"]['watch_reason']


def test_buy_needs_eps_rating(scanner):
    weak = ms.TIER_BUY_MIN_EPS_RATING - 10
    stocks = run(scanner, make_stock("WEAK", ibd_eps_rating=weak))
    assert tier_of(scanner, "WEAK") == 'watch'
    assert f"EPS Rtg {weak} < {ms.TIER_BUY_MIN_EPS_RATING}" in stocks["WEAK"]['watch_reason']


def test_buy_rejects_too_tight_stop(scanner):
    stocks = run(scanner, make_stock("TIGHT", stop_low=99.95))
    assert tier_of(scanner, "TIGHT") == 'watch'
    assert "too tight" in stocks["TIGHT"]['watch_reason']


def test_rs_leader_qualifies_at_reduced_size(scanner):
    stocks = run(scanner, make_stock("LEAD", rs_rating=97, ibd_rs=97, ibd_eps_rating=40,
                                     has_real_vcp=True, vcp_num_contractions=3, vcp_pivot_distance_pct=-2.0))
    assert tier_of(scanner, "LEAD") == 'rs_leader'
    assert stocks["LEAD"]['size_multiplier'] < 1.0


def test_rs_leader_past_pivot_goes_to_watch_with_reason(scanner):
    stocks = run(scanner, make_stock("PAST", rs_rating=97, ibd_rs=97, ibd_eps_rating=40,
                                     has_real_vcp=True, vcp_num_contractions=3, vcp_pivot_distance_pct=15.0))
    assert tier_of(scanner, "PAST") == 'watch'
    assert "15% past pivot" in stocks["PAST"]['watch_reason']


def test_rs_leader_needs_ibd_rs(scanner):
    """Homegrown RS alone never qualifies a stock as an RS leader."""
    run(scanner, make_stock("NOIBD", rs_rating=99, ibd_rs=None, ibd_eps_rating=None,
                            has_real_vcp=True, vcp_pivot_distance_pct=0.0))
    assert tier_of(scanner, "NOIBD") != 'rs_leader'


def test_extended_without_base_flag(scanner):
    stocks = run(scanner, make_stock("FAR", distance_from_200ma=70.0),
                 make_stock("FARBASE", distance_from_200ma=70.0, has_real_vcp=True))
    assert stocks["FAR"]['extended_no_base'] is True
    assert stocks["FARBASE"]['extended_no_base'] is False


def test_elite_blocked_by_distribution_even_without_eps_rating(scanner):
    """A/D is checked on its own: a blank IBD EPS Rating mustn't skip the distribution gate."""
    stocks = run(scanner, make_stock("ADONLY", rs_rating=99, ibd_rs=99, ytd_return=80,
                                     ibd_eps_rating=None, ibd_acc_dist='E', ibd_eps_est_yr_pct=150))
    assert tier_of(scanner, "ADONLY") not in ('elite_buy', 'elite_hold')
    assert "A/D E" in stocks["ADONLY"]['elite_blocked_reason']


def test_stop_just_under_minimum_is_not_rounded_up(scanner):
    """A stop 0.04 points under the minimum must stay under it, not round up to it and pass."""
    stocks = run(scanner, make_stock("NEARMIN", stop_low=100 - (ms.STOP_PCT_MIN - 0.04)))
    assert stocks["NEARMIN"]['stop_loss_pct'] < ms.STOP_PCT_MIN
    assert tier_of(scanner, "NEARMIN") != 'buy'


# ─── Market regime ───
class _Regime:
    def __init__(self, take_new):
        self.should_take_new_positions = take_new


class _Detector:
    def __init__(self, result=None, error=None):
        self.result, self.error = result, error

    def detect_regime(self):
        if self.error:
            raise self.error
        return self.result

    def display_regime(self, regime):
        pass


@pytest.mark.parametrize("detector", [_Detector(result=None), _Detector(error=RuntimeError("SPY fetch failed"))])
def test_regime_detection_failure_fails_closed(scanner, detector):
    scanner.regime_detector = detector
    scanner._set_regime_gating()
    assert scanner.watch_only_mode is True


@pytest.mark.parametrize("take_new, watch_only", [(True, False), (False, True)])
def test_regime_gating_follows_detected_regime(scanner, take_new, watch_only):
    scanner.regime_detector = _Detector(result=_Regime(take_new))
    scanner._set_regime_gating()
    assert scanner.watch_only_mode is watch_only


def test_rs_leader_entry_is_labeled_vcp(scanner):
    stocks = run(scanner, make_stock("LEADV", rs_rating=97, ibd_rs=97, ibd_eps_rating=40,
                                     has_real_vcp=True, vcp_num_contractions=3, vcp_pivot_distance_pct=-2.0))
    assert stocks["LEADV"]['setup_type'] == 'VCP'


def test_universe_snapshot_is_written(scanner, tmp_path, monkeypatch):
    import pandas as pd
    import momentum_scanner as ms
    monkeypatch.setattr(ms, "OUTPUT_DIR", str(tmp_path / "Scans"))
    scanner._save_universe_snapshot(["BBB", "AAA", "AAA"])
    files = list((tmp_path / "Data" / "snapshots").glob("universe_*.csv"))
    assert len(files) == 1
    import pandas as pd
    assert pd.read_csv(files[0])['symbol'].tolist() == ["AAA", "BBB"]


# ─── Entry label for the execution engine's stop band ───
def test_vcp_label_far_from_pivot_is_base_not_pivot():
    """Reviewer case: VCP score alone, no real VCP, 15% under the pivot -> must not get the pivot band."""
    from momentum_scanner import entry_setup_label
    assert entry_setup_label({'setup_type': 'VCP', 'has_real_vcp': False, 'vcp_pivot_distance_pct': -15.0}) == 'BASE'
    assert entry_setup_label({'setup_type': 'VCP', 'has_real_vcp': True, 'vcp_pivot_distance_pct': -15.0}) == 'BASE'


def test_vcp_loose_is_base():
    from momentum_scanner import entry_setup_label
    assert entry_setup_label({'setup_type': 'VCP_LOOSE', 'has_real_vcp': False}) == 'BASE'


def test_real_vcp_near_pivot_is_vcp_pivot():
    from momentum_scanner import entry_setup_label
    assert entry_setup_label({'setup_type': 'VCP', 'has_real_vcp': True, 'vcp_pivot_distance_pct': -2.0}) == 'VCP_PIVOT'


def test_elite_entry_label_wins():
    from momentum_scanner import entry_setup_label
    assert entry_setup_label({'elite_entry_setup': '10EMA', 'setup_type': 'VCP', 'has_real_vcp': True,
                              'vcp_pivot_distance_pct': 0.0}) == '10EMA'


def test_reviewer_case_end_to_end_goes_to_review(scanner):
    """No real VCP, 15% under the pivot, on the 10EMA, 6% stop: scanner + engine must not auto-buy."""
    import pandas as pd
    import execution_engine as ee
    from momentum_scanner import entry_setup_label
    stocks = run(scanner, make_stock('X', stop_low=94.0, vcp_score=4.5, adr=4.0, distance_from_10ema=0.5,
                                     vcp_pivot_distance_pct=-15.0, has_real_vcp=False))
    s = stocks['X']
    row = pd.DataFrame([{"Symbol": "X", "Tier": "TIER_BUY", "Setup_Type": s.get('setup_type', ''),
                         "Entry_Setup": entry_setup_label(s), "Stop_Loss_%": s['stop_loss_pct']}])
    decision = ee.build_execution_decisions(row, mode="afterhours", orb_window=10).iloc[0]
    assert decision["Entry_Type"] != "PIVOT_BREAKOUT"
    assert decision["Decision"] != ee.BUY


def test_days_above_21ema_counts_closes_since_the_last_close_below():
    import pandas as pd
    import momentum_scanner as ms
    ema = pd.Series([10.0] * 6)
    assert ms.days_above_ema(pd.Series([9, 11, 12, 13, 9, 11.0]), ema) == 1
    assert ms.days_above_ema(pd.Series([9, 11, 12, 13, 14, 15.0]), ema) == 5
    assert ms.days_above_ema(pd.Series([11, 11, 9.0]), pd.Series([10.0] * 3)) == 0


def test_default_stop_is_the_true_five_day_swing_low(scanner):
    # Outside the EMA / 50MA / extended contexts -> DEFAULT; yesterday's low 99, 5-day low 96
    pa = price_action(stop_low=96.0)
    pa['yesterday_low'] = 99.0
    stock = make_stock("DEF", price_action_override=pa, distance_from_10ema=8.0, distance_from_21ema=8.0,
                       distance_from_50ma=10.0)
    scanner._calculate_context_based_stop(stock)
    assert stock['stop_loss_pct'] == pytest.approx(4.0)     # 5-day low, not yesterday's 1%


def test_elites_trade_at_half_size_in_a_correction():
    import momentum_scanner as ms
    assert ms.effective_size_multiplier({}, 'TIER_ELITE_BUY', watch_only=True) == 0.5
    assert ms.effective_size_multiplier({}, 'TIER_ELITE_BUY', watch_only=False) == 1.0
    assert ms.effective_size_multiplier({'size_multiplier': 0.5}, 'TIER_RS_LEADER', watch_only=True) == 0.5
    assert ms.effective_size_multiplier({}, 'TIER_BUY', watch_only=True) == 1.0   # BUY is blocked separately


# ─── EPS growth is automatic (owner decision 2026-10-25, #5) ───
def test_eps_comes_from_ibd_forward_estimate(scanner):
    """IBD's current-year EPS estimate (universe export) wins over an IBD list file's value."""
    stocks = run(scanner, make_stock("EST", ibd_eps_est_yr_pct=45.0, ibd_eps_year_growth=300.0),
                 make_stock("LIST", ibd_eps_year_growth=60.0),
                 make_stock("ZERO", ibd_eps_est_yr_pct=0.0, ibd_eps_year_growth=300.0),
                 make_stock("NONE", eps_yoy=500.0))   # a stray value with no IBD source is dropped
    assert stocks["EST"]['eps_yoy'] == 45.0
    assert stocks["LIST"]['eps_yoy'] == 60.0
    assert stocks["ZERO"]['eps_yoy'] == 0.0
    assert stocks["NONE"]['eps_yoy'] is None


def test_forward_estimate_drives_the_fundamental_elite_path(scanner):
    # RS below the RS-only path's minimum: only the EPS-growth path can make it elite
    rs = sp.ELITE_RS_MIN
    assert rs < sp.ELITE_RS_ONLY_MIN
    stocks = run(scanner, make_stock("FWD", rs_rating=rs, ibd_rs=rs, ibd_eps_est_yr_pct=sp.ELITE_EPS_YOY_MIN + 50.0),
                 make_stock("NOFWD", rs_rating=rs, ibd_rs=rs))
    assert stocks["FWD"]['is_elite_leader'] is True
    assert stocks["NOFWD"]['is_elite_leader'] is False
