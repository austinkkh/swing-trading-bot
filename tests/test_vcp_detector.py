"""VCP detection on synthetic price paths with known contractions."""

from helpers import price_path
from vcp_detector import VCPDetector

RUN_UP = [(40, 100.0)]  # trend into the base high


def detect(legs):
    return VCPDetector().detect(price_path(legs), min_contractions=3)


def test_classic_three_contractions():
    # 20% -> 10% -> ~5%, then a quiet finish near the highs
    result = detect(RUN_UP + [(10, 80), (10, 99), (8, 89), (8, 98.5), (7, 94), (7, 98), (6, 97.6)])
    assert result.has_vcp
    assert result.num_contractions == 3
    assert result.final_contraction_pct < 6


def test_expanding_pullbacks_are_not_a_vcp():
    # 5% -> 10% -> 20%: volatility widening, the opposite of a VCP
    result = detect(RUN_UP + [(6, 95), (6, 99), (8, 89), (8, 99), (10, 79), (10, 98), (6, 97.6)])
    assert not result.has_vcp


def test_near_equal_pullbacks_still_count():
    # 20% -> 10.0% -> 10.4% -> 5%: the newer middle pullback is a hair deeper (within 10%),
    # which a strict "each shallower" rule treats as a reset
    result = detect(RUN_UP + [(10, 80), (10, 99), (8, 89.1), (8, 99), (8, 88.7), (8, 98.5), (6, 93.6), (6, 98),
                              (6, 97.6)])
    assert result.has_vcp
    assert result.num_contractions == 4


def test_in_progress_breakdown_cancels_an_old_pattern():
    # Same classic base, but the stock is falling hard right now (no confirmed swing low yet)
    result = detect(RUN_UP + [(10, 80), (10, 99), (8, 89), (8, 98.5), (7, 94), (7, 98), (4, 85)])
    assert not result.has_vcp
    assert result.pullbacks[-1].depth_pct > 10  # the live decline is visible to the detector


def test_steadily_widening_pullbacks_are_not_a_vcp():
    # 5.0 -> 5.5 -> 6.0 -> 6.6 -> 7.3%: each within 10% of the next, but volatility is expanding.
    # The near-equal tolerance must not compound across the whole sequence.
    legs = RUN_UP
    high = 100.0
    for depth in (5.0, 5.5, 6.0, 6.6, 7.3):
        legs = legs + [(7, high * (1 - depth / 100)), (7, high)]
    result = detect(legs + [(6, 99.6)])
    assert not result.has_vcp


def test_live_undercut_extends_the_last_pullback():
    # Clean 20 -> 10 -> 6% contractions, then a bounce to 95.5 and a drop below the last low (92.1).
    # Measured from the bounce high it looked like a tight 4% contraction with a 95.5 pivot, putting
    # price inside the buy range. The bounce never made a new swing: the last pullback just kept going.
    result = detect(RUN_UP + [(10, 80), (10, 99), (8, 89), (8, 98.5), (7, 92.1), (7, 95.5), (4, 91.5)])
    assert result.pivot_price > 98            # the real swing high, not the 95.5 bounce
    assert result.final_contraction_pct > 7   # 98.5 -> 91.5, deeper than the old 6%
    assert result.pivot_distance_pct < -5     # outside the ±5% buy range
