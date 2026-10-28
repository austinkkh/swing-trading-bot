"""Base counting on synthetic price paths."""

from base_count import count_bases
from helpers import price_path

STAGE = [(30, 100.0)]                  # stage low 50 -> 100
BASE_1 = [(10, 85), (10, 99), (2, 101)]  # 15% base, breakout above 100


def test_two_bases_after_a_real_advance():
    legs = STAGE + BASE_1 + [(15, 126), (10, 111), (10, 127)]  # +25% run, then a 12% base
    result = count_bases(price_path(legs))
    assert result['base_number'] == 2
    assert not result['base_forming']


def test_base_on_base_keeps_the_count():
    legs = STAGE + BASE_1 + [(8, 108), (10, 97), (10, 110)]  # only +8% before the next base
    assert count_bases(price_path(legs))['base_number'] == 1


def test_undercut_resets_to_first_base():
    legs = STAGE + BASE_1 + [(15, 126), (12, 80), (12, 128)]  # second base undercuts base 1's low
    result = count_bases(price_path(legs))
    assert result['base_number'] == 1
    assert 'undercut' in result['base_count_reason']


def test_base_still_forming():
    legs = STAGE + BASE_1 + [(15, 126), (10, 111), (8, 118)]  # no breakout yet
    result = count_bases(price_path(legs))
    assert result['base_number'] == 2
    assert result['base_forming']
