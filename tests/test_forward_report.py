"""Forward report: missing inputs, ISO-week bucketing, R stats, ledger R, PAPERCHECK exclusion, CI/status."""

import json
import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

import forward_report as fr
import order_ledger as ol

AS_OF = date(2026, 10, 30)


# ─── fixtures ───

def write_signals(path):
    rows = [
        # week 2026-W43 (Mon 10-19 .. Sun 10-25)
        dict(signal_date='2026-10-19', symbol='AAA', setup='VCP', fill_status='filled', adopted_status='closed',
             adopted_R=1.5, mfe3_R=1.0, mfe5_R=2.0),
        dict(signal_date='2026-10-21', symbol='BBB', setup='POCKET_PIVOT', fill_status='filled',
             adopted_status='closed', adopted_R=-1.0, mfe3_R=0.2, mfe5_R=0.4),
        dict(signal_date='2026-10-23', symbol='CCC', setup='POCKET_PIVOT', fill_status='filled',
             adopted_status='open', adopted_R=0.3, mfe3_R=0.5, mfe5_R=np.nan),
        dict(signal_date='2026-10-23', symbol='DDD', setup='POCKET_PIVOT', fill_status='no_entry',
             adopted_status=np.nan, adopted_R=np.nan, mfe3_R=np.nan, mfe5_R=np.nan),
        dict(signal_date='2026-10-25', symbol='EEE', setup='VCP', fill_status='filled',   # Sunday -> W43
             adopted_status='closed', adopted_R=0.5, mfe3_R=np.nan, mfe5_R=np.nan),
        # week 2026-W44
        dict(signal_date='2026-10-26', symbol='FFF', setup='VCP', fill_status='waiting',
             adopted_status=np.nan, adopted_R=np.nan, mfe3_R=np.nan, mfe5_R=np.nan),
        dict(signal_date='2026-10-27', symbol='GGG', setup='POCKET_PIVOT', fill_status='filled',
             adopted_status=np.nan, adopted_R=np.nan, mfe3_R=np.nan, mfe5_R=np.nan),       # filled, unscored
        # after as_of: dropped
        dict(signal_date='2026-11-02', symbol='HHH', setup='VCP', fill_status='filled',
             adopted_status='closed', adopted_R=9.0, mfe3_R=np.nan, mfe5_R=np.nan),
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def write_conviction(path):
    rows = [
        dict(date='2026-10-20', symbol='AAA', conviction=5, setup='VCP', late=False, fill_status='filled',
             status='closed', R=2.0),                          # overlaps bot AAA in W43
        dict(date='2026-10-21', symbol='ZZZ', conviction=5, setup='VCP', late=False, fill_status='filled',
             status='open', R=-0.5),
        dict(date='2026-10-22', symbol='YYY', conviction=2, setup='VCP', late=True, fill_status='filled',
             status='closed', R=5.0),                          # late: excluded
        dict(date='2026-10-27', symbol='XXX', conviction=3, setup='VCP', late=False, fill_status='waiting',
             status=np.nan, R=np.nan),                         # unscored
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def ev(trade_date, symbol, role, state, **kw):
    ref = kw.pop('order_ref', None) or ol.make_order_ref(role, trade_date, symbol)
    return {'trade_date': trade_date, 'symbol': symbol, 'setup': 'VCP', 'role': role, 'state': state,
            'order_ref': ref, 'order_id': kw.pop('order_id', None), **kw}


def write_ledger(orders_dir):
    d1, d2, d3 = '2026-10-19', '2026-10-20', '2026-10-27'
    events = [
        # AAA: 100 @ 50, stop 48 (later moved to 50 and filled), trim 33 @ 54 -> R 0.66
        ev(d1, 'AAA', 'entry', 'filled', order_id=1, qty=100, filled_qty=100, avg_fill_price=50.0),
        ev(d1, 'AAA', 'stop', 'working', order_id=2, qty=100, aux_price=48.0),
        ev(d1, 'AAA', 'trim', 'filled', order_id=3, qty=33, filled_qty=33, avg_fill_price=54.0),
        ev(d1, 'AAA', 'stop', 'filled', order_id=2, qty=67, aux_price=50.0, filled_qty=67, avg_fill_price=50.0),
        ev(d1, 'AAA', 'trade', 'closed', reason='flat'),
        # BBB: closed but no exit fill in the ledger -> R unknown
        ev(d2, 'BBB', 'entry', 'filled', order_id=4, qty=10, filled_qty=10, avg_fill_price=20.0),
        ev(d2, 'BBB', 'stop', 'expired', order_id=5, qty=10, aux_price=19.0, reason='trade_closed'),
        ev(d2, 'BBB', 'trade', 'closed', reason='position_gone'),
        # CCC: open; DDD: never reached TWS
        ev(d3, 'CCC', 'entry', 'filled', order_id=6, qty=5, filled_qty=5, avg_fill_price=30.0),
        ev(d3, 'CCC', 'stop', 'working', order_id=7, qty=5, aux_price=28.0),
        ev(d3, 'DDD', 'entry', 'cancelled', order_id=8, qty=5, reason='never_reached_tws'),
        # paper_checks test orders: must be ignored entirely
        ev(d3, 'EEE', 'entry', 'filled', order_id=9, qty=1, filled_qty=1, avg_fill_price=10.0,
           order_ref='PAPERCHECK-A-ENTRY-x-EEE'),
        ev(d1, 'AAA', 'exit', 'filled', order_id=10, qty=1, filled_qty=1, avg_fill_price=99.0,
           order_ref='PAPERCHECK-B-SELL-x-AAA'),
    ]
    orders_dir.mkdir(parents=True, exist_ok=True)
    (orders_dir / 'ledger.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in events))


@pytest.fixture
def full(tmp_path):
    write_signals(tmp_path / 's.csv')
    write_conviction(tmp_path / 'c.csv')
    write_ledger(tmp_path / 'orders')
    return fr.build(AS_OF, 3, tmp_path / 's.csv', tmp_path / 'c.csv', tmp_path / 'orders')


def week(rep, label):
    return rep['weeks'].set_index('week').loc[label]


# ─── empty / missing ───

def test_all_inputs_missing_gives_zeros_not_a_crash(tmp_path):
    rep = fr.build(AS_OF, 2, tmp_path / 'none.csv', tmp_path / 'none2.csv', tmp_path / 'no_orders')
    a = week(rep, 'ALL')
    assert a['sig_new'] == 0 and a['conv_ideas'] == 0 and a['ord_entries'] == 0 and a['ord_closed'] == 0
    assert list(rep['weeks']['week']) == ['2026-W43', '2026-W44', 'ALL']
    assert rep['paper_status'].startswith(f'n=0 of ~{fr.TARGET_N}')
    text = fr.render_text(rep)
    assert 'No order ledger yet' in text and 'No conviction log yet' in text
    csv_path, md_path = fr.write_outputs(rep, tmp_path / 'out')
    assert csv_path.name == 'forward_report_2026-10-30.csv' and md_path.exists()
    assert len(pd.read_csv(csv_path)) == 3


def test_empty_files_are_handled(tmp_path):
    pd.DataFrame(columns=['signal_date', 'symbol', 'setup']).to_csv(tmp_path / 's.csv', index=False)
    pd.DataFrame(columns=['date', 'symbol', 'conviction', 'late']).to_csv(tmp_path / 'c.csv', index=False)
    (tmp_path / 'orders').mkdir()
    (tmp_path / 'orders' / 'ledger.jsonl').write_text('')
    rep = fr.build(AS_OF, 1, tmp_path / 's.csv', tmp_path / 'c.csv', tmp_path / 'orders')
    assert week(rep, 'ALL')['sig_new'] == 0 and week(rep, 'ALL')['ord_entries'] == 0
    fr.render_text(rep)
    fr.render_md(rep)


def test_corrupt_ledger_is_reported_not_silently_zero(tmp_path):
    (tmp_path / 'orders').mkdir()
    (tmp_path / 'orders' / 'ledger.jsonl').write_text('{bad\n{"a": 1}\n')
    rep = fr.build(AS_OF, 1, tmp_path / 's.csv', tmp_path / 'c.csv', tmp_path / 'orders')
    assert rep['ledger_error'] and 'LEDGER UNREADABLE' in fr.render_text(rep)


def test_conviction_log_without_scores_counts_ideas(tmp_path):
    pd.DataFrame([dict(date='2026-10-27', symbol='abc', conviction=4, setup='VCP', entry=1, stop=0.9,
                       reason='x', recorded_at='t', late=False)]).to_csv(tmp_path / 'c.csv', index=False)
    rep = fr.build(AS_OF, 1, tmp_path / 's.csv', tmp_path / 'c.csv', tmp_path / 'o')
    w = week(rep, '2026-W44')
    assert w['conv_ideas'] == 1 and w['conv_scored'] == 0 and math.isnan(w['conv_mean_R'])


# ─── weeks ───

def test_iso_week_bucketing():
    assert fr.iso_week('2026-10-19') == fr.iso_week('2026-10-23') == fr.iso_week('2026-10-25') == '2026-W43'
    assert fr.iso_week('2026-10-26') == '2026-W44'
    assert fr.iso_week('2026-12-31') == '2026-W53' and fr.iso_week('2027-01-04') == '2027-W01'
    assert fr.week_list(date(2026, 10, 28), 3) == ['2026-W42', '2026-W43', '2026-W44']
    assert fr.week_start('2026-10-25') == date(2026, 10, 19)


def test_signals_bucketed_by_week_and_as_of(full):
    w43, w44, a = week(full, '2026-W43'), week(full, '2026-W44'), week(full, 'ALL')
    assert w43['sig_new'] == 5 and w44['sig_new'] == 2 and a['sig_new'] == 7      # 11-02 row dropped
    assert w43['week_start'] == '2026-10-19'
    assert a['sig_setup_VCP'] == 3 and a['sig_setup_POCKET_PIVOT'] == 4
    assert a['sig_fill_filled'] == 5 and a['sig_fill_waiting'] == 1 and a['sig_fill_no_entry'] == 1


# ─── R stats ───

def test_signal_r_stats_leave_pending_out(full):
    w43, w44, a = week(full, '2026-W43'), week(full, '2026-W44'), week(full, 'ALL')
    assert (a['sig_closed'], a['sig_open'], a['sig_pending'], a['sig_no_trade']) == (3, 1, 2, 1)
    assert a['sig_closed_mean_R'] == pytest.approx(1.0 / 3)                 # 1.5, -1.0, 0.5
    assert a['sig_closed_median_R'] == pytest.approx(0.5)
    assert a['sig_closed_win_rate'] == pytest.approx(2 / 3)
    assert a['sig_closed_expectancy_R'] == pytest.approx(a['sig_closed_mean_R'])
    assert a['sig_open_mean_R'] == pytest.approx(0.3)
    assert a['sig_scored_mean_R'] == pytest.approx((1.5 - 1.0 + 0.5 + 0.3) / 4)
    assert a['sig_mfe3_mean_R'] == pytest.approx((1.0 + 0.2 + 0.5) / 3)
    assert a['sig_mfe5_mean_R'] == pytest.approx(1.2)
    assert w44['sig_pending'] == 2 and math.isnan(w44['sig_closed_mean_R'])


def test_r_stats_basics():
    s = fr.r_stats(pd.Series([2.0, -1.0, -1.0, np.nan]))
    assert s['n'] == 3 and s['win_rate'] == pytest.approx(1 / 3)
    assert s['avg_win'] == 2.0 and s['avg_loss'] == 1.0
    assert s['expectancy'] == pytest.approx(0.0) and s['mean'] == pytest.approx(0.0)
    assert fr.r_stats(pd.Series([], dtype=float))['n'] == 0


def test_conviction_by_level_and_overlap(full):
    bl = full['by_level'].set_index('conviction')
    assert bl.loc[5, 'ideas'] == 2 and bl.loc[5, 'mean_R'] == pytest.approx(0.75)
    assert bl.loc[2, 'ideas'] == 0                                              # late idea excluded
    assert bl.loc[3, 'ideas'] == 1 and bl.loc[3, 'scored'] == 0
    w43 = week(full, '2026-W43')
    assert w43['conv_ideas'] == 2 and w43['conv_late_excluded'] == 1 and w43['conv_overlap_bot'] == 1
    pv = full['owner_vs_bot']
    assert pv['overlap'] == 1
    assert pv['owner_only_n'] == 1 and pv['owner_only_mean_R'] == pytest.approx(-0.5)
    assert pv['owner_overlap_mean_R'] == pytest.approx(2.0)
    assert pv['bot_overlap_mean_R'] == pytest.approx(1.5)
    assert pv['bot_only_n'] == 3 and pv['bot_only_mean_R'] == pytest.approx((-1.0 + 0.3 + 0.5) / 3)


# ─── ledger ───

def test_ledger_trades_r_and_paper_check_exclusion(full):
    t = full['trades'].set_index('symbol')
    assert 'EEE' not in t.index                                                 # PAPERCHECK entry ignored
    assert t.loc['AAA', 'R'] == pytest.approx(0.66)                             # PAPERCHECK exit not counted
    assert t.loc['AAA', 'initial_stop'] == 48.0                                 # first stop, not the moved one
    assert t.loc['AAA', 'status'] == 'closed' and t.loc['AAA', 'r_note'] == 'gross of commissions'
    assert t.loc['BBB', 'status'] == 'closed' and math.isnan(t.loc['BBB', 'R'])
    assert t.loc['CCC', 'status'] == 'open'
    assert t.loc['DDD', 'status'] == 'no_fill' and not t.loc['DDD', 'placed']
    a = week(full, 'ALL')
    assert (a['ord_entries'], a['ord_placed'], a['ord_filled'], a['ord_closed'], a['ord_closed_with_R'],
            a['ord_open']) == (4, 3, 3, 2, 1, 1)
    assert a['ord_mean_R'] == pytest.approx(0.66)
    assert week(full, '2026-W44')['ord_entries'] == 2
    assert full['closed_no_r'] == 1 and 'without a derivable R' in fr.render_text(full)


def test_exit_state_closes_trade_and_supplies_stop(tmp_path):
    d = '2026-10-26'
    events = [ev(d, 'QQQ', 'entry', 'filled', order_id=1, qty=10, filled_qty=10, avg_fill_price=100.0),
              ev(d, 'QQQ', 'exit', 'filled', order_id=2, qty=10, filled_qty=10, avg_fill_price=90.0)]
    trades = fr.ledger_trades(events, {(d, 'QQQ'): {'initial_stop': 95.0, 'status': 'closed'}})
    assert trades.loc[0, 'status'] == 'closed' and trades.loc[0, 'R'] == pytest.approx(-2.0)


def test_manual_touched_has_no_r():
    d = '2026-10-26'
    events = [ev(d, 'QQQ', 'entry', 'filled', order_id=1, qty=10, filled_qty=10, avg_fill_price=100.0),
              ev(d, 'QQQ', 'trade', 'manual_touched')]
    t = fr.ledger_trades(events, {})
    assert t.loc[0, 'status'] == 'manual_touched' and math.isnan(t.loc[0, 'R'])


# ─── CI and status line ───

def test_mean_ci_uses_t_on_small_samples_and_z_above_30():
    n, m, lo, hi = fr.mean_ci(pd.Series([1.0, -1.0, 1.0, -1.0]))
    half = 3.18 * np.std([1, -1, 1, -1], ddof=1) / 2                         # t(3 df), not 1.96
    assert n == 4 and m == 0 and lo == pytest.approx(-half) and hi == pytest.approx(half)
    assert math.isnan(fr.mean_ci(pd.Series([0.5]))[2])
    assert fr.mean_ci(pd.Series([], dtype=float))[0] == 0
    big = pd.Series([1.0, -1.0] * 20)
    assert fr.mean_ci(big)[3] == pytest.approx(1.96 * big.std(ddof=1) / math.sqrt(40))


def test_status_never_claims_an_edge_below_target():
    assert fr.verdict_status(pd.Series([], dtype=float)) == f"n=0 of ~{fr.TARGET_N}: no closed trades with R yet, nothing to judge"
    small = fr.verdict_status(pd.Series([3.0] * 6 + [2.0] * 6))                 # huge mean, tight CI, n=12
    assert small.startswith(f'n=12 of ~{fr.TARGET_N}') and 'too few to judge' in small
    mid = fr.verdict_status(pd.Series([2.0, 1.0] * 40))                         # n=80, clearly positive
    assert 'interim harm check only' in mid and 'not enough trades to confirm an edge' in mid
    for s in (small, mid):
        assert 'meets' not in s and 'significant' not in s


def test_status_flags_interim_harm():
    bad = fr.verdict_status(pd.Series([-1.0, -0.5] * 30))                       # n=60, mean -0.75
    assert 'FAILS the interim harm check' in bad


def test_status_at_target_applies_the_verdict_bar():
    rng = np.random.default_rng(0)
    n = fr.TARGET_N + 4
    good = fr.verdict_status(pd.Series(rng.normal(0.5, 1.0, n)))
    assert good.startswith(f'n={n} of ~{fr.TARGET_N}') and 'meets the protocol' in good
    flat = fr.verdict_status(pd.Series(rng.normal(0.0, 1.5, n)))
    assert 'does NOT meet' in flat


def test_harm_check_uses_the_90pct_bootstrap_interval():
    r = pd.Series([-0.6] * 40 + [0.5] * 10)                                    # n=50, mean -0.38
    lo, hi = fr.bootstrap_ci(r)
    assert lo < r.mean() < hi < 0
    assert 'FAILS the interim harm check' in fr.verdict_status(r) and '90% bootstrap' in fr.verdict_status(r)
    assert fr.bootstrap_ci(r) == fr.bootstrap_ci(r)                            # fixed seed


def test_unscored_closed_trades_block_the_verdict():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.5, 1.0, fr.TARGET_N + 4))
    s = fr.verdict_status(r, unscored=3)
    assert 'no verdict yet' in s and '3 closed trade(s) have no R' in s and 'meets' not in s
    assert '1 closed trade(s) have no R' in fr.verdict_status(pd.Series([], dtype=float), unscored=1)


def test_shadow_sample_is_never_a_verdict():
    rng = np.random.default_rng(0)
    s = fr.verdict_status(pd.Series(rng.normal(0.5, 1.0, 200)), shadow=True)
    assert 'never a verdict' in s and 'meets' not in s


def test_mfe_counts_only_trades_whose_window_has_passed(tmp_path):
    pd.DataFrame({'signal_date': ['2026-10-19', '2026-10-27'], 'symbol': ['A', 'B'], 'setup': ['VCP', 'VCP'],
                  'fill_status': ['filled', 'filled'], 'adopted_status': ['closed', 'closed'],
                  'adopted_R': [1.0, -1.0], 'mfe3_R': [2.0, 0.0], 'mfe5_R': [2.5, 0.0]}).to_csv(tmp_path / 's.csv', index=False)
    df = fr.load_signals(tmp_path / 's.csv', as_of=date(2026, 10, 29))
    assert df['mfe3_R'].tolist()[0] == 2.0 and math.isnan(df['mfe3_R'].tolist()[1])   # B: a fast stop-out, too early


def test_main_writes_reports(tmp_path, monkeypatch, capsys):
    write_signals(tmp_path / 's.csv')
    monkeypatch.setattr(fr, 'SIGNAL_LOG', tmp_path / 's.csv')
    monkeypatch.setattr(fr, 'CONVICTION_LOG', tmp_path / 'none.csv')
    monkeypatch.setattr(ol, 'ORDERS_DIR', tmp_path / 'orders')
    assert fr.main(['--weeks', '2', '--as-of', '2026-10-30', '--out', str(tmp_path / 'rep')]) == 0
    out = capsys.readouterr().out
    assert 'PROGRESS TO A VERDICT' in out and '2026-W44' in out
    df = pd.read_csv(tmp_path / 'rep' / 'forward_report_2026-10-30.csv')
    assert list(df['week']) == ['2026-W43', '2026-W44', 'ALL'] and df['sig_new'].iloc[-1] == 7
    assert '# Forward report' in (tmp_path / 'rep' / 'forward_report_latest.md').read_text()
