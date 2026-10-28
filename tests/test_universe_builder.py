import unittest

import pandas as pd

from universe_builder import UniverseBuilder


class UniverseBuilderTests(unittest.TestCase):
    def test_universe_filter_by_ibd_rs_parses_numeric_metadata(self):
        from universe_builder import Universe

        universe = Universe(
            symbols=['AAA', 'BBB', 'CCC', 'DDD'],
            metadata={
                'AAA': {'ibd_rs_rating': '95'},
                'BBB': {'ibd_rs_rating': 88.0},
                'CCC': {'ibd_rs_rating': 'N/A'},
                'DDD': {'ibd_rs_rating': None},
            },
        )

        self.assertEqual(universe.filter_by_ibd_rs(90), ['AAA'])
        self.assertEqual(universe.filter_by_ibd_rs(85), ['AAA', 'BBB'])

    def test_universe_filter_by_ibd_rs_skips_collection_values(self):
        from universe_builder import Universe

        universe = Universe(
            symbols=['AAA', 'BBB'],
            metadata={
                'AAA': {'ibd_rs_rating': []},
                'BBB': {'ibd_rs_rating': [95]},
            },
        )

        self.assertEqual(universe.filter_by_ibd_rs(90), [])

    def test_symbol_normalization_accepts_class_shares(self):
        builder = UniverseBuilder(data_dir='.')

        self.assertEqual(builder._normalize_symbol(' "brk.b" '), 'BRK.B')
        self.assertEqual(builder._normalize_symbol(float('nan')), '')
        self.assertTrue(builder._is_valid_symbol('BRK.B'))
        self.assertFalse(builder._is_valid_symbol('BRK-B'))

    def test_safe_float_and_metadata_parsing_skip_invalid_values(self):
        builder = UniverseBuilder(data_dir='.')
        row = pd.Series(
            {
                'Rank': '12',
                'Industry Rank': 'N/A',
                'RS Rating': '95',
                'Comp Rating': 'bad',
                'Price': '1,234.50',
            }
        )
        col_map = {
            'ibd_rank': 'Rank',
            'industry_rank': 'Industry Rank',
            'ibd_rs_rating': 'RS Rating',
            'ibd_composite_rating': 'Comp Rating',
            'price': 'Price',
        }

        metadata = builder._extract_ibd_metadata(row, col_map)

        self.assertEqual(metadata['ibd_rank'], 12.0)
        self.assertEqual(metadata['ibd_rs_rating'], 95.0)
        self.assertEqual(metadata['price'], 1234.5)
        self.assertNotIn('industry_rank', metadata)
        self.assertNotIn('ibd_composite_rating', metadata)

    def test_filtering_and_scoring_use_tuneable_thresholds(self):
        builder = UniverseBuilder(
            data_dir='.',
            max_industry_rank=20,
            apply_industry_filter=True,
            elite_industry_rank=5,
            score_weights={
                'rs_at_new_high': 1,
                'ibd_50': 1,
                'new_highs': 1,
                'elite_group': 7,
                'strong_group': 4,
                'multi_primary': 2,
                'rising_profits': 1,
            },
        )

        builder.universe.symbols = ['TVA', 'PRIM', 'BAD']
        builder.universe.source_lists = {
            'TVA': ['tv_watch.txt'],
            'PRIM': ['IBD_50.xlsx', 'New_Highs.xlsx'],
            'BAD': ['IBD_50.xlsx'],
        }
        builder.universe.metadata = {
            'TVA': {},
            'PRIM': {'industry_rank': 4},
            'BAD': {'industry_rank': 21},
        }

        builder._filter_and_score()

        self.assertEqual(builder.universe.symbols, ['PRIM', 'TVA'])
        self.assertEqual(builder.universe.rejected_count, 1)
        self.assertEqual(builder.universe.metadata['TVA']['inclusion_reason'], 'tradingview')
        self.assertEqual(builder.universe.metadata['PRIM']['inclusion_reason'], 'primary_filtered')
        self.assertEqual(builder.universe.metadata['PRIM']['universe_score'], 9)

    def test_legacy_positional_constructor_keeps_apply_industry_filter_third(self):
        builder = UniverseBuilder('~/swing-trading-bot/Data', 15, False)

        self.assertEqual(builder.max_industry_rank, 15)
        self.assertFalse(builder.apply_industry_filter)
        self.assertEqual(builder.elite_industry_rank, 10)


if __name__ == '__main__':
    unittest.main()


def test_ibd_universe_file_is_not_a_symbol_source(tmp_path):
    """The IBD data-table export is a ratings lookup; its ~12k symbols must not join the scan universe."""
    import pandas as pd
    from universe_builder import UniverseBuilder
    pd.DataFrame({"Symbol": ["AAA", "BBB"], "RS Rating": [99, 98]}).to_csv(
        tmp_path / "ibd_universe_2026-10-22.csv", index=False)
    pd.DataFrame({"Symbol": ["CCC"]}).to_csv(tmp_path / "my_list.csv", index=False)
    builder = UniverseBuilder(data_dir=str(tmp_path), apply_industry_filter=False)
    symbols = builder.build().symbols
    assert "CCC" in symbols
    assert "AAA" not in symbols and "BBB" not in symbols


def test_legacy_eps_list_adds_symbols_but_not_its_hand_typed_values(tmp_path):
    """Owner decision #5: old ibd_eps_*.csv values (EPS, RS, rank) are ignored; symbols still scanned."""
    path = tmp_path / "ibd_eps_40plus.csv"
    pd.DataFrame({"Symbol": ["CIEN"], "EPS_YoY": [102], "EPS_CurrQ": [81], "RS_Rating": [99],
                  "Industry_Group_Rank": [1]}).to_csv(path, index=False)
    builder = UniverseBuilder(data_dir=str(tmp_path))
    builder._ingest_legacy_ibd_csv(path)
    assert "CIEN" in builder.universe.symbols
    meta = builder.universe.metadata.get("CIEN", {})
    assert not {"eps_year_growth", "eps_qtr_growth", "ibd_rs_rating", "industry_rank"} & set(meta)
