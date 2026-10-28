"""
IBD Universe — ratings lookup from IBD's full data table export

Loads the newest ibd_universe_YYYY-MM-DD.csv (IBD "Data Tables" export,
~12,000 symbols) into a symbol -> ratings dict. Used as a lookup for the
stocks the scanner already analyzes, not as the scan universe: scanning
12,000 symbols would take hours under IBKR's request limits.

Name files with the export date so the scanner can tell how old they are.
"""

import glob
import os
import re
from datetime import date, datetime
from typing import Dict, Optional, Tuple

import pandas as pd

# IBD column -> stock field. Ratings are 1-99 unless noted.
COLUMNS = {
    'RS Rating': 'ibd_rs',
    'EPS Rating': 'ibd_eps_rating',              # earnings growth + stability
    'IBD Comp. Rating': 'ibd_composite_rating',
    'SMR Rating': 'ibd_smr_rating',              # sales, margins, ROE (A-E)
    'Acc/Dis Rating': 'ibd_acc_dist',            # institutional buying/selling (A-E)
    'Industry Group Rank': 'ibd_industry_rank',  # 1 = strongest of ~197 groups
    'Curr Qtr EPS Est. % Chg.': 'ibd_eps_est_qtr_pct',
    'Curr Yr EPS Est. % Chg.': 'ibd_eps_est_yr_pct',
}
GRADE_FIELDS = {'ibd_smr_rating', 'ibd_acc_dist'}


def _file_date(path: str) -> date:
    """Export date from the file name, else the file's modified time."""
    match = re.search(r"(\d{4}-\d{2}-\d{2})", os.path.basename(path))
    if match:
        return datetime.strptime(match.group(1), "%Y-%m-%d").date()
    return datetime.fromtimestamp(os.path.getmtime(path)).date()


def load_ibd_universe(pattern: str) -> Tuple[Dict[str, dict], Optional[date], Optional[str]]:
    """
    Newest file matching pattern -> ({symbol: {field: value}}, export date, path).
    Returns ({}, None, None) when no file matches.
    """
    paths = glob.glob(os.path.expanduser(pattern))
    if not paths:
        return {}, None, None
    path = max(paths, key=_file_date)

    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df = df[df['Symbol'].notna()]

    universe = {}
    for _, row in df.iterrows():
        ratings = {}
        for column, field in COLUMNS.items():
            value = row.get(column)
            if pd.isna(value):
                continue
            ratings[field] = str(value).strip() if field in GRADE_FIELDS else float(value)
        universe[str(row['Symbol']).strip().upper()] = ratings
    return universe, _file_date(path), path
