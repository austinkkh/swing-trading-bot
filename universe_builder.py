#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
UNIVERSE BUILDER — Dynamic Watchlist from IBD + TradingView Sources
─────────────────────────────────────────────────────────────────────────────
Replaces hardcoded STAGE2_WATCHLIST with a multi-source dynamic universe.

Sources supported:
  1. IBD .xlsx exports (IBD 50, Big Cap 20, Sector Leaders, etc.)
     - Auto-detects header row (skips IBD's title/description rows)
     - Extracts: Symbol, Rank, Industry Group Rank, EPS Rating, RS Rating,
       Composite Rating, Acc/Dis Rating, SMR Rating, Spon Rating, etc.
     
  2. Plain text symbol files (TradingView screener exports)
     - Format: comma-separated OR newline-separated symbols
     - Used for technical lists where you only need symbols

  3. Symbol CSV lists (TradingView export or manual list)
     - Requires one symbol-like column: Symbol/Ticker/Symbols/Tickers
      (or a single-column CSV)
     - Optional comma-separated values inside cells are supported

  4. Legacy IBD CSVs (old ibd_eps_*.csv lists): symbols only, their hand-typed values are ignored

Each symbol gets:
  - source_lists: which files mentioned this stock
  - merged metadata: best available data across sources
  - confidence boost: stocks in multiple lists rank higher

Usage:
    from universe_builder import UniverseBuilder
    
    builder = UniverseBuilder(data_dir='~/swing-trading-bot/Data')
    universe = builder.build()
    
    print(f"Total symbols: {len(universe.symbols)}")
    print(f"AAPL is in: {universe.source_lists.get('AAPL')}")
    print(f"AAPL IBD RS: {universe.metadata.get('AAPL', {}).get('ibd_rs_rating')}")
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import logging
import re
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# DATA STRUCTURE
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class Universe:
    """The merged universe of all watchlist sources"""
    symbols: List[str] = field(default_factory=list)
    metadata: Dict[str, Dict] = field(default_factory=dict)  # symbol -> attributes
    source_lists: Dict[str, List[str]] = field(default_factory=dict)  # symbol -> [source names]
    sources_loaded: List[str] = field(default_factory=list)
    build_timestamp: Optional[datetime] = None
    rejected_count: int = 0
    
    def get_metadata(self, symbol: str) -> Dict:
        """Get all known data for a symbol"""
        return self.metadata.get(symbol, {})
    
    def get_sources(self, symbol: str) -> List[str]:
        """Get list of source files this symbol appeared in"""
        return self.source_lists.get(symbol, [])
    
    def get_overlap_count(self, symbol: str) -> int:
        """How many sources contain this symbol (higher = stronger conviction)"""
        return len(self.source_lists.get(symbol, []))

    @staticmethod
    def _safe_float(value: object, default: Optional[float] = None) -> Optional[float]:
        if value is None:
            return default
        if not pd.api.types.is_scalar(value):
            return default
        try:
            if pd.isna(value):
                return default
        except TypeError:
            pass

        if isinstance(value, str):
            value = value.strip().replace(',', '').rstrip('%')

        try:
            numeric = pd.to_numeric(value, errors='coerce')
        except (TypeError, ValueError, OverflowError):
            return default
        if pd.isna(numeric):
            return default
        return float(numeric)

    def filter_by_min_overlap(self, min_count: int) -> List[str]:
        """Return symbols appearing in at least N source lists"""
        return [s for s in self.symbols if self.get_overlap_count(s) >= min_count]
    
    def filter_by_ibd_rs(self, min_rs: int) -> List[str]:
        """Return symbols with IBD RS >= threshold"""
        return [
            s for s in self.symbols
            if self._safe_float(self.metadata.get(s, {}).get('ibd_rs_rating'), default=-1) >= min_rs
        ]
    
    def summary(self) -> str:
        """Pretty summary"""
        lines = [
            f"Universe build summary",
            f"  Total unique symbols: {len(self.symbols)}",
            f"  Sources loaded: {len(self.sources_loaded)}",
        ]
        for src in self.sources_loaded:
            count = sum(1 for s in self.symbols if src in self.source_lists.get(s, []))
            lines.append(f"    • {src}: {count} symbols")
        
        # Overlap distribution
        lines.append(f"  Overlap distribution:")
        for n in range(1, 6):
            count = len([s for s in self.symbols if self.get_overlap_count(s) == n])
            if count > 0:
                lines.append(f"    {n} list(s): {count} stocks")
        more = len([s for s in self.symbols if self.get_overlap_count(s) >= 6])
        if more > 0:
            lines.append(f"    6+ lists: {more} stocks")
        
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# SOURCE CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════
# Primary sources: stocks here enter the universe (if they pass industry filter)
# Substring matching — so "IBD_50" matches "IBD_50-2.xlsx" etc.
PRIMARY_SOURCES = ['IBD_50', 'New_Highs', 'Relative_Strength_At_New_High']

# Enrichment sources: add metadata to existing symbols but don't add new ones
ENRICHMENT_SOURCES = [
    'IBD_Big_Cap_20', 'IBD_Sector_Leaders', 
    'Stock_Spotlight', 'Rising_Profit_Estimates'
]

# Filter thresholds
MAX_INDUSTRY_RANK = 20              # Hard cap for fundamentals-based stocks
ELITE_INDUSTRY_RANK = 10            # Priority boost threshold

DEFAULT_SCORE_WEIGHTS = {
    'rs_at_new_high': 5,      # In RS_At_New_High (strongest signal)
    'ibd_50': 4,              # In IBD 50 (editorial flagship)
    'new_highs': 3,           # In New Highs
    'elite_group': 5,         # Industry rank 1-10
    'strong_group': 3,        # Industry rank 11-20
    'multi_primary': 3,       # In 3+ primary sources
    'rising_profits': 2,      # In Rising Profit Estimates
}

SYMBOL_PATTERN = re.compile(r'^[A-Z]{1,6}(?:\.[A-Z]{1,2})?$')

# TradingView source prefix (text files, exempt from industry filter)
TV_SOURCE_PREFIX = 'tv_'


# ═══════════════════════════════════════════════════════════════════════════
# UNIVERSE BUILDER
# ═══════════════════════════════════════════════════════════════════════════
class UniverseBuilder:
    """
    Builds a unified watchlist from multiple data sources.
    
    Drop your files in the data directory and they get auto-detected:
      - *.xlsx     → IBD screener export
      - tv_*.txt   → TradingView symbol list
      - tv_*.csv   → TradingView symbol CSV list
      - ibd_*.csv  → Legacy IBD CSV (existing format)
      - *.txt      → Generic symbol list
    """
    
    def __init__(self, data_dir: str = '~/swing-trading-bot/Data',
                 max_industry_rank: int = MAX_INDUSTRY_RANK,
                 apply_industry_filter: bool = True,
                 elite_industry_rank: int = ELITE_INDUSTRY_RANK,
                 score_weights: Optional[Dict[str, int]] = None):
        self.data_dir = Path(os.path.expanduser(data_dir))
        self.max_industry_rank = max_industry_rank
        self.elite_industry_rank = elite_industry_rank
        self.score_weights = {**DEFAULT_SCORE_WEIGHTS, **(score_weights or {})}
        self.apply_industry_filter = apply_industry_filter
        self.universe = Universe()
    
    def build(self) -> Universe:
        """Main entry point — scan data dir, parse all sources, merge"""
        if not self.data_dir.exists():
            logger.warning(f"⚠️  Data directory not found: {self.data_dir}")
            self.data_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"   Created empty data dir — add files there")
            return self.universe
        
        logger.info(f"Building universe from {self.data_dir}")
        
        # Process xlsx files (IBD screener exports)
        for xlsx_path in sorted(self.data_dir.glob("*.xlsx")):
            try:
                self._ingest_ibd_xlsx(xlsx_path)
            except Exception as e:
                logger.error(f"❌ Failed to parse {xlsx_path.name}: {e}")
        
        # Process tv_*.txt files (TradingView symbol lists)
        for txt_path in sorted(self.data_dir.glob("tv_*.txt")):
            try:
                self._ingest_symbol_list(txt_path)
            except Exception as e:
                logger.error(f"❌ Failed to parse {txt_path.name}: {e}")

        # Process tv_*.csv files (TradingView symbol CSV exports)
        for csv_path in sorted(self.data_dir.glob("tv_*.csv")):
            try:
                self._ingest_symbol_csv(csv_path)
            except Exception as e:
                logger.error(f"❌ Failed to parse {csv_path.name}: {e}")
        
        # Process generic txt files (catch-all symbol lists)
        for txt_path in sorted(self.data_dir.glob("*.txt")):
            if txt_path.name.startswith("tv_"):
                continue  # Already processed
            try:
                self._ingest_symbol_list(txt_path)
            except Exception as e:
                logger.error(f"❌ Failed to parse {txt_path.name}: {e}")
        
        # Process legacy IBD csvs (your existing format)
        for csv_path in sorted(self.data_dir.glob("ibd_eps_*.csv")):
            try:
                self._ingest_legacy_ibd_csv(csv_path)
            except Exception as e:
                logger.error(f"❌ Failed to parse {csv_path.name}: {e}")

        # Process generic CSV symbol lists (excluding known legacy prefixes)
        for csv_path in sorted(self.data_dir.glob("*.csv")):
            # ibd_universe_*.csv is a ratings lookup (ibd_universe.py), not a list of stocks to scan
            if csv_path.name.startswith(("tv_", "ibd_eps_", "ibd_universe_")):
                continue
            try:
                self._ingest_symbol_csv(csv_path)
            except Exception as e:
                logger.error(f"❌ Failed to parse {csv_path.name}: {e}")
        
        # Finalize raw universe
        self.universe.symbols = sorted(set(self.universe.symbols))
        
        # Apply filtering and scoring
        if self.apply_industry_filter:
            self._filter_and_score()
        else:
            self._score_only()
        
        self.universe.build_timestamp = datetime.now()
        
        logger.info(self.universe.summary())
        return self.universe
    
    # ─────────────────────────────────────────────────────────────────
    # SOURCE CLASSIFICATION HELPERS
    # ─────────────────────────────────────────────────────────────────
    def _is_primary_source(self, source_name: str) -> bool:
        """Check if source is in primary list (substring match)"""
        return self._source_matches(source_name, PRIMARY_SOURCES)
    
    def _is_tv_source(self, source_name: str) -> bool:
        """TradingView symbol files — always included, no industry filter"""
        return source_name.startswith(TV_SOURCE_PREFIX)

    @staticmethod
    def _normalize_symbol(symbol: object) -> str:
        if symbol is None:
            return ''
        try:
            if pd.isna(symbol):
                return ''
        except TypeError:
            pass
        return str(symbol).strip().upper().strip('"').strip("'").rstrip(",;:")

    @staticmethod
    def _safe_float(value: object) -> Optional[float]:
        if value is None:
            return None
        try:
            if pd.isna(value):
                return None
        except TypeError:
            pass

        if isinstance(value, str):
            value = value.strip().replace(',', '').rstrip('%')

        try:
            numeric = pd.to_numeric(value, errors='coerce')
        except Exception:
            return None

        if pd.isna(numeric):
            return None
        return float(numeric)

    @staticmethod
    def _source_matches(source_name: str, patterns: List[str]) -> bool:
        return any(pattern in source_name for pattern in patterns)

    def _has_source(self, sources: List[str], patterns: List[str]) -> bool:
        return any(self._source_matches(source, patterns) for source in sources)

    def _get_primary_sources(self, symbol: str) -> list:
        """Return list of primary sources this symbol appears in"""
        return [s for s in self.universe.source_lists.get(symbol, [])
                if self._is_primary_source(s)]
    
    # ─────────────────────────────────────────────────────────────────
    # FILTER & SCORE
    # ─────────────────────────────────────────────────────────────────
    def _filter_and_score(self):
        """
        Apply the filtering rules and compute priority scores.
        
        Rules:
          - TV sources: always included (no filter)
          - Primary sources with industry_rank <= MAX: included
          - Primary sources with no rank or rank > MAX: rejected
          - Enrichment-only sources: rejected (add metadata to existing stocks only)
        """
        filtered = []
        rejected_no_primary = []
        rejected_high_industry = []
        
        for symbol in self.universe.symbols:
            sources = self.universe.source_lists.get(symbol, [])
            meta = self.universe.metadata.get(symbol, {})

            is_allowed, inclusion_reason, rejection_bucket = self._evaluate_symbol(symbol, sources, meta)
            if not is_allowed:
                if rejection_bucket == 'no_primary':
                    rejected_no_primary.append(symbol)
                else:
                    rejected_high_industry.append((symbol, meta.get('industry_rank')))
                continue

            filtered.append(symbol)
            meta['universe_score'] = self._compute_score(symbol)
            meta['inclusion_reason'] = inclusion_reason
        
        # Update universe with filtered set
        rejected_count = len(rejected_no_primary) + len(rejected_high_industry)
        logger.info(f"Filter results:")
        logger.info(f"  Accepted:                    {len(filtered)} stocks")
        logger.info(f"  Rejected (no primary src):   {len(rejected_no_primary)}")
        logger.info(f"  Rejected (industry > {self.max_industry_rank}):   {len(rejected_high_industry)}")
        
        # Rebuild universe with filtered list, preserving metadata for rejected stocks
        self.universe.symbols = sorted(filtered)
        self.universe.rejected_count = rejected_count
        
        # Sort filtered symbols by priority score
        self.universe.symbols.sort(
            key=lambda s: self.universe.metadata.get(s, {}).get('universe_score', 0),
            reverse=True
        )

    def _evaluate_symbol(self, symbol: str, sources: List[str], meta: Dict) -> tuple[bool, str, Optional[str]]:
        """Return (allowed, inclusion_reason, rejection_bucket)."""
        if any(self._is_tv_source(source) for source in sources):
            return True, 'tradingview', None

        if not self._get_primary_sources(symbol):
            return False, '', 'no_primary'

        ind_rank = self._safe_float(meta.get('industry_rank'))
        if ind_rank is None or ind_rank > self.max_industry_rank:
            return False, '', 'high_industry'

        return True, 'primary_filtered', None
    
    def _score_only(self):
        """Compute scores without filtering (for debug/analysis mode)"""
        for symbol in self.universe.symbols:
            meta = self.universe.metadata.get(symbol, {})
            meta['universe_score'] = self._compute_score(symbol)
    
    def _compute_score(self, symbol: str) -> int:
        """
        Priority score based on source attribution and industry rank.
        Higher = stronger conviction.
        """
        sources = self.universe.source_lists.get(symbol, [])
        meta = self.universe.metadata.get(symbol, {})
        score = 0

        has_rs_new_high = self._has_source(sources, ['Relative_Strength_At_New_High'])
        has_ibd_50 = self._has_source(sources, ['IBD_50'])
        has_new_highs = self._has_source(sources, ['New_Highs'])
        has_rising_profits = self._has_source(sources, ['Rising_Profit'])

        if has_rs_new_high:
            score += self.score_weights['rs_at_new_high']
        if has_ibd_50:
            score += self.score_weights['ibd_50']
        if has_new_highs and not has_rs_new_high:
            # Don't double-count RS_At_New_High as "New Highs"
            score += self.score_weights['new_highs']

        ind_rank = self._safe_float(meta.get('industry_rank'))
        if ind_rank is not None:
            if ind_rank <= self.elite_industry_rank:
                score += self.score_weights['elite_group']
            elif ind_rank <= self.max_industry_rank:
                score += self.score_weights['strong_group']

        primary_count = len(self._get_primary_sources(symbol))
        if primary_count >= 3:
            score += self.score_weights['multi_primary']

        if has_rising_profits:
            score += self.score_weights['rising_profits']
        
        return score
    
    # ─────────────────────────────────────────────────────────────────
    # IBD XLSX PARSER (handles IBD 50, Big Cap 20, Sector Leaders, etc.)
    # ─────────────────────────────────────────────────────────────────
    def _ingest_ibd_xlsx(self, path: Path):
        """
        Parse an IBD screener xlsx export.
        
        Format challenges:
          - First few rows are title/description text, not data
          - Header row appears at varying positions
          - Last few rows are footer/copyright text
          - Some rows may have missing fields
        
        Strategy: scan for the row containing 'Symbol' as first cell,
        use that as header, parse data below until first empty/footer row.
        """
        source_name = path.stem  # filename without .xlsx
        
        try:
            xl = pd.ExcelFile(path)
        except Exception as e:
            logger.error(f"❌ Cannot open {path.name}: {e}")
            return
        
        # Try the most likely sheet name first, then others
        sheet_names = xl.sheet_names
        target_sheet = None
        for name in ['Stock list', 'Stock List', 'Sheet1', 'Data']:
            if name in sheet_names:
                target_sheet = name
                break
        if target_sheet is None and sheet_names:
            target_sheet = sheet_names[0]
        
        # Read raw without header
        raw = pd.read_excel(path, sheet_name=target_sheet, header=None)
        
        # Find the header row — first row where col 0 is exactly 'Symbol'
        header_idx = None
        for i, row in raw.iterrows():
            if str(row.iloc[0]).strip() == 'Symbol':
                header_idx = i
                break
        
        if header_idx is None:
            logger.warning(f"⚠️  {path.name}: no 'Symbol' header found")
            return
        
        # Re-read with the discovered header row
        df = pd.read_excel(path, sheet_name=target_sheet, header=header_idx)
        
        # Normalize and validate symbols early so downstream merging is consistent.
        df['Symbol'] = df['Symbol'].apply(self._normalize_symbol)
        df = df[df['Symbol'].apply(self._is_valid_symbol)]
        
        if df.empty:
            logger.warning(f"⚠️  {path.name}: no valid stock rows")
            return
        
        # Map IBD column names to standardized internal names
        col_map = self._build_ibd_column_map(df.columns)
        
        count = 0
        for _, row in df.iterrows():
            symbol = self._normalize_symbol(row['Symbol'])
            if not symbol:
                continue
            
            metadata = self._extract_ibd_metadata(row, col_map)
            self._add_symbol(symbol, source_name, metadata)
            count += 1
        
        self.universe.sources_loaded.append(source_name)
        logger.info(f"✓ {path.name}: {count} symbols from IBD xlsx")
    
    def _build_ibd_column_map(self, columns: list) -> dict:
        """
        Map IBD column names to our standard internal names.
        IBD column names vary slightly across screen exports, so we do
        case-insensitive substring matching.
        """
        col_map = {}
        col_lower = {col: str(col).lower().strip() for col in columns}
        
        # Find each canonical field by looking for keywords
        for col, lower in col_lower.items():
            # Skip the 'Symbol' column itself
            if lower == 'symbol':
                continue
            
            if 'rank' in lower and 'industry' not in lower and 'group' not in lower:
                col_map['ibd_rank'] = col
            elif 'industry' in lower and 'rank' in lower:
                col_map['industry_rank'] = col
            elif 'industry' in lower and 'group' in lower and 'rank' in lower:
                col_map['industry_rank'] = col
            elif 'curr yr eps' in lower or 'curr year eps' in lower:
                col_map['eps_year_growth'] = col
            elif 'curr qtr eps' in lower or 'current qtr eps' in lower:
                col_map['eps_qtr_growth'] = col
            elif lower == 'price':
                col_map['price'] = col
            elif 'comp' in lower and 'rating' in lower:
                col_map['ibd_composite_rating'] = col
            elif 'eps rating' in lower or lower == 'eps':
                col_map['ibd_eps_rating'] = col
            elif lower == 'rs rating' or lower == 'rs':
                col_map['ibd_rs_rating'] = col
            elif 'ind grp rs' in lower or ('industry' in lower and 'rs' in lower):
                col_map['ind_group_rs_grade'] = col
            elif 'smr' in lower:
                col_map['ibd_smr_rating'] = col
            elif 'acc' in lower and ('dis' in lower or '/' in lower):
                col_map['ibd_acc_dist'] = col
            elif 'spon' in lower:
                col_map['ibd_sponsorship'] = col
            elif 'vol' in lower and '%' in lower:
                col_map['volume_change_pct'] = col
        
        return col_map

    @staticmethod
    def _is_valid_symbol(symbol: object) -> bool:
        return bool(symbol) and bool(SYMBOL_PATTERN.fullmatch(str(symbol)))
    
    def _extract_ibd_metadata(self, row, col_map: dict) -> dict:
        """Pull values from row using the column map"""
        metadata = {}
        
        for std_name, ibd_col in col_map.items():
            if ibd_col not in row.index:
                continue
            value = row[ibd_col]
            
            # Skip NaN
            if pd.isna(value):
                continue
            
            # Numeric fields
            if std_name in ('ibd_rank', 'industry_rank', 'ibd_composite_rating',
                           'ibd_eps_rating', 'ibd_rs_rating', 'eps_year_growth',
                           'eps_qtr_growth', 'volume_change_pct', 'price'):
                parsed = self._safe_float(value)
                if parsed is not None:
                    metadata[std_name] = parsed
            else:
                # Letter grade fields (A+, A, B-, etc.)
                metadata[std_name] = str(value).strip()
        
        return metadata
    
    # ─────────────────────────────────────────────────────────────────
    # SYMBOL LIST PARSER (TradingView, generic text)
    # ─────────────────────────────────────────────────────────────────
    @staticmethod
    def _normalize_symbol_token(raw_token: str) -> Optional[str]:
        token = str(raw_token).strip().strip('"').strip("'").upper()
        if not token:
            return None
        if 1 <= len(token) <= 6 and token.replace('.', '').isalpha():
            return token
        return None

    def _ingest_symbol_list(self, path: Path):
        """
        Parse a plain text file of symbols.
        Supports:
          - Comma-separated: AAPL,MSFT,NVDA
          - Newline-separated: one per line
          - Mixed (handles both)
          - Comments: lines starting with # are ignored
          - Quotes are stripped
        """
        source_name = path.stem
        
        with open(path, 'r') as f:
            content = f.read()
        
        # Strip comments
        lines = []
        for line in content.split('\n'):
            line = line.split('#')[0].strip()  # Remove inline comments
            if line:
                lines.append(line)
        
        # Try to extract symbols from each line (handles comma-separated within lines)
        symbols = set()
        invalid_count = 0
        for line in lines:
            for token in line.replace(',', ' ').replace('\t', ' ').split():
                normalized = self._normalize_symbol_token(token)
                if normalized:
                    symbols.add(normalized)
                else:
                    invalid_count += 1
        
        if not symbols:
            logger.warning(f"⚠️  {path.name}: no valid symbols found")
            return
        
        for symbol in symbols:
            self._add_symbol(symbol, source_name, metadata={})
        
        self.universe.sources_loaded.append(source_name)
        logger.info(f"✓ {path.name}: {len(symbols)} symbols from text list")
        if invalid_count:
            logger.warning(f"⚠️  {path.name}: ignored {invalid_count} invalid token(s)")

    def _ingest_symbol_csv(self, path: Path):
        """
        Parse a symbol CSV list.
        Accepted contract:
          - Has one of: Symbol, Symbols, Ticker, Tickers (case-insensitive), OR
          - Single-column CSV where that column is treated as symbols.
        """
        source_name = path.stem

        try:
            df = pd.read_csv(path, dtype=str)
        except Exception as e:
            logger.error(f"❌ Cannot read {path.name}: {e}")
            return

        if df.empty:
            logger.warning(f"⚠️  {path.name}: empty CSV")
            return

        normalized_cols = {str(c).strip().lower(): c for c in df.columns}
        symbol_col = None
        for candidate in ("symbol", "symbols", "ticker", "tickers"):
            if candidate in normalized_cols:
                symbol_col = normalized_cols[candidate]
                break

        if symbol_col is None:
            if len(df.columns) == 1:
                symbol_col = df.columns[0]
            else:
                logger.warning(
                    f"⚠️  {path.name}: no symbol column found "
                    f"(expected Symbol/Ticker). Skipping."
                )
                return

        symbols = set()
        invalid_count = 0
        for value in df[symbol_col].dropna():
            for token in str(value).replace(',', ' ').replace('\t', ' ').split():
                normalized = self._normalize_symbol_token(token)
                if normalized:
                    symbols.add(normalized)
                else:
                    invalid_count += 1

        if not symbols:
            logger.warning(f"⚠️  {path.name}: no valid symbols found")
            return

        for symbol in symbols:
            self._add_symbol(symbol, source_name, metadata={})

        self.universe.sources_loaded.append(source_name)
        logger.info(f"✓ {path.name}: {len(symbols)} symbols from CSV list ({symbol_col})")
        if invalid_count:
            logger.warning(f"⚠️  {path.name}: ignored {invalid_count} invalid token(s)")
    
    # ─────────────────────────────────────────────────────────────────
    # LEGACY IBD CSV PARSER (old ibd_eps_*.csv lists: symbols only)
    # ─────────────────────────────────────────────────────────────────
    def _ingest_legacy_ibd_csv(self, path: Path):
        """
        Old hand-typed IBD EPS lists (ibd_eps_*.csv): symbols only. Their EPS / RS / rank values
        are no longer used: EPS growth comes from the IBD universe
        export's current-year estimate, so a months-old hand-typed number can't make a stock elite.
        """
        source_name = path.stem
        try:
            df = pd.read_csv(path)
        except Exception as e:
            logger.error(f"❌ Cannot read {path.name}: {e}")
            return
        if 'Symbol' not in df.columns:
            logger.warning(f"⚠️  {path.name}: no 'Symbol' column")
            return
        for raw in df['Symbol']:
            if pd.isna(raw):
                continue
            symbol = self._normalize_symbol(raw)
            if symbol:
                self._add_symbol(symbol, source_name, {})
        self.universe.sources_loaded.append(source_name)
        logger.info(f"✓ {path.name}: {len(df)} symbols from legacy CSV (symbols only, values ignored)")

    # ─────────────────────────────────────────────────────────────────
    # MERGE LOGIC
    # ─────────────────────────────────────────────────────────────────
    def _add_symbol(self, symbol: str, source_name: str, metadata: dict):
        """
        Add a symbol from a source. Merges metadata across sources.
        
        Merge rules:
          - If symbol is new, add it with this metadata
          - If symbol exists, merge metadata (newer data wins on conflict)
          - Always append source to source_lists
        """
        if symbol not in self.universe.metadata:
            self.universe.metadata[symbol] = {}
            self.universe.source_lists[symbol] = []
            if symbol not in self.universe.symbols:
                self.universe.symbols.append(symbol)
        
        # Merge metadata — non-empty values overwrite empty ones
        for key, value in metadata.items():
            existing = self.universe.metadata[symbol].get(key)
            if existing is None or existing == '' or pd.isna(existing):
                self.universe.metadata[symbol][key] = value
            elif value is not None and value != '' and not pd.isna(value):
                # Both have values — keep the newer (current) one
                self.universe.metadata[symbol][key] = value
        
        # Track source
        if source_name not in self.universe.source_lists[symbol]:
            self.universe.source_lists[symbol].append(source_name)


# ═══════════════════════════════════════════════════════════════════════════
# STANDALONE TEST
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    print("\n" + "="*78)
    print("  UNIVERSE BUILDER - STANDALONE TEST")
    print("="*78)
    
    data_dir = '~/swing-trading-bot/Data'
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    
    builder = UniverseBuilder(data_dir=data_dir)
    universe = builder.build()
    
    print()
    print(universe.summary())
    
    if universe.symbols:
        # Top 20 by score
        print(f"\n{'─'*78}")
        print(f"  TOP 20 BY PRIORITY SCORE (highest conviction first)")
        print(f"{'─'*78}")
        print(f"  {'Symbol':<8} {'Score':<6} {'IndRnk':<7} {'IBD_RS':<7} {'IBD_Comp':<9} {'Sources'}")
        print(f"  {'-'*78}")
        
        for sym in universe.symbols[:20]:
            meta = universe.get_metadata(sym)
            score = meta.get('universe_score', 0)
            ind_rank = int(meta.get('industry_rank')) if meta.get('industry_rank') else '-'
            ibd_rs = int(meta.get('ibd_rs_rating')) if meta.get('ibd_rs_rating') else '-'
            ibd_comp = int(meta.get('ibd_composite_rating')) if meta.get('ibd_composite_rating') else '-'
            
            # Shorten source names for display
            sources = universe.get_sources(sym)
            short_sources = []
            for s in sources:
                if 'IBD_50' in s: short_sources.append('IBD50')
                elif 'Relative_Strength_At_New_High' in s: short_sources.append('RS@NH')
                elif 'New_Highs' in s: short_sources.append('NH')
                elif 'Big_Cap' in s: short_sources.append('BigCap')
                elif 'Sector_Leaders' in s: short_sources.append('SectLead')
                elif 'Spotlight' in s: short_sources.append('Spotlight')
                elif 'Rising_Profit' in s: short_sources.append('RisingP')
                elif s.startswith('tv_'): short_sources.append(f'TV:{s[3:]}')
                else: short_sources.append(s[:10])
            
            print(f"  {sym:<8} {score:<6} {ind_rank!s:<7} {ibd_rs!s:<7} {ibd_comp!s:<9} {','.join(short_sources)}")
        
        if len(universe.symbols) > 20:
            print(f"  ... +{len(universe.symbols) - 20} more")
        
        # Industry group distribution
        print(f"\n{'─'*78}")
        print(f"  INDUSTRY GROUP DISTRIBUTION")
        print(f"{'─'*78}")
        ind_groups = {}
        for sym in universe.symbols:
            rank = universe.metadata.get(sym, {}).get('industry_rank')
            if rank:
                rank = int(rank)
                ind_groups.setdefault(rank, []).append(sym)
        
        for rank in sorted(ind_groups.keys()):
            syms = ind_groups[rank]
            badge = '⭐ ELITE' if rank <= 10 else '✓ STRONG'
            syms_str = ', '.join(syms[:8])
            if len(syms) > 8:
                syms_str += f' (+{len(syms)-8})'
            print(f"  Rank {rank:>3} {badge:<10} ({len(syms):2}): {syms_str}")
    
    print("\n" + "="*78 + "\n")
