#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════════════════
RELATIVE STRENGTH CALCULATION MODULE
Calculates RS Rating vs SPY (market benchmark)
Based on IBD/Minervini methodology
═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

class RelativeStrengthCalculator:
    """
    Calculate Relative Strength ratings similar to IBD/Minervini methodology
    """
    
    def __init__(self, connector):
        self.connector = connector
        self.spy_data = None
    
    def fetch_spy_benchmark(self, retries: int = 3):
        """
        Fetch SPY historical data for RS comparison
        Cache it to avoid repeated requests. A timeout is retried (with a wait for the HMDS
        farm in between): without SPY the whole scan aborts, so one slow reply mustn't end it.
        """
        if self.spy_data is None:
            logger.info("Fetching SPY benchmark data...")
            bars = self.connector.fetch_historical_data("SPY", duration="1 Y", bar_size="1 day",
                                                        retries=retries)
            
            if bars:
                self.spy_data = self._bars_to_dataframe(bars)
                logger.info(f"✓ SPY benchmark loaded ({len(self.spy_data)} days)")
            else:
                logger.error("Failed to fetch SPY data")
                return False
        return True
    
    def calculate_rs_rating(self, symbol_bars, periods=[20, 50, 250]):
        """
        Calculate RS Rating for a stock vs SPY
        
        RS Formula (similar to IBD):
        - 40% weight on 250-day performance
        - 20% weight on 50-day performance  
        - 40% weight on 20-day performance
        
        Returns: RS Rating (0-100 scale)
        """
        if self.spy_data is None:
            if not self.fetch_spy_benchmark():
                return None
        
        try:
            # Convert bars to dataframe
            df = self._bars_to_dataframe(symbol_bars)
            
            if len(df) < max(periods):
                return None
            
            # Calculate returns for each period
            rs_scores = []
            weights = [0.4, 0.2, 0.4]  # [250-day, 50-day, 20-day] weights
            
            # SPY over the SAME dates as the stock: by position, a stock with a missing day or a
            # stale last bar would be compared with a different SPY window (review: date alignment)
            spy_close = self.spy_data.set_index('date')['close'].sort_index()
            for period, weight in zip(periods, weights):
                start_date, end_date = df['date'].iloc[-period], df['date'].iloc[-1]
                spy_start, spy_end = spy_close.asof(start_date), spy_close.asof(end_date)
                if pd.isna(spy_start) or pd.isna(spy_end):
                    return None   # SPY history doesn't cover the stock's window
                stock_return = (df['close'].iloc[-1] / df['close'].iloc[-period] - 1) * 100
                spy_return = (spy_end / spy_start - 1) * 100
                
                # Relative strength = stock performance - SPY performance
                rs_component = stock_return - spy_return
                rs_scores.append(rs_component * weight)
            
            # Combine weighted scores
            total_rs = sum(rs_scores)
            
            # Normalize to 0-100 scale (approximate distribution)
            # Stocks outperforming SPY by 50%+ get RS ~95-100
            # Stocks matching SPY get RS ~50
            # Stocks underperforming get RS <50
            rs_rating = 50 + (total_rs * 0.8)  # Scale factor
            rs_rating = max(0, min(100, rs_rating))  # Clamp to 0-100
            
            return round(rs_rating, 1)
            
        except Exception as e:
            logger.warning(f"RS calculation error: {e}")
            return None
    
    def calculate_sector_rs(self, sector_symbols, all_stocks_data):
        """
        Calculate sector relative strength
        Compare average sector performance vs market
        
        Returns: Sector RS Rating (0-100)
        """
        try:
            # Calculate average sector return
            sector_returns = []
            for symbol in sector_symbols:
                if symbol in all_stocks_data:
                    bars = all_stocks_data[symbol]
                    if len(bars) >= 50:
                        df = self._bars_to_dataframe(bars)
                        ret_50d = (df['close'].iloc[-1] / df['close'].iloc[-50] - 1) * 100
                        sector_returns.append(ret_50d)
            
            if not sector_returns:
                return None
            
            avg_sector_return = np.mean(sector_returns)
            
            # Compare to SPY
            spy_return_50d = (self.spy_data['close'].iloc[-1] / self.spy_data['close'].iloc[-50] - 1) * 100
            
            sector_rs = avg_sector_return - spy_return_50d
            sector_rating = 50 + (sector_rs * 1.0)
            sector_rating = max(0, min(100, sector_rating))
            
            return round(sector_rating, 1)
            
        except Exception as e:
            logger.warning(f"Sector RS calculation error: {e}")
            return None
    
    def _bars_to_dataframe(self, bars):
        """Convert IBKR bar data to pandas DataFrame"""
        data = {
            'date': [bar.date for bar in bars],
            'open': [bar.open for bar in bars],
            'high': [bar.high for bar in bars],
            'low': [bar.low for bar in bars],
            'close': [bar.close for bar in bars],
            'volume': [bar.volume for bar in bars]
        }
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        return df
    
    def rank_stocks_by_rs(self, rs_ratings_dict):
        """
        Rank all stocks by RS and assign percentile rankings
        Returns: Dict with symbol: percentile_rank
        """
        if not rs_ratings_dict:
            return {}
        
        # Sort by RS rating
        sorted_stocks = sorted(rs_ratings_dict.items(), key=lambda x: x[1], reverse=True)
        
        # Assign percentile ranks
        total_stocks = len(sorted_stocks)
        rankings = {}
        
        for idx, (symbol, rs_rating) in enumerate(sorted_stocks):
            percentile = ((total_stocks - idx) / total_stocks) * 100
            rankings[symbol] = {
                'rs_rating': rs_rating,
                'rs_percentile': round(percentile, 1),
                'rank': idx + 1
            }
        
        return rankings
