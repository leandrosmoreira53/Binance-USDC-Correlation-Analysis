import os
import pandas as pd
import ccxt.async_support as ccxt
from typing import Dict, List
from .base import BaseExchangeApp

class MexcCorrelationApp(BaseExchangeApp):
    """
    MEXC-specific implementation of correlation analysis
    """
    
    def __init__(self, pair_types: List[str] = None):
        # MEXC primarily uses USDT pairs
        super().__init__('mexc', pair_types or ['USDT'])
    
    async def create_exchange_instance(self) -> ccxt.Exchange:
        """Create MEXC exchange instance"""
        return ccxt.mexc({
            'enableRateLimit': True,
            'sandbox': False,
        })
    
    def get_exchange_specific_config(self) -> Dict:
        """Return MEXC-specific configuration"""
        return {
            'rate_limit': 1000,  # requests per minute (more conservative)
            'weight_limit': 1000,
            'supported_timeframes': ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w', '1M']
        }
    
    def _get_cache_path(self, pair_type: str, data_type: str) -> str:
        """Get cache file path for specific pair type and data type"""
        os.makedirs('data', exist_ok=True)
        return f'data/mexc_{pair_type.lower()}_{data_type}_{self.days}d.parquet'
    
    async def fetch_symbols(self, pair_type: str) -> List[str]:
        """Fetch all trading pairs for a specific pair type (MEXC-specific filtering)"""
        exchange = await self.create_exchange_instance()
        
        try:
            markets = await exchange.load_markets()
            # MEXC sometimes has different naming conventions
            symbols = []
            
            for symbol in markets.keys():
                market = markets[symbol]
                if (symbol.endswith(f'/{pair_type}') and 
                    market['active'] and 
                    market['type'] == 'spot'):  # Only spot markets
                    symbols.append(symbol)
            
            if self.symbol_cap > 0:
                symbols = symbols[:self.symbol_cap]
                
            return symbols
        finally:
            await exchange.close()
    
    def save_cache(self):
        """Save data to Parquet cache files organized by pair type"""
        os.makedirs('data', exist_ok=True)
        
        for pair_type in self.data_by_pair.keys():
            data = self.data_by_pair[pair_type]
            
            # Save main data
            data['closes'].reset_index().to_parquet(self._get_cache_path(pair_type, 'closes'), index=False)
            data['meta_df'].reset_index().to_parquet(self._get_cache_path(pair_type, 'meta'), index=False)
            
            # Save correlations
            correlations = self.correlations.get(pair_type, {})
            for corr_type, corr_matrix in correlations.items():
                if corr_matrix is not None:
                    corr_matrix.reset_index().to_parquet(
                        self._get_cache_path(pair_type, f'corr_{corr_type}'), 
                        index=False
                    )
        
        print(f"[MEXC] Data cached for pair types: {list(self.data_by_pair.keys())}")
    
    def load_cache(self) -> bool:
        """Load data from Parquet cache files"""
        try:
            # Try to load data for each configured pair type
            for pair_type in self.pair_types:
                if not self._load_cache_for_pair_type(pair_type):
                    print(f"[MEXC] Cache not found for {pair_type}")
                    continue
            
            if self.data_by_pair:
                print(f"[MEXC] Data loaded from cache for: {list(self.data_by_pair.keys())}")
                return True
            
            return False
            
        except Exception as e:
            print(f"[MEXC] Error loading cache: {e}")
            return False
    
    def _load_cache_for_pair_type(self, pair_type: str) -> bool:
        """Load cache for a specific pair type"""
        try:
            # Load main data
            closes_path = self._get_cache_path(pair_type, 'closes')
            meta_path = self._get_cache_path(pair_type, 'meta')
            
            if not os.path.exists(closes_path) or not os.path.exists(meta_path):
                return False
            
            closes_df = pd.read_parquet(closes_path)
            closes_df['date'] = pd.to_datetime(closes_df['date'])
            closes_df = closes_df.set_index('date')
            
            meta_df = pd.read_parquet(meta_path)
            meta_df = meta_df.set_index('symbol')
            
            returns_df = closes_df.pct_change().dropna()
            
            # Store data
            self.data_by_pair[pair_type] = {
                'closes': closes_df,
                'returns': returns_df,
                'meta_df': meta_df
            }
            
            # Load correlations
            self.correlations[pair_type] = {}
            correlation_types = ['close_pearson', 'returns_pearson', 'close_spearman', 
                               'returns_spearman', 'close_kendall', 'returns_kendall']
            
            for corr_type in correlation_types:
                try:
                    corr_path = self._get_cache_path(pair_type, f'corr_{corr_type}')
                    if os.path.exists(corr_path):
                        corr_df = pd.read_parquet(corr_path)
                        self.correlations[pair_type][corr_type] = corr_df.set_index('index')
                except Exception as e:
                    print(f"[MEXC] Warning: Could not load {corr_type} for {pair_type}: {e}")
            
            # Calculate missing correlations if needed
            if not self.correlations[pair_type]:
                print(f"[MEXC] Calculating missing correlations for {pair_type}...")
                # Note: This should be called from an async context
                # For now, calculate synchronously
                self._calculate_correlations_sync(pair_type)
            
            return True
            
        except Exception as e:
            print(f"[MEXC] Error loading cache for {pair_type}: {e}")
            return False
    
    def _calculate_correlations_sync(self, pair_type: str):
        """Synchronous version of correlation calculation for cache loading"""
        if pair_type not in self.data_by_pair:
            return
        
        data = self.data_by_pair[pair_type]
        closes = data['closes']
        returns = data['returns']
        
        print(f"[MEXC] Calculating {pair_type} correlations synchronously...")
        
        # Initialize correlations dict for this pair type
        self.correlations[pair_type] = {}
        
        # Pearson correlations
        self.correlations[pair_type]['close_pearson'] = closes.corr()
        self.correlations[pair_type]['returns_pearson'] = returns.corr()
        
        # Spearman correlations
        self.correlations[pair_type]['close_spearman'] = self.calculate_spearman_correlation(closes)
        self.correlations[pair_type]['returns_spearman'] = self.calculate_spearman_correlation(returns)
        
        # Kendall correlations
        self.correlations[pair_type]['close_kendall'] = self.calculate_kendall_correlation(closes)
        self.correlations[pair_type]['returns_kendall'] = self.calculate_kendall_correlation(returns)
        
        print(f"[MEXC] {pair_type} correlation calculations completed")
    
    def cache_exists(self) -> bool:
        """Check if cache files exist for at least one pair type"""
        for pair_type in self.pair_types:
            closes_path = self._get_cache_path(pair_type, 'closes')
            meta_path = self._get_cache_path(pair_type, 'meta')
            if os.path.exists(closes_path) and os.path.exists(meta_path):
                return True
        return False
    
    def export_csv(self):
        """Export all data to CSV files"""
        os.makedirs('data/exports', exist_ok=True)
        
        for pair_type in self.data_by_pair.keys():
            data = self.data_by_pair[pair_type]
            
            # Export main data
            data['closes'].to_csv(f'data/exports/mexc_{pair_type.lower()}_closes_{self.days}d.csv')
            data['returns'].to_csv(f'data/exports/mexc_{pair_type.lower()}_returns_{self.days}d.csv')
            data['meta_df'].to_csv(f'data/exports/mexc_{pair_type.lower()}_meta_{self.days}d.csv')
            
            # Export correlations
            correlations = self.correlations.get(pair_type, {})
            for corr_type, corr_matrix in correlations.items():
                if corr_matrix is not None:
                    corr_matrix.to_csv(f'data/exports/mexc_{pair_type.lower()}_{corr_type}_{self.days}d.csv')
        
        print(f"[MEXC] Data exported to CSV for pair types: {list(self.data_by_pair.keys())}")
    
    def export_parquet(self):
        """Export all data to Parquet files for external use"""
        os.makedirs('data/exports', exist_ok=True)
        
        for pair_type in self.data_by_pair.keys():
            data = self.data_by_pair[pair_type]
            
            # Export main data
            data['closes'].reset_index().to_parquet(f'data/exports/mexc_{pair_type.lower()}_closes_{self.days}d_export.parquet', index=False)
            data['returns'].reset_index().to_parquet(f'data/exports/mexc_{pair_type.lower()}_returns_{self.days}d_export.parquet', index=False)
            data['meta_df'].reset_index().to_parquet(f'data/exports/mexc_{pair_type.lower()}_meta_{self.days}d_export.parquet', index=False)
            
            # Export correlations
            correlations = self.correlations.get(pair_type, {})
            for corr_type, corr_matrix in correlations.items():
                if corr_matrix is not None:
                    corr_matrix.reset_index().to_parquet(f'data/exports/mexc_{pair_type.lower()}_{corr_type}_{self.days}d_export.parquet', index=False)
        
        print(f"[MEXC] Data exported to Parquet for pair types: {list(self.data_by_pair.keys())}")
    
    # Legacy property methods for compatibility (will return None if no data)
    @property
    def closes(self):
        """Legacy compatibility - returns USDT closes if available"""
        if 'USDT' in self.data_by_pair:
            return self.data_by_pair['USDT']['closes']
        elif self.data_by_pair:
            first_pair = list(self.data_by_pair.keys())[0]
            return self.data_by_pair[first_pair]['closes']
        return None
    
    @property
    def returns(self):
        """Legacy compatibility - returns USDT returns if available"""
        if 'USDT' in self.data_by_pair:
            return self.data_by_pair['USDT']['returns']
        elif self.data_by_pair:
            first_pair = list(self.data_by_pair.keys())[0]
            return self.data_by_pair[first_pair]['returns']
        return None
    
    @property
    def meta_df(self):
        """Legacy compatibility - returns USDT meta_df if available"""
        if 'USDT' in self.data_by_pair:
            return self.data_by_pair['USDT']['meta_df']
        elif self.data_by_pair:
            first_pair = list(self.data_by_pair.keys())[0]
            return self.data_by_pair[first_pair]['meta_df']
        return None