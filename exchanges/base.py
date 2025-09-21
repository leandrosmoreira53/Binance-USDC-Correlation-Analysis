import asyncio
import os
import pandas as pd
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
import ccxt.async_support as ccxt
from scipy.stats import spearmanr, kendalltau
from typing import List, Dict, Optional, Tuple

class BaseExchangeApp(ABC):
    """
    Abstract base class for cryptocurrency exchange correlation analysis
    """
    
    def __init__(self, exchange_name: str, pair_types: List[str] = None):
        self.exchange_name = exchange_name.lower()
        self.pair_types = pair_types or ['USDC', 'USDT']
        
        # Data storage
        self.data_by_pair = {}  # {pair_type: {closes: df, returns: df, meta_df: df}}
        self.correlations = {}  # {pair_type: {method: correlation_matrix}}
        
        # Configuration
        self.days = int(os.getenv('DAYS', 30))
        self.timeframe = os.getenv('TIMEFRAME', '1d')
        self.concurrency = int(os.getenv('CONCURRENCY', 50))
        self.symbol_cap = int(os.getenv('SYMBOL_CAP', 120))
        self.vol_threshold = float(os.getenv('VOL_THRESHOLD', 0))
    
    @abstractmethod
    async def create_exchange_instance(self) -> ccxt.Exchange:
        """Create and return exchange instance"""
        pass
    
    @abstractmethod
    def get_exchange_specific_config(self) -> Dict:
        """Return exchange-specific configuration"""
        pass
    
    async def fetch_symbols(self, pair_type: str) -> List[str]:
        """Fetch all trading pairs for a specific pair type (USDC/USDT)"""
        exchange = await self.create_exchange_instance()
        
        try:
            markets = await exchange.load_markets()
            symbols = [symbol for symbol in markets.keys() 
                      if symbol.endswith(f'/{pair_type}') and markets[symbol]['active']]
            
            if self.symbol_cap > 0:
                symbols = symbols[:self.symbol_cap]
                
            return symbols
        finally:
            await exchange.close()
    
    async def fetch_ohlcv_data(self, symbol: str, exchange: ccxt.Exchange, 
                              progress_info: Optional[Tuple[int, int]] = None) -> Optional[Dict]:
        """Fetch OHLCV data for a single symbol"""
        try:
            if progress_info:
                current, total = progress_info
                percentage = (current / total) * 100
                remaining = total - current
                print(f"[{self.exchange_name.upper()}] Downloading {symbol} ({current}/{total}) - {percentage:.1f}% complete - {remaining} remaining")
            
            since = int((datetime.now() - timedelta(days=self.days + 5)).timestamp() * 1000)
            ohlcv = await exchange.fetch_ohlcv(symbol, self.timeframe, since=since, limit=self.days + 5)
            
            if len(ohlcv) < self.days:
                return None
                
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['date'] = pd.to_datetime(df['timestamp'], unit='ms').dt.date
            df = df.tail(self.days)
            
            return {
                'symbol': symbol,
                'closes': df.set_index('date')['close'],
                'volumes': df.set_index('date')['volume'],
                'mean_volume': df['volume'].mean(),
                'mean_price': df['close'].mean()
            }
        except Exception as e:
            print(f"[{self.exchange_name.upper()}] Error fetching {symbol}: {e}")
            return None
    
    async def collect_data_for_pair_type(self, pair_type: str):
        """Collect OHLCV data for all symbols of a specific pair type"""
        print(f"[{self.exchange_name.upper()}] Fetching {pair_type} symbols...")
        symbols = await self.fetch_symbols(pair_type)
        print(f"[{self.exchange_name.upper()}] Found {len(symbols)} {pair_type} pairs")
        
        exchange = await self.create_exchange_instance()
        semaphore = asyncio.Semaphore(self.concurrency)
        total_symbols = len(symbols)
        
        async def fetch_with_semaphore(symbol, index):
            async with semaphore:
                return await self.fetch_ohlcv_data(symbol, exchange, (index + 1, total_symbols))
        
        try:
            print(f"[{self.exchange_name.upper()}] Collecting {pair_type} data with concurrency limit of {self.concurrency}...")
            
            tasks = [fetch_with_semaphore(symbol, i) for i, symbol in enumerate(symbols)]
            results = await asyncio.gather(*tasks)
            
            valid_results = [r for r in results if r is not None]
            print(f"[{self.exchange_name.upper()}] Successfully collected {pair_type} data for {len(valid_results)} symbols")
            
            if not valid_results:
                print(f"[{self.exchange_name.upper()}] Warning: No valid {pair_type} data collected")
                return
            
            # Process data
            closes_data = {}
            meta_data = []
            
            for result in valid_results:
                symbol = result['symbol']
                closes_data[symbol] = result['closes']
                meta_data.append({
                    'symbol': symbol,
                    'mean_volume_30d': result['mean_volume'],
                    'mean_price_30d': result['mean_price']
                })
            
            closes_df = pd.DataFrame(closes_data)
            closes_df.index = pd.to_datetime(closes_df.index)
            closes_df = closes_df.sort_index()
            
            meta_df = pd.DataFrame(meta_data).set_index('symbol')
            returns_df = closes_df.pct_change().dropna()
            
            # Store data
            self.data_by_pair[pair_type] = {
                'closes': closes_df,
                'returns': returns_df,
                'meta_df': meta_df
            }
            
            # Calculate correlations
            await self.calculate_correlations_for_pair_type(pair_type)
            
        finally:
            await exchange.close()
    
    async def calculate_correlations_for_pair_type(self, pair_type: str):
        """Calculate all correlation types for a specific pair type"""
        if pair_type not in self.data_by_pair:
            return
        
        data = self.data_by_pair[pair_type]
        closes = data['closes']
        returns = data['returns']
        
        print(f"[{self.exchange_name.upper()}] Calculating {pair_type} correlations...")
        
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
        
        print(f"[{self.exchange_name.upper()}] {pair_type} correlation calculations completed")
    
    def calculate_spearman_correlation(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Spearman rank correlation matrix"""
        corr_matrix = pd.DataFrame(index=data.columns, columns=data.columns)
        
        for i, col1 in enumerate(data.columns):
            for j, col2 in enumerate(data.columns):
                if i <= j:  # Only calculate upper triangle
                    corr, _ = spearmanr(data[col1].dropna(), data[col2].dropna())
                    corr_matrix.loc[col1, col2] = corr
                    corr_matrix.loc[col2, col1] = corr  # Symmetric matrix
        
        return corr_matrix.astype(float)
    
    def calculate_kendall_correlation(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Kendall tau correlation matrix"""
        corr_matrix = pd.DataFrame(index=data.columns, columns=data.columns)
        
        for i, col1 in enumerate(data.columns):
            for j, col2 in enumerate(data.columns):
                if i <= j:  # Only calculate upper triangle
                    corr, _ = kendalltau(data[col1].dropna(), data[col2].dropna())
                    corr_matrix.loc[col1, col2] = corr
                    corr_matrix.loc[col2, col1] = corr  # Symmetric matrix
        
        return corr_matrix.astype(float)
    
    async def collect_all_data(self):
        """Collect data for all configured pair types"""
        for pair_type in self.pair_types:
            await self.collect_data_for_pair_type(pair_type)
    
    def get_data(self, pair_type: str) -> Dict:
        """Get data for a specific pair type"""
        return self.data_by_pair.get(pair_type, {})
    
    def get_correlations(self, pair_type: str, correlation_type: str) -> Optional[pd.DataFrame]:
        """Get correlation matrix for specific pair type and correlation type"""
        return self.correlations.get(pair_type, {}).get(correlation_type)
    
    def get_available_pair_types(self) -> List[str]:
        """Get list of available pair types with data"""
        return list(self.data_by_pair.keys())
    
    def get_symbols_for_pair_type(self, pair_type: str) -> List[str]:
        """Get list of symbols for a specific pair type"""
        data = self.get_data(pair_type)
        if 'closes' in data:
            return data['closes'].columns.tolist()
        return []
    
    def save_cache(self):
        """Save data to cache files - to be implemented by subclasses"""
        pass
    
    def load_cache(self) -> bool:
        """Load data from cache files - to be implemented by subclasses"""
        return False
    
    def cache_exists(self) -> bool:
        """Check if cache files exist - to be implemented by subclasses"""
        return False