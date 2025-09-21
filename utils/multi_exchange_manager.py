import asyncio
import pandas as pd
from typing import Dict, List, Optional, Tuple
from exchanges.binance import BinanceCorrelationApp
from exchanges.mexc import MexcCorrelationApp
from exchanges.base import BaseExchangeApp

class MultiExchangeManager:
    """
    Manager class for coordinating multiple exchange data collection and analysis
    """
    
    def __init__(self):
        self.exchanges = {}
        self.initialize_exchanges()
    
    def initialize_exchanges(self):
        """Initialize all supported exchanges"""
        # Binance with both USDC and USDT
        self.exchanges['binance'] = BinanceCorrelationApp(pair_types=['USDC', 'USDT'])
        
        # MEXC primarily with USDT
        self.exchanges['mexc'] = MexcCorrelationApp(pair_types=['USDT'])
    
    async def collect_all_data(self, exchanges: List[str] = None):
        """
        Collect data from specified exchanges (or all if None)
        """
        target_exchanges = exchanges or list(self.exchanges.keys())
        
        print(f"Starting data collection for exchanges: {target_exchanges}")
        
        # Collect data from exchanges in parallel
        tasks = []
        for exchange_name in target_exchanges:
            if exchange_name in self.exchanges:
                exchange = self.exchanges[exchange_name]
                tasks.append(self._collect_exchange_data(exchange, exchange_name))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        print("Multi-exchange data collection completed")
    
    async def _collect_exchange_data(self, exchange: BaseExchangeApp, exchange_name: str):
        """Collect data for a single exchange with error handling"""
        try:
            print(f"Starting data collection for {exchange_name.upper()}")
            
            # Check cache first
            if exchange.cache_exists():
                print(f"[{exchange_name.upper()}] Loading from cache...")
                if exchange.load_cache():
                    print(f"[{exchange_name.upper()}] Successfully loaded from cache")
                    return
            
            # Collect fresh data
            print(f"[{exchange_name.upper()}] Collecting fresh data...")
            await exchange.collect_all_data()
            
            # Save to cache
            exchange.save_cache()
            print(f"[{exchange_name.upper()}] Data collection and caching completed")
            
        except Exception as e:
            print(f"[{exchange_name.upper()}] Error during data collection: {e}")
            # Continue with other exchanges even if one fails
    
    def get_exchange(self, exchange_name: str) -> Optional[BaseExchangeApp]:
        """Get specific exchange instance"""
        return self.exchanges.get(exchange_name.lower())
    
    def get_available_exchanges(self) -> List[str]:
        """Get list of available exchange names"""
        return list(self.exchanges.keys())
    
    def get_exchange_data(self, exchange_name: str, pair_type: str) -> Dict:
        """Get data for specific exchange and pair type"""
        exchange = self.get_exchange(exchange_name)
        if exchange:
            return exchange.get_data(pair_type)
        return {}
    
    def get_exchange_correlations(self, exchange_name: str, pair_type: str, correlation_type: str) -> Optional[pd.DataFrame]:
        """Get correlation matrix for specific exchange, pair type, and correlation method"""
        exchange = self.get_exchange(exchange_name)
        if exchange:
            return exchange.get_correlations(pair_type, correlation_type)
        return None
    
    def get_combined_symbols(self, pair_type: str, exchanges: List[str] = None) -> List[str]:
        """Get combined list of symbols across exchanges for a specific pair type"""
        target_exchanges = exchanges or list(self.exchanges.keys())
        all_symbols = set()
        
        for exchange_name in target_exchanges:
            exchange = self.get_exchange(exchange_name)
            if exchange:
                symbols = exchange.get_symbols_for_pair_type(pair_type)
                all_symbols.update(symbols)
        
        return sorted(list(all_symbols))
    
    def get_common_symbols(self, pair_type: str, exchanges: List[str] = None) -> List[str]:
        """Get symbols that are common across all specified exchanges"""
        target_exchanges = exchanges or list(self.exchanges.keys())
        
        if not target_exchanges:
            return []
        
        # Start with symbols from first exchange
        first_exchange = self.get_exchange(target_exchanges[0])
        if not first_exchange:
            return []
        
        common_symbols = set(first_exchange.get_symbols_for_pair_type(pair_type))
        
        # Intersect with symbols from other exchanges
        for exchange_name in target_exchanges[1:]:
            exchange = self.get_exchange(exchange_name)
            if exchange:
                exchange_symbols = set(exchange.get_symbols_for_pair_type(pair_type))
                common_symbols = common_symbols.intersection(exchange_symbols)
        
        return sorted(list(common_symbols))
    
    def compare_correlations(self, pair_type: str, correlation_type: str, 
                           symbol1: str, symbol2: str, exchanges: List[str] = None) -> Dict[str, float]:
        """Compare correlation between two symbols across exchanges"""
        target_exchanges = exchanges or list(self.exchanges.keys())
        results = {}
        
        for exchange_name in target_exchanges:
            correlation_matrix = self.get_exchange_correlations(exchange_name, pair_type, correlation_type)
            if correlation_matrix is not None and symbol1 in correlation_matrix.index and symbol2 in correlation_matrix.columns:
                results[exchange_name] = correlation_matrix.loc[symbol1, symbol2]
        
        return results
    
    def get_cross_exchange_comparison(self, pair_type: str, correlation_type: str, 
                                    top_n: int = 20) -> pd.DataFrame:
        """
        Create a comparison matrix showing correlations across exchanges
        """
        common_symbols = self.get_common_symbols(pair_type)
        
        if len(common_symbols) < 2:
            return pd.DataFrame()
        
        # Limit to top_n symbols to avoid huge matrices
        selected_symbols = common_symbols[:min(top_n, len(common_symbols))]
        
        comparison_data = []
        
        for exchange_name in self.exchanges.keys():
            correlation_matrix = self.get_exchange_correlations(exchange_name, pair_type, correlation_type)
            if correlation_matrix is not None:
                # Extract relevant subset
                available_symbols = [s for s in selected_symbols if s in correlation_matrix.index and s in correlation_matrix.columns]
                if len(available_symbols) >= 2:
                    subset_matrix = correlation_matrix.loc[available_symbols, available_symbols]
                    
                    # Flatten upper triangle (excluding diagonal)
                    for i, symbol1 in enumerate(available_symbols):
                        for j, symbol2 in enumerate(available_symbols):
                            if i < j:  # Upper triangle only
                                correlation_value = subset_matrix.loc[symbol1, symbol2]
                                comparison_data.append({
                                    'exchange': exchange_name,
                                    'symbol1': symbol1,
                                    'symbol2': symbol2,
                                    'correlation': correlation_value,
                                    'pair': f"{symbol1} vs {symbol2}"
                                })
        
        if comparison_data:
            return pd.DataFrame(comparison_data)
        return pd.DataFrame()
    
    def get_exchange_status(self) -> Dict[str, Dict]:
        """Get status information for all exchanges"""
        status = {}
        
        for exchange_name, exchange in self.exchanges.items():
            status[exchange_name] = {
                'available_pair_types': exchange.get_available_pair_types(),
                'cache_exists': exchange.cache_exists(),
                'total_symbols': {
                    pair_type: len(exchange.get_symbols_for_pair_type(pair_type))
                    for pair_type in exchange.get_available_pair_types()
                }
            }
        
        return status
    
    def export_all_data(self, format_type: str = 'csv'):
        """Export data from all exchanges"""
        for exchange_name, exchange in self.exchanges.items():
            print(f"Exporting data for {exchange_name.upper()}...")
            if format_type.lower() == 'csv':
                exchange.export_csv()
            elif format_type.lower() == 'parquet':
                exchange.export_parquet()
    
    def get_btc_comparison_data(self, exchange_name: str, pair_type: str, 
                              correlation_type: str, method: str = 'pearson') -> Optional[pd.Series]:
        """Get BTC comparison data for a specific exchange and pair type"""
        btc_symbol = f'BTC/{pair_type}'
        
        # Map correlation type and method to the internal naming convention
        corr_key = f"{correlation_type}_{method}"
        
        correlation_matrix = self.get_exchange_correlations(exchange_name, pair_type, corr_key)
        
        if correlation_matrix is not None:
            # Check if BTC is available
            if btc_symbol in correlation_matrix.index:
                # Get BTC correlations with all other symbols
                btc_correlations = correlation_matrix.loc[btc_symbol].copy()
                # Remove self-correlation
                if btc_symbol in btc_correlations.index:
                    btc_correlations = btc_correlations.drop(btc_symbol)
                return btc_correlations.sort_values(ascending=False)
            
            # If BTC not available, try alternative major cryptocurrencies
            alternative_symbols = [f'ETH/{pair_type}', f'SOL/{pair_type}', f'BNB/{pair_type}']
            
            for alt_symbol in alternative_symbols:
                if alt_symbol in correlation_matrix.index:
                    print(f"[{exchange_name.upper()}] BTC/{pair_type} not available, using {alt_symbol} as reference")
                    alt_correlations = correlation_matrix.loc[alt_symbol].copy()
                    # Remove self-correlation
                    if alt_symbol in alt_correlations.index:
                        alt_correlations = alt_correlations.drop(alt_symbol)
                    return alt_correlations.sort_values(ascending=False)
        
        return None