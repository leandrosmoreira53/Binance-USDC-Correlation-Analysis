#!/usr/bin/env python3
"""
Test script for the multi-exchange integration
"""
import asyncio
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.multi_exchange_manager import MultiExchangeManager

async def test_integration():
    """Test the multi-exchange integration"""
    print("🚀 Starting Multi-Exchange Integration Test")
    print("=" * 50)
    
    # Initialize manager
    manager = MultiExchangeManager()
    
    # Test 1: Check available exchanges
    print("\n📊 Available Exchanges:")
    exchanges = manager.get_available_exchanges()
    for exchange in exchanges:
        print(f"  ✓ {exchange.upper()}")
    
    # Test 2: Check exchange status (before data collection)
    print("\n📈 Exchange Status (Before Data Collection):")
    status = manager.get_exchange_status()
    for exchange_name, info in status.items():
        print(f"  {exchange_name.upper()}:")
        print(f"    - Available pair types: {info['available_pair_types']}")
        print(f"    - Cache exists: {info['cache_exists']}")
        print(f"    - Total symbols: {info['total_symbols']}")
    
    # Test 3: Try to collect data (will use cache if available)
    print("\n🔄 Testing Data Collection:")
    try:
        # Test both exchanges
        await manager.collect_all_data(['binance', 'mexc'])
        print("  ✓ Binance and MEXC data collection completed")
    except Exception as e:
        print(f"  ❌ Data collection failed: {e}")
    
    # Test 4: Check data availability
    print("\n📊 Data Availability Check:")
    
    # Check Binance
    binance_exchange = manager.get_exchange('binance')
    if binance_exchange:
        available_pairs = binance_exchange.get_available_pair_types()
        print(f"  Binance available pair types: {available_pairs}")
        
        for pair_type in available_pairs:
            symbols = binance_exchange.get_symbols_for_pair_type(pair_type)
            print(f"    {pair_type}: {len(symbols)} symbols")
            
            # Test correlation data
            corr_matrix = binance_exchange.get_correlations(pair_type, 'returns_pearson')
            if corr_matrix is not None:
                print(f"    {pair_type} correlations: {corr_matrix.shape}")
            else:
                print(f"    {pair_type} correlations: Not available")
    
    # Check MEXC
    mexc_exchange = manager.get_exchange('mexc')
    if mexc_exchange:
        available_pairs = mexc_exchange.get_available_pair_types()
        print(f"  MEXC available pair types: {available_pairs}")
        
        for pair_type in available_pairs:
            symbols = mexc_exchange.get_symbols_for_pair_type(pair_type)
            print(f"    {pair_type}: {len(symbols)} symbols")
            
            # Test correlation data
            corr_matrix = mexc_exchange.get_correlations(pair_type, 'returns_pearson')
            if corr_matrix is not None:
                print(f"    {pair_type} correlations: {corr_matrix.shape}")
            else:
                print(f"    {pair_type} correlations: Not available")
    
    # Test 5: Test BTC comparison
    print("\n🪙 Testing BTC Comparison:")
    try:
        btc_data = manager.get_btc_comparison_data('binance', 'USDC', 'returns', 'pearson')
        if btc_data is not None and not btc_data.empty:
            print(f"  ✓ BTC/USDC comparison data: {len(btc_data)} correlations")
            print(f"    Top 3 correlations:")
            for symbol, corr in btc_data.head(3).items():
                print(f"      {symbol}: {corr:.4f}")
        else:
            print("  ❌ BTC/USDC comparison data not available")
    except Exception as e:
        print(f"  ❌ BTC comparison test failed: {e}")
    
    # Test 6: Test symbol filtering
    print("\n🔍 Testing Symbol Filtering:")
    try:
        binance_data = manager.get_exchange_data('binance', 'USDC')
        if binance_data and 'meta_df' in binance_data:
            meta_df = binance_data['meta_df']
            total_symbols = len(meta_df)
            
            # Filter by volume threshold
            filtered_symbols = meta_df[meta_df['mean_volume_30d'] >= 100000]
            filtered_count = len(filtered_symbols)
            
            print(f"  Total USDC symbols: {total_symbols}")
            print(f"  Symbols with volume >= 100K: {filtered_count}")
            
            if filtered_count > 0:
                print(f"  Top 3 by volume:")
                top_volume = filtered_symbols.nlargest(3, 'mean_volume_30d')
                for symbol, data in top_volume.iterrows():
                    volume = data['mean_volume_30d']
                    print(f"    {symbol}: {volume:,.0f}")
        else:
            print("  ❌ Symbol filtering test failed - no data available")
    except Exception as e:
        print(f"  ❌ Symbol filtering test failed: {e}")
    
    # Test 7: Final status check
    print("\n📈 Final Exchange Status:")
    final_status = manager.get_exchange_status()
    for exchange_name, info in final_status.items():
        print(f"  {exchange_name.upper()}:")
        print(f"    - Available pair types: {info['available_pair_types']}")
        print(f"    - Cache exists: {info['cache_exists']}")
        print(f"    - Total symbols: {info['total_symbols']}")
    
    print("\n🎉 Integration test completed!")
    print("=" * 50)

if __name__ == "__main__":
    asyncio.run(test_integration())