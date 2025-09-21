#!/usr/bin/env python3
"""
Test script specifically for MEXC data collection
"""
import asyncio
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from exchanges.mexc import MexcCorrelationApp

async def test_mexc():
    """Test MEXC data collection specifically"""
    print("🚀 Testing MEXC Exchange Data Collection")
    print("=" * 50)
    
    # Initialize MEXC app
    mexc_app = MexcCorrelationApp(pair_types=['USDT'])
    
    print("📊 MEXC Configuration:")
    config = mexc_app.get_exchange_specific_config()
    print(f"  Rate limit: {config['rate_limit']}")
    print(f"  Supported timeframes: {config['supported_timeframes']}")
    
    try:
        print("\n🔄 Testing MEXC API Connection...")
        
        # Test symbol fetching
        symbols = await mexc_app.fetch_symbols('USDT')
        print(f"  ✓ Found {len(symbols)} USDT symbols")
        
        if len(symbols) > 0:
            print(f"  First 5 symbols: {symbols[:5]}")
            
            # Test data collection for a few symbols
            print("\n📈 Testing data collection for top 5 symbols...")
            mexc_app.symbol_cap = 5  # Limit to 5 for testing
            
            await mexc_app.collect_all_data()
            
            # Check if data was collected
            if mexc_app.data_by_pair:
                for pair_type, data in mexc_app.data_by_pair.items():
                    print(f"  ✓ {pair_type} data collected:")
                    print(f"    - Close prices shape: {data['closes'].shape}")
                    print(f"    - Returns shape: {data['returns'].shape}")
                    print(f"    - Metadata symbols: {len(data['meta_df'])}")
            else:
                print("  ❌ No data collected")
        else:
            print("  ❌ No symbols found")
            
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🎉 MEXC test completed!")

if __name__ == "__main__":
    asyncio.run(test_mexc())