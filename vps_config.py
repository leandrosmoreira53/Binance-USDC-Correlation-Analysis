#!/usr/bin/env python3
"""
VPS Configuration Script for Binance API Project
Optimized settings for better performance on VPS
"""

import os

# VPS Optimized Environment Variables
VPS_CONFIG = {
    # Increased concurrency for better VPS performance
    'CONCURRENCY': '100',
    
    # More symbols for comprehensive analysis
    'SYMBOL_CAP': '500',
    
    # Longer time period for better correlation analysis
    'DAYS': '90',
    
    # Higher volume threshold to focus on liquid pairs
    'VOL_THRESHOLD': '100000',
    
    # Use 4-hour timeframe for more data points
    'TIMEFRAME': '4h'
}

def setup_vps_environment():
    """Set up environment variables for VPS optimization"""
    print("Setting up VPS-optimized environment...")
    
    for key, value in VPS_CONFIG.items():
        os.environ[key] = value
        print(f"Set {key} = {value}")
    
    print("\nVPS configuration applied!")
    print("Expected improvements:")
    print("- 10x faster data collection (100 concurrent requests)")
    print("- 4x more symbols analyzed (500 vs 120)")
    print("- 3x more historical data (90 days vs 30)")
    print("- Better correlation accuracy with more data points")

if __name__ == "__main__":
    setup_vps_environment()
