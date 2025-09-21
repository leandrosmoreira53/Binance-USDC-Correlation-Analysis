# Multi-Exchange Cryptocurrency Correlation Analysis Dashboard

A powerful web application for analyzing price and returns correlations between cryptocurrency trading pairs across multiple exchanges (Binance and MEXC), featuring advanced correlation analysis, efficient data caching, interactive visualizations, and flexible export options.

## Features

### 🚀 **Multi-Exchange Support**
- **Binance Integration**: Full support for USDC and USDT pairs
- **MEXC Integration**: Comprehensive USDT pairs analysis
- **Cross-Exchange Comparison**: Compare correlations across different exchanges
- **Unified Interface**: Single dashboard for multiple exchanges

### 📊 **Advanced Correlation Analysis**
- **Multiple Correlation Methods**: Pearson, Spearman, and Kendall correlations
- **Dual Data Types**: Price correlation and returns correlation analysis
- **Reference Symbol Analysis**: BTC comparison with automatic fallback (SOL/ETH for MEXC)
- **Interactive Filtering**: Volume threshold, symbol selection, and correlation type filters

### 🎯 **Two Application Versions**
- **app.py**: Single-exchange Binance analysis (legacy version)
- **app_expanded.py**: Multi-exchange analysis with enhanced features (recommended)

### ⚡ **Performance & Efficiency**
- **Async Data Collection**: Concurrent API requests with configurable limits
- **Smart Caching**: Parquet format storage for optimal performance
- **Modular Architecture**: Object-oriented design with exchange-specific implementations
- **Rate Limiting**: Built-in protection against API limits

## Requirements

- Python 3.10 or higher
- UV package manager (recommended) or pip
- Internet connection for exchange API access

## Installation

### Using UV (Recommended)

```bash
# Navigate to the project directory
cd /home/snayder/Documents/api_binance

# Create virtual environment with UV
uv venv .venv

# Activate virtual environment
# Linux/macOS:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt
```

### Using pip

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Linux/macOS:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 🎯 **Recommended: Multi-Exchange Application**

```bash
python app_expanded.py
```

**Features:**
- Multi-exchange support (Binance + MEXC)
- Advanced correlation methods (Pearson, Spearman, Kendall)
- Tabbed interface with specialized analysis views
- BTC/SOL comparison charts with filtering
- Export tools and detailed exchange status

### 📊 **Legacy: Single Exchange Application**

```bash
python app.py
```

**Features:**
- Binance USDC pairs analysis
- Basic correlation analysis
- Simple interface
- BTC comparison functionality

### Application Startup Process

Both applications will:
1. Check for cached data in the `data/` directory
2. Load from cache if available (instant startup)
3. If no cache, collect fresh data from exchanges (may take several minutes)
4. Start the Dash server at `http://localhost:8050`

### Environment Configuration

Customize application behavior using environment variables:

```bash
# Data collection parameters
export DAYS=30                    # Number of days to collect (default: 30)
export TIMEFRAME='1d'             # Timeframe for OHLCV data (default: 1d)
export CONCURRENCY=50             # API request concurrency (default: 50)
export SYMBOL_CAP=120             # Maximum symbols to collect (default: 120)
export VOL_THRESHOLD=0            # Default volume filter (default: 0)

# Run the multi-exchange application
python app_expanded.py
```

## Dashboard Features (app_expanded.py)

### 🏢 **Exchange Selection**
- **Binance**: Access to both USDC and USDT pairs
- **MEXC**: USDT pairs with alternative reference symbols

### 💱 **Pair Type Filtering**
- **USDC Pairs**: Available on Binance (120 symbols)
- **USDT Pairs**: Available on both Binance (120) and MEXC (113)

### 📈 **Analysis Tabs**

#### 1. **Correlation Matrix**
- Interactive heatmap with color-coded correlations
- Dynamic font sizing based on matrix dimensions
- Responsive design for different screen sizes
- Hover tooltips with exact correlation values

#### 2. **BTC Comparison**
- Reference symbol analysis (BTC/SOL/ETH depending on availability)
- Advanced filtering options:
  - All correlations
  - Positive correlations only
  - Negative correlations only
  - Strong correlations (|r| > 0.5)
  - Moderate correlations (0.3 < |r| < 0.7)
  - Weak correlations (|r| < 0.3)

#### 3. **Export & Tools**
- CSV export for Excel compatibility
- Parquet export for data science workflows
- Data refresh functionality
- Exchange status monitoring

### 🎛️ **Interactive Controls**
- **Volume Threshold Slider**: Filter by 30-day average volume
- **Symbol Selection**: Multi-select with "Select All" option
- **Correlation Methods**: Choose between Pearson, Spearman, Kendall
- **Correlation Types**: Price vs Returns correlation

## Architecture

### 🏗️ **Modular Design**

```
├── app.py                           # Legacy single-exchange application
├── app_expanded.py                  # Multi-exchange application (recommended)
├── exchanges/
│   ├── __init__.py
│   ├── base.py                      # Abstract base class for exchanges
│   ├── binance.py                   # Binance implementation
│   └── mexc.py                      # MEXC implementation
├── utils/
│   ├── __init__.py
│   └── multi_exchange_manager.py    # Central coordinator
└── test_integration.py             # Integration testing
```

### 🗄️ **Data Storage Structure**

```
data/
├── binance_usdc_closes_30d.parquet       # Binance USDC price data
├── binance_usdc_meta_30d.parquet         # Binance USDC metadata
├── binance_usdc_corr_*_30d.parquet       # Binance USDC correlations
├── binance_usdt_closes_30d.parquet       # Binance USDT price data
├── binance_usdt_meta_30d.parquet         # Binance USDT metadata
├── binance_usdt_corr_*_30d.parquet       # Binance USDT correlations
├── mexc_usdt_closes_30d.parquet          # MEXC USDT price data
├── mexc_usdt_meta_30d.parquet            # MEXC USDT metadata
├── mexc_usdt_corr_*_30d.parquet          # MEXC USDT correlations
└── exports/                              # CSV/Parquet exports
```

## Cache Management

### ⚡ **Automatic Caching**
- Data automatically cached in Parquet format after collection
- Subsequent runs load from cache for instant startup
- Separate cache files for each exchange and pair type
- Correlation matrices cached for all methods (Pearson, Spearman, Kendall)

### 🔄 **Manual Cache Refresh**
- Use the "Refresh Data" button in the dashboard
- Or delete specific cache files to force fresh collection:

```bash
# Refresh all data
rm data/*.parquet

# Refresh only MEXC data
rm data/mexc_*.parquet

# Refresh only Binance USDC data
rm data/binance_usdc_*.parquet
```

### 🏆 **Cache Benefits**
- **Parquet Format**: 50-90% smaller file size vs CSV
- **Fast Loading**: 10x faster read times than CSV
- **Type Preservation**: Maintains data types and index information
- **Compression**: Built-in compression for storage efficiency

## API Rate Limiting & Performance

### 🛡️ **Rate Limiting Best Practices**
- **enableRateLimit=True**: Built-in ccxt rate limiting
- **Concurrency Control**: Configurable semaphore limiting (default: 50)
- **Exchange-Specific Limits**: Optimized for each exchange's requirements
- **Error Handling**: Graceful handling of API timeouts and errors

### ⚙️ **Recommended Settings**
- **CONCURRENCY=50**: Optimal for most use cases
- **CONCURRENCY=100**: For faster collection (monitor for rate limit errors)
- **CONCURRENCY=20**: Conservative setting for shared networks

### 📊 **Performance Metrics**
- **Data Collection**: ~2-5 minutes for full dataset (353 symbols)
- **Cache Loading**: ~2-3 seconds for complete dataset
- **Memory Usage**: ~200-500MB depending on dataset size
- **Storage**: ~50-100MB for complete cached dataset

## Testing

### 🧪 **Integration Testing**

```bash
# Test multi-exchange integration
python test_integration.py

# Test specific exchange
python test_mexc_specific.py
```

### ✅ **Expected Results**
- Binance: 240 symbols (120 USDC + 120 USDT)
- MEXC: ~113 symbols (USDT only)
- Total: ~353 cryptocurrency pairs
- All correlation matrices: 120x120 (Binance), 113x113 (MEXC)

## Troubleshooting

### 🚨 **Common Issues**

1. **Rate Limit Errors**
   - Reduce CONCURRENCY value
   - Check exchange API status
   - Verify network stability

2. **BTC Comparison "Data Not Available"**
   - Fixed in app_expanded.py with automatic fallback
   - MEXC uses SOL/USDT as reference when BTC/USDT unavailable
   - Check exchange-specific symbol availability

3. **Memory Issues**
   - Reduce SYMBOL_CAP parameter
   - Reduce DAYS parameter
   - Monitor system memory during collection

4. **Cache Corruption**
   - Delete cache files and restart
   - Check disk space availability
   - Verify file permissions

5. **Network Connectivity**
   - Verify internet connection
   - Check firewall settings for exchange APIs
   - Test exchange API accessibility

### 🔧 **Performance Optimization**

- **Use app_expanded.py**: Recommended for all new users
- **Keep Cache Files**: Avoid repeated API calls
- **Adjust Concurrency**: Balance speed vs stability
- **Use Volume Filters**: Focus on liquid trading pairs
- **Monitor Resources**: Watch memory and network usage

## Dependencies

### 📦 **Core Dependencies**
- **ccxt>=4.3.0**: Cryptocurrency exchange API library
- **pandas>=2.2.2**: Data manipulation and analysis
- **numpy>=1.26.4**: Numerical computing
- **dash>=2.17.0**: Web application framework
- **plotly>=5.22.0**: Interactive plotting library
- **pyarrow>=16.0.0**: Parquet file format support
- **dash-bootstrap-components**: Enhanced UI components

### 🔗 **Additional Libraries**
- **asyncio**: Asynchronous programming support
- **typing**: Type hints for better code quality
- **os**: Operating system interface

## Roadmap

### ✅ **Completed Features**
- [x] Multi-exchange support (Binance + MEXC)
- [x] Advanced correlation metrics (Pearson, Spearman, Kendall)
- [x] Modular architecture with exchange abstraction
- [x] Responsive UI with tabbed interface
- [x] BTC comparison with automatic fallback
- [x] Comprehensive caching system

### 🚧 **Planned Features**
- [ ] Real-time data streaming
- [ ] Additional exchanges (Coinbase, Kraken)
- [ ] Portfolio optimization tools
- [ ] Alert system for correlation changes
- [ ] Database integration for historical analysis
- [ ] REST API for external integrations
- [ ] Mobile-responsive improvements

## Contributing

1. Fork the repository
2. Create a feature branch
3. Follow the modular architecture patterns
4. Add tests for new exchanges or features
5. Update documentation
6. Submit a pull request

## License

MIT License - see LICENSE file for details

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review exchange API documentation
3. Run integration tests to verify setup
4. Create an issue with detailed error information and logs

---

**💡 Quick Start Recommendation**: Use `python app_expanded.py` for the best experience with multi-exchange analysis and advanced features!