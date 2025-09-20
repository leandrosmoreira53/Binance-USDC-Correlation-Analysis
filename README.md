# Binance USDC Pairs Correlation Analysis Dashboard

A powerful web application for analyzing price and returns correlations between Binance USDC trading pairs, featuring efficient data caching, interactive visualizations, and flexible export options.

## Features

- **Async Data Collection**: Efficiently collects OHLCV data for all USDC pairs using ccxt with configurable concurrency
- **Dual Correlation Analysis**: 
  - Price correlation (closes.corr())
  - Returns correlation (returns.pct_change().corr())
- **Interactive Dashboard**: Filter by volume threshold, select specific symbols, toggle correlation types
- **Efficient Caching**: Primary storage in Parquet format for optimal performance
- **Multiple Export Formats**: CSV and Parquet exports for all datasets
- **Real-time Updates**: Refresh data on demand with rate limit respect

## Requirements

- Python 3.10 or higher
- UV package manager (recommended) or pip
- Internet connection for Binance API access

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

### Running the Application

```bash
python app.py
```

The application will:
1. Check for cached data in the `data/` directory
2. If no cache exists, collect fresh data from Binance (this may take a few minutes)
3. Start the Dash server at `http://localhost:8050`

### Environment Configuration

You can customize the application behavior using environment variables:

```bash
# Data collection parameters
export DAYS=30                    # Number of days to collect (default: 30)
export TIMEFRAME='1d'             # Timeframe for OHLCV data (default: 1d)
export CONCURRENCY=10             # API request concurrency (default: 10)
export SYMBOL_CAP=120             # Maximum symbols to collect (default: 120)
export VOL_THRESHOLD=0            # Default volume filter (default: 0)

# Run the application
python app.py
```

## Dashboard Features

### Volume Threshold Filter
- Slider to filter symbols by 30-day average volume
- Automatically updates available symbols in the dropdown

### Symbol Selection
- Multi-select dropdown for choosing specific trading pairs
- Defaults to top 20 symbols by volume when threshold changes

### Correlation Type Toggle
- **Price Correlation**: Correlation between closing prices
- **Returns Correlation**: Correlation between daily returns (recommended for financial analysis)

### Interactive Heatmap
- Color-coded correlation matrix using Plotly
- Hover tooltips showing exact correlation values
- Responsive design for different screen sizes

### Statistics Table
- Symbol-wise 30-day average volume and price
- Sortable columns for easy analysis

### Export Options
- **CSV Export**: Compatible with Excel and other tools
- **Parquet Export**: Efficient binary format for data science workflows

## Data Storage Structure

```
data/
├── closes_30d.parquet          # Primary cache: closing prices
├── meta_30d.parquet            # Primary cache: volume/price metadata
├── closes_30d.csv              # Export: closing prices
├── returns_30d.csv             # Export: daily returns
├── corr_close_30d.csv          # Export: price correlation matrix
├── corr_returns_30d.csv        # Export: returns correlation matrix
├── mean_price_30d.csv          # Export: 30-day average prices
├── mean_volume_30d.csv         # Export: 30-day average volumes
├── closes_30d_export.parquet   # Optional: Parquet exports
├── returns_30d_export.parquet
├── corr_close_30d_export.parquet
├── corr_returns_30d_export.parquet
├── mean_price_30d_export.parquet
└── mean_volume_30d_export.parquet
```

## Cache Management

### Automatic Caching
- Data is automatically cached in Parquet format after collection
- Subsequent runs load from cache for instant startup
- Cache includes both price data and metadata

### Manual Cache Refresh
- Use the "Refresh Data" button in the dashboard
- Or delete cache files to force fresh collection:
```bash
rm data/closes_30d.parquet data/meta_30d.parquet
```

### Cache Benefits
- **Parquet Format**: 50-90% smaller file size compared to CSV
- **Fast Loading**: 10x faster read times than CSV
- **Type Preservation**: Maintains data types and index information

## API Rate Limiting

The application implements several rate limiting best practices:

- **enableRateLimit=True**: Built-in ccxt rate limiting
- **Concurrency Control**: Configurable semaphore limiting
- **Batch Processing**: Processes symbols in controlled batches
- **Error Handling**: Graceful handling of API timeouts and errors

### Recommended Settings
- **CONCURRENCY=10**: Safe for most use cases
- **CONCURRENCY=20**: For faster collection (monitor for rate limit errors)
- **CONCURRENCY=5**: Conservative setting for shared IP addresses

## Troubleshooting

### Common Issues

1. **Rate Limit Errors**
   - Reduce CONCURRENCY value
   - Add delays between requests
   - Check Binance API status

2. **Memory Issues with Large Datasets**
   - Reduce SYMBOL_CAP
   - Reduce DAYS parameter
   - Monitor system memory usage

3. **Cache Corruption**
   - Delete cache files and restart
   - Check disk space availability

4. **Network Connectivity**
   - Verify internet connection
   - Check firewall settings
   - Test Binance API accessibility

### Performance Optimization

- **Use Parquet**: Always prefer Parquet over CSV for large datasets
- **Adjust Concurrency**: Find optimal balance between speed and stability
- **Cache Strategy**: Keep cache files to avoid repeated API calls
- **Symbol Filtering**: Use volume thresholds to focus on liquid pairs

## Dependencies

- **ccxt>=4.3.0**: Cryptocurrency exchange API library
- **pandas>=2.2.2**: Data manipulation and analysis
- **numpy>=1.26.4**: Numerical computing
- **dash>=2.17.0**: Web application framework
- **plotly>=5.22.0**: Interactive plotting library
- **pyarrow>=16.0.0**: Parquet file format support

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review Binance API documentation
3. Create an issue with detailed error information

## Roadmap

- [ ] Real-time data streaming
- [ ] Additional correlation metrics (Spearman, Kendall)
- [ ] Portfolio optimization features
- [ ] Alert system for correlation changes
- [ ] Database integration for historical analysis