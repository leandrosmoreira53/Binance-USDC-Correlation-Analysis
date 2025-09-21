import asyncio
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import ccxt.async_support as ccxt
from dash import Dash, dcc, html, Input, Output, callback, dash_table
import dash_bootstrap_components as dbc
from scipy.stats import spearmanr, kendalltau

DAYS = int(os.getenv('DAYS', 30))
TIMEFRAME = os.getenv('TIMEFRAME', '1d')
CONCURRENCY = int(os.getenv('CONCURRENCY', 50))  # Increased for VPS
SYMBOL_CAP = int(os.getenv('SYMBOL_CAP', 120))
VOL_THRESHOLD = float(os.getenv('VOL_THRESHOLD', 0))

class BinanceCorrelationApp:
    def __init__(self):
        self.closes = None
        self.returns = None
        self.meta_df = None
        self.corr_close = None
        self.corr_returns = None
        self.corr_close_spearman = None
        self.corr_returns_spearman = None
        self.corr_close_kendall = None
        self.corr_returns_kendall = None
        
    async def fetch_symbols(self):
        """Fetch all USDC trading pairs from Binance"""
        exchange = ccxt.binance({
            'enableRateLimit': True,
            'sandbox': False,
        })
        
        try:
            markets = await exchange.load_markets()
            usdc_symbols = [symbol for symbol in markets.keys() 
                          if symbol.endswith('/USDC') and markets[symbol]['active']]
            
            if SYMBOL_CAP > 0:
                usdc_symbols = usdc_symbols[:SYMBOL_CAP]
                
            return usdc_symbols
        finally:
            await exchange.close()
    
    async def fetch_ohlcv_data(self, symbol, exchange, progress_info=None):
        """Fetch OHLCV data for a single symbol"""
        try:
            if progress_info:
                current, total = progress_info
                percentage = (current / total) * 100
                remaining = total - current
                print(f"Downloading {symbol} ({current}/{total}) - {percentage:.1f}% complete - {remaining} remaining")
            
            since = int((datetime.now() - timedelta(days=DAYS + 5)).timestamp() * 1000)
            ohlcv = await exchange.fetch_ohlcv(symbol, TIMEFRAME, since=since, limit=DAYS + 5)
            
            if len(ohlcv) < DAYS:
                return None
                
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['date'] = pd.to_datetime(df['timestamp'], unit='ms').dt.date
            df = df.tail(DAYS)
            
            return {
                'symbol': symbol,
                'closes': df.set_index('date')['close'],
                'volumes': df.set_index('date')['volume'],
                'mean_volume': df['volume'].mean(),
                'mean_price': df['close'].mean()
            }
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            return None
    
    async def collect_data(self):
        """Collect OHLCV data for all USDC pairs asynchronously"""
        print("Fetching USDC symbols...")
        symbols = await self.fetch_symbols()
        print(f"Found {len(symbols)} USDC pairs")
        
        exchange = ccxt.binance({
            'enableRateLimit': True,
            'sandbox': False,
        })
        
        semaphore = asyncio.Semaphore(CONCURRENCY)
        completed_count = 0
        total_symbols = len(symbols)
        
        async def fetch_with_semaphore(symbol, index):
            nonlocal completed_count
            async with semaphore:
                result = await self.fetch_ohlcv_data(symbol, exchange, (index + 1, total_symbols))
                completed_count += 1
                return result
        
        try:
            print(f"Collecting data with concurrency limit of {CONCURRENCY}...")
            print(f"Starting download of {total_symbols} symbols...")
            
            tasks = [fetch_with_semaphore(symbol, i) for i, symbol in enumerate(symbols)]
            results = await asyncio.gather(*tasks)
            
            valid_results = [r for r in results if r is not None]
            print(f"\nDownload completed! Successfully collected data for {len(valid_results)} symbols")
            
            if not valid_results:
                raise ValueError("No valid data collected")
            
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
            
            self.closes = pd.DataFrame(closes_data)
            self.closes.index = pd.to_datetime(self.closes.index)
            self.closes = self.closes.sort_index()
            
            self.meta_df = pd.DataFrame(meta_data).set_index('symbol')
            
            self.returns = self.closes.pct_change().dropna()
            
            print("Calculating price correlations...")
            self.corr_close = self.closes.corr()
            print("Calculating returns correlations...")
            self.corr_returns = self.returns.corr()
            
            print("Calculating Spearman correlations...")
            self.corr_close_spearman = self.calculate_spearman_correlation(self.closes)
            self.corr_returns_spearman = self.calculate_spearman_correlation(self.returns)
            
            print("Calculating Kendall correlations...")
            self.corr_close_kendall = self.calculate_kendall_correlation(self.closes)
            self.corr_returns_kendall = self.calculate_kendall_correlation(self.returns)
            
            print("All correlation calculations completed")
            
        finally:
            await exchange.close()
    
    def calculate_spearman_correlation(self, data):
        """Calculate Spearman rank correlation matrix"""
        corr_matrix = pd.DataFrame(index=data.columns, columns=data.columns)
        
        for i, col1 in enumerate(data.columns):
            for j, col2 in enumerate(data.columns):
                if i <= j:  # Only calculate upper triangle
                    corr, _ = spearmanr(data[col1].dropna(), data[col2].dropna())
                    corr_matrix.loc[col1, col2] = corr
                    corr_matrix.loc[col2, col1] = corr  # Symmetric matrix
        
        return corr_matrix.astype(float)
    
    def calculate_kendall_correlation(self, data):
        """Calculate Kendall tau correlation matrix"""
        corr_matrix = pd.DataFrame(index=data.columns, columns=data.columns)
        
        for i, col1 in enumerate(data.columns):
            for j, col2 in enumerate(data.columns):
                if i <= j:  # Only calculate upper triangle
                    corr, _ = kendalltau(data[col1].dropna(), data[col2].dropna())
                    corr_matrix.loc[col1, col2] = corr
                    corr_matrix.loc[col2, col1] = corr  # Symmetric matrix
        
        return corr_matrix.astype(float)
    
    def save_cache(self):
        """Save data to Parquet cache files"""
        os.makedirs('data', exist_ok=True)
        
        self.closes.reset_index().to_parquet('data/closes_30d.parquet', index=False)
        self.meta_df.reset_index().to_parquet('data/meta_30d.parquet', index=False)
        
        # Save pre-calculated correlations to cache
        if self.corr_close is not None:
            self.corr_close.reset_index().to_parquet('data/corr_close_30d.parquet', index=False)
        if self.corr_returns is not None:
            self.corr_returns.reset_index().to_parquet('data/corr_returns_30d.parquet', index=False)
        if self.corr_close_spearman is not None:
            self.corr_close_spearman.reset_index().to_parquet('data/corr_close_spearman_30d.parquet', index=False)
        if self.corr_returns_spearman is not None:
            self.corr_returns_spearman.reset_index().to_parquet('data/corr_returns_spearman_30d.parquet', index=False)
        if self.corr_close_kendall is not None:
            self.corr_close_kendall.reset_index().to_parquet('data/corr_close_kendall_30d.parquet', index=False)
        if self.corr_returns_kendall is not None:
            self.corr_returns_kendall.reset_index().to_parquet('data/corr_returns_kendall_30d.parquet', index=False)
        
        print("Data and all correlation types cached to Parquet files")
    
    def load_cache(self):
        """Load data from Parquet cache files"""
        try:
            closes_df = pd.read_parquet('data/closes_30d.parquet')
            closes_df['date'] = pd.to_datetime(closes_df['date'])
            self.closes = closes_df.set_index('date')
            
            meta_df = pd.read_parquet('data/meta_30d.parquet')
            self.meta_df = meta_df.set_index('symbol')
            
            self.returns = self.closes.pct_change().dropna()
            
            # Try to load pre-calculated correlations
            try:
                corr_close_df = pd.read_parquet('data/corr_close_30d.parquet')
                self.corr_close = corr_close_df.set_index('index')
                
                corr_returns_df = pd.read_parquet('data/corr_returns_30d.parquet')
                self.corr_returns = corr_returns_df.set_index('index')
                
                # Load Spearman correlations
                try:
                    corr_close_spearman_df = pd.read_parquet('data/corr_close_spearman_30d.parquet')
                    self.corr_close_spearman = corr_close_spearman_df.set_index('index')
                    
                    corr_returns_spearman_df = pd.read_parquet('data/corr_returns_spearman_30d.parquet')
                    self.corr_returns_spearman = corr_returns_spearman_df.set_index('index')
                except FileNotFoundError:
                    print("Spearman correlations not found, calculating...")
                    self.corr_close_spearman = self.calculate_spearman_correlation(self.closes)
                    self.corr_returns_spearman = self.calculate_spearman_correlation(self.returns)
                
                # Load Kendall correlations
                try:
                    corr_close_kendall_df = pd.read_parquet('data/corr_close_kendall_30d.parquet')
                    self.corr_close_kendall = corr_close_kendall_df.set_index('index')
                    
                    corr_returns_kendall_df = pd.read_parquet('data/corr_returns_kendall_30d.parquet')
                    self.corr_returns_kendall = corr_returns_kendall_df.set_index('index')
                except FileNotFoundError:
                    print("Kendall correlations not found, calculating...")
                    self.corr_close_kendall = self.calculate_kendall_correlation(self.closes)
                    self.corr_returns_kendall = self.calculate_kendall_correlation(self.returns)
                
                print("Data and all correlation types loaded from cache")
            except FileNotFoundError:
                print("Pre-calculated correlations not found, calculating now...")
                self.corr_close = self.closes.corr()
                self.corr_returns = self.returns.corr()
                self.corr_close_spearman = self.calculate_spearman_correlation(self.closes)
                self.corr_returns_spearman = self.calculate_spearman_correlation(self.returns)
                self.corr_close_kendall = self.calculate_kendall_correlation(self.closes)
                self.corr_returns_kendall = self.calculate_kendall_correlation(self.returns)
                print("All correlations calculated")
            
            return True
        except FileNotFoundError:
            print("Cache files not found")
            return False
    
    def cache_exists(self):
        """Check if cache files exist"""
        return (os.path.exists('data/closes_30d.parquet') and 
                os.path.exists('data/meta_30d.parquet'))
    
    def export_csv(self):
        """Export all data to CSV files"""
        os.makedirs('data', exist_ok=True)
        
        self.closes.to_csv('data/closes_30d.csv')
        self.returns.to_csv('data/returns_30d.csv')
        self.corr_close.to_csv('data/corr_close_30d.csv')
        self.corr_returns.to_csv('data/corr_returns_30d.csv')
        self.meta_df.to_csv('data/mean_price_30d.csv')
        self.meta_df[['mean_volume_30d']].to_csv('data/mean_volume_30d.csv')
        
        print("Data exported to CSV files")
    
    def export_parquet(self):
        """Export all data to Parquet files"""
        os.makedirs('data', exist_ok=True)
        
        self.closes.reset_index().to_parquet('data/closes_30d_export.parquet', index=False)
        self.returns.reset_index().to_parquet('data/returns_30d_export.parquet', index=False)
        self.corr_close.reset_index().to_parquet('data/corr_close_30d_export.parquet', index=False)
        self.corr_returns.reset_index().to_parquet('data/corr_returns_30d_export.parquet', index=False)
        self.meta_df.reset_index().to_parquet('data/mean_price_30d_export.parquet', index=False)
        self.meta_df[['mean_volume_30d']].reset_index().to_parquet('data/mean_volume_30d_export.parquet', index=False)
        
        print("Data exported to Parquet files")

app_instance = BinanceCorrelationApp()

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], meta_tags=[
    {"name": "viewport", "content": "width=device-width, initial-scale=1"}
])

# Custom CSS for responsive design
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            @media (max-width: 576px) {
                .card { margin-bottom: 1rem !important; }
                .btn { font-size: 0.875rem; }
                .form-label { font-size: 0.875rem; }
            }
            @media (max-width: 768px) {
                .container-fluid { padding: 0.5rem; }
                .row { margin: 0; }
                .col { padding: 0.25rem; }
            }
            .heatmap-container {
                width: 100%;
                overflow-x: auto;
                overflow-y: auto;
                max-height: 80vh;
            }
            .correlation-matrix {
                min-width: 600px;
                width: 100%;
                height: auto;
            }
            @media (max-width: 768px) {
                .heatmap-container {
                    max-height: 60vh;
                }
                .correlation-matrix {
                    min-width: 400px;
                }
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

app.layout = dbc.Container([
        dbc.Row([
        dbc.Col([
            html.H1("Binance USDC Pairs Correlation Analysis", 
                   className="text-center mb-4", 
                   style={'fontSize': 'clamp(1.5rem, 4vw, 2.5rem)'}),
            
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Volume Threshold:"),
                            dcc.Slider(
                                id='volume-threshold',
                                min=0,
                                max=1000000,
                                step=10000,
                                value=VOL_THRESHOLD,
                                marks={i: f'{i/1000:.0f}K' for i in range(0, 1000001, 200000)},
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ], width=12, lg=6),
                        
                        dbc.Col([
                            html.Label("Correlation Type:"),
                            dcc.RadioItems(
                                id='correlation-type',
                                options=[
                                    {'label': 'Price Correlation (Pearson)', 'value': 'close'},
                                    {'label': 'Returns Correlation (Pearson)', 'value': 'returns'},
                                    {'label': 'Price Correlation (Spearman)', 'value': 'close_spearman'},
                                    {'label': 'Returns Correlation (Spearman)', 'value': 'returns_spearman'},
                                    {'label': 'Price Correlation (Kendall)', 'value': 'close_kendall'},
                                    {'label': 'Returns Correlation (Kendall)', 'value': 'returns_kendall'}
                                ],
                                value='returns',
                                style={'fontSize': 'clamp(0.8rem, 2vw, 1rem)'}
                            )
                        ], width=12, lg=6)
                    ]),
                    
                    html.Hr(),
                    
                    dbc.Row([
                        dbc.Col([
                            html.Label("Select Symbols:"),
                            dcc.Dropdown(
                                id='symbol-dropdown',
                                multi=True,
                                placeholder="Select symbols to analyze"
                            )
                        ], width=12)
                    ])
                ])
            ], className="mb-4"),
            
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Button("Export CSV", id="btn-csv", color="primary", className="me-2 mb-2"),
                            dbc.Button("Export Parquet", id="btn-parquet", color="secondary", className="me-2 mb-2"),
                            dbc.Button("Refresh Data", id="btn-refresh", color="warning", className="mb-2")
                        ])
                    ])
                ])
            ], className="mb-4"),
            
            html.Div([
                dcc.Graph(id='correlation-heatmap', className='correlation-matrix')
            ], className='heatmap-container'),
            
            html.Div(id='stats-table')
            
        ], width=12)
    ])
], fluid=True)

@callback(
    Output('symbol-dropdown', 'options'),
    Output('symbol-dropdown', 'value'),
    Input('volume-threshold', 'value')
)
def update_symbol_options(volume_threshold):
    if app_instance.meta_df is None:
        return [], []
    
    filtered_symbols = app_instance.meta_df[
        app_instance.meta_df['mean_volume_30d'] >= volume_threshold
    ].index.tolist()
    
    options = [{'label': symbol, 'value': symbol} for symbol in sorted(filtered_symbols)]
    default_values = filtered_symbols[:20] if len(filtered_symbols) > 20 else filtered_symbols
    
    return options, default_values

@callback(
    Output('correlation-heatmap', 'figure'),
    [Input('symbol-dropdown', 'value'),
     Input('correlation-type', 'value')]
)
def update_heatmap(selected_symbols, corr_type):
    if not selected_symbols or app_instance.corr_close is None:
        return go.Figure()
    
    # Select the appropriate correlation matrix
    if corr_type == 'close':
        corr_matrix = app_instance.corr_close.loc[selected_symbols, selected_symbols]
        title = "Price Correlation Matrix (Pearson)"
    elif corr_type == 'returns':
        corr_matrix = app_instance.corr_returns.loc[selected_symbols, selected_symbols]
        title = "Returns Correlation Matrix (Pearson)"
    elif corr_type == 'close_spearman':
        corr_matrix = app_instance.corr_close_spearman.loc[selected_symbols, selected_symbols]
        title = "Price Correlation Matrix (Spearman)"
    elif corr_type == 'returns_spearman':
        corr_matrix = app_instance.corr_returns_spearman.loc[selected_symbols, selected_symbols]
        title = "Returns Correlation Matrix (Spearman)"
    elif corr_type == 'close_kendall':
        corr_matrix = app_instance.corr_close_kendall.loc[selected_symbols, selected_symbols]
        title = "Price Correlation Matrix (Kendall)"
    elif corr_type == 'returns_kendall':
        corr_matrix = app_instance.corr_returns_kendall.loc[selected_symbols, selected_symbols]
        title = "Returns Correlation Matrix (Kendall)"
    else:
        corr_matrix = app_instance.corr_returns.loc[selected_symbols, selected_symbols]
        title = "Returns Correlation Matrix (Pearson)"
    
    fig = px.imshow(
        corr_matrix,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        color_continuous_scale='RdBu_r',
        aspect='auto',
        title=title,
        zmin=-1,
        zmax=1
    )
    
    # Calculate responsive dimensions
    num_symbols = len(selected_symbols)
    base_size = 40  # Base cell size
    min_size = 25   # Minimum cell size
    max_size = 60   # Maximum cell size
    
    # Adjust cell size based on number of symbols
    cell_size = max(min_size, min(max_size, base_size - (num_symbols - 10) * 2))
    height = max(400, min(800, num_symbols * cell_size + 100))
    
    # Font size calculation
    font_size = max(8, min(14, 120 / num_symbols))
    
    fig.update_layout(
        height=height,
        xaxis_title="Symbol",
        yaxis_title="Symbol",
        font=dict(size=font_size),
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis=dict(
            tickangle=45,
            tickfont=dict(size=font_size)
        ),
        yaxis=dict(
            tickfont=dict(size=font_size)
        ),
        coloraxis_colorbar=dict(
            title="Correlation",
            dtick=0.2,
            tickvals=[-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1],
            ticktext=["-1.0", "-0.8", "-0.6", "-0.4", "-0.2", "0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]
        )
    )
    
    # Update hovertemplate for better mobile experience
    fig.update_traces(
        hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>" +
                      "Correlation: %{z:.3f}<br>" +
                      "<extra></extra>"
    )
    
    return fig

@callback(
    Output('stats-table', 'children'),
    Input('symbol-dropdown', 'value')
)
def update_stats_table(selected_symbols):
    if not selected_symbols or app_instance.meta_df is None:
        return html.Div()
    
    stats_data = app_instance.meta_df.loc[selected_symbols].reset_index()
    stats_data['mean_volume_30d'] = stats_data['mean_volume_30d'].round(0)
    stats_data['mean_price_30d'] = stats_data['mean_price_30d'].round(4)
    
    table = dash_table.DataTable(
        data=stats_data.to_dict('records'),
        columns=[
            {'name': 'Symbol', 'id': 'symbol'},
            {'name': 'Mean Volume (30d)', 'id': 'mean_volume_30d', 'type': 'numeric', 'format': {'specifier': ',.0f'}},
            {'name': 'Mean Price (30d)', 'id': 'mean_price_30d', 'type': 'numeric', 'format': {'specifier': '.4f'}}
        ],
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left', 'fontSize': 'clamp(0.7rem, 2vw, 0.9rem)'},
        style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
        sort_action='native'
    )
    
    return dbc.Card([
        dbc.CardHeader("Symbol Statistics"),
        dbc.CardBody(table)
    ])

@callback(
    Output('btn-csv', 'children'),
    Input('btn-csv', 'n_clicks'),
    prevent_initial_call=True
)
def export_csv_callback(n_clicks):
    if n_clicks:
        app_instance.export_csv()
        return "CSV Exported ✓"
    return "Export CSV"

@callback(
    Output('btn-parquet', 'children'),
    Input('btn-parquet', 'n_clicks'),
    prevent_initial_call=True
)
def export_parquet_callback(n_clicks):
    if n_clicks:
        app_instance.export_parquet()
        return "Parquet Exported ✓"
    return "Export Parquet"

@callback(
    Output('btn-refresh', 'children'),
    Input('btn-refresh', 'n_clicks'),
    prevent_initial_call=True
)
def refresh_data_callback(n_clicks):
    if n_clicks:
        asyncio.run(app_instance.collect_data())
        app_instance.save_cache()
        return "Data Refreshed ✓"
    return "Refresh Data"

async def initialize_app():
    """Initialize the app with data"""
    print("Initializing Binance USDC Correlation App...")
    
    if app_instance.cache_exists():
        print("Loading data from cache...")
        app_instance.load_cache()
    else:
        print("No cache found. Collecting fresh data...")
        await app_instance.collect_data()
        app_instance.save_cache()
    
    print(f"Data loaded: {len(app_instance.closes.columns)} symbols, {len(app_instance.closes)} days")

if __name__ == '__main__':
    asyncio.run(initialize_app())
    app.run(debug=True, host='0.0.0.0', port=8050)