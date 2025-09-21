import asyncio
import os
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from dash import Dash, dcc, html, Input, Output, callback, dash_table
import dash_bootstrap_components as dbc

# Import our new modular architecture
from utils.multi_exchange_manager import MultiExchangeManager

# Configuration
DAYS = int(os.getenv('DAYS', 30))
TIMEFRAME = os.getenv('TIMEFRAME', '1d')
CONCURRENCY = int(os.getenv('CONCURRENCY', 50))
SYMBOL_CAP = int(os.getenv('SYMBOL_CAP', 120))
VOL_THRESHOLD = float(os.getenv('VOL_THRESHOLD', 0))

# Initialize multi-exchange manager
manager = MultiExchangeManager()

# Initialize Dash app
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
            html.H1("Multi-Exchange Cryptocurrency Correlation Analysis", 
                   className="text-center mb-4", 
                   style={'fontSize': 'clamp(1.5rem, 4vw, 2.5rem)'}),
            
            # Main Control Panel
            dbc.Card([
                dbc.CardHeader("Analysis Configuration"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Exchange:"),
                            dcc.Dropdown(
                                id='exchange-dropdown',
                                options=[
                                    {'label': 'Binance', 'value': 'binance'},
                                    {'label': 'MEXC', 'value': 'mexc'}
                                ],
                                value='binance',
                                clearable=False
                            )
                        ], width=12, lg=3),
                        
                        dbc.Col([
                            html.Label("Pair Type:"),
                            dcc.Dropdown(
                                id='pair-type-dropdown',
                                options=[
                                    {'label': 'USDC Pairs', 'value': 'USDC'},
                                    {'label': 'USDT Pairs', 'value': 'USDT'}
                                ],
                                value='USDC',
                                clearable=False
                            )
                        ], width=12, lg=3),
                        
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
                        ], width=12, lg=3),
                        
                        dbc.Col([
                            html.Label("Correlation Method:"),
                            dcc.RadioItems(
                                id='correlation-method',
                                options=[
                                    {'label': 'Pearson', 'value': 'pearson'},
                                    {'label': 'Spearman', 'value': 'spearman'},
                                    {'label': 'Kendall', 'value': 'kendall'}
                                ],
                                value='pearson',
                                inline=True,
                                style={'fontSize': 'clamp(0.8rem, 2vw, 1rem)'}
                            )
                        ], width=12, lg=3)
                    ], className="mb-3"),
                    
                    dbc.Row([
                        dbc.Col([
                            html.Label("Correlation Type:"),
                            dcc.RadioItems(
                                id='correlation-type',
                                options=[
                                    {'label': 'Price Correlation', 'value': 'close'},
                                    {'label': 'Returns Correlation', 'value': 'returns'}
                                ],
                                value='returns',
                                inline=True,
                                style={'fontSize': 'clamp(0.8rem, 2vw, 1rem)'}
                            )
                        ], width=12, lg=6),
                        
                        dbc.Col([
                            dcc.Store(id='current-symbols-store'),
                            html.Div(id='exchange-status', className="text-muted")
                        ], width=12, lg=6)
                    ])
                ])
            ], className="mb-4"),
            
            # Symbol Selection
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Select Symbols:"),
                            dbc.Row([
                                dbc.Col([
                                    dcc.Dropdown(
                                        id='symbol-dropdown',
                                        multi=True,
                                        placeholder="Select symbols to analyze"
                                    )
                                ], width=9),
                                dbc.Col([
                                    dbc.Checklist(
                                        id='select-all-checkbox',
                                        options=[
                                            {"label": "Select All", "value": "all"}
                                        ],
                                        value=[],
                                        inline=True,
                                        style={'margin-top': '5px'}
                                    )
                                ], width=3)
                            ])
                        ], width=12)
                    ])
                ])
            ], className="mb-4"),
            
            # Tabs for Different Analysis Views
            dbc.Tabs([
                dbc.Tab(label="Correlation Matrix", tab_id="matrix-tab"),
                dbc.Tab(label="BTC Comparison", tab_id="btc-tab"),
                dbc.Tab(label="Export & Tools", tab_id="tools-tab")
            ], id="analysis-tabs", active_tab="matrix-tab"),
            
            # Pre-define all tab contents to avoid ID conflicts
            html.Div([
                # Matrix Tab Content
                html.Div([
                    html.Div([
                        dcc.Graph(id='correlation-heatmap', className='correlation-matrix')
                    ], className='heatmap-container'),
                    html.Div(id='stats-table', className="mt-4")
                ], id="matrix-tab-content", style={'display': 'block'}),
                
                # BTC Tab Content
                html.Div([
                    dbc.Card([
                        dbc.CardHeader("BTC Correlation Comparison"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Correlation Filter:"),
                                    dcc.Dropdown(
                                        id='btc-correlation-filter',
                                        options=[
                                            {'label': 'TODAS', 'value': 'all'},
                                            {'label': 'CORRELAÇÃO POSITIVA (>0)', 'value': 'positive'},
                                            {'label': 'CORRELAÇÃO NEGATIVA (<0)', 'value': 'negative'},
                                            {'label': 'CORRELAÇÃO FORTE (|r|>0.5)', 'value': 'strong'},
                                            {'label': 'CORRELAÇÃO MODERADA (0.3<|r|<0.7)', 'value': 'moderate'},
                                            {'label': 'CORRELAÇÃO FRACA (|r|<0.3)', 'value': 'weak'}
                                        ],
                                        value='all',
                                        clearable=False
                                    )
                                ], width=12)
                            ]),
                            html.Hr(),
                            html.Div([
                                dcc.Graph(id='btc-comparison-chart')
                            ], className='mt-3')
                        ])
                    ])
                ], id="btc-tab-content", style={'display': 'none'}),
                
                # Tools Tab Content
                html.Div([
                    dbc.Card([
                        dbc.CardHeader("Export & Analysis Tools"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    dbc.Button("Export CSV", id="btn-csv", color="primary", className="me-2 mb-2"),
                                    dbc.Button("Export Parquet", id="btn-parquet", color="secondary", className="me-2 mb-2"),
                                    dbc.Button("Refresh Data", id="btn-refresh", color="warning", className="mb-2")
                                ])
                            ]),
                            html.Hr(),
                            html.Div(id="exchange-status-detailed")
                        ])
                    ])
                ], id="tools-tab-content", style={'display': 'none'})
                
            ], className="mt-4")
        ], width=12)
    ])
], fluid=True)

# Tab visibility control to avoid ID conflicts
@callback(
    [Output('matrix-tab-content', 'style'),
     Output('btc-tab-content', 'style'),
     Output('tools-tab-content', 'style')],
    Input('analysis-tabs', 'active_tab')
)
def update_tab_visibility(active_tab):
    matrix_style = {'display': 'block'} if active_tab == "matrix-tab" else {'display': 'none'}
    btc_style = {'display': 'block'} if active_tab == "btc-tab" else {'display': 'none'}
    tools_style = {'display': 'block'} if active_tab == "tools-tab" else {'display': 'none'}
    
    return matrix_style, btc_style, tools_style

# Update pair type options based on selected exchange
@callback(
    Output('pair-type-dropdown', 'options'),
    Output('pair-type-dropdown', 'value'),
    Input('exchange-dropdown', 'value')
)
def update_pair_type_options(selected_exchange):
    if selected_exchange == 'binance':
        options = [
            {'label': 'USDC Pairs', 'value': 'USDC'},
            {'label': 'USDT Pairs', 'value': 'USDT'}
        ]
        default_value = 'USDC'
    elif selected_exchange == 'mexc':
        options = [
            {'label': 'USDT Pairs', 'value': 'USDT'}
        ]
        default_value = 'USDT'
    else:  # fallback
        options = [
            {'label': 'USDT Pairs', 'value': 'USDT'}
        ]
        default_value = 'USDT'
    
    return options, default_value

# Update symbol dropdown options
@callback(
    Output('symbol-dropdown', 'options'),
    [Input('exchange-dropdown', 'value'),
     Input('pair-type-dropdown', 'value'),
     Input('volume-threshold', 'value')]
)
def update_symbol_options(exchange, pair_type, volume_threshold):
    if not exchange or not pair_type:
        return []
    
    exchange_obj = manager.get_exchange(exchange)
    if not exchange_obj:
        return []
    
    data = exchange_obj.get_data(pair_type)
    if not data or 'meta_df' not in data:
        return []
    
    meta_df = data['meta_df']
    if meta_df is None or meta_df.empty:
        return []
    
    filtered_symbols = meta_df[
        meta_df['mean_volume_30d'] >= volume_threshold
    ].index.tolist()
    
    options = [{'label': symbol, 'value': symbol} for symbol in sorted(filtered_symbols)]
    return options

# Update symbol values based on select all checkbox
@callback(
    Output('symbol-dropdown', 'value'),
    [Input('select-all-checkbox', 'value'),
     Input('exchange-dropdown', 'value'),
     Input('pair-type-dropdown', 'value'),
     Input('volume-threshold', 'value')]
)
def update_symbol_values(select_all, exchange, pair_type, volume_threshold):
    if not exchange or not pair_type:
        return []
    
    exchange_obj = manager.get_exchange(exchange)
    if not exchange_obj:
        return []
    
    data = exchange_obj.get_data(pair_type)
    if not data or 'meta_df' not in data:
        return []
    
    meta_df = data['meta_df']
    if meta_df is None or meta_df.empty:
        return []
    
    filtered_symbols = meta_df[
        meta_df['mean_volume_30d'] >= volume_threshold
    ].index.tolist()
    
    if select_all and 'all' in select_all:
        return filtered_symbols
    else:
        # Default to first 20 symbols
        return filtered_symbols[:20] if len(filtered_symbols) > 20 else filtered_symbols

# Update correlation heatmap
@callback(
    Output('correlation-heatmap', 'figure'),
    [Input('symbol-dropdown', 'value'),
     Input('correlation-type', 'value'),
     Input('correlation-method', 'value'),
     Input('exchange-dropdown', 'value'),
     Input('pair-type-dropdown', 'value')]
)
def update_heatmap(selected_symbols, corr_type, method, exchange, pair_type):
    if not selected_symbols or not exchange or not pair_type:
        return go.Figure()
    
    # Get correlation matrix
    corr_key = f"{corr_type}_{method}"
    corr_matrix = manager.get_exchange_correlations(exchange, pair_type, corr_key)
    
    if corr_matrix is None:
        return go.Figure()
    
    # Filter matrix to selected symbols
    available_symbols = [s for s in selected_symbols if s in corr_matrix.index and s in corr_matrix.columns]
    
    if len(available_symbols) < 2:
        return go.Figure()
    
    corr_matrix = corr_matrix.loc[available_symbols, available_symbols]
    
    title = f"{pair_type} {corr_type.title()} Correlation Matrix ({method.title()}) - {exchange.upper()}"
    
    fig = px.imshow(
        corr_matrix,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        color_continuous_scale='RdBu_r',
        aspect='auto',
        title=title,
        zmin=-1,
        zmax=1,
        text_auto=True
    )
    
    # Calculate dynamic text size
    num_symbols = len(available_symbols)
    if num_symbols <= 5:
        text_size = 24
    elif num_symbols <= 10:
        text_size = 20
    elif num_symbols <= 15:
        text_size = 18
    elif num_symbols <= 20:
        text_size = 16
    elif num_symbols <= 30:
        text_size = 14
    elif num_symbols <= 50:
        text_size = 12
    else:
        text_size = 10
    
    # Update text properties
    fig.update_traces(
        texttemplate="%{z:.2f}",
        textfont=dict(
            size=text_size,
            color="black",
            family="Arial Black"
        )
    )
    
    # Calculate responsive dimensions
    base_size = 40
    min_size = 25
    max_size = 60
    
    cell_size = max(min_size, min(max_size, base_size - (num_symbols - 10) * 2))
    height = max(400, min(800, num_symbols * cell_size + 100))
    
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
    
    # Update hovertemplate
    fig.update_traces(
        hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>" +
                      "Correlation: %{z:.3f}<br>" +
                      "<extra></extra>"
    )
    
    return fig

# BTC comparison chart
@callback(
    Output('btc-comparison-chart', 'figure'),
    [Input('btc-correlation-filter', 'value'),
     Input('correlation-type', 'value'),
     Input('correlation-method', 'value'),
     Input('exchange-dropdown', 'value'),
     Input('pair-type-dropdown', 'value'),
     Input('symbol-dropdown', 'value')]
)
def update_btc_comparison(correlation_filter, corr_type, method, exchange, pair_type, selected_symbols):
    if not exchange or not pair_type:
        return go.Figure()
    
    btc_symbol = f'BTC/{pair_type}'
    
    # Get BTC comparison data
    btc_correlations = manager.get_btc_comparison_data(exchange, pair_type, corr_type, method)
    
    if btc_correlations is None or btc_correlations.empty:
        return go.Figure().add_annotation(
            text=f"Comparison data not available for {exchange.upper()}/{pair_type}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
    
    # Determine which symbol is being used as reference
    reference_symbol = btc_symbol
    if exchange.lower() == 'mexc':
        # Check which symbol is actually being used
        exchange_obj = manager.get_exchange(exchange)
        if exchange_obj:
            available_symbols = exchange_obj.get_symbols_for_pair_type(pair_type)
            if f'BTC/{pair_type}' not in available_symbols:
                alternatives = [f'SOL/{pair_type}', f'ETH/{pair_type}', f'BNB/{pair_type}']
                for alt in alternatives:
                    if alt in available_symbols:
                        reference_symbol = alt
                        break
    
    # Filter to selected symbols if any
    if selected_symbols:
        available_symbols = [s for s in selected_symbols if s in btc_correlations.index and s != btc_symbol]
        if available_symbols:
            btc_correlations = btc_correlations.loc[available_symbols]
    
    # Apply correlation filter
    title = f"{reference_symbol} {corr_type.title()} Correlation ({method.title()}) - {exchange.upper()}"
    
    if correlation_filter == 'positive':
        btc_correlations = btc_correlations[btc_correlations > 0]
        title += " - Positive Correlations"
    elif correlation_filter == 'negative':
        btc_correlations = btc_correlations[btc_correlations < 0]
        title += " - Negative Correlations"
    elif correlation_filter == 'strong':
        btc_correlations = btc_correlations[abs(btc_correlations) > 0.5]
        title += " - Strong Correlations (|r|>0.5)"
    elif correlation_filter == 'moderate':
        btc_correlations = btc_correlations[(abs(btc_correlations) > 0.3) & (abs(btc_correlations) < 0.7)]
        title += " - Moderate Correlations (0.3<|r|<0.7)"
    elif correlation_filter == 'weak':
        btc_correlations = btc_correlations[abs(btc_correlations) < 0.3]
        title += " - Weak Correlations (|r|<0.3)"
    
    if len(btc_correlations) == 0:
        return go.Figure().add_annotation(
            text="No correlations found for the selected filter",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
    
    # Create bar chart
    fig = go.Figure()
    
    # Color code bars
    colors = ['red' if x >= 0.7 else 'orange' if x >= 0.4 else 'yellow' if x >= 0.1 
              else 'lightblue' if x >= -0.1 else 'blue' if x >= -0.4 else 'darkblue' 
              for x in btc_correlations.values]
    
    fig.add_trace(go.Bar(
        x=btc_correlations.index,
        y=btc_correlations.values,
        marker=dict(color=colors),
        text=[f'{val:.3f}' for val in btc_correlations.values],
        textposition='outside',
        textfont=dict(size=10),
        hovertemplate="<b>%{x}</b><br>Correlation: %{y:.3f}<extra></extra>"
    ))
    
    # Calculate dimensions
    num_symbols = len(btc_correlations)
    height = max(500, min(1200, num_symbols * 8 + 200))
    tick_font_size = max(6, min(10, 120 / max(1, num_symbols)))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis_title="Cryptocurrency Pairs",
        yaxis_title=f"Correlation with BTC/{pair_type}",
        height=height,
        margin=dict(l=50, r=20, t=80, b=150),
        xaxis=dict(
            tickangle=45,
            tickfont=dict(size=tick_font_size),
            showticklabels=True if num_symbols <= 50 else False
        ),
        yaxis=dict(
            range=[-1, 1],
            tickfont=dict(size=10),
            zeroline=True,
            zerolinecolor='black',
            zerolinewidth=1
        ),
        font=dict(size=10),
        showlegend=False
    )
    
    # Add annotation with count
    fig.add_annotation(
        text=f"Showing {num_symbols} symbols",
        xref="paper", yref="paper",
        x=1, y=1, xanchor='right', yanchor='top',
        showarrow=False, font=dict(size=10),
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="gray",
        borderwidth=1
    )
    
    # Add reference lines
    fig.add_hline(y=0.7, line_dash="dash", line_color="red", opacity=0.5)
    fig.add_hline(y=0.4, line_dash="dash", line_color="orange", opacity=0.5)
    fig.add_hline(y=-0.4, line_dash="dash", line_color="blue", opacity=0.5)
    fig.add_hline(y=-0.7, line_dash="dash", line_color="darkblue", opacity=0.5)
    
    return fig

# Stats table
@callback(
    Output('stats-table', 'children'),
    [Input('symbol-dropdown', 'value'),
     Input('exchange-dropdown', 'value'),
     Input('pair-type-dropdown', 'value')]
)
def update_stats_table(selected_symbols, exchange, pair_type):
    if not selected_symbols or not exchange or not pair_type:
        return html.Div()
    
    exchange_obj = manager.get_exchange(exchange)
    if not exchange_obj:
        return html.Div()
    
    data = exchange_obj.get_data(pair_type)
    if not data or 'meta_df' not in data:
        return html.Div()
    
    meta_df = data['meta_df']
    if meta_df is None or meta_df.empty:
        return html.Div()
    
    available_symbols = [s for s in selected_symbols if s in meta_df.index]
    if not available_symbols:
        return html.Div()
    
    stats_data = meta_df.loc[available_symbols].reset_index()
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
        dbc.CardHeader(f"Symbol Statistics - {exchange.upper()} {pair_type}"),
        dbc.CardBody(table)
    ])

# Exchange status
@callback(
    Output('exchange-status', 'children'),
    [Input('exchange-dropdown', 'value'),
     Input('pair-type-dropdown', 'value')]
)
def update_exchange_status(exchange, pair_type):
    if not exchange or not pair_type:
        return ""
    
    exchange_obj = manager.get_exchange(exchange)
    if not exchange_obj:
        return "Exchange not available"
    
    symbols = exchange_obj.get_symbols_for_pair_type(pair_type)
    return f"Available symbols: {len(symbols)}"

# Export callbacks
@callback(
    Output('btn-csv', 'children'),
    Input('btn-csv', 'n_clicks'),
    prevent_initial_call=True
)
def export_csv_callback(n_clicks):
    if n_clicks:
        manager.export_all_data('csv')
        return "CSV Exported ✓"
    return "Export CSV"

@callback(
    Output('btn-parquet', 'children'),
    Input('btn-parquet', 'n_clicks'),
    prevent_initial_call=True
)
def export_parquet_callback(n_clicks):
    if n_clicks:
        manager.export_all_data('parquet')
        return "Parquet Exported ✓"
    return "Export Parquet"

@callback(
    Output('btn-refresh', 'children'),
    Input('btn-refresh', 'n_clicks'),
    prevent_initial_call=True
)
def refresh_data_callback(n_clicks):
    if n_clicks:
        asyncio.run(manager.collect_all_data())
        return "Data Refreshed ✓"
    return "Refresh Data"

async def initialize_app():
    """Initialize the app with multi-exchange data"""
    print("Initializing Multi-Exchange Correlation Analysis App...")
    
    # Collect data from all exchanges
    await manager.collect_all_data()
    
    status = manager.get_exchange_status()
    print("Exchange Status:")
    for exchange_name, info in status.items():
        print(f"  {exchange_name.upper()}: {info}")

if __name__ == '__main__':
    asyncio.run(initialize_app())
    app.run(debug=True, host='0.0.0.0', port=8050)