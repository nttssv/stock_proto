import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import glob
import os
from datetime import datetime, timedelta
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Stock Sentiment Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 1rem;
        opacity: 0.9;
    }
    
    .chart-container {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    
    .analysis-container {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 15px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
    }
    
    .ticker-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
    
    .positive { color: #28a745; }
    .negative { color: #dc3545; }
    .neutral { color: #6c757d; }
    
    .stSelectbox > div > div > div {
        background-color: white;
    }
    
    .stDateInput > div > div > div {
        background-color: white;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 📊 Dashboard Controls")
    
    # Date selection
    score_files = glob.glob('score/*_score.csv')
    if not score_files:
        st.error("No score files found in the 'score' folder.")
        st.stop()

    latest_score_file = max(score_files, key=os.path.getmtime)
    default_date = os.path.basename(latest_score_file).split('_')[0]
    
    selected_date = st.date_input(
        "📅 Select Analysis Date",
        pd.to_datetime(default_date),
        help="Choose the date for sentiment analysis"
    )
    
    st.markdown("---")
    
    # Filter options
    st.markdown("### 🔍 Filters")
    
    # Load data for filters
    sentiment_date_str = selected_date.strftime('%Y%m%d')
    sentiment_file = f'llm_analysis/daily_sentiment_summary_{sentiment_date_str}.csv'
    score_file = f'score/{selected_date.strftime("%Y-%m-%d")}_score.csv'
    
    try:
        sentiment_df = pd.read_csv(sentiment_file)
        score_df = pd.read_csv(score_file)
        merged = pd.merge(sentiment_df, score_df, on='symbol', how='inner')
        
        # Outlook filter
        outlook_options = ['All'] + merged['outlook'].unique().tolist()
        selected_outlook = st.selectbox("Sentiment Outlook", outlook_options)
        
        # Indicator range filter
        min_indicator = merged['Indicator'].min()
        max_indicator = merged['Indicator'].max()
        indicator_range = st.slider(
            "Indicator Range",
            min_value=float(min_indicator),
            max_value=float(max_indicator),
            value=(float(min_indicator), float(max_indicator)),
            step=0.1
        )
        
        # Apply filters
        if selected_outlook != 'All':
            merged = merged[merged['outlook'] == selected_outlook]
        
        merged = merged[
            (merged['Indicator'] >= indicator_range[0]) & 
            (merged['Indicator'] <= indicator_range[1])
        ]
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()
    
    st.markdown("---")
    
    # Quick stats
    st.markdown("### 📈 Quick Stats")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Stocks", len(merged))
        st.metric("Positive", len(merged[merged['outlook'] == 'positive']))
    
    with col2:
        st.metric("Avg Indicator", f"{merged['Indicator'].mean():.2f}")
        st.metric("Negative", len(merged[merged['outlook'] == 'negative']))

# Main content
st.markdown('<h1 class="main-header">📈 Stock Sentiment Analysis Dashboard</h1>', unsafe_allow_html=True)

# Top metrics row
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{len(merged)}</div>
        <div class="metric-label">Total Stocks Analyzed</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    positive_count = len(merged[merged['outlook'] == 'positive'])
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{positive_count}</div>
        <div class="metric-label">Positive Outlook</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    negative_count = len(merged[merged['outlook'] == 'negative'])
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{negative_count}</div>
        <div class="metric-label">Negative Outlook</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    avg_indicator = merged['Indicator'].mean()
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{avg_indicator:.2f}</div>
        <div class="metric-label">Avg Indicator</div>
    </div>
    """, unsafe_allow_html=True)

# Main chart section
st.markdown("## 🎯 Interactive Analysis Chart")

# Prepare bubble chart data
oversold_col = 'Indicator'
outlook_col = 'outlook'
ticker_col = 'symbol'

outlook_order = ['negative', 'neutral', 'positive']
outlook_map = {k: v for v, k in enumerate(outlook_order)}

merged = merged.dropna(subset=[outlook_col])
merged['outlook_numeric'] = merged[outlook_col].map(outlook_map)

# Group by oversold indicator and outlook
grouped_df = merged.groupby([oversold_col, outlook_col]).apply(lambda df: df.to_dict('records')).reset_index()

# Prepare bubble chart
xs, ys, sizes, colors, hover_texts = [], [], [], [], []
for _, row in grouped_df.iterrows():
    group_rows = row[0]
    xs.append(row[oversold_col])
    ys.append(outlook_map[row[outlook_col]])
    count = len(group_rows)
    sizes.append(count * 20)  # Increased bubble size
    colors.append('#dc3545' if row[oversold_col] <= 0 else '#28a745')  # Better colors
    hover_texts.append("<br>".join([d[ticker_col] for d in group_rows]))

# Create bubble chart
trace_bubbles = go.Scatter(
    x=xs, y=ys, mode='markers',
    marker=dict(
        size=sizes, 
        sizemode='area',
        sizeref=2.*max(sizes)/(40.**2),
        sizemin=8,
        color=colors,
        line=dict(width=2, color='white'),
        opacity=0.8
    ),
    text=hover_texts,
    hoverinfo='text',
    name='Stock Groups',
    showlegend=False
)

# Add labels
text_x, text_y, text_labels = [], [], []
left_offset, right_offset, vertical_spacing = 0.4, 0.1, 0.2

for _, row in grouped_df.iterrows():
    indicator = row[oversold_col]
    center_y = outlook_map[row[outlook_col]]
    group_rows = row[0]
    count = len(group_rows)
    half = (count + 1)//2
    left_symbols = group_rows[:half]
    right_symbols = group_rows[half:]
    left_start_y = center_y + vertical_spacing*(len(left_symbols)-1)/2
    right_start_y = center_y + vertical_spacing*(len(right_symbols)-1)/2

    for i, ticker_data in enumerate(left_symbols):
        text_x.append(indicator - left_offset)
        text_y.append(left_start_y - i*vertical_spacing)
        text_labels.append(ticker_data[ticker_col])

    for i, ticker_data in enumerate(right_symbols): 
        text_x.append(indicator + right_offset)
        text_y.append(right_start_y - i*vertical_spacing)
        text_labels.append(ticker_data[ticker_col])

trace_labels = go.Scatter(
    x=text_x, y=text_y, mode='text',
    text=text_labels,
    textposition='middle right',
    textfont=dict(size=10, color='#333'),
    name='Tickers',
    showlegend=False
)

# Enhanced layout
layout = go.Layout(
    title=dict(
        text=f'Stock Sentiment Analysis - {selected_date.strftime("%B %d, %Y")}',
        font=dict(size=20, color='#333')
    ),
    xaxis=dict(
        title='Oversold-Overbought Indicator',
        titlefont=dict(size=14),
        gridcolor='#f0f0f0',
        zerolinecolor='#ccc'
    ),
    yaxis=dict(
        title='Sentiment Outlook',
        tickvals=list(outlook_map.values()),
        ticktext=[o.title() for o in outlook_map.keys()],
        titlefont=dict(size=14),
        gridcolor='#f0f0f0'
    ),
    hovermode='closest',
    height=600,
    plot_bgcolor='white',
    paper_bgcolor='white',
    margin=dict(l=50, r=50, t=80, b=50)
)

fig = go.Figure(data=[trace_bubbles, trace_labels], layout=layout)

# Display chart with interaction
with st.container():
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    
    # Initialize session state
    if 'selected_ticker' not in st.session_state:
        st.session_state.selected_ticker = None

    # Add ticker selection dropdown
    available_tickers = merged['symbol'].unique().tolist()
    selected_ticker = st.selectbox(
        "Select a stock for detailed analysis:",
        ["Click to select..."] + available_tickers,
        key="ticker_selector"
    )
    
    if selected_ticker and selected_ticker != "Click to select...":
        st.session_state.selected_ticker = selected_ticker
    
    # Display the chart
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Analysis section
st.markdown("## 📋 Detailed Analysis")

def display_ticker_analysis(ticker):
    """Display enhanced analysis for selected ticker"""
    if not ticker:
        st.info("👆 Click on any stock in the chart above to view detailed analysis")
        return
    
    st.markdown(f'<div class="analysis-container">', unsafe_allow_html=True)
    
    # Header with ticker info
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown(f"### 📊 {ticker} Analysis")
    
    with col2:
        ticker_data = merged[merged['symbol'] == ticker]
        if not ticker_data.empty:
            outlook = ticker_data.iloc[0]['outlook']
            outlook_class = outlook.lower()
            st.markdown(f'<span class="{outlook_class}">**Outlook: {outlook.title()}**</span>', unsafe_allow_html=True)
    
    with col3:
        if not ticker_data.empty:
            indicator = ticker_data.iloc[0]['Indicator']
            st.metric("Indicator", f"{indicator:.2f}")
    
    # Load detailed analysis
    file_path = f'llm_analysis/{ticker}_analysis_{sentiment_date_str}.csv'
    
    try:
        df_details = pd.read_csv(file_path)
        analysis_cols = [col for col in df_details.columns if 'analysis' in col.lower()]
        
        if analysis_cols:
            full_text = "\n\n".join(
                df_details[analysis_cols].astype(str).agg("\n".join, axis=1)
            )
            
            # Enhanced text area
            st.text_area(
                "📝 Sentiment Analysis Report",
                full_text,
                height=300,
                key=f"analysis_{ticker}",
                help="Detailed sentiment analysis based on news data"
            )
            
            # Add download button
            csv = df_details.to_csv(index=False)
            st.download_button(
                label="📥 Download Analysis Data",
                data=csv,
                file_name=f'{ticker}_analysis_{sentiment_date_str}.csv',
                mime='text/csv'
            )
        else:
            st.warning(f"⚠️ No analysis content found for {ticker}")
            
    except FileNotFoundError:
        st.error(f"❌ Analysis file not found for {ticker}")
    except Exception as e:
        st.error(f"❌ Error loading data for {ticker}: {e}")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Display analysis
display_ticker_analysis(st.session_state.selected_ticker)

# Additional insights section
st.markdown("## 📈 Market Insights")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Sentiment Distribution")
    sentiment_counts = merged['outlook'].value_counts()
    fig_pie = px.pie(
        values=sentiment_counts.values,
        names=sentiment_counts.index,
        color_discrete_map={
            'positive': '#28a745',
            'neutral': '#6c757d',
            'negative': '#dc3545'
        }
    )
    fig_pie.update_layout(height=400, showlegend=True)
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    st.markdown("### Indicator Distribution")
    fig_hist = px.histogram(
        merged, 
        x='Indicator',
        nbins=20,
        color_discrete_sequence=['#667eea']
    )
    fig_hist.update_layout(
        height=400,
        xaxis_title="Indicator Value",
        yaxis_title="Number of Stocks"
    )
    st.plotly_chart(fig_hist, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p>📊 Stock Sentiment Analysis Dashboard | Powered by Streamlit & Plotly</p>
        <p>Data updated daily from financial news sources</p>
    </div>
    """,
    unsafe_allow_html=True
)
