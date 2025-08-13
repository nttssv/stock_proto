import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import glob
import os

st.set_page_config(page_title="Interactive Bubble Chart", layout="wide")

# --- Sidebar: Select Date ---
score_files = glob.glob('score/*_score.csv')
if not score_files:
    st.error("No score files found in the 'score' folder.")
    st.stop()

# Sort files by modification time descending and pick latest
latest_score_file = max(score_files, key=os.path.getmtime)
default_date = os.path.basename(latest_score_file).split('_')[0]

selected_date = st.sidebar.date_input("Select Date", pd.to_datetime(default_date))

# Prepare file paths
sentiment_date_str = selected_date.strftime('%Y%m%d')
sentiment_file = f'llm_analysis/daily_sentiment_summary_{sentiment_date_str}.csv'
score_file = f'score/{selected_date.strftime("%Y-%m-%d")}_score.csv'

try:
    sentiment_df = pd.read_csv(sentiment_file)
    score_df = pd.read_csv(score_file)
except Exception as e:
    st.error(f"Error loading files for {selected_date}: {e}")
    st.stop()

merged = pd.merge(sentiment_df, score_df, on='symbol', how='inner')

oversold_col = 'Indicator'
outlook_col = 'outlook'
ticker_col = 'symbol'

# Map outlook to numeric
outlook_order = ['negative', 'neutral', 'positive']
outlook_map = {k: v for v, k in enumerate(outlook_order)}
merged['outlook_order'] = merged[outlook_col].map(outlook_map)
merged = merged.dropna(subset=['outlook_order'])

grouped_df = merged.groupby([oversold_col, 'outlook_order']).apply(lambda df: df.to_dict('records')).reset_index()

# --- Prepare bubble chart ---
xs, ys, sizes, colors, hover_texts = [], [], [], [], []
for _, row in grouped_df.iterrows():
    group_rows = row[0]
    xs.append(row[oversold_col])
    ys.append(row['outlook_order'])
    count = len(group_rows)
    sizes.append(count * 15)
    colors.append('red' if row[oversold_col] <= 0 else 'green')
    hover_texts.append("<br>".join([d[ticker_col] for d in group_rows]))

trace_bubbles = go.Scatter(
    x=xs, y=ys, mode='markers',
    marker=dict(size=sizes, sizemode='area',
                sizeref=2.*max(sizes)/(40.**2),
                sizemin=4,
                color=colors,
                line=dict(width=1, color='black')),
    text=hover_texts,
    hoverinfo='text',
    name='Bubbles',
    showlegend=False
)

# Labels
text_x, text_y, text_labels = [], [], []
left_offset, right_offset, vertical_spacing = 0.5, 0.15, 0.15

for _, row in grouped_df.iterrows():
    indicator = row[oversold_col]
    net_score = row['outlook_order']
    group_rows = row[0]
    count = len(group_rows)
    center_y = net_score
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
    x=text_x, y=text_y, mode='markers+text',
    marker=dict(size=12, color='rgba(0,0,0,0)', line=dict(width=0)),
    text=text_labels,
    textposition='middle right',
    name='Tickers',
    showlegend=False
)

layout = go.Layout(
    title=f'Oversold-Overbought vs Outlook Bubble Chart ({selected_date})',
    xaxis=dict(title='Oversold-Overbought Indicator'),
    yaxis=dict(title='Sentiment Outlook'),
    hovermode='closest',
    height=700
)

fig = go.Figure(data=[trace_bubbles, trace_labels], layout=layout)
st.plotly_chart(fig, use_container_width=True)

# --- Ticker Details ---
st.subheader("Ticker Details")
ticker_selected = st.selectbox("Select Ticker", merged['symbol'].unique())

file_path = f'llm_analysis/{ticker_selected}_analysis_{sentiment_date_str}.csv'
try:
    df_details = pd.read_csv(file_path)
    st.dataframe(df_details)
except Exception as e:
    st.error(f"No data available for {ticker_selected}: {e}")