import os
import glob
from flask import Flask, render_template_string, request
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as pyo
import math

app = Flask(__name__)

@app.route('/')
def index():
    folder = "score"
    all_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.csv')]

    df_list = []
    for file in all_files:
        df_temp = pd.read_csv(file)
        df_list.append(df_temp)

    df = pd.concat(df_list, ignore_index=True)
    print(df)
    
    df['Net_Score'] = df['Bullish_Signal_Count'] - df['Bearish_Signal_Count']

    # Get unique dates and symbols for filter dropdowns
    unique_dates = sorted(df['date'].unique())
    unique_symbols = sorted(df['symbol'].unique())

    # Get filters from query parameters
    selected_date = request.args.get('date', default=None)
    selected_symbol = request.args.get('symbol', default=None)

    # Filter dataframe based on selected filters
    if selected_date:
        df = df[df['date'] == selected_date]
    #if selected_symbol:
    #    df = df[df['symbol'] == selected_symbol]

    grouped = df.groupby(['Indicator', 'Net_Score'])

    scatter_x = []
    scatter_y = []
    scatter_text = []
    scatter_hover = []
    scatter_marker_size = []
    scatter_marker_color = []

    for (indicator, net_score), group in grouped:
        symbols = group['symbol'].tolist()
        bullish = group['Bullish_Signal_Count'].tolist()
        bearish = group['Bearish_Signal_Count'].tolist()
        dates = group['date'].tolist()

        count = len(symbols)
        base_x = indicator
        base_y = net_score

        color = 'green' if indicator >= 3 else 'red' if indicator <= -3 else 'blue'
        size = count * 10

        scatter_x.append(base_x)
        scatter_y.append(base_y)
        scatter_text.append('')  # no label on bubble
        hover_text = "<br>".join([
            f"{sym}: Net={net_score}, +{b}, -{br}, Date={dt}"
            for sym, b, br, dt in zip(symbols, bullish, bearish, dates)
        ])
        scatter_hover.append(hover_text)
        scatter_marker_size.append(size)
        scatter_marker_color.append(color)

    text_x = []
    text_y = []
    text_labels = []
    arrow_annotations = []

    horizontal_offset = 0.3
    vertical_spacing = 0.25
    vertical_down_shift = 0.1

    for (indicator, net_score), group in grouped:
        symbols = group['symbol'].tolist()
        count = len(symbols)
        size = count * 10
        radius = math.sqrt(size / math.pi) / 40

        if net_score >= 0:
            base_y = net_score + radius - vertical_down_shift
        else:
            base_y = net_score - radius - vertical_down_shift

        half = (count + 1) // 2

        for i, sym in enumerate(symbols[:half]):
            label_x = indicator - horizontal_offset
            label_y = base_y + i * vertical_spacing
            text_x.append(label_x)
            text_y.append(label_y)
            text_labels.append(sym)

            arrow_annotations.append(dict(
                x=indicator, y=net_score,
                ax=label_x, ay=label_y,
                xref='x', yref='y',
                axref='x', ayref='y',
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1,
                arrowcolor='gray',
            ))

        for i, sym in enumerate(symbols[half:]):
            label_x = indicator + horizontal_offset
            label_y = base_y + i * vertical_spacing
            text_x.append(label_x)
            text_y.append(label_y)
            text_labels.append(sym)

            arrow_annotations.append(dict(
                x=indicator, y=net_score,
                ax=label_x, ay=label_y,
                xref='x', yref='y',
                axref='x', ayref='y',
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1,
                arrowcolor='gray',
            ))

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=scatter_x,
        y=scatter_y,
        mode='markers',
        marker=dict(
            size=scatter_marker_size,
            color=scatter_marker_color,
            line=dict(width=1, color='black'),
            sizemode='area',
            sizeref=2.0 * max(scatter_marker_size) / (40. ** 2) if scatter_marker_size else 1,
            sizemin=4,
        ),
        hoverinfo='text',
        hovertext=scatter_hover,
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=text_x,
        y=text_y,
        mode='text',
        text=text_labels,
        textposition='middle center',
        textfont=dict(size=10, color='black'),
        hoverinfo='skip',
        showlegend=False
    ))

    fig.update_layout(
        title='Net Score vs Oversold/Overbought Indicator',
        xaxis=dict(
            title='Oversold - Overbought Indicators',
            tickmode='array',
            tickvals=list(range(-6, 7)),
            ticktext=['Deep Oversold', 'Deep Oversold', 'Oversold', 'Oversold', 'Mild Oversold', 'Mild Oversold',
                      'Neutral', 'Mild Overbought', 'Mild Overbought', 'Overbought', 'Overbought', 'Deep Overbought',
                      'Deep Overbought'],
            tickangle=45
        ),
        yaxis=dict(
            title='Net Score (Bullish - Bearish)',
        ),
        height=700,
        margin=dict(l=40, r=40, t=60, b=150),
        annotations=arrow_annotations
    )

    graph_html = pyo.plot(fig, output_type='div')

    # Load LLM analysis for selected_symbol
    llm_analysis_text = ""
    if selected_symbol:
        import glob
        analysis_files = sorted(
            glob.glob(f"llm_analysis/{selected_symbol}_analysis_*.csv"), reverse=True
        )
        if analysis_files:
            try:
                df_analysis = pd.read_csv(analysis_files[0])
                llm_analysis_text = df_analysis.at[0, 'analysis']
            except Exception as e:
                llm_analysis_text = f"Error loading analysis file: {e}"

    # Pass llm_analysis_text and selected_symbol to the template
    return render_template_string("""
    <!-- your existing HTML template -->

    <form method="get">
    <label for="date">Filter by Date:</label>
    <select id="date" name="date" onchange="this.form.submit()">
        <option value="">All Dates</option>
        {% for d in unique_dates %}
        <option value="{{ d }}" {% if d == selected_date %}selected{% endif %}>{{ d }}</option>
        {% endfor %}
    </select>

    <label for="symbol">Select Ticker for Analysis:</label>
    <select id="symbol" name="symbol" onchange="this.form.submit()">
        <option value="">Select Ticker</option>
        {% for s in unique_symbols %}
        <option value="{{ s }}" {% if s == selected_symbol %}selected{% endif %}>{{ s }}</option>
        {% endfor %}
    </select>
    </form>

    <div>
    {{ graph_html|safe }}
    </div>

    {% if llm_analysis_text %}
    <h2>LLM Analysis for {{ selected_symbol }}</h2>
    <textarea rows="15" cols="100" readonly style="width:100%; font-family: monospace;">{{ llm_analysis_text }}</textarea>
    {% endif %}

    <!-- rest of your HTML -->
    """, graph_html=graph_html, unique_dates=unique_dates, unique_symbols=unique_symbols,
        selected_date=selected_date, selected_symbol=selected_symbol, llm_analysis_text=llm_analysis_text)


if __name__ == '__main__':
    app.run(debug=True)