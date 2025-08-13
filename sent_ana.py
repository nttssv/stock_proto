import pandas as pd
import subprocess
from datetime import datetime, timedelta
import os
import re

# Load watchlist symbols
watchlist_path = "watchlist.csv"
df_watchlist = pd.read_csv(watchlist_path)
symbols = df_watchlist['Ticker'].dropna().unique().tolist()

# Ensure output folder exists
output_folder = 'llm_analysis'
os.makedirs(output_folder, exist_ok=True)

def analyze_with_ollama(prompt, model="gemma3n:e4b"):
    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt.encode('utf-8'),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        return result.stdout.decode('utf-8').strip()
    except subprocess.CalledProcessError as e:
        print("Error calling Ollama:", e.stderr.decode())
        return None

def sentiment_distribution(df_subset):
    counts = df_subset['sentiment'].value_counts(normalize=True) * 100
    counts = counts.round(2).to_dict()
    for key in ['positive', 'neutral', 'negative']:
        counts.setdefault(key, 0.0)
    return counts

summary_records = []
from datetime import datetime
import pytz

ny_tz = pytz.timezone('America/New_York')

today_str = datetime.now(ny_tz).strftime('%Y-%m-%d')
today_str_file = datetime.now(ny_tz).strftime('%Y%m%d')

for symbol in symbols:
    print(f"\nProcessing symbol: {symbol}")
    file_path = f"news/{symbol}_with_sentiment.csv"
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"File not found: {file_path}, skipping.")
        continue

    # Normalize and parse datetime
    df['sentiment'] = df['sentiment'].str.lower()
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')

    now = datetime.now()
    today_date = datetime(now.year, now.month, now.day)
    weekday = today_date.weekday()

    start_this_week = today_date - timedelta(days=weekday)
    start_last_week = start_this_week - timedelta(days=7)
    end_last_week = start_this_week
    start_today = today_date
    end_today = start_today + timedelta(days=1)

    df_today = df[(df['datetime'] >= start_today) & (df['datetime'] < end_today)]
    df_this_week = df[(df['datetime'] >= start_this_week) & (df['datetime'] < end_today)]
    df_last_week = df[(df['datetime'] >= start_last_week) & (df['datetime'] < end_last_week)]

    def format_rows(df_subset):
        return df_subset[['datetime', 'sentiment', 'headline']].astype(str).apply(
            lambda row: f"{row['datetime']} | {row['sentiment']} | {row['headline']}", axis=1
        ).tolist()

    today_lines = format_rows(df_today)
    this_week_lines = format_rows(df_this_week)
    last_week_lines = format_rows(df_last_week)
    today_sentiment_dist = sentiment_distribution(df_today)

    prompt = f"""
You are a financial news analyst.

You must base your entire analysis strictly on the news data provided below. 
If a detail is not explicitly in the data, state "No information available." 
Do NOT speculate, guess, or hallucinate.

Here is recent news data for ticker {symbol}.

Today’s news (date | sentiment | headline):
{chr(10).join(today_lines) if today_lines else 'No news for today.'}

This week’s news (Monday till today):
{chr(10).join(this_week_lines) if this_week_lines else 'No news for this week.'}

Last week’s news (Monday to Sunday):
{chr(10).join(last_week_lines) if last_week_lines else 'No news for last week.'}

Analyze factually how sentiment evolved from last week through this week up to today. 
Highlight only the key risks or negative factors and positive developments that are explicitly stated in the data. 
Provide a sentiment outlook (positive, neutral, or negative or no news) for today based solely on the provided sentiment values.

At the very end, output ONLY one line in the following exact CSV format, and nothing else:
{symbol},YYYY-MM-DD,positive_pct,neutral_pct,negative_pct,outlook

Example (for illustration only — replace values with the actual analysis):
{symbol},2025-08-12,40.00,35.00,25.00,positive

Where:
- "{symbol}" must appear exactly as shown here (do not substitute it with company name or another ticker).
- date is in YYYY-MM-DD format for today’s date.
- positive_pct, neutral_pct, negative_pct are today's sentiment percentages with 2 decimals, no "%" sign.
- outlook is the dominant sentiment for today: positive, neutral, or negative.
- Do not include any extra spaces, commentary, or text before or after this line.
"""

    llm_response = analyze_with_ollama(prompt)

    print("LLM Analysis:\n")
    print(llm_response if llm_response else "No analysis generated.")

    # Save individual analysis
    df_output = pd.DataFrame({
        'symbol': [symbol],
        'date': [today_str],
        'analysis': [llm_response if llm_response else "No analysis generated."]
    })
    output_file = os.path.join(output_folder, f"{symbol}_analysis_{today_str_file}.csv")
    df_output.to_csv(output_file, index=False)
    print(f"Saved LLM analysis to {output_file}")

    # Extract final summary line from LLM output
    match = re.search(rf"^{symbol},\d{{4}}-\d{{2}}-\d{{2}},([\d\.]+)%?,([\d\.]+)%?,([\d\.]+)%?,(positive|neutral|negative)$",
                      llm_response.strip(), re.MULTILINE | re.IGNORECASE)
    if match:
        positive_pct, neutral_pct, negative_pct, outlook = match.groups()
        summary_records.append({
            'symbol': symbol,
            'date': today_str,
            'positive_pct': float(positive_pct),
            'neutral_pct': float(neutral_pct),
            'negative_pct': float(negative_pct),
            'outlook': outlook.lower()
        })
    else:
        print(f"⚠️ No valid summary line found for {symbol}")

# Save aggregated summary
if summary_records:
    df_summary = pd.DataFrame(summary_records)
    summary_file = os.path.join(output_folder, f"daily_sentiment_summary_{today_str_file}.csv")
    df_summary.to_csv(summary_file, index=False)
    print(f"\nSaved aggregated daily summary to {summary_file}")
else:
    print("\n⚠️ No summary records generated.")