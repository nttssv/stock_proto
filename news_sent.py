import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from datetime import datetime
import pytz

# Load watchlist symbols
watchlist_path = "watchlist.csv"
df_watchlist = pd.read_csv(watchlist_path)
symbols = df_watchlist['Ticker'].dropna().unique().tolist()

# Load FinBERT model and tokenizer once
model_name = "yiyanghkust/finbert-tone"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

def fix_and_parse_datetime(dt_str, original_str):
    if pd.isna(dt_str) or dt_str == "":
        # No datetime info, keep original string
        return original_str
    if "Today" in dt_str:
        time_part = dt_str.split("Today")[-1].strip()



        ny_tz = pytz.timezone('America/New_York')
        current_date = datetime.now(ny_tz).strftime("%b-%d-%y")
        fixed_str = f"{current_date} {time_part}"
    else:
        fixed_str = dt_str

    for fmt in ("%b-%d-%y %I:%M%p", "%b-%d-%y %H:%M:%S"):
        try:
            return datetime.strptime(fixed_str, fmt)
        except ValueError:
            continue
    # If parsing fails, return original string
    return original_str

for symbol in symbols:
    print(f"Processing {symbol} ...")
    file_path = f"news/{symbol}.csv"
    try:
        df_news = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"File not found for {symbol}: {file_path}, skipping.")
        continue

    # Apply fix_and_parse_datetime row-wise, keeping original datetime string if parsing fails
    df_news['datetime'] = df_news.apply(
        lambda row: fix_and_parse_datetime(row['datetime'], row['datetime']), axis=1
    )

    headlines = df_news['headline'].tolist()
    results = sentiment_analyzer(headlines)
    df_news['sentiment'] = [result['label'] for result in results]

    output_file = f"news/{symbol}_with_sentiment.csv"
    df_news.to_csv(output_file, index=False)

    print(df_news[['datetime', 'headline', 'sentiment']])
    print(f"Saved sentiment analysis results to {output_file}\n")