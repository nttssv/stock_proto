import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import pandas as pd
from datetime import datetime

# Read symbols from watchlist.csv (assumes a column named 'symbol')
watchlist_path = 'watchlist.csv'
df_watchlist = pd.read_csv(watchlist_path)
symbols = df_watchlist['Ticker'].dropna().unique().tolist()

headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/114.0.0.0 Safari/537.36"
}

base_url = "https://finviz.com"
folder_name = "news"
os.makedirs(folder_name, exist_ok=True)

for symbol in symbols:
    print(f"Fetching news for {symbol} ...")
    url = f"https://finviz.com/quote.ashx?t={symbol}"

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Failed to fetch news for {symbol}: {e}")
        continue

    soup = BeautifulSoup(response.text, "html.parser")
    news_table = soup.find("table", id="news-table")

    if news_table is None:
        print(f"No news table found for {symbol}, skipping.")
        continue

    news_items = []
    current_date = None

    for row in news_table.find_all("tr"):
        cols = row.find_all("td")
        if len(cols) >= 2:
            datetime_str = cols[0].get_text(strip=True)  # e.g. "08:52AM" or "Aug-11-25 07:00PM"

            # Determine date and time
            if '-' in datetime_str:
                parts = datetime_str.split()
                current_date = parts[0]  # e.g. "Aug-11-25"
                time_part = parts[1] if len(parts) > 1 else "12:00AM"
            else:
                time_part = datetime_str
                if current_date is None:
                    current_date = datetime.now().strftime("%b-%d-%y")

            full_datetime_str = f"{current_date} {time_part}"

            dt_obj = None
            for fmt in ("%b-%d-%y %I:%M%p", "%b-%d-%y %H:%M:%S"):
                try:
                    dt_obj = datetime.strptime(full_datetime_str, fmt)
                    break
                except ValueError:
                    continue

            link_tag = cols[1].find("a", class_="tab-link-news")
            headline = link_tag.get_text(strip=True) if link_tag else None
            link = urljoin(base_url, link_tag["href"]) if link_tag else None

            news_items.append({
                "datetime": dt_obj if dt_obj else full_datetime_str,
                "headline": headline,
                "url": link
            })

    df_news = pd.DataFrame(news_items)

    # Format datetime column for CSV export
    df_news['datetime'] = df_news['datetime'].apply(
        lambda x: x.strftime("%Y-%m-%d %H:%M:%S") if isinstance(x, datetime) else x
    )

    file_path = os.path.join(folder_name, f"{symbol}.csv")
    df_news.to_csv(file_path, index=False)
    print(f"Saved news data for {symbol} to {file_path}\n")