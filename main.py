import warnings
import time
import os
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import numpy as np

# Suppress warnings
warnings.filterwarnings("ignore")

# Track execution time
start_time = time.time()

# Set working directory
my_path = os.getcwd()
print(my_path)

# --- Technical Indicator Calculations ---
def get_stochf(df, k_period=5, avg=3):
    df = df.copy()
    df['Lp'] = df['Low'].rolling(window=k_period).min()
    df['Hp'] = df['High'].rolling(window=k_period).max()
    df['%K'] = 100 * ((df['Close'].squeeze() - df['Lp']) / (df['Hp'] - df['Lp']))
    df['FullK'] = df['%K'].rolling(window=avg).mean()
    df['FullD'] = df['FullK'].rolling(window=3).mean()
    return df

def get_bband(df, no_of_std=2):
    window = 21
    rolling_mean = df['Close'].rolling(window).mean()
    rolling_std = df['Close'].rolling(window).std()
    df['BBM'] = rolling_mean
    df['BBH'] = rolling_mean + (rolling_std * no_of_std)
    df['BBL'] = rolling_mean - (rolling_std * no_of_std)
    df['Volume_pct_change'] = df['Volume'].pct_change() * 100
    df['Close_price_pct_change'] = df['Close'].pct_change() * 100
    return df

def get_price_df(symbol, duration=0):
    end = datetime.today()
    start = end - timedelta(days=duration)
    return yf.download(symbol, start=start, end=end)

# --- State Classification ---
def state_check(df):
    KD_OS_threshold = 30
    KD_OB_threshold = 100 - KD_OS_threshold

    df['BBL_trig'] = df['BBL'] >= df['Close'].squeeze()
    df['BBH_trig'] = df['BBH'] <= df['Close'].squeeze()

    df['StL_t'] = 0
    df.loc[(df.FullK <= KD_OS_threshold) & (df.FullD > KD_OS_threshold), 'StL_t'] = 1
    df.loc[(df.FullK <= KD_OS_threshold) & (df.FullD <= KD_OS_threshold), 'StL_t'] = 2
    df.loc[(df.FullK > KD_OS_threshold) & (df.FullD <= KD_OS_threshold), 'StL_t'] = 3

    df['StH_t'] = 0
    df.loc[(df.FullK >= KD_OB_threshold) & (df.FullD < KD_OB_threshold), 'StH_t'] = 1
    df.loc[(df.FullK >= KD_OB_threshold) & (df.FullD >= KD_OB_threshold), 'StH_t'] = 2
    df.loc[(df.FullK < KD_OB_threshold) & (df.FullD >= KD_OB_threshold), 'StH_t'] = 3

    df['OS'] = 0
    df.loc[((df.StL_t == 1) | (df.StL_t == 3)) & (~df.BBL_trig) | ((df.StL_t == 0) & df.BBL_trig), 'OS'] = 1
    df.loc[((df.StL_t == 1) & df.BBL_trig) | ((df.StL_t >= 2) & ~df.BBL_trig), 'OS'] = 2
    df.loc[(df.StL_t >= 2) & df.BBL_trig, 'OS'] = 3
    df['OSa'] = df['OS'].rolling(2).sum().fillna(0)
    df.loc[df.OS == 0, 'OSa'] = 0

    df['OB'] = 0
    df.loc[((df.StH_t == 1) | (df.StH_t == 3)) & (~df.BBH_trig) | ((df.StH_t == 0) & df.BBH_trig), 'OB'] = 1
    df.loc[((df.StH_t == 1) & df.BBH_trig) | ((df.StH_t >= 2) & ~df.BBH_trig), 'OB'] = 2
    df.loc[(df.StH_t >= 2) & df.BBH_trig, 'OB'] = 3
    df['OBa'] = df['OB'].rolling(2).sum().fillna(0)
    df.loc[df.OB == 0, 'OBa'] = 0

    return df

# --- Main Indicator Checker ---
def osb_check_db(watchlist, duration=100):
    df_all = pd.DataFrame()
    for count, symbol in enumerate(watchlist, 1):
        if count % 5 == 0:
            print(f"Processed {count} symbols")

        df = get_price_df(symbol, duration)
        df = get_stochf(df)
        df = get_bband(df)
        df = state_check(df)

        final_row = {
            "symbol": symbol,
            "OSa": df["OSa"].iloc[-1] * -1,
            "OSb": df["OBa"].iloc[-1],
            "Indicator": df["OSa"].iloc[-1] * -1 + df["OBa"].iloc[-1]
        }
        df_all = pd.concat([df_all, pd.DataFrame([final_row])], ignore_index=True)

    return df_all

# --- Watchlist ---

watchlist_full = pd.read_csv('watchlist.csv')['Ticker'].tolist()

# --- Run Analysis ---
df = osb_check_db(watchlist_full)
unique_values = sorted(df["Indicator"].unique())
list_symbol = [[val, df[df["Indicator"] == val]["symbol"].sort_values().values] for val in unique_values]

# --- Plotting ---
fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
df['Indicator'].hist(rwidth=0.5, color='w', align='mid')
plt.xlim(xmin=-6, xmax=6)

for value, symbols in list_symbol:
    color = 'g' if value >= 3 else 'r' if value <= -3 else 'b'
    for idx, sym in enumerate(symbols):
        ax.annotate(sym, xy=(value, idx + 1), color=color)

# Set X-axis labels
labels = ['Deep Oversold'] * 2 + ['Oversold'] * 2 + ['Mild Oversold'] * 2 + [0] + ['Mild Overbought'] * 2 + ['Overbought'] * 2 + ['Deep Overbought'] * 2
ax.set_xticklabels(labels)
ax.xaxis.set_ticks(np.arange(*ax.get_xlim(), 1))
plt.xticks(rotation=45)

plt.title('Stochastic & BB indicators', weight='bold')
plt.xlabel('Oversold - Overbought Indicators', weight='bold')
plt.ylabel('# of securities', weight='bold')

filename = f"/static/{datetime.now().strftime('%Y-%m-%d')}.png"
plt.savefig(my_path + filename)
print("My program took", time.time() - start_time, "to run")