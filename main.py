from random import random
import warnings
import time
import os
import sqlite3
from datetime import datetime, timedelta
import talib    
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")
start_time = time.time()
my_path = os.getcwd()
print(my_path)

def get_reversal_score(df):
    import talib
    import numpy as np

    bullish_patterns = [
        talib.CDLENGULFING,
        talib.CDLHAMMER,
        talib.CDLMORNINGSTAR,
        talib.CDLMORNINGDOJISTAR,
        talib.CDLPIERCING,
        talib.CDLINVERTEDHAMMER,
        talib.CDLHARAMI,
        talib.CDLHARAMICROSS,
        talib.CDLLADDERBOTTOM,
        talib.CDLCOUNTERATTACK,
        talib.CDLUNIQUE3RIVER,
        talib.CDLTHRUSTING,
        talib.CDL3INSIDE,
        talib.CDLBELTHOLD,
        talib.CDLONNECK,
        talib.CDLABANDONEDBABY
    ]

    bearish_patterns = [
        talib.CDLENGULFING,
        talib.CDLSHOOTINGSTAR,
        talib.CDLEVENINGSTAR,
        talib.CDLEVENINGDOJISTAR,
        talib.CDLDARKCLOUDCOVER,
        talib.CDLHAMMER,
        talib.CDLINVERTEDHAMMER,
        talib.CDLHANGINGMAN,
        talib.CDLKICKING,
        talib.CDLKICKINGBYLENGTH,
        talib.CDL3BLACKCROWS,
        talib.CDLADVANCEBLOCK,
        talib.CDLCOUNTERATTACK,
        talib.CDLBELTHOLD,
        talib.CDLABANDONEDBABY,
        talib.CDLONNECK
    ]

    open_arr = df['Open'].squeeze()
    high_arr = df['High'].squeeze()
    low_arr = df['Low'].squeeze()
    close_arr = df['Close'].squeeze()

    bullish_occurrences = []
    for pattern_func in bullish_patterns:
        pattern_result = pattern_func(open_arr, high_arr, low_arr, close_arr)
        # Check if pattern occurred at least once in last 5 days
        occurred = np.any(pattern_result[-5:] != 0)
        bullish_occurrences.append(occurred)

    bearish_occurrences = []
    for pattern_func in bearish_patterns:
        pattern_result = pattern_func(open_arr, high_arr, low_arr, close_arr)
        occurred = np.any(pattern_result[-5:] != 0)
        bearish_occurrences.append(occurred)

    bullish_count = sum(bullish_occurrences)
    bearish_count = sum(bearish_occurrences)

    df['Bullish_Signal_Count'] = 0
    df['Bearish_Signal_Count'] = 0
    df.at[df.index[-1], 'Bullish_Signal_Count'] = bullish_count
    df.at[df.index[-1], 'Bearish_Signal_Count'] = bearish_count

    print(f"Number of bullish patterns present in last 5 days: {bullish_count}")
    print(f"Number of bearish patterns present in last 5 days: {bearish_count}")
    #print(df[['Bullish_Signal_Count', 'Bearish_Signal_Count']].iloc[-1])

    return df

# ---------------- Technical Indicator Calculations ----------------
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
    print(symbol,start,end)
    return yf.download(symbol, start=start, end=end)

# ---------------- State Classification ----------------
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

import os
import pandas as pd


# ---------------- Main Analysis Update ----------------
def osb_check_db_to_db(watchlist, duration=100):
    df_all = pd.DataFrame()
    csv_path = os.path.join(my_path, "daily_data.csv")

    for count, symbol in enumerate(watchlist, 1):
        if count % 5 == 0:
            print(f"Processed {count} symbols")

        df = get_price_df(symbol, duration)
        if df.empty:
            print(f"No data for {symbol}, skipping")
            continue

        df = get_stochf(df)
        df = get_bband(df)
        df = state_check(df)
        df = get_reversal_score(df)

        OSa_val = df["OSa"].iloc[-1] * -1
        OBa_val = df["OBa"].iloc[-1]
        indicator_val = OSa_val + OBa_val

        reversal_val = df["Bullish_Signal_Count"].iloc[-1] - df["Bearish_Signal_Count"].iloc[-1]

        bullish_count = df["Bullish_Signal_Count"].iloc[-1]
        bearish_count = df["Bearish_Signal_Count"].iloc[-1]

        from datetime import datetime
        import pytz

        ny_tz = pytz.timezone('America/New_York')
        today_str = datetime.now(ny_tz).strftime("%Y-%m-%d")
        
        final_row = {
            "date": today_str,
            "symbol": symbol,
            "OSa": OSa_val,
            "OBa": OBa_val,
            "Indicator": indicator_val,
            "Reversal_Score": reversal_val,
            "Bullish_Signal_Count": bullish_count,
            "Bearish_Signal_Count": bearish_count
        }

        
        csv_path = os.path.join("score", f"{today_str}_score.csv")

        new_row_df = pd.DataFrame([final_row])

        # Append mode if file exists, else write with header
        if os.path.exists(csv_path):
            new_row_df.to_csv(csv_path, mode='a', header=False, index=False)
        else:
            new_row_df.to_csv(csv_path, mode='w', header=True, index=False)

        print(new_row_df)
        print(f"Appended row to {csv_path}")
                
        df_all = pd.concat([df_all, pd.DataFrame([final_row])], ignore_index=True)

    return df_all

# ---------------- Watchlist ----------------
watchlist_full = pd.read_csv('watchlist.csv')['Ticker'].tolist()

# ---------------- Run Analysis ----------------
df = osb_check_db_to_db(watchlist_full)
unique_values = sorted(df["Indicator"].unique())
list_symbol = [[val, df[df["Indicator"] == val]["symbol"].sort_values().values] for val in unique_values]

# ---------------- Plotting OSB Indicator ----------------
fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
df['Indicator'].hist(rwidth=0.5, color='w', align='mid')
plt.xlim(xmin=-6, xmax=6)

for value, symbols in list_symbol:
    color = 'g' if value >= 3 else 'r' if value <= -3 else 'b'
    for idx, sym in enumerate(symbols):
        ax.annotate(sym, xy=(value, idx + 2), color=color)

labels = ['Deep Oversold'] * 2 + ['Oversold'] * 2 + ['Mild Oversold'] * 2 + [0] + ['Mild Overbought'] * 2 + ['Overbought'] * 2 + ['Deep Overbought'] * 2
ax.set_xticklabels(labels)
ax.xaxis.set_ticks(np.arange(*ax.get_xlim(), 1))
plt.xticks(rotation=45)

plt.title('Stochastic & BB indicators', weight='bold')
plt.xlabel('Oversold - Overbought Indicators', weight='bold')
plt.ylabel('# of securities', weight='bold')

filename = f"/static/{datetime.now().strftime('%Y-%m-%d')}.png"
plt.savefig(my_path + filename)



# ---------------- Prepare Labels with Scores ----------------
# Calculate net score
df['Net_Score'] = df['Bullish_Signal_Count'] - df['Bearish_Signal_Count']

score_labels = {
    row['symbol']: f"(Net: {row['Net_Score']}, +: {row['Bullish_Signal_Count']}, -: {row['Bearish_Signal_Count']})"
    for _, row in df.iterrows()
}

# ---------------- Plotting OSB Indicator ----------------
fig3, ax3 = plt.subplots(figsize=(12, 10), constrained_layout=True)  # bigger figure for more vertical space
df['Indicator'].hist(rwidth=0.5, color='w', align='mid')
plt.xlim(xmin=-6, xmax=6)

for value, symbols in list_symbol:
    color = 'g' if value >= 3 else 'r' if value <= -3 else 'b'
    for idx, sym in enumerate(symbols):
        y_pos = idx * 1.1 +1 # increase vertical spacing by multiplying idx by 1.5
        # Plot ticker symbol
        ax3.annotate(sym, xy=(value, y_pos), color=color, fontsize=9)
        # Plot score label one line below (offset y by -0.3)
        ax3.annotate(score_labels.get(sym, ''), xy=(value, y_pos - 0.3), color='black', fontsize=7)

labels = (
    ['Deep Oversold'] * 2 +
    ['Oversold'] * 2 +
    ['Mild Oversold'] * 2 +
    [0] +
    ['Mild Overbought'] * 2 +
    ['Overbought'] * 2 +
    ['Deep Overbought'] * 2
)
ax3.set_xticklabels(labels)
ax3.xaxis.set_ticks(np.arange(*ax.get_xlim(), 1))
plt.xticks(rotation=45)

plt.title('Stochastic & BB indicators', weight='bold')
plt.xlabel('Oversold - Overbought Indicators', weight='bold')
plt.ylabel('# of securities', weight='bold')

plt.tight_layout()  # better layout

filename = f"/static/{datetime.now().strftime('%Y-%m-%d')}_2nd.png"
plt.savefig(my_path + filename)


import collections
import numpy as np
# ---------------- New Chart: Net Score vs Indicator ----------------
fig4, ax4 = plt.subplots(figsize=(20, 14), constrained_layout=True)

x = df['Indicator']
y = df['Net_Score']

# Color coding same as first chart (green for high, red for low, white neutral)
colors = ['green' if val >= 3 else 'red' if val <= -3 else 'white' for val in x]

# All dots black
dot_colors = ['black'] * len(x)
scatter = ax4.scatter(x, y, c=dot_colors, s=60)

ax4.set_xlim(-6, 6)
ax4.set_ylim(-6, 6)
ax4.set_xlabel('Oversold - Overbought Indicators', weight='bold')
ax4.set_ylabel('Net Score (Bullish - Bearish)', weight='bold')
ax4.set_title('Net Score vs Oversold/Overbought Indicator', weight='bold')

# Add x-tick labels
ax4.set_xticks(np.arange(-6, 7, 1))
labels = (
    ['Deep Oversold'] * 2 +
    ['Oversold'] * 2 +
    ['Mild Oversold'] * 2 +
    [0] +
    ['Mild Overbought'] * 2 +
    ['Overbought'] * 2 +
    ['Deep Overbought'] * 2
)
ax4.set_xticklabels(labels, rotation=45)

# Group points by (x,y) to spread labels
grouped_points = {}
for i, row in df.iterrows():
    key = (row['Indicator'], row['Net_Score'])
    grouped_points.setdefault(key, []).append((row['symbol'], row['Bullish_Signal_Count'], row['Bearish_Signal_Count'], colors[i]))

spacing_v = 30  # vertical spacing
spacing_h = 40  # horizontal spacing

import matplotlib.transforms as mtransforms
import random

placed_bboxes = []

for (x_val, y_val), points in grouped_points.items():
    n = len(points)
    start_v = - (n - 1) / 2 * spacing_v  # vertical start for centering

    for idx, (sym, bcount, scount, color) in enumerate(points):
        base_offset_x = 25
        base_offset_y = 20

        offset_y = start_v + idx * spacing_v
        offset_x = base_offset_x if idx % 2 == 0 else -base_offset_x

        text_color = 'white' if color in ['red', 'green'] else 'black'
        label = f"{sym}\nNet:{y_val}, +{bcount}, -{scount}"

        xytext = np.array([offset_x, offset_y + base_offset_y])

        attempt = 0
        max_attempts = 10
        ann = None

        while True:
            # If there's a previous annotation, remove it before creating a new one
            if ann is not None:
                ann.remove()

            ann = ax4.annotate(
                label,
                (x_val, y_val),
                textcoords="offset points",
                xytext=xytext,
                ha='center',
                fontsize=9,
                color=text_color,
                bbox=dict(boxstyle="round,pad=0.2", fc=color, alpha=0.8),
                arrowprops=dict(
                    arrowstyle="->",
                    color='gray',
                    lw=1,
                    connectionstyle="arc3,rad=0.2"
                )
            )

            fig4.canvas.draw()  # force draw to get bbox updated

            renderer = fig4.canvas.get_renderer()
            bbox = ann.get_window_extent(renderer=renderer)

            overlaps = any(bbox.overlaps(other_bbox) for other_bbox in placed_bboxes)

            if overlaps:
                if overlaps:
                    # Move randomly left/right by ±10 and up by +15 points
                    xytext[0] += random.choice([-10, 10])
                    # Move vertically: up if net>0, down if net<=0
                    if y_val > 0:
                        xytext[1] += 15
                    else:
                        xytext[1] -= 30

                    attempt += 1
                    if attempt >= max_attempts:
                        print(f"Max attempts reached for label {label}, placing anyway.")
                        placed_bboxes.append(bbox)
                        break
            else:
                placed_bboxes.append(bbox)
                break

plt.tight_layout()
filename2 = f"/static/{datetime.now().strftime('%Y-%m-%d')}_net_vs_indicator.png"
plt.savefig(my_path + filename2)