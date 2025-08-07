# stock_check

⸻

Stock Oversold/Overbought Indicator Plot

This script analyzes stock indicators (like RSI/Stochastic) from a watchlist.csv file and visualizes symbols that are currently oversold or overbought based on specified thresholds.

⸻

Features
	•	Automatically loads stock indicator values from watchlist.csv
	•	Filters out unwanted tickers (e.g., index funds, certain ETFs)
	•	Annotates stocks on a histogram:
	•	Red: Deep Oversold (≤ -3)
	•	Green: Deep Overbought (≥ 3)
	•	Blue: Neutral / Mid-range

Input
	•	watchlist.csv: Must contain at least the following columns:
	•	symbol

How to use
1. Clone this repo
git clone https://github.com/your-username/your-repo.git
cd your-repo

2. Add your watchlist.csv in the same folder as the script.
3. Run the script: