import pandas as pd

# Load the CSV file into a DataFrame
file_path = 'finviz.csv'  # Update this path with the actual location of your CSV file
df = pd.read_csv(file_path)

# Extract the 'Ticker' column
tickers = df['Ticker'].tolist()
df2 = pd.read_csv('watchlist.csv')
# Print the list of tickers
ticker2s = df2['Ticker'].tolist()

#print(tickers)
#print(ticker2s)
mergedlist= list(set(tickers).union(set(ticker2s)))
print(mergedlist)

# Save the merged list to a new CSV file
output_file_path = 'merged_tickers.csv'
merged_df = pd.DataFrame(mergedlist, columns=['Ticker'])
merged_df.to_csv(output_file_path, index=False)
print(f"Merged list saved to {output_file_path}")
# Ensure the Flask app is running before testing this script
# You can run this script to generate the merged CSV file with tickers from both sources.
# Ensure the Flask app is running before testing this script
# You can run this script to generate the merged CSV file with tickers from both sources.
# Ensure the Flask app is running before testing this script
# You can run this script to generate the merged CSV file with tickers from both sources.
# Ensure the Flask app is running before testing this script   


df['Ticker'].to_csv('watchlist.csv', index=False)