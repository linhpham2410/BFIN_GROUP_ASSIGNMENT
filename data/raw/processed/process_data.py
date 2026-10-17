import os
import pandas as pd

os.makedirs("data/processed", exist_ok=True)

prices = pd.read_csv("data/raw/stock_monthly_prices.csv", index_col=0, parse_dates=True)

monthly_returns = prices.pct_change().dropna()

monthly_returns.to_csv("data/processed/stock_monthly_returns.csv")

print("Monthly stock returns successfully calculated and saved!")
print(monthly_returns.head())

