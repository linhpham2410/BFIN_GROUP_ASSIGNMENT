import os
import pandas as pd
import yfinance as yf

tickers = [
    "AAPL",
    "MSFT",
    "JPM",
    "BAC",
    "JNJ",
    "PFE",
    "KO",
    "TSLA",
    "SPY",
    "QQQ",
]

start_date = "2015-01-01"
end_date = "2022-12-31"

os.makedirs("data/raw", exist_ok=True)

data = yf.download(
    tickers, start=start_date, end=end_date, interval="1mo", auto_adjust=True
)

prices = data["Close"]

prices.to_csv("data/raw/stock_monthly_prices.csv")

print("Raw stock price data successfully downloaded and saved to: data/raw/stock_monthly_prices.csv")
print(prices.head())
