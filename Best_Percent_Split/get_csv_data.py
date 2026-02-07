import yfinance as yf
import pandas as pd
from fredapi import Fred

# -------- CONFIG --------
START = "2025-01-01"
END = "2026-01-01"
STOCKS = ["NVDA", "AAPL", "MSFT", "GOOGL", "AMZN"]
FRED_API_KEY = "PUT_YOUR_FRED_API_KEY_HERE"
# ------------------------

# Pull stock prices (CLOSE)
prices = yf.download(STOCKS, start=START, end=END)["Close"]
prices.reset_index(inplace=True)

# Initialize FRED
fred = Fred(api_key=FRED_API_KEY)

# Economic indicators
cpi = fred.get_series("CPIAUCSL")
unemp = fred.get_series("UNRATE")
inflation = fred.get_series("CPIAUCSL").pct_change() * 100
gdp = fred.get_series("GDP")
rates = fred.get_series("FEDFUNDS")

# Convert to DataFrames
def prep(series, name):
    df = series.reset_index()
    df.columns = ["Date", name]
    df["Date"] = pd.to_datetime(df["Date"])
    return df

cpi = prep(cpi, "CPI")
unemp = prep(unemp, "Unemployment")
inflation = prep(inflation, "Inflation_MoM")
gdp = prep(gdp, "GDP")
rates = prep(rates, "Interest_Rate")

# Merge & forward-fill
df = prices.copy()
for econ in [cpi, unemp, inflation, gdp, rates]:
    df = pd.merge(df, econ, on="Date", how="left")

df.ffill(inplace=True)

# Save CSV
df.to_csv("stocks_macro_2025.csv", index=False)

print("CSV GENERATED: stocks_macro_2025.csv")
