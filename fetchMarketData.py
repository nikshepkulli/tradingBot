import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime

import os
# Alpaca API Credentials
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")


# Initialize the data client
data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

# Define request parameters
request_params = StockBarsRequest(
    symbol_or_symbols=["AAPL"],
    timeframe=TimeFrame.Day,
    start=datetime(2021, 1, 1)
)

# Fetch historical data
bars = data_client.get_stock_bars(request_params)

# Convert to DataFrame
df = bars.df
print(df.head())
