import logging
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
from dotenv import load_dotenv
import ta
import pandas as pd
import time
import os

# Load the .env file
load_dotenv()

# Fetch credentials from environment variables
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

if not API_KEY or not SECRET_KEY:
    raise ValueError("API Key or Secret Key not found in the environment.")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
    logging.StreamHandler(),  # Log to console
    logging.FileHandler("trading_bot.log")  # Log to file
])

# Initialize Alpaca Clients
data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)  # Set paper=False for live trading

# Check if a Symbol is Tradable on IEX
def is_symbol_tradable(symbol):
    try:
        asset = trading_client.get_asset(symbol)
        if asset.tradable and asset.exchange == "IEX":
            logging.info(f"{symbol} is tradable on IEX.")
            return True
        logging.warning(f"{symbol} is not tradable on IEX.")
        return False
    except Exception as e:
        logging.error(f"Error checking symbol {symbol}: {e}")
        return False

# Fetch Historical Data with IEX Feed
def fetch_data(symbol, start_date, end_date):
    try:
        # Adjust the end_date to 15 minutes before the current time for SIP compliance
        adjusted_end_date = min(end_date, datetime.now() - timedelta(minutes=15))
        request_params = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=TimeFrame.Day,
            start=start_date,
            end=adjusted_end_date,
            feed='iex'  # Use IEX data feed
        )
        bars = data_client.get_stock_bars(request_params)
        df = bars.df.reset_index()
        df['close'] = df['close'].astype(float)
        logging.info(f"Fetched {len(df)} rows of data for {symbol}. Adjusted end date: {adjusted_end_date}.")
        return df
    except Exception as e:
        logging.error(f"Error fetching data for {symbol}: {e}")
        raise

# Add Technical Indicators
def add_indicators(data):
    try:
        data['rsi'] = ta.momentum.RSIIndicator(close=data['close']).rsi()
        data['ema_10'] = ta.trend.EMAIndicator(close=data['close'], window=10).ema_indicator()
        data['macd'] = ta.trend.MACD(close=data['close']).macd()
        data['target'] = (data['close'].shift(-1) > data['close']).astype(int)  # Predict if price will go up
        logging.info("Technical indicators added successfully.")
        return data.dropna()
    except Exception as e:
        logging.error(f"Error adding indicators: {e}")
        raise

# Train ML Model
def train_model(data):
    try:
        features = ['rsi', 'ema_10', 'macd']
        target = 'target'
        X = data[features]
        y = data[target]

        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        logging.info(f"Model trained successfully with accuracy: {accuracy:.4f}")
        return model, scaler
    except Exception as e:
        logging.error(f"Error training model: {e}")
        raise

# Check if Market is Open
def is_market_open():
    try:
        clock = trading_client.get_clock()
        if clock.is_open:
            logging.info("Market is open.")
            return True
        logging.info(f"Market closed. Next open: {clock.next_open}, Next close: {clock.next_close}")
        return False
    except Exception as e:
        logging.error(f"Error checking market status: {e}")
        raise

# Get Account Balance
def get_account_balance():
    try:
        account = trading_client.get_account()
        cash_balance = float(account.cash)  # Your available cash balance
        logging.info(f"Current cash balance: ${cash_balance:.2f}")
        return cash_balance
    except Exception as e:
        logging.error(f"Error fetching account balance: {e}")
        raise

# Main Function
if __name__ == "__main__":
    stock_symbol = "NVDA"  # NVIDIA stock symbol
    budget = 50  # Budget per trade
    retrain_interval = timedelta(days=1)  # Retrain every 1 day
    last_trained = None

    # Check if the symbol is tradable on IEX before proceeding
    if not is_symbol_tradable(stock_symbol):
        logging.error(f"Symbol {stock_symbol} is not tradable on IEX. Exiting.")
        exit(1)

    while True:
        now = datetime.now()

        # Retrain model if needed
        if last_trained is None or (now - last_trained) >= retrain_interval:
            start_date = datetime(2021, 1, 1)  # Historical data start
            end_date = now - timedelta(days=1)  # Up to yesterday
            model, scaler = train_model(fetch_data(stock_symbol, start_date, end_date))
            last_trained = now

        # Check market status
        if is_market_open():
            logging.info("Market is open. Fetching live data for trading.")

            # Fetch live data for predictions
            live_data_start = now - timedelta(days=7)  # Recent data
            live_data_end = now
            try:
                live_data = fetch_data(stock_symbol, live_data_start, live_data_end)
                live_data = add_indicators(live_data)
            except Exception as e:
                logging.warning(f"Skipping trading due to live data error: {e}")
                continue

            # Predict and trade
            # Add prediction and order execution logic as per your use case
            logging.info("Predictions and trade logic should be added here.")
        else:
            logging.info("Market is closed. Predictions on historical data only.")

        # Sleep for 10 minutes before the next check
        time.sleep(600)
