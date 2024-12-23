import logging
import yfinance as yf
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
import os
import time

# Load the .env file
load_dotenv()

API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

if not API_KEY or not SECRET_KEY:
    raise ValueError("API Key or Secret Key not found in the environment.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Example log messages
logging.info("Logging is configured successfully.")

# Initialize Alpaca Trading Client
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)  # Use paper=True for testing


# Fetch Historical Data using Yahoo Finance
def fetch_data(symbol, start_date, end_date):
    try:
        df = yf.download(symbol, start=start_date, end=end_date, progress=False)
        df.reset_index(inplace=True)
        df.rename(columns={'Adj Close': 'close'}, inplace=True)
        df['close'] = df['close'].astype(float)
        logging.info(f"Fetched {len(df)} rows of data for {symbol} using Yahoo Finance.")
        return df
    except Exception as e:
        logging.error(f"Error fetching data for {symbol} from Yahoo Finance: {e}")
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
        # Features and target column
        features = ['rsi', 'ema_10', 'macd']
        target = 'target'

        # Extract features and target
        X = data[features]
        y = data[target].values.ravel()  # Ensure y is 1-dimensional

        # Scale features
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Train the model
        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        # Evaluate model accuracy
        accuracy = model.score(X_test, y_test)
        logging.info(f"Model trained successfully with accuracy: {accuracy:.4f}")
        return model, scaler, accuracy
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
        cash_balance = float(account.cash)
        logging.info(f"Current cash balance: ${cash_balance:.2f}")
        return cash_balance
    except Exception as e:
        logging.error(f"Error fetching account balance: {e}")
        raise


# Place Order with Alpaca API
def place_order(symbol, qty, side):
    try:
        order = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.GTC
        )
        trading_client.submit_order(order)
        logging.info(f"Order placed: {side} {qty} shares of {symbol}")
    except Exception as e:
        logging.error(f"Error placing order for {symbol}: {e}")
        raise


# Main Function
if __name__ == "__main__":
    stock_symbol = "NVDA"  # NVIDIA stock symbol
    budget = 50  # Budget per trade
    retrain_interval = timedelta(days=1)  # Retrain every 1 day
    last_trained = None

    while True:
        now = datetime.now()

        # Retrain model if needed
        if last_trained is None or (now - last_trained) >= retrain_interval:
            start_date = datetime(2021, 1, 1)
            end_date = now - timedelta(days=1)
            historical_data = fetch_data(stock_symbol, start_date, end_date)
            historical_data = add_indicators(historical_data)
            model, scaler, accuracy = train_model(historical_data)
            logging.info(f"Model retrained with accuracy: {accuracy:.4f}")
            last_trained = now

        # Check if market is open
        if is_market_open():
            logging.info("Market is open. Fetching live data for trading.")

            # Fetch live data
            live_data_start = now - timedelta(days=7)
            live_data_end = now
            try:
                live_data = fetch_data(stock_symbol, live_data_start, live_data_end)
                live_data = add_indicators(live_data)
                logging.info(f"Live data fetched successfully.")
            except Exception as e:
                logging.warning(f"Skipping trading due to live data error: {e}")
                continue

            # Prediction and trading logic placeholder
            logging.info("Trading logic based on predictions goes here.")
        else:
            logging.info("Market is closed. Predictions on historical data only.")

        # Sleep for 10 minutes before next iteration
        time.sleep(600)
