import logging
from flask import Flask, jsonify, render_template
import threading

# Check if Alpaca is installed, if not, provide guidance
try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
except ModuleNotFoundError:
    raise ModuleNotFoundError("Alpaca module not found. Please ensure 'alpaca-trade-api' is installed in your environment.")

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
from dotenv import load_dotenv
import ta
import pandas as pd
import numpy as np
import time
import os

# Flask app for monitoring
app = Flask(__name__)
log_data = []  # Store log entries for the web interface

# Flask Route for the Dashboard
@app.route("/")
def dashboard():
    return render_template("dashboard.html", logs=log_data)

# Flask Route for API Logs
@app.route("/logs")
def get_logs():
    return jsonify(log_data)

# Custom logger to capture logs in memory
class InMemoryLogger(logging.Handler):
    def emit(self, record):
        log_entry = self.format(record)
        log_data.append(log_entry)
        if len(log_data) > 1000:  # Limit stored logs to 1000 entries
            log_data.pop(0)

# Load the .env file
load_dotenv()

# Fetch credentials from environment variables
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

if not API_KEY or not SECRET_KEY:
    raise ValueError("API Key or Secret Key not found in the environment.")

# Set up logging
in_memory_logger = InMemoryLogger()
in_memory_logger.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
    logging.StreamHandler(),
    logging.FileHandler("enhanced_trading_bot.log"),
    in_memory_logger
])

# Initialize Alpaca Clients
data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=False)

# Fetch Historical Data with SIP Compliance
def fetch_data(symbol, start_date, end_date):
    try:
        adjusted_end_date = min(end_date, datetime.now() - timedelta(minutes=15))
        request_params = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=TimeFrame.Day,
            start=start_date,
            end=adjusted_end_date,
            feed='iex'
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
        data['bollinger_high'] = ta.volatility.BollingerBands(close=data['close']).bollinger_hband()
        data['bollinger_low'] = ta.volatility.BollingerBands(close=data['close']).bollinger_lband()
        data['atr'] = ta.volatility.AverageTrueRange(high=data['high'], low=data['low'], close=data['close']).average_true_range()
        data['target'] = (data['close'].shift(-1) > data['close']).astype(int)
        logging.info("Technical indicators added successfully.")
        return data.dropna()
    except Exception as e:
        logging.error(f"Error adding indicators: {e}")
        raise

# Train ML Model
def train_model(data):
    try:
        features = ['rsi', 'ema_10', 'macd', 'bollinger_high', 'bollinger_low', 'atr']
        target = 'target'

        # Validate required columns
        missing_features = [col for col in features if col not in data.columns]
        if missing_features:
            raise ValueError(f"Missing required features for training: {missing_features}")
        if target not in data.columns:
            raise ValueError("Target column 'target' is missing from the dataset.")

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

# Get Current Price
def get_current_price(symbol):
    try:
        latest_bar = data_client.get_stock_bars(StockBarsRequest(
            symbol_or_symbols=symbol, timeframe=TimeFrame.Minute, start=datetime.now() - timedelta(minutes=5)
        ))
        price = float(latest_bar.df['close'].iloc[-1])
        return price
    except Exception as e:
        logging.error(f"Error fetching current price for {symbol}: {e}")
        raise

# Dynamic Position Sizing
def calculate_position_size(balance, risk_percentage, price):
    position_size = (balance * risk_percentage) / price
    return round(position_size, 4)

# Place Order with Stop-Loss and Take-Profit
def place_order_with_risk_management(symbol, balance, risk_percentage, side, stop_loss_pct=0.02, take_profit_pct=0.05):
    try:
        price = get_current_price(symbol)
        quantity = calculate_position_size(balance, risk_percentage, price)
        if quantity <= 0:
            logging.warning(f"Not enough funds to place an order for {symbol}.")
            return

        stop_loss = price * (1 - stop_loss_pct) if side == OrderSide.BUY else price * (1 + stop_loss_pct)
        take_profit = price * (1 + take_profit_pct) if side == OrderSide.BUY else price * (1 - take_profit_pct)

        order = MarketOrderRequest(
            symbol=symbol,
            qty=quantity,
            side=side,
            time_in_force=TimeInForce.GTC,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        trading_client.submit_order(order)
        logging.info(f"Order placed with Stop Loss: {stop_loss}, Take Profit: {take_profit}")
    except Exception as e:
        logging.error(f"Error placing order with risk management for {symbol}: {e}")
        raise

# Trading Bot Main Logic

def trading_bot():
    stock_symbol = "NVDA"
    risk_percentage = 0.02  # Risk 2% of account balance per trade
    retrain_interval = timedelta(days=1)
    last_trained = None

    while True:
        now = datetime.now()

        # Retrain model if needed
        if last_trained is None or (now - last_trained) >= retrain_interval:
            start_date = datetime(2021, 1, 1)
            end_date = now - timedelta(days=1)
            historical_data = fetch_data(stock_symbol, start_date, end_date)
            historical_data = add_indicators(historical_data)
            model, scaler = train_model(historical_data)
            last_trained = now

        # Check if market is open
        if is_market_open():
            logging.info("Market is open. Fetching live data for trading.")
            live_data = fetch_data(stock_symbol, now - timedelta(days=7), now)
            live_data = add_indicators(live_data)

            # Predict and place orders
            features = ['rsi', 'ema_10', 'macd', 'bollinger_high', 'bollinger_low', 'atr']
            live_data_scaled = scaler.transform(live_data[features])
            probabilities = model.predict_proba(live_data_scaled)

            for i, prob in enumerate(probabilities):
                confidence_buy = prob[1] > 0.7
                confidence_sell = prob[0] > 0.7
                
                balance = get_account_balance()
                if confidence_buy:
                    logging.info(f"High confidence buy signal: {prob[1]*100:.2f}% for {stock_symbol}")
                    place_order_with_risk_management(stock_symbol, balance, risk_percentage, OrderSide.BUY)
                elif confidence_sell:
                    logging.info(f"High confidence sell signal: {prob[0]*100:.2f}% for {stock_symbol}")
                    place_order_with_risk_management(stock_symbol, balance, risk_percentage, OrderSide.SELL)

        # Sleep before next iteration
        time.sleep(600)

# Run Flask and Trading Bot in Parallel
if __name__ == "__main__":
    bot_thread = threading.Thread(target=trading_bot)
    bot_thread.start()
    app.run(host="0.0.0.0", port=5000)
