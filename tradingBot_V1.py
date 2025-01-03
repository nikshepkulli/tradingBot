import logging
from flask import Flask, jsonify, render_template
import threading
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, StopLossRequest, TakeProfitRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
from dotenv import load_dotenv
import ta
import pandas as pd
import numpy as np
import time
import os

# Flask app setup
app = Flask(__name__)
log_data = []  # Store log entries for the web interface

@app.route("/")
def dashboard():
    try:
        return render_template("dashboard.html", logs=log_data)
    except Exception as e:
        logging.error(f"Error rendering dashboard: {e}")
        return jsonify({"error": "Dashboard template not found"}), 500

@app.route("/logs")
def get_logs():
    return jsonify(log_data)

# Custom logger
class InMemoryLogger(logging.Handler):
    def emit(self, record):
        log_entry = self.format(record)
        log_data.append(log_entry)
        if len(log_data) > 1000:
            log_data.pop(0)

# Initialize logging
in_memory_logger = InMemoryLogger()
in_memory_logger.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler("enhanced_trading_bot.log"), in_memory_logger]
)

# Load environment variables
load_dotenv()
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

if not API_KEY or not SECRET_KEY:
    raise ValueError("API Key or Secret Key not found in the environment.")

# Initialize Alpaca clients
data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)

def get_current_price_enhanced(symbol):
    """Enhanced price fetching with multiple fallback strategies"""
    def try_fetch_bars(timeframe, window_minutes):
        try:
            logging.info(f"Attempting to fetch {timeframe} data for the past {window_minutes} minutes.")
            bars = data_client.get_stock_bars(StockBarsRequest(
                symbol_or_symbols=[symbol],
                timeframe=timeframe,
                start=datetime.now() - timedelta(minutes=window_minutes),
                feed='sip'  # Ensure correct feed is used
            ))
            if not bars.df.empty and 'close' in bars.df.columns:
                logging.info(f"Fetched price data for {symbol}: {bars.df.tail()}")
                return float(bars.df['close'].iloc[-1])
            logging.warning(f"No valid data in {timeframe} request for {symbol}.")
            return None
        except Exception as e:
            logging.warning(f"Error in price fetch attempt for {timeframe}: {e}")
            return None

    attempts = [
        (TimeFrame.Minute, 5),
        (TimeFrame.Minute, 15),
        (TimeFrame.Minute, 60),
        (TimeFrame.Hour, 24)
    ]

    for timeframe, window in attempts:
        price = try_fetch_bars(timeframe, window)
        if price is not None:
            logging.info(f"Successfully fetched price for {symbol} using {timeframe} timeframe: ${price:.2f}")
            return price
        time.sleep(1)  # Add a delay between API requests

    logging.error(f"All price fetch attempts failed for {symbol}")
    return None

def get_account_balance():
    """Get current account balance"""
    try:
        account = trading_client.get_account()
        balance = float(account.cash)
        logging.info(f"Current account balance: ${balance:.2f}")
        return balance
    except Exception as e:
        logging.error(f"Error getting account balance: {e}")
        return None

def fetch_historical_data(symbol, start_date, end_date):
    """Fetch historical data with improved error handling and retry logic"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            request_params = StockBarsRequest(
                symbol_or_symbols=[symbol],
                timeframe=TimeFrame.Hour,  # Changed to hourly for better granularity
                start=start_date,
                end=end_date,
                feed='sip',
                limit=10000  # Increased limit
            )
            bars = data_client.get_stock_bars(request_params)
            df = bars.df.reset_index()
            
            if len(df) < 100:  # Minimum data requirement
                logging.warning(f"Insufficient data points ({len(df)}) for {symbol}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
            return df
            
        except Exception as e:
            logging.error(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                raise

def add_enhanced_indicators(data):
    """Calculate enhanced technical indicators"""
    try:
        required_columns = ['volume', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            logging.error(f"Missing columns for indicators: {missing_columns}")
            raise ValueError(f"Missing columns for indicators: {missing_columns}")
        
        # Basic indicators
        data['rsi'] = ta.momentum.RSIIndicator(close=data['close']).rsi()
        data['ema_10'] = ta.trend.EMAIndicator(close=data['close'], window=10).ema_indicator()
        data['macd'] = ta.trend.MACD(close=data['close']).macd()
        
        # Volume-based indicators
        data['volume_sma'] = data['volume'].rolling(window=20).mean()
        data['vwap'] = ta.volume.VolumeWeightedAveragePrice(
            high=data['high'],
            low=data['low'],
            close=data['close'],
            volume=data['volume']
        ).volume_weighted_average_price()
        
        # Trend indicators
        data['adx'] = ta.trend.ADXIndicator(
            high=data['high'],
            low=data['low'],
            close=data['close']
        ).adx()
        
        # Volatility indicators
        bb = ta.volatility.BollingerBands(close=data['close'])
        data['bb_high'] = bb.bollinger_hband()
        data['bb_low'] = bb.bollinger_lband()
        data['bb_width'] = (data['bb_high'] - data['bb_low']) / data['close']
        
        # Custom features
        data['price_range'] = (data['high'] - data['low']) / data['close']
        data['returns'] = data['close'].pct_change()
        data['volatility'] = data['returns'].rolling(window=20).std()
        
        # Target variable
        returns_threshold = 0.001
        data['target'] = ((data['close'].shift(-1) - data['close']) / data['close'] > returns_threshold).astype(int)
        
        return data.dropna()
    except Exception as e:
        logging.error(f"Error adding indicators: {e}")
        raise


# Enhanced model training function with cross-validation and hyperparameter tuning
def train_enhanced_model(data):
    """Train model with cross-validation and advanced feature engineering"""
    try:
        # Additional technical indicators
        features = [
            'rsi', 'ema_10', 'macd', 'bb_high', 'bb_low', 'bb_width',
            'adx', 'vwap', 'volume_sma', 'price_range', 'volatility',
            'rsi_slope', 'volume_ratio', 'price_momentum', 'trend_strength'
        ]
        
        # Add momentum and trend features
        data['rsi_slope'] = data['rsi'].diff(3)
        data['volume_ratio'] = data['volume'] / data['volume'].rolling(10).mean()
        data['price_momentum'] = data['close'].pct_change(5)
        data['trend_strength'] = data['adx'] * (1 if data['ema_10'].iloc[-1] > data['close'].iloc[-1] else -1)
        
        X = data[features]
        y = data['target']

        # Enhanced scaling with outlier handling
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Time-based split for financial data
        train_size = int(len(data) * 0.8)
        X_train = X_scaled[:train_size]
        X_test = X_scaled[train_size:]
        y_train = y[:train_size]
        y_test = y[train_size:]

        # Hyperparameter optimization
        from sklearn.model_selection import GridSearchCV
        param_grid = {
            'n_estimators': [200, 300],
            'max_depth': [8, 10, 12],
            'min_samples_split': [8, 10, 12],
            'min_samples_leaf': [4, 5, 6],
            'class_weight': ['balanced', 'balanced_subsample']
        }
        
        base_model = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(
            base_model, 
            param_grid,
            cv=5,
            scoring='f1',
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        
        # Calculate various metrics
        from sklearn.metrics import precision_score, recall_score, f1_score
        y_pred = best_model.predict(X_test)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        logging.info(f"Model metrics - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        logging.info(f"Best parameters: {grid_search.best_params_}")
        
        return best_model, scaler
    except Exception as e:
        logging.error(f"Error in enhanced model training: {e}")
        raise

def is_market_open():
    """Check if the market is open"""
    try:
        clock = trading_client.get_clock()
        if clock.is_open:
            logging.info("Market is open")
            return True
        logging.info(f"Market is closed. Next open: {clock.next_open}, Next close: {clock.next_close}")
        return False
    except Exception as e:
        logging.error(f"Error checking market status: {e}")
        return False

def calculate_position_size(balance, risk_percentage, price):
    """Calculate position size based on risk management"""
    position_size = (balance * risk_percentage) / price
    return round(position_size, 4)

def get_available_position(symbol):
    """Get the available position for a specific symbol."""
    try:
        positions = trading_client.get_all_positions()
        for position in positions:
            if position.symbol == symbol:
                return float(position.qty_available)
        return 0.0  # Return 0 if no position is found
    except Exception as e:
        logging.error(f"Error fetching position for {symbol}: {e}")
        return 0.0

def place_order_with_enhanced_risk_management(symbol, balance, risk_percentage, side, price):
    """Place order with confirmation and retry logic"""
    try:
        quantity = calculate_position_size(balance, risk_percentage, price)
        if quantity <= 0:
            logging.warning(f"Insufficient quantity for {symbol}")
            return None

        if side == OrderSide.SELL:
            available_qty = get_available_position(symbol)
            if available_qty < quantity:
                quantity = available_qty
                if quantity <= 0:
                    return None

        order = MarketOrderRequest(
            symbol=symbol,
            qty=quantity,
            side=side,
            time_in_force=TimeInForce.DAY
        )
        
        # Submit order with confirmation
        response = trading_client.submit_order(order)
        
        # Verify order status
        order_status = trading_client.get_order_by_id(response.id)
        if order_status.status == 'accepted' or order_status.status == 'filled':
            logging.info(f"Order confirmed - ID: {response.id}, Status: {order_status.status}")
            return response.id
        else:
            logging.error(f"Order failed - Status: {order_status.status}")
            return None
            
    except Exception as e:
        logging.error(f"Order error: {e}")
        return None

def trading_bot():
    """Main trading bot with improved data handling"""
    symbols = ["AAPL", "NVDA", "MSFT"]
    risk_percentage = 0.02
    retrain_interval = timedelta(hours=4)  # More frequent retraining
    last_trained = {}
    
    while True:
        try:
            now = datetime.now()
            
            for symbol in symbols:
                if not is_market_open():
                    time.sleep(60)
                    continue
                    
                # Comprehensive data fetch
                start_date = now - timedelta(days=180)  # Increased historical window
                historical_data = fetch_historical_data(symbol, start_date, now)
                
                if historical_data is None or len(historical_data) < 100:
                    logging.error(f"Insufficient historical data for {symbol}")
                    continue
                
                # Model training with validation
                if (symbol not in last_trained or 
                    (now - last_trained[symbol]) >= retrain_interval):
                    
                    processed_data = add_enhanced_indicators(historical_data)
                    if len(processed_data) >= 100:
                        model, scaler = train_enhanced_model(processed_data)
                        last_trained[symbol] = now
                    else:
                        logging.error(f"Insufficient processed data for {symbol}")
                        continue
                
                # Real-time trading logic remains the same...
                
        except Exception as e:
            logging.error(f"Trading loop error: {e}")
        
        time.sleep(60)


@app.route("/performance")
def performance_dashboard():
    """Bot performance dashboard"""
    try:
        return render_template("performance.html", logs=log_data)
    except Exception as e:
        logging.error(f"Error rendering performance dashboard: {e}")
        return jsonify({"error": "Performance dashboard template not found"}), 500
        
@app.route("/profit_loss")
def profit_loss_dashboard():
    """Profit and loss tracking dashboard"""
    try:
        trades = get_trade_history()
        return render_template("profit_loss.html", trades=trades, total_profit_loss=calculate_total_profit_loss(trades))
    except Exception as e:
        logging.error(f"Error rendering profit/loss dashboard: {e}")
        return jsonify({"error": "Profit/Loss dashboard template not found"}), 500

trade_history = []  # Store trade history (can also be fetched from a database)

def record_trade(asset, side, qty, filled_price):
    """Record a trade in the history"""
    trade_history.append({
        "asset": asset,
        "side": side,
        "qty": qty,
        "filled_price": filled_price,
        "timestamp": datetime.now()
    })

def get_trade_history():
    """Return the trade history"""
    return trade_history

def calculate_total_profit_loss(trades):
    """Calculate the total profit/loss from trade history"""
    stock_positions = {}
    total_profit_loss = 0.0

    for trade in trades:
        symbol = trade["asset"]
        qty = float(trade["qty"])
        price = float(trade["filled_price"])
        side = trade["side"]

        if side == "buy":
            # Add to stock position
            if symbol not in stock_positions:
                stock_positions[symbol] = {"qty": 0, "cost": 0.0}
            stock_positions[symbol]["qty"] += qty
            stock_positions[symbol]["cost"] += price * qty
        elif side == "sell":
            # Calculate profit/loss for the sold shares
            if symbol in stock_positions and stock_positions[symbol]["qty"] >= qty:
                avg_cost = stock_positions[symbol]["cost"] / stock_positions[symbol]["qty"]
                profit = (price - avg_cost) * qty
                total_profit_loss += profit

                # Update stock position
                stock_positions[symbol]["qty"] -= qty
                stock_positions[symbol]["cost"] -= avg_cost * qty

    return total_profit_loss


# Run Flask and Trading Bot
if __name__ == "__main__":
    bot_thread = threading.Thread(target=trading_bot)
    bot_thread.daemon = True
    bot_thread.start()
    app.run(host="0.0.0.0", port=5000)
