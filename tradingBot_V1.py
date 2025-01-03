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
    """Fetch historical data with proper error handling"""
    try:
        adjusted_end_date = min(end_date, datetime.now() - timedelta(minutes=15))
        request_params = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=TimeFrame.Day,
            start=start_date,
            end=adjusted_end_date,
            feed='sip'
        )
        bars = data_client.get_stock_bars(request_params)
        df = bars.df.reset_index()
        df['close'] = df['close'].astype(float)
        logging.info(f"Fetched {len(df)} rows of data for {symbol}")
        return df
    except Exception as e:
        logging.error(f"Error fetching historical data: {e}")
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
    """Place order with enhanced risk management"""
    try:
        if price is None:
            logging.warning(f"Skipping order for {symbol} due to missing price data.")
            return

        # Dynamic stop loss and take profit based on volatility
        volatility = 0.02  # You can calculate this dynamically
        stop_loss_pct = max(0.01, volatility)
        take_profit_pct = max(0.02, volatility * 2)

        quantity = calculate_position_size(balance, risk_percentage, price)
        if quantity <= 0:
            logging.warning(f"Not enough funds to place an order for {symbol}.")
            return
        
        # Ensure quantity is fractional (if required by account type)
        quantity = round(quantity, 6)  # Alpaca supports up to 6 decimal places for fractional shares

        # Record the trade
        record_trade(symbol, side, quantity, price)
        
        # Check available position for sell orders
        if side == OrderSide.SELL:
            available_qty = get_available_position(symbol)
            if available_qty < quantity:
                logging.warning(f"Reducing sell quantity for {symbol} from {quantity} to {available_qty} due to insufficient available shares.")
                quantity = available_qty
                if quantity <= 0:
                    logging.warning(f"No shares available to sell for {symbol}. Skipping order.")
                    return

        # Wrap stop_loss and take_profit in their respective request classes
        stop_loss = StopLossRequest(stop_price=price * (1 - stop_loss_pct))
        take_profit = TakeProfitRequest(limit_price=price * (1 + take_profit_pct))

        order = MarketOrderRequest(
            symbol=symbol,
            qty=quantity,
            side=side,
            time_in_force=TimeInForce.DAY,  # Use DAY for fractional orders
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        trading_client.submit_order(order)
        logging.info(f"Order placed - Symbol: {symbol}, Side: {side}, Quantity: {quantity}, Stop Loss: ${stop_loss.stop_price:.2f}, Take Profit: ${take_profit.limit_price:.2f}")
    except Exception as e:
        logging.error(f"Error placing order: {e}")

def trading_bot():
    """Enhanced trading bot with improved accuracy"""
    symbols = ["AAPL", "NVDA", "MSFT"]
    risk_percentage = 0.02
    retrain_interval = timedelta(days=1)
    last_trained = {}
    
    # Dynamic confidence threshold based on market volatility
    def get_confidence_threshold(volatility):
        base_threshold = 0.75
        return min(0.85, base_threshold + volatility * 0.5)
    
    # Enhanced position tracking with profit targets
    positions = {symbol: {'active': False, 'entry_price': 0, 'profit_target': 0} for symbol in symbols}
    
    while True:
        try:
            now = datetime.now()
            
            for symbol in symbols:
                if not is_market_open():
                    logging.info(f"Market closed. Skipping {symbol}.")
                    continue
                
                # Get market data
                price = get_current_price_enhanced(symbol)
                if price is None:
                    continue
                    
                # Calculate market volatility
                live_data = fetch_historical_data(symbol, now - timedelta(days=10), now)
                if live_data.empty:
                    continue
                    
                volatility = live_data['close'].pct_change().std()
                confidence_threshold = get_confidence_threshold(volatility)
                
                # Retrain model if needed
                if symbol not in last_trained or (now - last_trained[symbol]) >= retrain_interval:
                    historical_data = fetch_historical_data(symbol, datetime(2021, 1, 1), now)
                    if not historical_data.empty:
                        historical_data = add_enhanced_indicators(historical_data)
                        model, scaler = train_enhanced_model(historical_data)
                        last_trained[symbol] = now
                
                # Enhanced prediction with market condition checks
                live_data = add_enhanced_indicators(live_data)
                features = live_data.columns[:-1]  # Exclude target column
                live_data_scaled = scaler.transform(live_data[features].iloc[-1:])
                probabilities = model.predict_proba(live_data_scaled)[0]
                
                # Trading logic with enhanced risk management
                balance = get_account_balance()
                if balance is None:
                    continue
                
                position = positions[symbol]
                
                # Buy signal with trend confirmation
                if (probabilities[1] > confidence_threshold and 
                    not position['active'] and 
                    live_data['ema_10'].iloc[-1] > live_data['close'].iloc[-1]):
                    
                    logging.info(f"{symbol}: Strong buy signal ({probabilities[1]*100:.2f}%)")
                    place_order_with_enhanced_risk_management(symbol, balance, risk_percentage, OrderSide.BUY, price)
                    position['active'] = True
                    position['entry_price'] = price
                    position['profit_target'] = price * (1 + volatility * 2)
                
                # Sell signals with multiple conditions
                elif position['active'] and (
                    probabilities[0] > confidence_threshold or  # Strong sell signal
                    price >= position['profit_target'] or      # Profit target reached
                    price <= position['entry_price'] * (1 - volatility)  # Stop loss
                ):
                    logging.info(f"{symbol}: Sell signal triggered")
                    place_order_with_enhanced_risk_management(symbol, balance, risk_percentage, OrderSide.SELL, price)
                    position['active'] = False
        
        except Exception as e:
            logging.error(f"Error in main trading loop: {e}")
        
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
