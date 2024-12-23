import logging
from flask import Flask, jsonify, render_template
import threading
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
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
log_data = []

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
    """Enhanced price fetching using IEX feed with improved error handling"""
    def try_fetch_bars(timeframe, window_minutes, retries=3):
        for attempt in range(retries):
            try:
                request_params = StockBarsRequest(
                    symbol_or_symbols=[symbol],
                    timeframe=timeframe,
                    start=datetime.now() - timedelta(minutes=window_minutes),
                    end=datetime.now(),
                    feed='iex'  # Using IEX feed only
                )
                bars = data_client.get_stock_bars(request_params)
                
                if not bars.df.empty and 'close' in bars.df.columns:
                    price = float(bars.df['close'].iloc[-1])
                    logging.info(f"Successfully fetched price for {symbol}: ${price:.2f}")
                    return price
                
                if attempt < retries - 1:  # Don't log for last attempt
                    logging.warning(f"Empty data received for {symbol} on attempt {attempt + 1}")
                time.sleep(2)
                
            except Exception as e:
                if attempt < retries - 1:  # Don't log for last attempt
                    logging.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                time.sleep(2)
        return None

    # Modified time windows
    attempts = [
        (TimeFrame.Minute, 1),    # Try most recent minute first
        (TimeFrame.Minute, 5),
        (TimeFrame.Minute, 15),
        (TimeFrame.Hour, 1),
        (TimeFrame.Day, 1)
    ]

    for timeframe, window in attempts:
        price = try_fetch_bars(timeframe, window)
        if price is not None:
            return price

    logging.error(f"All price fetch attempts failed for {symbol}")
    return None

def prepare_features(data):
    """Prepare and select features based on importance"""
    try:
        # Calculate basic price-based features
        data['returns'] = data['close'].pct_change()
        data['log_returns'] = np.log(data['close']).diff()
        data['volatility'] = data['returns'].rolling(window=20).std()
        
        # Momentum indicators
        data['rsi'] = ta.momentum.RSIIndicator(close=data['close']).rsi()
        data['roc'] = ta.momentum.ROCIndicator(close=data['close']).roc()
        
        # Trend indicators
        data['macd'] = ta.trend.MACD(close=data['close']).macd()
        data['adx'] = ta.trend.ADXIndicator(high=data['high'], low=data['low'], close=data['close']).adx()
        
        # Volume indicators
        data['volume_momentum'] = data['volume'].pct_change()
        data['volume_ma'] = data['volume'].rolling(window=20).mean()
        data['volume_ratio'] = data['volume'] / data['volume_ma']
        
        # Price range and volatility
        data['price_range'] = (data['high'] - data['low']) / data['close']
        
        # Target calculation with multiple thresholds
        returns_threshold = data['returns'].rolling(window=20).std() * 0.5  # Dynamic threshold
        data['target'] = ((data['close'].shift(-1) - data['close']) / data['close'] > returns_threshold).astype(int)
        
        return data.dropna()
    except Exception as e:
        logging.error(f"Error preparing features: {e}")
        raise

def fetch_historical_data(symbol, start_date, end_date):
    """Fetch historical data with proper error handling"""
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
        logging.info(f"Fetched {len(df)} rows of data for {symbol}")
        return df
    except Exception as e:
        logging.error(f"Error fetching historical data: {e}")
        raise

def add_enhanced_indicators(data):
    """Calculate enhanced technical indicators with corrected volume calculations"""
    try:
        # Basic indicators
        data['rsi'] = ta.momentum.RSIIndicator(close=data['close']).rsi()
        data['ema_10'] = ta.trend.EMAIndicator(close=data['close'], window=10).ema_indicator()
        data['macd'] = ta.trend.MACD(close=data['close']).macd()
        
        # Volume-based indicators (corrected)
        data['volume_sma'] = data['volume'].rolling(window=20).mean()  # Simple moving average of volume
        data['volume_ratio'] = data['volume'] / data['volume_sma']
        
        # VWAP calculation
        data['vwap'] = ta.volume.VolumeWeightedAveragePrice(
            high=data['high'],
            low=data['low'],
            close=data['close'],
            volume=data['volume'],
            window=14
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

def train_enhanced_model(data):
    """Train model with improved feature selection and ensemble approach"""
    try:
        features = [
            'rsi', 'roc', 'macd', 'adx', 'volume_momentum', 
            'volume_ratio', 'price_range', 'volatility'
        ]
        
        X = data[features]
        y = data['target']

        # Handle missing and infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(method='ffill').fillna(method='bfill')
        
        # Feature scaling
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Time-based split with validation set
        train_size = int(len(data) * 0.7)
        val_size = int(len(data) * 0.15)
        
        X_train = X_scaled[:train_size]
        X_val = X_scaled[train_size:train_size+val_size]
        X_test = X_scaled[train_size+val_size:]
        
        y_train = y[:train_size]
        y_val = y[train_size:train_size+val_size]
        y_test = y[train_size+val_size:]

        # Random Forest with balanced parameters
        model = RandomForestClassifier(
            n_estimators=50,           # Reduced number of trees
            max_depth=4,               # Shallow trees
            min_samples_split=50,      # Increased minimum samples
            min_samples_leaf=20,       # Increased minimum leaf samples
            random_state=42,
            class_weight='balanced',
            max_features='sqrt'
        )
        
        model.fit(X_train, y_train)
        
        # Calculate metrics
        train_accuracy = model.score(X_train, y_train)
        val_accuracy = model.score(X_val, y_val)
        test_accuracy = model.score(X_test, y_test)
        
        # Validation checks
        if test_accuracy < 0.45 or (train_accuracy - test_accuracy) > 0.2:
            logging.warning("Model shows signs of poor generalization")
            
        feature_importance = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logging.info(f"""Model metrics:
Train accuracy: {train_accuracy:.4f}
Validation accuracy: {val_accuracy:.4f}
Test accuracy: {test_accuracy:.4f}""")
        
        logging.info("\nFeature importance:\n" + feature_importance.to_string())
        
        return model, scaler
    except Exception as e:
        logging.error(f"Error training model: {e}")
        raise

def get_current_positions():
    """Get current positions"""
    try:
        positions = trading_client.get_all_positions()
        return {p.symbol: p for p in positions}
    except Exception as e:
        logging.error(f"Error getting positions: {e}")
        return {}

def calculate_position_size(balance, risk_percentage, price):
    """Calculate position size based on risk management"""
    position_size = (balance * risk_percentage) / price
    return round(position_size, 4)

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
        logging.info(f"Order placed - Side: {side}, Quantity: {quantity}, Stop Loss: ${stop_loss:.2f}, Take Profit: ${take_profit:.2f}")
    except Exception as e:
        logging.error(f"Error placing order: {e}")

def is_market_open():
    """Check if market is open"""
    try:
        clock = trading_client.get_clock()
        return clock.is_open
    except Exception as e:
        logging.error(f"Error checking market status: {e}")
        return False

def trading_bot():
    """Main trading bot logic for a list of symbols."""
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]  # List of stocks to monitor
    risk_percentage = 0.01  # Reduced risk percentage
    retrain_interval = timedelta(hours=12)  # Retrain every 12 hours
    last_trained = None
    confidence_threshold = 0.65  # Adjusted confidence threshold
    max_positions = 2  # Maximum positions
    consecutive_failures = {symbol: 0 for symbol in symbols}
    max_consecutive_failures = 5

    while True:
        try:
            now = datetime.now()

            # Retrain model if needed
            if last_trained is None or (now - last_trained) >= retrain_interval:
                start_date = datetime(2021, 1, 1)
                end_date = now - timedelta(days=1)

                for symbol in symbols:
                    try:
                        historical_data = fetch_historical_data(symbol, start_date, end_date)
                        if not historical_data.empty:
                            historical_data = prepare_features(historical_data)
                            model, scaler = train_enhanced_model(historical_data)
                            last_trained = now
                            consecutive_failures[symbol] = 0
                        else:
                            logging.error(f"Failed to fetch historical data for training: {symbol}")
                            consecutive_failures[symbol] += 1
                    except Exception as e:
                        logging.error(f"Error training model for {symbol}: {e}")
                        consecutive_failures[symbol] += 1

            # Check for too many consecutive failures
            for symbol in symbols:
                if consecutive_failures[symbol] >= max_consecutive_failures:
                    logging.error(f"Too many consecutive failures for {symbol}. Skipping for 15 minutes.")
                    time.sleep(900)  # Wait 15 minutes
                    consecutive_failures[symbol] = 0

            if is_market_open():
                for symbol in symbols:
                    price = get_current_price_enhanced(symbol)
                    if price is None:
                        consecutive_failures[symbol] += 1
                        continue

                    consecutive_failures[symbol] = 0
                    current_positions = get_current_positions()

                    if len(current_positions) >= max_positions:
                        logging.info("Maximum position limit reached")
                        continue

                    live_data = fetch_historical_data(symbol, now - timedelta(days=5), now)
                    if not live_data.empty:
                        live_data = prepare_features(live_data)
                        if all(feature in live_data.columns for feature in model.feature_names_in_):
                            latest_data = live_data[model.feature_names_in_].iloc[-1:]
                            latest_scaled = scaler.transform(latest_data)
                            probabilities = model.predict_proba(latest_scaled)[0]

                            logging.info(f"{symbol} - Prediction probabilities - Buy: {probabilities[1]:.4f}, Sell: {probabilities[0]:.4f}")
                            balance = get_account_balance()
                            if probabilities[1] > confidence_threshold:
                                place_order_with_enhanced_risk_management(
                                    symbol, balance, risk_percentage, OrderSide.BUY, price
                                )
                            elif probabilities[0] > confidence_threshold:
                                place_order_with_enhanced_risk_management(
                                    symbol, balance, risk_percentage, OrderSide.SELL, price
                                )
            else:
                logging.info("Market is closed")
                time.sleep(300)

        except Exception as e:
            logging.error(f"Error in main trading loop: {e}")
            time.sleep(30)

        time.sleep(30)  # Reduced main loop wait time


if __name__ == "__main__":
    # Create template directory and dashboard.html if they don't exist
    os.makedirs('templates', exist_ok=True)
    if not os.path.exists('templates/dashboard.html'):
        with open('templates/dashboard.html', 'w') as f:
            f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>Trading Bot Dashboard</title>
    <meta http-equiv="refresh" content="30">
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .log-entry { margin: 5px 0; padding: 5px; border-bottom: 1px solid #eee; }
        .error { color: red; }
        .warning { color: orange; }
        .info { color: blue; }
    </style>
</head>
<body>
    <h1>Trading Bot Dashboard</h1>
    <div id="logs">
        {% for log in logs %}
        <div class="log-entry {% if 'ERROR' in log %}error{% elif 'WARNING' in log %}warning{% else %}info{% endif %}">
            {{ log }}
        </div>
        {% endfor %}
    </div>
</body>
</html>
            """)

    # Start trading bot in a separate thread
    bot_thread = threading.Thread(target=trading_bot)
    bot_thread.daemon = True
    bot_thread.start()

    # Start Flask app
    app.run(host="0.0.0.0", port=5000)
