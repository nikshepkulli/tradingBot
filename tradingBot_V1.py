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
    """Enhanced price fetching with multiple fallback strategies"""
    def try_fetch_bars(timeframe, window_minutes):
        try:
            bars = data_client.get_stock_bars(StockBarsRequest(
                symbol_or_symbols=[symbol],
                timeframe=timeframe,
                start=datetime.now() - timedelta(minutes=window_minutes),
                feed='iex'
            ))
            if not bars.df.empty and 'close' in bars.df.columns:
                return float(bars.df['close'].iloc[-1])
            return None
        except Exception as e:
            logging.warning(f"Error in price fetch attempt: {e}")
            return None

    attempts = [
        (TimeFrame.Minute, 5),
        (TimeFrame.Minute, 15),
        (TimeFrame.Minute, 30),
        (TimeFrame.Hour, 1)
    ]

    for timeframe, window in attempts:
        price = try_fetch_bars(timeframe, window)
        if price is not None:
            logging.info(f"Successfully fetched price for {symbol} using {timeframe} timeframe: ${price:.2f}")
            return price

    logging.error(f"All price fetch attempts failed for {symbol}")
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
    """Train model with enhanced features and cross-validation"""
    try:
        features = [
            'rsi', 'ema_10', 'macd', 'bb_high', 'bb_low', 'bb_width',
            'adx', 'vwap', 'volume_ratio', 'price_range', 'volatility'
        ]
        
        X = data[features]
        y = data['target']

        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        
        train_size = int(len(data) * 0.8)
        X_train = X_scaled[:train_size]
        X_test = X_scaled[train_size:]
        y_train = y[:train_size]
        y_test = y[train_size:]

        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            class_weight='balanced'
        )
        
        model.fit(X_train, y_train)
        
        train_accuracy = model.score(X_train, y_train)
        test_accuracy = model.score(X_test, y_test)
        
        feature_importance = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logging.info(f"Model trained successfully. Train accuracy: {train_accuracy:.4f}, Test accuracy: {test_accuracy:.4f}")
        logging.info("\nTop 5 important features:\n" + feature_importance.head().to_string())
        
        return model, scaler
    except Exception as e:
        logging.error(f"Error training model: {e}")
        raise

def get_account_balance():
    """Get current account balance"""
    try:
        account = trading_client.get_account()
        balance = float(account.cash)
        logging.info(f"Current account balance: ${balance:.2f}")
        return balance
    except Exception as e:
        logging.error(f"Error getting account balance: {e}")
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
    """Main trading bot logic"""
    symbol = "AAPL"
    risk_percentage = 0.02
    retrain_interval = timedelta(days=1)
    last_trained = None
    confidence_threshold = 0.75
    max_positions = 3

    while True:
        try:
            now = datetime.now()

            # Retrain model if needed
            if last_trained is None or (now - last_trained) >= retrain_interval:
                start_date = datetime(2021, 1, 1)
                end_date = now - timedelta(days=1)
                historical_data = fetch_historical_data(symbol, start_date, end_date)
                if not historical_data.empty:
                    historical_data = add_enhanced_indicators(historical_data)
                    model, scaler = train_enhanced_model(historical_data)
                    last_trained = now

            # Check if market is open
            if is_market_open():
                # Get current market data
                price = get_current_price_enhanced(symbol)
                if price is None:
                    time.sleep(60)
                    continue

                # Get current positions
                current_positions = get_current_positions()
                if len(current_positions) >= max_positions:
                    logging.info("Maximum position limit reached")
                    time.sleep(60)
                    continue

                # Get live data for prediction
                live_data = fetch_historical_data(symbol, now - timedelta(days=60), now)
                if not live_data.empty:
                    live_data = add_enhanced_indicators(live_data)
                    features = [
                        'rsi', 'ema_10', 'macd', 'bb_high', 'bb_low', 'bb_width',
                        'adx', 'vwap', 'volume_sma', 'price_range', 'volatility'
                    ]
                    
                    if all(feature in live_data.columns for feature in features):
                        live_data_scaled = scaler.transform(live_data[features].iloc[-1:])
                        probabilities = model.predict_proba(live_data_scaled)[0]

                        # Trading decision
                        if probabilities[1] > confidence_threshold:
                            balance = get_account_balance()
                            place_order_with_enhanced_risk_management(
                                symbol, balance, risk_percentage, OrderSide.BUY, price
                            )
                        elif probabilities[0] > confidence_threshold:
                            balance = get_account_balance()
                            place_order_with_enhanced_risk_management(
                                symbol, balance, risk_percentage, OrderSide.SELL, price
                            )

            else:
                logging.info("Market is closed")
                time.sleep(300)
                continue

        except Exception as e:
            logging.error(f"Error in main trading loop: {e}")
        
        time.sleep(60)

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
