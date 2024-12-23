import logging
from datetime import datetime, timedelta
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from flask import Flask, jsonify, render_template
import time

# Flask app setup
app = Flask(__name__)
log_data = []

@app.route("/")
def dashboard():
    return render_template("dashboard.html", logs=log_data)

@app.route("/logs")
def get_logs():
    return jsonify(log_data)

# Initialize logging
class InMemoryLogger(logging.Handler):
    def emit(self, record):
        log_entry = self.format(record)
        log_data.append(log_entry)
        if len(log_data) > 1000:
            log_data.pop(0)

in_memory_logger = InMemoryLogger()
in_memory_logger.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), in_memory_logger]
)

# Function to fetch current price with retries
def try_fetch_bars(data_client, symbol, timeframe, window_minutes, retries=3):
    for attempt in range(retries):
        try:
            request_params = StockBarsRequest(
                symbol_or_symbols=[symbol],
                timeframe=timeframe,
                start=datetime.now() - timedelta(minutes=window_minutes),
                end=datetime.now(),
                feed='iex'  # Default to IEX feed
            )
            bars = data_client.get_stock_bars(request_params)
            if not bars.df.empty and 'close' in bars.df.columns:
                price = float(bars.df['close'].iloc[-1])
                logging.info(f"Successfully fetched price for {symbol}: ${price:.2f}")
                return price
            else:
                logging.warning(f"Empty data received for {symbol} on attempt {attempt + 1}")
        except Exception as e:
            logging.warning(f"Attempt {attempt + 1} failed for {symbol}: {e}")
        time.sleep(2)  # Delay between retries
    logging.error(f"Failed to fetch price for {symbol} after {retries} attempts.")
    return None

# Fallback mechanism for live price fetching
def get_current_price_with_fallback(data_client, symbol):
    for feed in ['iex', 'sip']:
        try:
            price = try_fetch_bars(data_client, symbol, TimeFrame.Minute, 1)
            if price is not None:
                return price
        except Exception as e:
            logging.warning(f"Failed to fetch price for {symbol} using {feed} feed: {e}")
    logging.error(f"All feeds failed for {symbol}.")
    return None

# Check market status
def is_market_open(trading_client):
    try:
        clock = trading_client.get_clock()
        return clock.is_open
    except Exception as e:
        logging.error(f"Error checking market status: {e}")
        return False

# Main trading bot logic
def trading_bot(data_client, trading_client, symbols):
    risk_percentage = 0.01
    max_consecutive_failures = 5
    consecutive_failures = {symbol: 0 for symbol in symbols}

    while True:
        if not is_market_open(trading_client):
            logging.info("Market is closed. Waiting until open.")
            time.sleep(300)
            continue

        for symbol in symbols:
            if consecutive_failures[symbol] >= max_consecutive_failures:
                logging.warning(f"Skipping {symbol} due to repeated failures.")
                continue

            price = get_current_price_with_fallback(data_client, symbol)
            if price is None:
                consecutive_failures[symbol] += 1
                continue

            consecutive_failures[symbol] = 0
            logging.info(f"Processing trading logic for {symbol} with price ${price:.2f}")

        time.sleep(30)

if __name__ == "__main__":
    # Placeholder for initializing data_client and trading_client
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.trading.client import TradingClient

    data_client = StockHistoricalDataClient('<API_KEY>', '<SECRET_KEY>')
    trading_client = TradingClient('<API_KEY>', '<SECRET_KEY>', paper=True)

    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    
    bot_thread = threading.Thread(target=trading_bot, args=(data_client, trading_client, symbols))
    bot_thread.daemon = True
    bot_thread.start()

    app.run(host="0.0.0.0", port=5000)
