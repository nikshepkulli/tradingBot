import logging
import yfinance as yf
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from dotenv import load_dotenv
from datetime import datetime, timedelta
import os

# Load environment variables
load_dotenv()

API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

if not API_KEY or not SECRET_KEY:
    raise ValueError("API Key or Secret Key not found in the environment.")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
    logging.StreamHandler(),
    logging.FileHandler("trading_bot.log")
])

# Initialize Alpaca Trading Client
try:
    trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)  # Use paper=True for testing
    logging.info("Connected to Alpaca Trading API.")
except Exception as e:
    logging.error(f"Error initializing Alpaca client: {e}")
    raise

# Fetch Historical Data using Yahoo Finance
def fetch_data(symbol, start_date, end_date):
    try:
        # Adjust end_date for SIP compliance
        adjusted_end_date = min(end_date, datetime.now() - timedelta(minutes=15))
        df = yf.download(symbol, start=start_date, end=adjusted_end_date, progress=False)
        df.reset_index(inplace=True)
        df.rename(columns={'Adj Close': 'close'}, inplace=True)
        logging.info(f"Fetched {len(df)} rows of data for {symbol} from Yahoo Finance.")
        return df
    except Exception as e:
        logging.error(f"Error fetching data for {symbol} from Yahoo Finance: {e}")
        raise

# Example: Check if Market is Open
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

# Example Order Placement
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
    start_date = datetime(2021, 1, 1)
    end_date = datetime.now()

    try:
        # Fetch historical data
        historical_data = fetch_data(stock_symbol, start_date, end_date)
        
        # Example: Check market status
        if is_market_open():
            logging.info("Market is open. Example trading logic goes here.")
        else:
            logging.info("Market is closed. No trades can be executed.")
    except Exception as e:
        logging.error(f"Error in main execution: {e}")
