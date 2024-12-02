from alpaca.trading.client import TradingClient
import os
# Alpaca API Credentials
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

# Initialize the client with paper trading enabled
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=False)

# Fetch account details
try:
    account = trading_client.get_account()
    print(account)
except Exception as e:
    print(f"Error fetching account: {e}")
