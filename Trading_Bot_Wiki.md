
# Trading Bot Development and Deployment Wiki

## Introduction
This documentation outlines the development and deployment process for a Python-based trading bot using Alpaca's API. The bot utilizes machine learning to make predictions based on historical stock data and executes trades automatically when market conditions meet specific criteria.

---

## Setting Up the Trading Bot

### Prerequisites
1. Python 3.9+ installed on your local or server environment.
2. AWS EC2 instance for hosting the bot.
3. Alpaca API credentials for paper/live trading.
4. GitHub repository to manage the bot's code.

### Environment Setup
1. **Install Python Dependencies**:
   - Use a virtual environment:
     ```bash
     python3 -m venv alpaca_env
     source alpaca_env/bin/activate
     pip install -r requirements.txt
     ```

2. **Create `.env` File**:
   - Store Alpaca credentials in `.env`:
     ```plaintext
     ALPACA_API_KEY=your_api_key
     ALPACA_SECRET_KEY=your_secret_key
     ```

3. **Verify the Setup**:
   - Test connectivity to Alpaca's API:
     ```python
     from alpaca.trading.client import TradingClient

     client = TradingClient("your_api_key", "your_secret_key", paper=True)
     print(client.get_account())
     ```

---

## Script Enhancements

### Logging Integration
The script includes logging for debugging and monitoring:
```python
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
    logging.FileHandler("bot.log"), logging.StreamHandler()
])
```

### Dynamic Environment Variable Management
Using `python-dotenv`, the `.env` file is loaded automatically:
```python
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
```

---

## Deploying the Bot on AWS EC2

### Service Configuration
The bot is configured as a systemd service for automatic execution:
1. Create the service file:
   ```bash
   sudo nano /etc/systemd/system/trading-bot.service
   ```
2. Add the following content:
   ```plaintext
   [Unit]
   Description=Trading Bot Service
   After=network.target

   [Service]
   User=ec2-user
   Group=ec2-user
   WorkingDirectory=/home/ec2-user/tradingBot
   ExecStart=/home/ec2-user/tradingBot/alpaca_env/bin/python3 /home/ec2-user/tradingBot/tradingBot_V1.py
   Restart=always
   EnvironmentFile=/home/ec2-user/tradingBot/.env

   [Install]
   WantedBy=multi-user.target
   ```
3. Reload and start the service:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl start trading-bot
   sudo systemctl enable trading-bot
   ```

### Debugging Common Issues
1. **ModuleNotFoundError**:
   - Ensure the virtual environment is used in the service:
     ```plaintext
     ExecStart=/home/ec2-user/tradingBot/alpaca_env/bin/python3 /home/ec2-user/tradingBot/tradingBot_V1.py
     ```

2. **Environment Variables Not Loaded**:
   - Use `EnvironmentFile` in the service file or set variables directly.

3. **Service Fails with Exit Code**:
   - Check logs:
     ```bash
     sudo journalctl -u trading-bot.service -f
     ```

---

## CI/CD Pipeline for GitHub Actions

### Workflow Configuration
Save the following workflow file as `.github/workflows/deploy.yml`:

```yaml
name: Deploy to EC2

on:
  push:
    branches:
      - main

jobs:
  deploy:
    runs-on: ubuntu-latest

    steps:
    - name: Checkout code
      uses: actions/checkout@v3

    - name: Set up SSH
      uses: webfactory/ssh-agent@v0.5.3
      with:
        ssh-private-key: ${{ secrets.EC2_SSH_KEY }}

    - name: Deploy to EC2
      run: |
        ssh -o StrictHostKeyChecking=no ec2-user@${{ secrets.EC2_HOST }} << 'EOF'
          cd /home/ec2-user/tradingBot
          git pull origin main
          sudo systemctl restart trading-bot.service
        EOF
```

### GitHub Secrets Required
- `EC2_SSH_KEY`: Private SSH key for EC2 access.
- `EC2_HOST`: Public IP or hostname of the EC2 instance.

---

## FAQ

### How do I debug service failures?
Use `journalctl` to view detailed logs:
```bash
sudo journalctl -u trading-bot.service -f
```

### How do I manually test the bot?
Run the script manually in the virtual environment:
```bash
source alpaca_env/bin/activate
python tradingBot_V1.py
```

### What happens when the market is closed?
The bot logs the market status and performs predictions based on historical data without executing trades.

---
