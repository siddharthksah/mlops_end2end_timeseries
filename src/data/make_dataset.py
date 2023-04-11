import yfinance as yf
from datetime import datetime, timedelta

ticker = "^GSPC"  # S&P 500 ticker
# start_date = "2015-01-01"
# Calculate the date 35 years ago
start_date = (datetime.now() - timedelta(days=365*35)).strftime("%Y-%m-%d")
end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

data = yf.download(ticker, start=start_date, end=end_date)