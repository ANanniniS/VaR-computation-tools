import yfinance as yf

# portfolio tickers and historical period
TICKERS = ["YPF", "GGAL", "BMA", "PAM", "AGRO", "BIOX"]
PERIOD = "1y"

def download_prices(tickers = TICKERS,period = PERIOD):
    """Download the adjusted closing prices for each ticker."""
    data = yf.download(tickers=tickers,period=period,auto_adjust=True)["Close"]
    return data.dropna()


if __name__ == "__main__":
    data = download_prices()
    data.to_csv("data/precios.csv")
