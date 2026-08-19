import yfinance as yf

#tickers del portafolio y periodo historico
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN"]
PERIOD = "3y"

def download_prices(tickers = TICKERS,period = PERIOD):
    "Descargo los precios de cierre ajustados para cada TICKER"
    data = yf.download(tickers=tickers,period=period,auto_adjust=True)["Close"]
    print(data.isna().mean())
    return data.dropna()


if __name__ == "__main__":
    data = download_prices()
    data.to_csv("data/precios.csv")
    print(data.head())