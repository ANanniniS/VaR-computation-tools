"""
Download adjusted closing prices used as input by the rest of the project.
"""

import yfinance as yf

# Portfolio tickers and the date range to download.
# Companies recognizable to someone from Rosario, Argentina, all priced in
# USD (ADRs): YPF (oil), GGAL and BMA (banks), PAM (energy), AGRO (agriculture,
# core region), BIOX (Bioceres, agricultural biotech founded in Rosario).
TICKERS = ["YPF", "GGAL", "BMA", "PAM", "AGRO", "BIOX"]

# Fixed dates (not relative to "today") so results are reproducible
# regardless of when the script is run.
START = "2020-08-01"
END = "2025-08-01"


def download_prices(tickers=TICKERS, start=START, end=END):
    """
    Download the adjusted closing price of each ticker between two dates.

    Rows with any missing price are dropped, so every returned row has a
    price for every ticker.

    Parameters
    ----------
    tickers : list of str, default=TICKERS
        Tickers to download.
    start : str, default=START
        Start date, format 'YYYY-MM-DD' (inclusive).
    end : str, default=END
        End date, format 'YYYY-MM-DD' (exclusive, as per yfinance).

    Returns
    -------
    pd.DataFrame
        Adjusted closing prices indexed by date, with one column per
        ticker.
    """
    data = yf.download(tickers, start=start, end=end, auto_adjust=True)["Close"]
    return data.dropna()  # drop rows with missing data


if __name__ == "__main__":
    prices = download_prices()
    prices.to_csv("data/precios.csv")
    print(prices.tail())