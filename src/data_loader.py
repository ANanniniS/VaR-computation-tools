import yfinance as yf

# Tickers del portafolio y rango de fechas a descargar.
# Empresas reconocibles para alguien de Rosario, Argentina, todas en USD (ADRs):
# YPF (petrolera), GGAL y BMA (bancos), PAM (energia), AGRO (agro zona nucleo),
# BIOX (Bioceres, biotecnologia agricola fundada en Rosario)
TICKERS = ["YPF", "GGAL", "BMA", "PAM", "AGRO", "BIOX"]

# Fechas fijas (no relativas a "hoy") para que los resultados sean
# reproducibles sin importar cuándo se corra el script.
START = "2020-08-01"
END = "2025-08-01"


def download_prices(tickers=TICKERS, start=START, end=END):
    """
    Descarga precios de cierre ajustado para cada ticker entre start y end.

    Parameters
    ----------
    tickers : list of str
        Tickers a descargar.
    start : str
        Fecha de inicio, formato 'YYYY-MM-DD' (inclusive).
    end : str
        Fecha de fin, formato 'YYYY-MM-DD' (exclusive, según yfinance).
    """
    data = yf.download(tickers, start=start, end=end, auto_adjust=True)["Close"]
    return data.dropna()  # saca filas con datos faltantes


if __name__ == "__main__":
    prices = download_prices()
    prices.to_csv("data/precios.csv")
    print(prices.tail())