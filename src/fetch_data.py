import yfinance as yf
import pandas as pd

def get_ohlcv(ticker: str, start: str, end: str | None = None) -> pd.DataFrame:
    # Force single-level columns
    df = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column",   # <- important
        threads=False
    )

    if df is None or df.empty:
        raise ValueError(f"No data returned for {ticker}")

    # If Yahoo gives MultiIndex columns anyway, flatten them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Ensure we have the expected columns (case-insensitive)
    df = df.rename(columns=str.lower)

    # Some yfinance versions use 'adj close' even with auto_adjust=True; handle both
    expected = {"open","high","low","close","volume"}
    have = set(df.columns)
    if not expected.issubset(have):
        # try to map 'adj close' to 'close' if 'close' is missing
        if "adj close" in have and "close" not in have:
            df["close"] = df["adj close"]
        missing = expected - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns from Yahoo for {ticker}: {missing}")

    # Put the DatetimeIndex into a column named 'date'
    df.index.name = "date"
    df = df.reset_index()

    df["ticker"] = ticker
    return df[["date","ticker","open","high","low","close","volume"]]
