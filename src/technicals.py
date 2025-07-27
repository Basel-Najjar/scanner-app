from ta.volatility import average_true_range
from .constants import default_lookback
import pandas as pd
import pandas_market_calendars as mcal


def calculate_atr(df: pd.DataFrame, window: int = default_lookback) -> pd.Series:
    return average_true_range(df["High"], df["Low"], df["Close"], window=window)
