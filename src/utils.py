from ta.volatility import average_true_range
from .constants import default_lookback
import pandas as pd
import pandas_market_calendars as mcal


def calculate_atr(df: pd.DataFrame, window: int = default_lookback) -> pd.Series:
    return average_true_range(df["High"], df["Low"], df["Close"], window=window)


def get_last_market_day(exchange: str = "NYSE", lookback: int = 7) -> str:
    nyse = mcal.get_calendar(exchange)
    end_date = pd.Timestamp.now(tz="UTC")
    start_date = end_date - pd.Timedelta(days=lookback)
    schedule = nyse.schedule(start_date=start_date, end_date=end_date)
    if end_date < schedule.iloc[-1]["market_open"]:
        return schedule.iloc[-2]["market_close"].strftime("%Y-%m-%d")
    else:
        return schedule.iloc[-1]["market_close"].strftime("%Y-%m-%d")
