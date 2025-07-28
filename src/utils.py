import pandas_market_calendars as mcal
from src.constants import string_date_format, exch
from datetime import datetime
from functools import lru_cache
import pandas as pd


def import_watchlist(fpath: str) -> list:
    with open(fpath) as f:
        return f.read().split(" ")


@lru_cache(maxsize=128)
def get_market_hours(date: str, exchange: str = "NYSE") -> tuple:
    if not isinstance(date, str):
        date = date.strftime(string_date_format)
    schedule = exch.schedule(start_date=date, end_date=date)
    market_open = schedule.at[date, "market_open"]
    market_close = schedule.at[date, "market_close"]
    return market_open, market_close


def get_last_market_day(exchange: str = "NYSE", lookback: int = 7) -> datetime.date:
    today = datetime.today().date()
    nyse = mcal.get_calendar(exchange)
    start_date = today - pd.Timedelta(days=lookback)
    schedule = nyse.schedule(start_date=start_date, end_date=today)
    last_day = schedule.iloc[-1]["market_open"].date()
    if today == last_day:
        return today
    else:
        return last_day


def reaggregate_bars(df, resample: str = "1d") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Open": df["Open"].resample(resample).first(),
            "High": df["High"].resample(resample).max(),
            "Low": df["Low"].resample(resample).min(),
            "Close": df["Close"].resample(resample).last(),
            "Volume": df["Volume"].resample(resample).sum(),
        }
    ).dropna()
