import yfinance as yf
import pandas as pd
from datetime import datetime


def pull_watchlist_data(
    watchlist: list, start: datetime.date, end: datetime.date, interval: str = "30m"
) -> pd.DataFrame:
    return yf.download(
        watchlist,
        auto_adjust=False,
        interval=interval,
        start=start,
        end=end,
        group_by="ticker",
    )
