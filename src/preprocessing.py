import pandas as pd


def preprocess_data(
    df: pd.DataFrame,
    columns: list | str = "all",
    hv_window: int = 20,
    return_window: int = 20,
) -> pd.DataFrame:
    """
    Melts raw yfinance dataframe and adds rolling returns
    """
    data = pd.melt(
        df if columns == "all" else df[columns],  # type: ignore
        var_name="Ticker",
        value_name="Value",
        ignore_index=False,
    ).reset_index(drop=False)
    data["Returns"] = data.groupby("Ticker")["Close"].pct_change()
    data["Volatility"] = data.groupby("Ticker")["Returns"].transform(
        lambda x: x.rolling(window=hv_window).std()
    )
    data["RollingReturns"] = data.groupby("Ticker")["Returns"].transform(
        lambda x: x.rolling(window=return_window).sum()
    )
    return data.dropna().reset_index(drop=True).sort_values("Date")
