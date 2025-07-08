import pandas as pd
from typing import List, Optional, Dict, Union, Tuple


class SMAScanner:
    def __init__(
        self,
        sma_periods: List[int],
        hv_window: int = 20,
        return_window: int = 50,
        consolidate_periods: int = 10,
        sentiment_map: Optional[Dict[str, str]] = None,
    ) -> None:
        self.sma_periods = sma_periods
        self.hv_window = hv_window
        self.return_window = return_window
        self.window_days = consolidate_periods
        self.sentiment_map = sentiment_map or {
            "support hold": "bullish",
            "support break": "bearish",
            "resistance hold": "bearish",
            "resistance break": "bullish",
        }
        self.events_df = pd.DataFrame()

    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Melts raw yfinance dataframe and adds a volatility
        """
        data = pd.melt(
            df["Close"].dropna(),  # type: ignore
            var_name="Ticker",
            value_name="Close",
            ignore_index=False,
        ).reset_index(drop=False)
        data["Returns"] = data.groupby("Ticker")["Close"].pct_change()
        data["Volatility"] = data.groupby("Ticker")["Returns"].transform(
            lambda x: x.rolling(window=self.hv_window).std()
        )
        data["RollingReturns"] = data.groupby("Ticker")["Returns"].transform(
            lambda x: x.rolling(window=self.return_window).sum()
        )
        return data.dropna().reset_index(drop=True).sort_values("Date")

    def _calculate_sma(
        self, df: pd.DataFrame, period: int
    ) -> Union[pd.DataFrame, pd.Series]:
        df = df.sort_values(["Ticker", "Date"])
        return df.groupby("Ticker")["Close"].transform(
            lambda x: x.rolling(window=period).mean()
        )

    def _add_sma_thresholds(
        self, df: pd.DataFrame, period: int
    ) -> Tuple[pd.DataFrame, str]:
        sma_col = f"SMA_{period}"
        df[sma_col] = self._calculate_sma(df, period)
        near_thresh = df[sma_col] * df["Volatility"]
        df[f"{sma_col}_lower"] = df[sma_col] - (near_thresh)
        df[f"{sma_col}_upper"] = df[sma_col] + (near_thresh)
        df = df.dropna(subset=[sma_col, f"{sma_col}_lower", f"{sma_col}_upper"])
        return df, sma_col

    def _get_band_tests(
        self, df: pd.DataFrame, sma_col: str
    ) -> Union[pd.DataFrame, pd.Series]:
        lower = f"{sma_col}_lower"
        upper = f"{sma_col}_upper"
        return df[(df["Close"] > df[lower]) & (df["Close"] < df[upper])]

    def run_scan(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        data = self._preprocess_data(df_raw)
        last_day = data["Date"].max()
        scan_results = []
        for period in self.sma_periods:
            df, sma_col = self._add_sma_thresholds(data, period)
            near_df = self._get_band_tests(df, sma_col)
            results = near_df[near_df["Date"] == last_day].copy()[["Date", "Ticker"]]

            results["MA"] = sma_col
            scan_results.append(results)

        return (
            pd.concat(scan_results)
            .sort_values(["Ticker", "Date"])
            .reset_index(drop=True)
        )
