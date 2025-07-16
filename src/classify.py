import numpy as np
import pandas as pd
from .constants import default_lookback
from .utils import calculate_atr, get_last_market_day


def calculate_sma_slope(df: pd.DataFrame, lookback: int = default_lookback) -> float:
    atr = calculate_atr(df, lookback).to_numpy()
    close = df.sort_values("Timestamp")["Close"]
    sma = close.rolling(lookback).mean().iloc[-lookback:].to_numpy()
    raw_slope = np.polyfit(range(lookback), sma[-lookback:], deg=1)[0]
    return raw_slope / atr[-1]


def calculate_cumulative_pivot_levels(
    df: pd.DataFrame, n_pivots: int = 3, thresh: float = 0.0
) -> dict:
    pivot_high_periods = df[df["pivot_type"] == "high"]
    pivot_low_periods = df[df["pivot_type"] == "low"]

    pivot_high_pct_change = (
        pivot_high_periods.iloc[-n_pivots:]["pivot_level"].pct_change().dropna()
    )
    pivot_low_pct_change = (
        pivot_low_periods.iloc[-n_pivots:]["pivot_level"].pct_change().dropna()
    )

    higher_highs = (pivot_high_pct_change > thresh).all()
    lower_highs = (pivot_low_pct_change < thresh).all()
    higher_lows = (pivot_high_pct_change > thresh).all()
    lower_lows = (pivot_low_pct_change < thresh).all()

    return {
        "cumul_pivot_highs": pivot_high_pct_change.sum(),
        "cumul_pivot_lows": pivot_low_pct_change.sum(),
        "higher_highs_higher_lows": (higher_highs and higher_lows),
        "lower_highs_lower_lows": (lower_highs and lower_lows),
        "pivot_high_dates": str(
            [
                x.strftime("%Y-%m-%d")
                for x in pivot_high_periods["Timestamp"].iloc[-n_pivots:].tolist()
            ]
        ),
        "pivot_low_dates": str(
            [
                x.strftime("%Y-%m-%d")
                for x in pivot_low_periods["Timestamp"].iloc[-n_pivots:].tolist()
            ]
        ),
    }


def identify_pivots(
    df: pd.DataFrame,
    threshold: str | float = 0.05,
    atr_scaling: float = 1.0,
    use_high_low: bool = False,
) -> pd.DataFrame:
    if use_high_low:
        highs = df["High"].values
        lows = df["Low"].values
        candidate_extreme_price = highs[0]
        last_pivot_price = highs[0]
    else:
        close = df["Close"].values
        candidate_extreme_price = close[0]
        last_pivot_price = close[0]

    if threshold == "atr":
        atr = calculate_atr(df, window=default_lookback).to_numpy()
        atr_pct = atr / df["Close"]
        atr_pct = atr_pct.replace(0, np.nan)
        threshold = atr_pct.fillna(value=atr_pct.mean()).values * atr_scaling
    elif type(threshold) is float:
        threshold = [threshold] * len(df)

    zz_price = [np.nan] * len(df)
    zz_type = [np.nan] * len(df)
    last_pivot_idx = 0
    candidate_extreme_idx = 0
    direction = None  # initialize

    for i in range(len(df)):
        thresh = threshold[i]
        if use_high_low:
            if direction == "up":
                price = highs[i]
            elif direction == "down":
                price = lows[i]
            else:
                price = highs[i]
        else:
            price = close[i]

        change_pct = (price - last_pivot_price) / last_pivot_price

        if direction is None:
            # No direction yet: look for first significant move
            if abs(change_pct) >= thresh:
                direction = "up" if change_pct > 0 else "down"
                candidate_extreme_idx = i
                candidate_extreme_price = price
                zz_price[last_pivot_idx] = last_pivot_price
                zz_type[last_pivot_idx] = "high" if direction == "up" else "low"

        elif direction == "up":
            if price > candidate_extreme_price:
                candidate_extreme_price = price
                candidate_extreme_idx = i
            elif price < candidate_extreme_price * (1 - thresh):
                # Commit the previous high as a pivot
                zz_price[candidate_extreme_idx] = candidate_extreme_price
                zz_type[candidate_extreme_idx] = "high"
                last_pivot_price = price
                last_pivot_idx = i
                candidate_extreme_price = price
                candidate_extreme_idx = i
                direction = "down"

        elif direction == "down":
            if price < candidate_extreme_price:
                candidate_extreme_price = price
                candidate_extreme_idx = i
            elif price > candidate_extreme_price * (1 + thresh):
                # Commit the previous low as a pivot
                zz_price[candidate_extreme_idx] = candidate_extreme_price
                zz_type[candidate_extreme_idx] = "low"
                last_pivot_price = price
                last_pivot_idx = i
                candidate_extreme_price = price
                candidate_extreme_idx = i
                direction = "up"

    df["pivot_level"] = zz_price
    df["pivot_type"] = zz_type
    df["threshold"] = threshold
    return df
