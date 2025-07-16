import numpy as np
import pandas as pd
from .constants import default_lookback
from .utils import calculate_atr, get_last_market_day


def calculate_scaled_slope(df: pd.DataFrame, lookback: int = default_lookback) -> float:
    atr = calculate_atr(df, lookback).to_numpy()
    close = df.sort_values("Timestamp")["Close"]
    sma = close.rolling(lookback).mean().iloc[-lookback:].to_numpy()
    raw_slope = np.polyfit(range(lookback), sma[-lookback:], deg=1)[0]
    return raw_slope / atr[-1]


def calculate_cumulative_pivot_levels(
    df: pd.DataFrame,
    n_pivots: int = 4,
    max_lookback: pd.Timestamp = 90,
    # thresh: float = 0.0,
) -> dict:
    last_market_day = get_last_market_day()
    lookback_thresh = pd.Timestamp(last_market_day) - pd.Timedelta(f"{max_lookback}d")
    pivot_high_periods = df[df["pivot_type"] == "high"]
    pivot_low_periods = df[df["pivot_type"] == "low"]
    # higher_pivot_highs = (
    #     pivot_high_periods.iloc[-n_pivots:]["Close"].pct_change().sum() > thresh
    # )
    # higher_pivot_lows = (
    #     pivot_low_periods.iloc[-n_pivots:]["Close"].pct_change().sum() > thresh
    # )
    # lower_pivot_highs = (
    #     pivot_high_periods.iloc[-n_pivots:]["Close"].pct_change().sum() < -thresh
    # )
    # lower_pivot_lows = (
    #     pivot_low_periods.iloc[-n_pivots:]["Close"].pct_change().sum() < -thresh
    # )
    if (pivot_high_periods["Timestamp"] > lookback_thresh).sum() < n_pivots:
        return (np.nan, np.nan)
    elif (pivot_low_periods["Timestamp"] > lookback_thresh).sum() < n_pivots:
        return (np.nan, np.nan)
    else:
        return (
            pivot_high_periods.iloc[-n_pivots:]["Close"].pct_change().sum(),
            pivot_low_periods.iloc[-n_pivots:]["Close"].pct_change().sum(),
        )
    # if higher_pivot_highs and higher_pivot_lows:
    #     return "uptrend"
    # elif lower_pivot_highs and lower_pivot_lows:
    #     return "downtrend"
    # else:
    #     return "no trend"


# def categorize_slope(
#     value: float,
#     sentiment_value_map={
#         "bullish": 0.2,
#         "non-bearish": 0.1,
#         "neutral": -0.1,
#         "non-bullish": -0.2,
#     },
# ):
#     sentiment = None
#     if value >= sentiment_value_map["bullish"]:
#         sentiment = "uptrend"
#     elif (value < sentiment_value_map["bullish"]) and (
#         value >= sentiment_value_map["non-bearish"]
#     ):
#         sentiment = "weak uptrend"
#     elif (value < sentiment_value_map["non-bearish"]) and (
#         value >= sentiment_value_map["neutral"]
#     ):
#         sentiment = "neutral"
#     elif (value < sentiment_value_map["neutral"]) and (
#         sentiment_value_map["non-bullish"] >= -0.2
#     ):
#         sentiment = "weak downtrend"
#     elif value < sentiment_value_map["non-bullish"]:
#         sentiment = "downtrend"
#     return sentiment


def identify_pivots(
    df: pd.DataFrame, threshold: str | float = 0.05, use_high_low: bool = False
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
        threshold = atr_pct.fillna(value=atr_pct.mean()).values
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
