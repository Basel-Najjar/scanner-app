import pandas as pd
import numpy as np
from .technicals import calculate_atr
from .utils import get_market_hours, get_last_market_day, reaggregate_bars
from .constants import extreme_threshold, breakout_threshold, string_date_format
from datetime import timedelta
import plotly.express as px


def calculate_tpo(
    df_symbol: pd.DataFrame,
    session_date: str = get_last_market_day(),
    atr_scaler: float = 25.0,
) -> pd.Series:

    df_daily = reaggregate_bars(df_symbol, "1d")
    df = df_symbol.loc[session_date].copy()
    max_price = df["High"].max().round(2)
    min_price = df["Low"].min().round(2)
    atr = calculate_atr(df_daily).iloc[-1]
    tpo_bin_size = round(atr / atr_scaler, 2)
    bin_edges = np.arange(min_price, max_price + tpo_bin_size, tpo_bin_size)
    bin_labels = bin_edges[:-1]
    tpo_counts = pd.Series(0, index=bin_labels)
    for _, row in df.iterrows():
        low = row["Low"]
        high = row["High"]
        price_range = np.arange(low, high + tpo_bin_size, tpo_bin_size)
        binned = pd.cut(price_range, bins=bin_edges, labels=bin_labels, right=False)
        binned = binned.dropna().unique()
        for bin_label in binned:
            tpo_counts[bin_label] += 1
    return tpo_counts


def calculate_value_area(tpo_counts: pd.Series, value_area_pct: float = 0.7) -> dict:
    sorted_tpo = tpo_counts.sort_values(ascending=False)
    total_tpos = sorted_tpo.sum()
    target_tpos = total_tpos * value_area_pct
    max_count = sorted_tpo.iloc[0]
    poc_candidates = sorted_tpo[sorted_tpo == max_count].index
    price_center = np.average(tpo_counts.index, weights=tpo_counts.values)
    poc = min(poc_candidates, key=lambda x: abs(x - price_center))  # closest to center
    value_area_bins = [poc]
    cumulative = tpo_counts[poc]
    remaining = tpo_counts.drop(index=poc)  # remove POC bin
    sorted_bins = sorted(remaining.index)
    bin_size = sorted_bins[1] - sorted_bins[0]
    above = poc + bin_size
    below = poc - bin_size

    while cumulative < target_tpos:
        next_bins = []
        if above in remaining:
            next_bins.append((above, tpo_counts[above]))
        if below in remaining:
            next_bins.append((below, tpo_counts[below]))

        if not next_bins:
            break

        # Choose the bin with higher TPO count; if tied, prefer lower price
        next_bins.sort(key=lambda x: (-x[1], x[0]))
        chosen_price, count = next_bins[0]

        value_area_bins.append(chosen_price)
        cumulative += count
        remaining = remaining.drop(index=chosen_price)

        # Move pointers
        if chosen_price == above:
            above += bin_size
        else:
            below -= bin_size

    val = min(value_area_bins)
    vah = max(value_area_bins)

    return {
        "POC": round(poc, 2),
        "VAH": round(vah, 2),
        "VAL": round(val, 2),
        # 'Value Area Bins': sorted(value_area_bins),
        # 'Total TPOs': total_tpos,
        # 'Captured TPOs': cumulative
    }


def calculate_initial_balance(df_session: pd.DataFrame) -> tuple:
    ib_low = df_session[df_session["First Hour"]]["Low"].min()
    ib_high = df_session[df_session["First Hour"]]["High"].max()
    return ib_low, ib_high


def calculate_session_high_low_close(df_session: pd.DataFrame) -> tuple:
    session_high = df_session["High"].max()
    session_low = df_session["Low"].min()
    session_close = df_session.iloc[-1]["Close"]
    return session_high, session_low, session_close


def rank_momentum(
    df_current_session: pd.DataFrame, va_current: dict, va_previous: dict
) -> tuple:
    session_high, session_low, session_close = calculate_session_high_low_close(
        df_current_session
    )
    ib_low, ib_high = calculate_initial_balance(df_current_session)
    opening = rank_opening(session_high, session_low, ib_low, ib_high, va_previous)
    extension = rank_extension(session_high, session_low, ib_low, ib_high, va_previous)
    closing = rank_closing(session_close, va_current)
    return opening, extension, closing


def rank_opening(session_high, session_low, ib_low, ib_high, va_previous):
    rank = 0.0
    # bullish
    if (ib_low == session_low) & (ib_high != session_high):
        if session_low > va_previous["VAH"]:
            rank = 4.0
        elif (session_low <= va_previous["VAH"]) & (session_low > va_previous["POC"]):
            rank = 3.0
        elif (session_low <= va_previous["POC"]) & (session_low >= va_previous["VAL"]):
            rank = 2.0
        elif session_low < va_previous["VAL"]:
            rank = 1.0

    # bearish
    elif (ib_high == session_high) & (ib_low != session_low):
        if session_high < va_previous["VAL"]:
            rank = -4.0
        elif (session_high >= va_previous["VAL"]) & (session_high < va_previous["POC"]):
            rank = -3.0
        elif (session_high >= va_previous["POC"]) & (
            session_high <= va_previous["VAH"]
        ):
            rank = -2.0
        elif session_high >= va_previous["VAH"]:
            rank = -1.0

    # session high and low made in first hour
    elif (ib_high == session_high) & (ib_low == session_low):
        rank = 0.0
    return rank


def rank_extension(session_high, session_low, ib_low, ib_high, va_previous) -> float:
    rank = 0.0
    # bullish
    if session_high > ib_high:
        if session_high > va_previous["VAH"]:
            rank = 4.0
        elif (session_high <= va_previous["VAH"]) & (session_high > va_previous["POC"]):
            rank = 3.0
        elif (session_high <= va_previous["POC"]) & (
            session_high >= va_previous["VAL"]
        ):
            rank = 2.0
        elif session_high < va_previous["VAL"]:
            rank = 1.0
    # bearish
    elif session_low < ib_low:
        if session_low < va_previous["VAL"]:
            rank = -4.0
        elif (session_low >= va_previous["VAL"]) & (session_low < va_previous["POC"]):
            rank = -3.0
        elif (session_low >= va_previous["POC"]) & (session_low <= va_previous["VAH"]):
            rank = -2.0
        elif session_low > va_previous["VAH"]:
            rank = -1.0
    return rank


def rank_closing(session_close, va_current) -> float:
    rank = 0.0
    # bullish
    if session_close > va_current["VAH"]:
        rank = 3.0
    elif session_close == va_current["VAH"]:
        rank = 2.0
    elif session_close > va_current["POC"]:
        rank = 1.0
    # neutral
    elif session_close == va_current["POC"]:
        rank = 0.0
    # bearish
    elif session_close > va_current["VAL"]:
        rank = -1.0
    elif session_close == va_current["VAL"]:
        rank = -2.0
    elif session_close < va_current["VAL"]:
        rank = -3.0
    return rank


def calculate_momentum_ranks(df: pd.DataFrame) -> pd.DataFrame:
    results = {}
    available_dates = (
        pd.Series(df.index.date).sort_values(ascending=True).unique().tolist()
    )
    market_hours = {date: get_market_hours(date) for date in available_dates}

    for symbol in df.columns.levels[0]:
        results[symbol] = {}
        df_symbol = df[symbol].copy()

        df_symbol["First Hour"] = False
        df_symbol["Last Half Hour"] = False

        for date in available_dates:
            opening_bell, closing_bell = market_hours[date]
            first_30 = opening_bell
            second_30 = opening_bell + timedelta(minutes=30)
            last_30 = closing_bell - timedelta(minutes=30)
            if first_30 in df_symbol.index:
                df_symbol.at[first_30, "First Hour"] = True
            if second_30 in df_symbol.index:
                df_symbol.at[second_30, "First Hour"] = True
            if last_30 in df_symbol.index:
                df_symbol.at[last_30, "Last Half Hour"] = True
        value_areas = {}

        for date in available_dates:
            str_date = date.strftime(string_date_format)
            prev_date = (date - timedelta(days=1)).strftime(string_date_format)

            tpo = calculate_tpo(df_symbol, str_date)
            value_areas[str_date] = calculate_value_area(tpo)

            if prev_date in value_areas:
                df_session = df_symbol.loc[str_date].copy()

                opening, extension, closing = rank_momentum(
                    df_session, value_areas[str_date], value_areas[prev_date]
                )
                results[symbol][str_date] = {
                    "Opening Rank": opening,
                    "Extension Rank": extension,
                    "Closing Rank": closing,
                    "Total Rank": opening + extension + closing,
                }
                # results[symbol][current_date]['Value Area'] = value_areas[current_date]

    return pd.concat(
        {
            symbol: pd.DataFrame.from_dict(inner_dict, orient="index")
            for symbol, inner_dict in results.items()
        },
        axis=1,
    )


def scan_momentum_ranks(
    df,
    extreme_threshold: float = extreme_threshold,
    breakout_threshold: float = breakout_threshold,
) -> pd.DataFrame:
    signals = dict()
    for symbol in df.columns.levels[0]:
        symbol_rank = df[symbol]["Total Rank"]
        last_rank = symbol_rank.iloc[-1]
        if last_rank >= extreme_threshold:
            signals[f"{symbol}:bull"] = df[symbol]["Total Rank"]
        elif last_rank <= -extreme_threshold:
            signals[f"{symbol}:bear"] = df[symbol]["Total Rank"]
        elif (
            symbol_rank.iloc[-3:].between(-breakout_threshold, breakout_threshold).all()
        ):
            signals[f"{symbol}:breakout"] = df[symbol]["Total Rank"]

    return pd.DataFrame(signals)


def plot_scan_results(df: pd.DataFrame, symbol: str):
    fig = px.bar(df, x=df.index, y=symbol, title=symbol, text_auto=True)
    fig.add_hline(y=5, line_color="blue", line_dash="dot", line_width=0.5)
    fig.add_hline(y=-5, line_color="blue", line_dash="dot", line_width=0.5)
    fig.add_hline(y=9, line_color="red", line_dash="dot", line_width=0.5)
    fig.add_hline(y=-9, line_color="red", line_dash="dot", line_width=0.5)
    fig.update_yaxes(range=[-11, 11])
    return fig
