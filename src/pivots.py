import pandas as pd
import numpy as np
from ta.volatility import average_true_range

def identify_pivots(df:pd.DataFrame, threshold:float|str=0.05, use_high_low:bool=False) -> pd.DataFrame:
    if use_high_low:
        highs = df['High'].values
        lows = df['Low'].values
        candidate_price = highs[0]
    else:
        close = df['Close'].values
        candidate_price = close[0]

    if threshold == 'atr':
        atr = average_true_range(df['High'], df['Low'], df['Close'], window=14, fillna=False)
        atr_pct = atr / df['Close']
        atr_pct = atr_pct.replace(0, np.nan)
        threshold = atr_pct.fillna(value=atr_pct.mean()).values
    elif type(threshold) is float:
        threshold = [threshold] * len(df)

    zz_price = [np.nan] * len(df)
    zz_type = [np.nan] * len(df)
    last_pivot_idx = 0
    last_pivot_price = highs[0] if use_high_low else close[0]
    candidate_extreme_idx = 0
    candidate_extreme_price = highs[0] if use_high_low else close[0]
    direction = None # initialize

    for i in range(len(df)):
        thresh = threshold[i]
        if use_high_low:
            if direction == 'up':
                price = highs[i]
            elif direction == 'down':
                price = lows[i]
            else:
                price = highs[i]
        else:
            price = close[i]

        change_pct = (price - last_pivot_price) / last_pivot_price

        if direction is None:
            # No direction yet: look for first significant move
            if abs(change_pct) >= thresh:
                direction = 'up' if change_pct > 0 else 'down'
                candidate_extreme_idx = i
                candidate_extreme_price = price
                zz_price[last_pivot_idx] = last_pivot_price
                zz_type[last_pivot_idx] = 'high' if direction == 'up' else 'low'

        elif direction == 'up':
            if price > candidate_extreme_price:
                candidate_extreme_price = price
                candidate_extreme_idx = i
            elif price < candidate_extreme_price * (1 - thresh):
                # Commit the previous high as a pivot
                zz_price[candidate_extreme_idx] = candidate_extreme_price
                zz_type[candidate_extreme_idx] = 'high'
                last_pivot_price = price
                last_pivot_idx = i
                candidate_extreme_price = price
                candidate_extreme_idx = i
                direction = 'down'

        elif direction == 'down':
            if price < candidate_extreme_price:
                candidate_extreme_price = price
                candidate_extreme_idx = i
            elif price > candidate_extreme_price * (1 + thresh):
                # Commit the previous low as a pivot
                zz_price[candidate_extreme_idx] = candidate_extreme_price
                zz_type[candidate_extreme_idx] = 'low'
                last_pivot_price = price
                last_pivot_idx = i
                candidate_extreme_price = price
                candidate_extreme_idx = i
                direction = 'up'

    df['Pivot Level'] = zz_price
    df['Pivot Type'] = zz_type
    df['Threshold'] = threshold
    return df