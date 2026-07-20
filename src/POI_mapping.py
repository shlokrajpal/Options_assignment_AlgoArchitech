import pandas as pd
import numpy as np

def POI(df: pd.DataFrame, curr_datetime, left_n=15, right_n=2) -> tuple:
    data = df.loc[:curr_datetime]

    min_bars = left_n + 1 + right_n
    if len(data) < min_bars:
        return None, None, None, None, None, []

    left_max_high = data['high'].shift(1).rolling(window=left_n).max()
    left_min_low = data['low'].shift(1).rolling(window=left_n).min()

    right_max_high = data['high'].rolling(window=right_n).max().shift(-right_n)
    right_min_low = data['low'].rolling(window=right_n).min().shift(-right_n)

    prev_close = data['close'].shift(1)
    tr = np.maximum(data['high'] - data['low'],
        np.maximum(np.abs(data['high'] - prev_close), np.abs(data['low'] - prev_close)))
    atr = tr.rolling(window=14).mean()

    atr_filter_high = (data['high'] - data['close'].shift(-right_n)) >= atr.shift(-right_n)
    atr_filter_low = (data['close'].shift(-right_n) - data['low']) >= atr.shift(-right_n)

    is_swing_high = (data['high'] > left_max_high) & (data['high'] > right_max_high) & atr_filter_high
    is_swing_low = (data['low'] < left_min_low) & (data['low'] < right_min_low) & atr_filter_low

    recent_high_idx = data[is_swing_high].index[-1] if not data[is_swing_high].empty else None
    recent_low_idx = data[is_swing_low].index[-1] if not data[is_swing_low].empty else None

    if recent_high_idx is None or recent_low_idx is None:
        return recent_high_idx, recent_low_idx, None, None, None, []

    high_price = data.loc[recent_high_idx, 'high']
    low_price = data.loc[recent_low_idx, 'low']
    price_range = high_price - low_price

    if recent_low_idx < recent_high_idx:
        swing_direction = 'uptrend'
        fib_50 = high_price - (price_range * 0.50)
        fib_786 = high_price - (price_range * 0.786)
    else:
        swing_direction = 'downtrend'
        fib_50 = low_price + (price_range * 0.50)
        fib_786 = low_price + (price_range * 0.786)

    zone_max = max(fib_50, fib_786)
    zone_min = min(fib_50, fib_786)

    end_idx = max(recent_high_idx, recent_low_idx)
    start_idx = min(recent_high_idx, recent_low_idx)

    post_swing_data = data.loc[end_idx:].iloc[1:]

    if not post_swing_data.empty:
        if (post_swing_data['high'].max() > high_price) or (post_swing_data['low'].min() < low_price):
            return recent_high_idx, recent_low_idx, swing_direction, fib_50, fib_786, []

    swing_data = data.loc[start_idx:end_idx]
    candidate_pois = []

    if len(swing_data) >= 3:
        for i in range(2, len(swing_data)):
            c1 = swing_data.iloc[i-2]
            c3 = swing_data.iloc[i]

            if swing_direction == 'uptrend':
                if c1['high'] < c3['low']:
                    gap_bottom, gap_top = c1['high'], c3['low']
                    if gap_top >= zone_min and gap_bottom <= zone_max:
                        candidate_pois.append([gap_bottom, gap_top])
            else:
                if c1['low'] > c3['high']:
                    gap_top, gap_bottom = c1['low'], c3['high']
                    if gap_top >= zone_min and gap_bottom <= zone_max:
                        candidate_pois.append([gap_top, gap_bottom])

    pois = []
    for fvg in candidate_pois:
        if swing_direction == 'uptrend':
            if post_swing_data.empty or post_swing_data['low'].min() >= fvg[0]:
                pois.append(fvg)
        else:
            if post_swing_data.empty or post_swing_data['high'].max() <= fvg[0]:
                pois.append(fvg)

    if pois:
        largest_fvg = max(pois, key=lambda fvg: abs(fvg[1] - fvg[0]))
        merged_poi = [min(largest_fvg), max(largest_fvg)]
    else:
        merged_poi = []

    return recent_high_idx, recent_low_idx, swing_direction, fib_50, fib_786, merged_poi