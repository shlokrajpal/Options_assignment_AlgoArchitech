import pandas as pd
import numpy as np

from POI_mapping import POI

def get_entry_signal(curr_ltf_time: pd.Timestamp, curr_idx: int, ltf_df: pd.DataFrame, htf_df: pd.DataFrame, cache: dict, in_position: bool) -> dict:

    start_idx = max(0, curr_idx - 6)
    ltf_data = ltf_df.iloc[start_idx : curr_idx + 1]
    
    if ltf_data.empty:
        return None
    curr_ltf_candle = ltf_data.iloc[-1]
    
    if curr_ltf_time.minute == 15:
        htf_data = htf_df.loc[:curr_ltf_time]
        
        if not htf_data.empty and curr_ltf_time in htf_data.index:
            high_idx, low_idx, direction, _, _, merged_poi = POI(htf_df, curr_ltf_time)
            
            if merged_poi:
                cache['poi'] = merged_poi
                cache['direction'] = direction
                
                if direction == 'uptrend':
                    cache['tp'] = htf_data.loc[high_idx, 'high']
                    cache['sl'] = merged_poi[0]  # POI Low
                elif direction == 'downtrend':
                    cache['tp'] = htf_data.loc[low_idx, 'low']
                    cache['sl'] = merged_poi[1]  # POI High
            else:
                cache.update({'poi': [], 'direction': None, 'tp': None, 'sl': None})

    if in_position or not cache.get('poi'):
        return None

    poi_min, poi_max = cache['poi'][0], cache['poi'][1]
    direction = cache['direction']

    if direction == 'uptrend' and curr_ltf_candle['close'] < poi_min:
        cache['poi'] = []
        return None
    elif direction == 'downtrend' and curr_ltf_candle['close'] > poi_max:
        cache['poi'] = []
        return None

    if len(ltf_data) >= 3:
        c1 = ltf_data.iloc[-3]
        c3 = ltf_data.iloc[-1]

        gap_low = gap_high = None
        if direction == 'uptrend' and (c1['high'] < c3['low']):
            gap_low, gap_high = c1['high'], c3['low']
        elif direction == 'downtrend' and (c1['low'] > c3['high']):
            gap_low, gap_high = c3['high'], c1['low']

        if gap_low is not None and gap_high >= poi_min and gap_low <= poi_max:
            return {
                'timestamp': curr_ltf_time,
                'direction': direction,
                'entry_price': curr_ltf_candle['close'],
                'poi_zone': cache['poi'],
                'take_profit': cache['tp'],
                'stop_loss': cache['sl']
            }

    return None