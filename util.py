import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


def to_standard(series):
    return np.apply_along_axis(
        lambda entry: (entry - np.min(series)) / (np.max(series) - np.min(series)), 0, series
    )


def format_df(df: pd.DataFrame):
    df['new_building'] = [int(value == 'Yeni tikili') for value in df['category']]
    df['hypothec'] = [int(value == 'var') for value in df['hypothec']]
    df['repairs'] = [int(value == 'var') for value in df['repairs']]
    df['kupcha'] = [int(value == 'var') for value in df['kupcha']]
    df['area'] = [float(value[:-3]) for value in df['area']]
    df['floors'] = [int(s.split(' / ')[1]) for s in df['floor']]
    df['floor'] = [int(s.split(' / ')[0]) for s in df['floor']]

    df['urgent'] = [
        int(
            'təci̇li̇' in str(text).lower() or
            'teci̇li̇' in str(text).lower() or
            'təcili' in str(text).lower() or
            'tecili' in str(text).lower()
        ) for text in df['description']
    ]

    if 'price' in df.keys():
        df = df[[
            'latitude', 'longitude', 'floor', 'floors', 'rooms', 'hypothec', 'repairs', 'kupcha', 'area',
            'new_building', 'urgent', 'price'
        ]]
    else:
        df = df[[
            'latitude', 'longitude', 'floor', 'floors', 'rooms', 'hypothec', 'repairs', 'kupcha', 'area',
            'new_building', 'urgent',
        ]]

    for key in df.keys():
        if key != 'price':
            df.loc[:, key] = to_standard(df[key])

    return df
