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
    df['area'] = [float(value[:-3]) for value in df['area']]
    df['floors'] = [int(s.split(' / ')[1]) for s in df['floor']]
    df['floor'] = [int(s.split(' / ')[0]) for s in df['floor']]

    df = df[[
        'latitude', 'longitude', 'floor', 'floors', 'area', 'new_building', 'price'
    ]]

    for key in df.keys():
        if key != 'price':
            df.loc[:, key] = to_standard(df[key])

    return df
