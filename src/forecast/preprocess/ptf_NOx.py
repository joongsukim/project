import numpy as np


def preprocess(df):
    df.insert(0, 'time_idx', np.array(list(range(len(df))), dtype=int))
    df.insert(0, 'group_id', 0)

    return df
