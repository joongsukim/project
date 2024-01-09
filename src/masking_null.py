import os.path
import sys

import numpy as np
import pandas as pd

@property
def manipulated_columns(self):
    return [
        ( 'COR (촉매반응로)', '2차 Air')
    ]

@property
def disturbance_columns(self):
    return [
        (    'Acid Gas',     '유량'),
        (    'Acid Gas',     '압력'),
        (    'Acid Gas',     '온도'),
        ( 'COR (촉매반응로)', '1차 Air'),
        ( 'COR (촉매반응로)', 'Air 압력'),
        ( 'COR (촉매반응로)', '연소 Air'),
        ( 'COR (촉매반응로)',  'COG유량'),
        ( 'COR (촉매반응로)',   '상부온도'),
        ( 'COR (촉매반응로)',   '중부온도'),
        (    '탈황 고압보일러',   '출구온도'),
        (    '탈황 고압보일러',   '스팀유량'),
        (    '탈황 고압보일러',     '압력'),
        (    '탈황 고압보일러',   '급수유량'),
        (    '탈황 고압보일러',     '레벨'),
        (       '저압보일러',     '압력'),
        (       '저압보일러',   '급수유량'),
        (       '저압보일러',   '스팀유량'),
        (     '반응기#1온도',     '입구'),
        (     '반응기#1온도',     '출구'),
        (     '반응기#2온도',     '입구'),
        (     '반응기#2온도',     '출구'),
        (         '콘덴서',     '압력'),
        (         '콘덴서',     '레벨'),
        (         '콘덴서',   '급수유량'),
        (         '유입량',   '스팀유량'),
        (         '유입량',  '순수유입량'),
        (   'PG Heat온도',   '#1출구'),
        (    'Tail Gas',    '유량1'),
        (    'Tail Gas',    '온도1'),
        (  'Drain Tank',     '레벨'),
        (  'Drain Tank',     '온도'),
        (   '저장Tank 온도',   '#100'),
        ('Desuper Heat',     '압력'),
        ('Desuper Heat',     '레벨')
    ]

@property
def controlled_columns(self):
    return [
        ('TAIL GAS 분석기',    'H2S'),
        ('TAIL GAS 분석기',    'SO2')
    ]

if __name__ == "__main__":
# 1. load df
    df_path = sys.argv[1]
    df = pd.read_pickle(df_path)

    # 2. replace missing value with nan
    df_diff = df.diff()
    missing_mask = df_diff < 1e-6
    masked_df = df.copy()
    masked_df[missing_mask] = np.nan

    # 3. save it
    out_dir = sys.argv[2]
    data_path = os.path.join(out_dir, "missing_data.pkl")
    mask_path = os.path.join(out_dir, "missing_mask.pkl")

    masked_df.to_pickle(data_path)
missing_mask.to_pickle(mask_path)

