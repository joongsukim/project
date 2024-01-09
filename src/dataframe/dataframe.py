from abc import ABCMeta, abstractmethod
from typing import Tuple, Optional, List
import pandas as pd
import os
import numpy as np

class DataFrameUtils:
    def __init__(self):
        pass

    @staticmethod
    def join_dfs(data_df, missing_mask_df):
        df = pd.merge(data_df, missing_mask_df, on=['time_idx'], suffixes=(None, '_missing_mask'))
        # df = pd.merge(df, corrupt_mask_df, on=[ 'time_idx'], suffixes=(None, '_corrupt_mask'))
        # df = pd.merge(df, intact_df, on=[ 'time_idx'], suffixes=(None, '_intact'))
        return df

    @staticmethod
    def decompose_df(df, column_names):
        data_df = df[column_names + ['time_idx']].copy()
        missing_mask_df = df[
            [f"{column_name}_missing_mask" for column_name in column_names] + ['time_idx']].copy()
        #corrupt_mask_df = df[
        #    [f"{column_name}_corrupt_mask" for column_name in column_names] + ['time_idx']].copy()
        #intact_df = df[[f"{column_name}_intact" for column_name in column_names] + ['time_idx']].copy()

        missing_mask_columns_mapping = dict(
            [(f"{column_name}_missing_mask", column_name) for column_name in column_names])
        #corrupt_mask_columns_mapping = dict(
        #    [(f"{column_name}_corrupt_mask", column_name) for column_name in column_names])
        #intact_columns_mapping = dict([(f"{column_name}_intact", column_name) for column_name in column_names])

        missing_mask_df = missing_mask_df.rename(columns=missing_mask_columns_mapping)
        #corrupt_mask_df = corrupt_mask_df.rename(columns=corrupt_mask_columns_mapping)
        #intact_df = intact_df.rename(columns=intact_columns_mapping)

        return data_df, missing_mask_df
    #corrupt_mask_df, intact_df

    @staticmethod
    def get_column_names(df):
        return [column_name for column_name in df.columns if
                not column_name.endswith(('_missing_mask', '_corrupt_mask', '_intact')) and column_name != 'time_idx'] 


# ----- Classes for General Data Load & Integration ----- #
class DataFrameLoader(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def load(self) -> pd.DataFrame:
        pass


class CSVDataFrameLoader(DataFrameLoader):
    def __init__(
            self,
            dataset_dir: str,
            names: Optional[List[str]],
            sep: str = ',',
            header: Optional[int] = None,
            drop_names: Optional[List[str]] = None,
            timestamp_name: Optional[str] = None
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.names = names
        self.sep = sep
        self.header = header
        self.drop_names = drop_names
        self.timestamp_name = timestamp_name

    def enumerate_csv_files(self):
        """

        Returns: [CSV File Path String]

        """
        csv_files = []

        # self.dataset_dir 안의 모든 파일 중에서
        for file_name in os.listdir(self.dataset_dir):
            file_path = os.path.join(self.dataset_dir, file_name)

            # 확장자가 csv인 파일을 csv_files에 모음
            if file_path.endswith('.csv'):
                csv_files.append(file_path)

        # csv_files를 알파벳 순서대로 정렬
        csv_files.sort()

        # csv 파일 경로 리스트 반환
        return csv_files

    def load(self) -> pd.DataFrame:
        # csv file 경로 리스트
        csv_files = self.enumerate_csv_files()

        # 각각의 csv 파일을 Pandas dataframe으로 변환
        dfs = [pd.read_csv(csv_path, sep=self.sep, header=self.header, names=self.names) for csv_path in csv_files]
        # csv에 있는 column 중에 dataframe에 포함시키지 않을 column names (drop_names)가 있으면 배제
        if self.drop_names:
            dfs = map(lambda x: x.drop(columns=self.drop_names), dfs)
        # dataframes를 병합 (csv 파일 이름 순서대로 정렬되었으므로 합쳤을 때도 csv 파일 이름 순서대로 병합될 가능성이 높음)
        df = pd.concat(dfs)
        
        # column 중에 시간을 나타내는 column이 있으면 그것을 기반으로 병합된 df를 정렬
        if self.timestamp_name and self.timestamp_name in df.columns: 
            df = df.sort_values(by=[self.timestamp_name])
        return df


class ExcelDataFrameLoader(DataFrameLoader):
    def __init__(self):
        super().__init__()
        pass

    # TODO: implement later
    def load(self) -> pd.DataFrame:
        pass


class PickleDataFrameLoader(DataFrameLoader):
    def __init__(self, train_dir, val_dir, test_dir):
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir

    # enumerate
    def enumerate_pkl_files(self, directory):
        pkl_files = []

        # self.dataset_dir 안의 모든 파일 중에서
        for file_name in os.listdir(directory):
            file_path = os.path.join(directory, file_name)

            # 확장자가 csv인 파일을 csv_files에 모음
            if file_path.endswith('.pkl'):
                pkl_files.append(file_path)

        # csv_files를 알파벳 순서대로 정렬
        pkl_files.sort()

        # csv 파일 경로 리스트 반환
        return pkl_files

    def load(self):
        train_pkl_files = self.enumerate_pkl_files(self.train_dir)
        val_pkl_files = self.enumerate_pkl_files(self.val_dir)
        test_pkl_files = self.enumerate_pkl_files(self.test_dir)

        train_dfs = [pd.read_pickle(pkl_path) for pkl_path in train_pkl_files]
        val_dfs = [pd.read_pickle(pkl_path) for pkl_path in val_pkl_files]
        test_dfs = [pd.read_pickle(pkl_path) for pkl_path in test_pkl_files]

        return train_dfs, val_dfs, test_dfs


class DataFrameWriter:
    def __init__(
            self,
            dataset_dir: str,
            file_name: str
    ):
        self.fpath = os.path.join(dataset_dir, file_name)

    def save(self, df):
        df.to_pickle(self.fpath)

