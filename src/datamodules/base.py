# TODO: DEPRECATED


from typing import Any, Dict, Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from abc import ABCMeta, abstractmethod  
from typing import List, Union
from pytorch_forecasting import TimeSeriesDataSet 
import pandas as pd 
from sklearn.model_selection import train_test_split
from src.dataframe.dataframe import DataFrameUtils 
import numpy as np 
from sklearn.preprocessing import StandardScaler


class DataModuleUtils:
    def __init__(self):
        pass 

    @staticmethod 
    def split(df, train_ratio, val_ratio, test_ratio): 
        ratio_sum = 0 

        if train_ratio is not None:
            ratio_sum += train_ratio 
        if val_ratio is not None:
            ratio_sum += val_ratio 
        if test_ratio is not None:
            ratio_sum += test_ratio 

        # normalize ratio
        if train_ratio is not None:
            train_ratio /= ratio_sum 
        if val_ratio is not None:
            val_ratio /= ratio_sum 
        if test_ratio is not None:
            test_ratio /= ratio_sum 

        if train_ratio is not None:
            train_df, remain_df = train_test_split(df, train_size=train_ratio, shuffle=False) 
        else:
            train_df = None
            remain_df = df 

        if val_ratio is not None and test_ratio is None:
            val_df = remain_df 
            test_df = None 
        elif val_ratio is not None and test_ratio is not None:
            ratio_sum = val_ratio + test_ratio 
            val_ratio /= ratio_sum 
            test_ratio /= ratio_sum 

            val_df, test_df = train_test_split(remain_df, train_size=val_ratio, shuffle=False)
        else:
            val_df = None 
            test_df = None 
        
        return train_df, val_df, test_df 

    @staticmethod 
    def get_sliding_windowed_array(array, n_features, n_steps, n_strides=1):
        rows = array.shape[0] 

        # 0 * n_stride : 0 * n_stride + n_steps - 1
        # 1 * n_stride :  1 * n_stride + n_steps - 1
        # ... 
        # (n-1) * n_stride : (n-1) * n_stride + n_steps - 1
        # (n-1) * n_stride + n_steps - 1 <= len_train_X - 1
        # (n-1) <= (len_train_X - n_steps) / n_stride
        # n <= (len_train_X - n-steps) / n_stride + 1
        num_batches = int(np.floor((rows - n_steps) / n_strides + 1) )

        result = np.zeros((num_batches, n_steps, n_features))

        for i in range(num_batches):
            result[i] = array[i * n_strides: i * n_strides + n_steps]

        return result 

    @staticmethod 
    def get_num_batches(array, n_steps, n_strides):
        rows = array.shape[0] 
        return int(np.floor((rows - n_steps)/n_strides + 1))

    @staticmethod 
    def set_ith_batch(array, i, n_steps, n_strides, batch):
        array[i * n_strides: i * n_strides + n_steps] = batch 
        return array 

    @staticmethod 
    def get_ith_batch(array, i, n_steps, n_strides):
        return array[i * n_strides: i* n_strides + n_steps]


# used for imputation 
# X (corrupted => ground-truth is known, or missing => ground-truth is unknown)
# X_corrupted_mask: missing values whose ground-truth is known (corrupted)
# X_missing_mask: missing values

# 
#'train_X'
#'val_X'
#'test_X'
#'test_X_intact'
#'test_X_indicating_mask'
#'n_steps'
#'n_features'

# 
# train_ratio: Optional[float]=0.8,
# val_ratio: Optional[float]=0.1,
# test_ratio: Optional[float]=0.1,
# batch_size: int=32,
# num_workers: int = 0
class ImputeDataModule:
    def __init__(
        self, 
        df: pd.DataFrame, 
        n_steps: int, 
        n_strides: int=1, 
        #train_ratio: Optional[float]=0.8, 
        #val_ratio: Optional[float]=0.1, 
        #test_ratio: Optional[float]=0.1, 
        batch_size: int=32, 
        num_workers: int=0
        ):
        #self.train_ratio = train_ratio 
        #self.val_ratio = val_ratio 
        #self.test_ratio = test_ratio 

        self.batch_size = batch_size 
        self.num_workers = num_workers 

        # train / val / test split 
        #train_df, val_df, test_df = self.split(df)
        column_names = DataFrameUtils.get_column_names(df) 
        n_features = len(column_names) 

        #
        #train_data_df, _, _, _ = DataFrameUtils.decompose_df(train_df, column_names)
        #val_data_df, _, _, _ = DataFrameUtils.decompose_df(val_df, column_names)
        #test_data_df, test_missing_mask_df, test_corrupt_mask_df, test_intact_df = DataFrameUtils.decompose_df(test_df, column_names)
        # mask 추출 
        data_df, missing_mask_df = DataFrameUtils.decompose_df(df, column_names)

        # 
        #train_X_unprocessed = train_data_df[column_names].to_numpy() 
        #val_X_unprocessed = val_data_df[column_names].to_numpy() 
        #test_X_unprocessed = test_data_df[column_names].to_numpy() 
        #test_intact_X_unprocessed = test_intact_df[column_names].to_numpy() 
        #test_missing_mask_unprocessed = test_missing_mask_df[column_names].to_numpy() 
        #test_corrupt_mask_unprocessed = test_corrupt_mask_df[column_names].to_numpy() 

        self.train_X_unprocessed = data_df[column_names].to_numpy() 
        self.test_X_unprocessed = data_df[column_names].to_numpy() 
        #self.test_intact_X_unprocessed = intact_df[column_names].to_numpy() 
        self.test_missing_mask_unprocessed = missing_mask_df[column_names].to_numpy() 
        #self.test_corrupt_mask_unprocessed = corrupt_mask_df[column_names].to_numpy() 

        # standardization / normalization 
        # 표준화 진행 
        scaler = StandardScaler() 
        scaler.fit(self.train_X_unprocessed) 
        
        self.train_X_unprocessed = scaler.transform(self.train_X_unprocessed)
        #val_X_unprocessed = scaler.transform(val_X_unprocessed)
        self.test_X_unprocessed = scaler.transform(self.test_X_unprocessed)
        #self.test_intact_X_unprocessed = scaler.transform(self.test_intact_X_unprocessed)

        self.scaler = scaler 

        # 
        num_train_batches = DataModuleUtils.get_num_batches(self.train_X_unprocessed, n_steps, n_strides) 
        train_X = [DataModuleUtils.get_ith_batch(self.train_X_unprocessed, i, n_steps, n_strides) for i in range(num_train_batches)]
        val_X = None 

        #test_X = DataModuleUtils.get_sliding_windowed_array(test_X_unprocessed, n_features=n_features, n_steps=n_steps, n_strides=n_steps)
        #test_X_intact = DataModuleUtils.get_sliding_windowed_array(test_intact_X_unprocessed, n_features=n_features, n_steps=n_steps, n_strides=n_steps)
        #test_X_indicating_mask = test_X_intact = DataModuleUtils.get_sliding_windowed_array(test_corrupt_mask_unprocessed, n_features=n_features, n_steps=n_steps, n_strides=n_steps)

        # 
        self.df = df 
        self.column_names = column_names 

        self.train_X = train_X 
        self.val_X = val_X 
        #self.test_X = test_X 
        #self.test_X_intact = test_X_intact 
        #self.test_X_indicating_mask = test_X_indicating_mask 
        self.n_steps = n_steps 
        self.n_features = n_features 
        self.n_strides = n_strides 

    #def split(self, df):
    #    return DataModuleUtils.split(df, self.train_ratio, self.val_ratio, self.test_ratio) 



# used for prediction
# X (imputed) 
# y (imputed)
# X_missing_mask 
# y_missing_mask 
class ForecastDataModule(LightningDataModule):
    def __init__(
        self, 
        df: pd.DataFrame,
        target: Union[str, List[str]],
        forecast_horizon: int, 
        context_length: int, 
        is_timestamp_indexed: bool,
        train_ratio: Optional[float]=0.8,
        val_ratio: Optional[float]=0.1,
        test_ratio: Optional[float]=0.1,
        batch_size: int=32,
        num_workers: int = 0
        ):
        self.df = df 
        self.target = target 
        self.forecast_horizon = forecast_horizon
        self.context_length = context_length 
        self.is_timestamp_indexed = is_timestamp_indexed
        self.train_ratio = train_ratio 
        self.val_ratio = val_ratio 
        self.test_ratio = test_ratio 
        self.batch_size = batch_size 
        self.num_workers = num_workers 
        
        self.train_df = None 
        self.val_df = None 
        self.test_df = None 

        self.data_train = None 
        self.data_val = None 
        self.data_test = None 

        self.setup() 

    @property 
    def targets(self) -> List[str]: 
        return [self.target] if not isinstance(self.target, list) else self.target 

    def setup(self, stage: Optional[str] = None):
        if self.data_train or self.data_val or self.data_test:
            return

        # train / val / test split 
        train_df, val_df, test_df = self.split(self.df)
        column_names = DataFrameUtils.get_column_names(self.df) 

        self.column_names= column_names 

        train_data_df, _, _, _ = DataFrameUtils.decompose_df(train_df, column_names) 
        val_data_df, _, _, _ = DataFrameUtils.decompose_df(val_df, column_names) 
        test_data_df, test_missing_mask_df, test_corrupt_mask_df, test_intact_df = DataFrameUtils.decompose_df(test_df, column_names) 

        self.train_df = train_df 
        self.val_df = val_df 
        self.test_df = test_df 

        max_prediction_length = self.forecast_horizon
        max_encoder_length = self.context_length 

        if self.is_timestamp_indexed:
            # TODO: modify this 
            timestamp_columns = []
        else:
            timestamp_columns = []

        self.data_train = TimeSeriesDataSet(
            train_data_df, 
            time_idx="time_idx", 
            target=self.target, 
            group_ids=["group_id"],
            max_encoder_length=max_encoder_length,
            max_prediction_length=max_prediction_length,
            static_categoricals=[],
            static_reals=[],
            time_varying_known_categoricals=timestamp_columns,
            time_varying_known_reals=["time_idx"],
            time_varying_unknown_categoricals=[],
            time_varying_unknown_reals=self.targets
        )

        self.data_val = TimeSeriesDataSet.from_dataset(
            self.data_train, 
            data=val_data_df, 
            stop_randomization=True
        )

        self.data_test = TimeSeriesDataSet.from_dataset(
            self.data_train, 
            data=test_data_df, 
            stop_randomization=True
        )

        self.test_missing_mask_df = test_missing_mask_df 
        self.test_corrupt_mask_df = test_corrupt_mask_df 
        self.test_intact_df = test_intact_df 

    def split(self, df):
        return DataModuleUtils.split(df, self.train_ratio, self.val_ratio, self.test_ratio) 

    def train_dataloader(self):
        return self.data_train.to_dataloader(
            train=True,
            batch_size=self.batch_size, 
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return self.data_val.to_dataloader(
            train=False,
            batch_size=self.batch_size, 
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return self.data_test.to_dataloader(
            train=False,
            batch_size=self.batch_size, 
            num_workers=self.num_workers
        )
