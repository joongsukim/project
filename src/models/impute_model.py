from pypots.imputation import (
    SAITS,
    BRITS,
)
from abc import ABCMeta, abstractmethod 
import numpy as np  
import torch 
import pandas as pd 
from src.dataframe.dataframe import DataFrameUtils
from src.datamodules.base import DataModuleUtils
from src.utils.plot import plot_dataframes


class ImputeModelUtils:
    def __init__(self):
        pass 

    @staticmethod 
    def cal_mae(inputs, target, mask=None):
        """calculate Mean Absolute Error"""
        assert type(inputs) == type(target), (
            f"types of inputs and target must match, but got"
            f"type(inputs)={type(inputs)}, type(target)={type(target)}"
        )
        lib = np if isinstance(inputs, np.ndarray) else torch
        if mask is not None:
            return lib.sum(lib.abs(inputs - target) * mask) / (lib.sum(mask) + 1e-9)
        else:
            return lib.mean(lib.abs(inputs - target))

    @staticmethod 
    def eval_mae(imputed_test_X_array, intact_test_X_array, indicating_mask_test_X_array, logger=None):
        mae = ImputeModelUtils.cal_mae(imputed_test_X_array, intact_test_X_array, indicating_mask_test_X_array)

        if logger is not None: 
            logger.log_metrics({"test_mae": mae })

        return mae 

    @staticmethod 
    def plot_result(imputed_test_X_array, indicating_mask_test_X_array, column_names, df, logger=None):
        imputed_df = pd.DataFrame(index=df.index, columns=column_names)
        imputed_df[column_names] = imputed_test_X_array 
        imputed_df[['group_id', 'time_idx']] = df[['group_id', 'time_idx']]

        missing_mask_df = pd.DataFrame(index=df.index, columns=column_names) 
        missing_mask_df[column_names] = indicating_mask_test_X_array 
        missing_mask_df[['group_id', 'time_idx']] = df[['group_id', 'time_idx']]

        _, _, _, intact_df = DataFrameUtils.decompose_df(df, column_names)

        # shade ? 
        shade = {}
        for column in column_names: 
            missing_mask = missing_mask_df[column].to_numpy() 
            missing_mask = missing_mask.astype(float) 
            # [0, 1, 1, 1, 1, 1, ...., 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0]
            # 0->1 : start index 
            # 1->0 : end index 
            start_inds = [] 
            end_inds = [] 
            if missing_mask[0]:
                start_inds.append(0) 
            
            missing_mask_diff = np.diff(missing_mask) 
            start_inds_list = np.where(missing_mask_diff == 1)[0]
            end_inds_list = np.where(missing_mask_diff == -1)[0] 
            start_inds = start_inds + start_inds_list.tolist() 
            end_inds = end_inds + end_inds_list.tolist() 

            if missing_mask[-1]:
                end_inds.append(len(missing_mask)-1)

            shade[column] = list(zip(start_inds, end_inds))

        df_tuples = [('raw', intact_df), ('imputed', imputed_df)]
        figs = [plot_dataframes(column, column, 'time_idx', False, *df_tuples, shade=shade[column]) for column in column_names]

        # log using figs 
        if logger is not None:
            print(type(logger.experiment)) 
            logger.experiment.add_figure('Impute Result', figs) 

            for column, fig in zip(column_names, figs):
                logger.experiment.add_figure(column, fig)




class BaseImputer(metaclass=ABCMeta):
    def __init__(self):
        pass 

    @abstractmethod
    def fit(self, train_X, val_X):
        pass 

    @abstractmethod 
    def impute(self, X):
        pass 

    def test(self, test_X_array, intact_test_X_array, test_missing_mask, n_steps, n_strides, column_names, df, scaler, logger=None):
        # 1. imputed by sliding window
        num_test_batches = DataModuleUtils.get_num_batches(test_X_array, n_steps, n_strides)
        imputed_test_X_array = test_X_array.copy() 
        for i in range(num_test_batches):
            test_X = DataModuleUtils.get_ith_batch(test_X_array, i, n_steps, n_strides) 
            # insert dimension 
            test_X = np.expand_dims(test_X, axis=0) 

            imputed_test_X = self.impute(test_X) 

            imputed_test_X_array = DataModuleUtils.set_ith_batch(imputed_test_X_array, i, n_steps, n_strides, imputed_test_X[0])

        test_metrics = {} 
        # 2. eval mae 
        mae = ImputeModelUtils.eval_mae(imputed_test_X_array, intact_test_X_array, test_missing_mask, logger)
        test_metrics["mae"] = mae 

        # 3. plot 
        unscaled_imputed_test_X_array = scaler.inverse_transform(imputed_test_X_array)
        ImputeModelUtils.plot_result(unscaled_imputed_test_X_array, test_missing_mask, column_names, df, logger)

        # 4. recover df 
        imputed_df = pd.DataFrame(index=df.index, columns=column_names)
        imputed_df[column_names] = unscaled_imputed_test_X_array 
        imputed_df[['group_id', 'time_idx']] = df[['group_id', 'time_idx']]

        _, missing_mask_df, corrupt_mask_df, intact_df = DataFrameUtils.decompose_df(df, column_names)

        result_df = DataFrameUtils.join_dfs(imputed_df, missing_mask_df, corrupt_mask_df, intact_df) 

        return result_df, test_metrics 

class SAITSImputer(BaseImputer):
    def __init__(
        self,
        n_steps,
        n_features,
        n_layers,
        d_model,
        d_inner,
        n_head,
        d_k,
        d_v,
        dropout,
        learning_rate=1e-3,
        epochs=100,
        patience=10,
        batch_size=32,
        weight_decay=1e-5,
        device=None
        ):
        super().__init__()
        self.saits = SAITS(
            n_steps,
            n_features,
            n_layers=n_layers,
            d_model=d_model,
            d_inner=d_inner,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
            learning_rate=learning_rate,
            epochs=epochs,
            patience=patience,
            batch_size=batch_size,
            weight_decay=weight_decay,
            device=device
        )

    def fit(self, train_X, val_X):
        self.saits.fit(train_X, val_X) 

    def impute(self, test_X):
        return self.saits.impute(test_X) 


class BRITSImputer(BaseImputer):
    def __init__(
        self,
        n_steps,
        n_features,
        rnn_hidden_size,
        learning_rate=1e-3,
        epochs=100,
        patience=10,
        batch_size=32,
        weight_decay=1e-5,
        device=None
        ):
        super().__init__()
        self.brits = BRITS(
            n_steps,
            n_features, 
            rnn_hidden_size,
            learning_rate=learning_rate,
            epochs=epochs,
            patience=patience,
            batch_size=batch_size,
            weight_decay=weight_decay,
            device=device
        )

    def fit(self, train_X, val_X):
        self.brits.fit(train_X, val_X) 

    def impute(self, test_X):
        return self.brits.impute(test_X) 
