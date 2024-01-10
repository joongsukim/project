from abc import ABCMeta, abstractmethod
from src.dataframe.dataframe import DataFrameUtils 
from src.utils import plot_dataframes 
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
import numpy as np 


class BaseForecastModelHelper(metaclass=ABCMeta):
    def __init__(self, ptf_model_cls, **kwargs):
        self.ptf_model_cls = ptf_model_cls 
        self.kwargs = kwargs
        self.model = None 

    def get_model_from_dataset(self, dataset):
        kwargs = self.instantiate_kwargs()
        
        self.model = self.ptf_model_cls.from_dataset(dataset, **kwargs)

    def get_model_from_datamodule(self, datamodule):
        self.get_model_from_dataset(dataset=datamodule.data_train)

    @abstractmethod
    def instantiate_kwargs(self):
        pass 

    def load_checkpoint(self, ckpt_path):
        self.model = self.ptf_model_cls.load_from_checkpoint(ckpt_path) 

    def test(self, datamodule, logger):
        model = self.model 

        predictions, index = model.predict(datamodule.test_dataloader(), return_index=True)

        targets = datamodule.targets 

        for i, target in enumerate(targets):
            index[target] = predictions[:, i]

        predicted_df = index.copy()

        # plot prediction results
        df = datamodule.df 
        column_names = datamodule.column_names 

        imputed_df, missing_mask_df, _, intact_df = DataFrameUtils.decompose_df(df, column_names) 

        # shade ? 
        shade = {}
        for column in targets: 
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

        df_tuples = [('raw', intact_df), ('imputed', imputed_df), ('forecasted', predicted_df)]

        figs = [plot_dataframes(target, target, 'time_idx', False, *df_tuples, shade=shade[column]) for target in targets]

        logger.experiment.add_figure('Forecasting Result', figs) 

        # metrics 
        merged_df = predicted_df.join(intact_df, on='time_idx', lsuffix="_predicted", rsuffix="_raw")

        raw_target_values = merged_df[[f"{target}_raw" for target in targets]]
        predicted_target_values = merged_df[[f"{target}_predicted" for target in targets]]

        raw_targets_np = raw_target_values.to_numpy() 
        predicted_targets_np = predicted_target_values.to_numpy() 

        mae = mean_absolute_error(raw_targets_np, predicted_targets_np) 
        mape = mean_absolute_percentage_error(raw_targets_np, predicted_targets_np) 
        mse = mean_squared_error(raw_targets_np, predicted_targets_np) 

        logger.log_metrics({"test_mae": mae, "test_mape": mape, "test_mse": mse})
