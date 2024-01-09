# DEPRECATED

from abc import ABCMeta, abstractmethod

import pandas as pd

from src.forecast.forecast import BaseForecastingModule
from pytorch_forecasting import TemporalFusionTransformer, RecurrentNetwork, TimeSeriesDataSet


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
                end_inds.append(len(missing_mask) - 1)

            shade[column] = list(zip(start_inds, end_inds))

        df_tuples = [('raw', intact_df), ('imputed', imputed_df), ('forecasted', predicted_df)]

        figs = [plot_dataframes(target, target, 'time_idx', False, *df_tuples, shade=shade[column]) for target in
                targets]

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


class Dataloader...()

class BaseTorchForecastingModule(BaseForecastingModule, metaclass=ABCMeta):
    def __init__(
            self,
            model,
            trainer,
            dataset
    ):
        self.model = None
        pass

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    def prepare_dataset(self, df: pd.DataFrame) -> TimeSeriesDataSet:
        training = TimeSeriesDataSet(
            으음... 으음... 으음...
            비율로 해버릴까보다? ㅋㅋ
        )
        pass

    @abstractmethod
    def set_model(self, dataset):
        pass

    def train(self, df: pd.DataFrame, trainer):
        model = self.model

        # df -> TimeSeriesDataSet instance
        dataset = self.prepare_dataset(df)

        trainer.fit(
            model,
            train_dataloaders=blabla,
            val_dataloaders=blabla
        )
        model, trainer, ...
        pass

    def forecast(self):
        pass


class LSTMForecastingModule(BaseTorchForecastingModule):
    def __init__(
            self,
            hidden_size=10,
            rnn_layers=2,
            dropout=0.1,
            loss=None
    ):
        self.hidden_size = hidden_size
        self.rnn_layers = rnn_layers
        self.dropout = dropout
        self.loss = loss
        pass

    def set_model(self, dataset):
        self.model = RecurrentNetwork.from_dataset(
            dataset,
            cell_type='LSTM',
            hidden_size=self.hidden_size,
            rnn_layers=self.rnn_layers,
            dropout=self.dropout,
            loss=self.loss
        )


class TemporalFusionTransformerForecastingModule(BaseTorchForecastingModule):
    def __init__(
            self,
            lr=0.001,
            hidden_size=16,
            attention_head_size=1,
            dropout=0.1,
            hidden_continuous_size=8,
            output_size=7,
            loss=None,
            reduce_on_plateau_patience=4,
            lstm_layers=1
    ):
        self.lr = lr
        self.hidden_size =hidden_size
        self.attention_head_size = attention_head_size
        self.dropout = dropout
        self.hidden_continuous_size = hidden_continuous_size
        self.output_size = output_size
        self.loss = loss
        self.reduce_on_plateau_patience = reduce_on_plateau_patience
        self.lstm_layers = lstm_layers

    def set_model(self, dataset):
        self.model = TemporalFusionTransformer.from_dataset(
            dataset,
            learning_rate=self.lr,
            hidden_size=self.hidden_size,
            attention_head_size=self.attention_head_size,
            dropout=self.dropout,
            hidden_continuous_size=self.hidden_continuous_size,
            output_size=self.output_size,
            loss=self.loss,
            reduce_on_plateau_patience=self.reduce_on_plateau_patience,
            lstm_layers=self.lstm_layers
        )
