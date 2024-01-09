from abc import ABCMeta, abstractmethod
import numpy as np
from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet


class PTFForecastingDataModule(metaclass=ABCMeta):
    def __init__(self, df, context_length, horizon_length, batch_size=128, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
        self.df = df
        self.context_length = context_length
        self.horizon_length = horizon_length
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.train_df = None
        self.val_df = None
        self.test_df = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        # split train_df, val_df, test_df
        self.split()
        # dataset 만들기
        self.make_datasets()

    def split(self):
        num_rows = len(self.df)

        last_train_idx = int(np.floor(self.train_ratio * num_rows))
        last_val_idx = int(np.floor((self.train_ratio + self.val_ratio) * num_rows))

        self.train_df = self.df[:last_train_idx]
        self.val_df = self.df[last_train_idx - self.context_length - self.horizon_length + 1:last_val_idx]
        self.test_df = self.df[last_val_idx - self.context_length - self.horizon_length + 1:]

    def make_datasets(self):
        self.train_dataset = TimeSeriesDataSet(
            self.train_df,
            time_idx="time_idx",
            group_ids=["group_id"],
            target=self.target,
            max_encoder_length=self.context_length,
            max_prediction_length=self.horizon_length,
            time_varying_known_reals=self.time_varying_known_reals,
            time_varying_known_categoricals=self.time_varying_known_categoricals,
            time_varying_unknown_categoricals=self.time_varying_unknown_categoricals,
            time_varying_unknown_reals=self.time_varying_unknown_reals
        )

        self.val_dataset = TimeSeriesDataSet.from_dataset(self.train_dataset, self.val_df, predict=False, stop_randomization=False)
        self.test_dataset = TimeSeriesDataSet.from_dataset(self.train_dataset, self.test_df, predict=False, stop_randomization=False)

    @property
    def train_dataloader(self):
        return self.train_dataset.to_dataloader(train=True, batch_size=self.batch_size, num_workers=0)

    @property
    def val_dataloader(self):
        return self.val_dataset.to_dataloader(train=False, batch_size=self.batch_size, num_workers=0)

    @property
    def test_dataloader(self):
        return self.test_dataset.to_dataloader(train=False, batch_size=self.batch_size, num_workers=0)

    @property
    @abstractmethod
    def target(self):
        pass

    @property
    @abstractmethod
    def time_varying_known_reals(self):
        pass

    @property
    @abstractmethod
    def time_varying_known_categoricals(self):
        pass

    @property
    @abstractmethod
    def time_varying_unknown_reals(self):
        pass

    @property
    @abstractmethod
    def time_varying_unknown_categoricals(self):
        pass
