from src.forecast.datamodule.datamodule import PTFForecastingDataModule


class PTFNOxDataModule(PTFForecastingDataModule):
    def __init__(self, df, context_length, horizon_length, batch_size=128, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
        super().__init__(df=df,
                         context_length=context_length,
                         horizon_length=horizon_length,
                         batch_size=batch_size,
                         train_ratio=train_ratio,
                         val_ratio=val_ratio,
                         test_ratio=test_ratio)

    @property
    def target(self):
        return "NOx emission concentration"

    @property
    def time_varying_known_reals(self):
        return ["boiler load", "total coal feed rate", "inlet NOx concentration",
              "NH3 injection flow", "inlet flue gas temperature", "inlet flue gas flow",
              "inlet O2 content"]

    @property
    def time_varying_known_categoricals(self):
        return []

    @property
    def time_varying_unknown_reals(self):
        return ["NOx emission concentration"]

    @property
    def time_varying_unknown_categoricals(self):
        return []
