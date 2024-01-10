from pytorch_forecasting.models.rnn import RecurrentNetwork
from pytorch_forecasting.metrics import RMSE, MAE
from src.models.base import BaseForecastModelHelper

class LSTMForecastModelHelper(BaseForecastModelHelper):
    def __init__(
        self,
        cell_type: str = "LSTM",
        hidden_size: int = 64, 
        rnn_layers: int = 1,
        dropout: float = 0.0, 
        learning_rate: float = 0.001, 
        weight_decay: float = 0.0,
        optimizer: str = "ranger", 
        loss="rmse"
        ):
        kwargs = dict()
        
        kwargs["cell_type"] = cell_type
        kwargs["hidden_size"] = hidden_size
        kwargs["rnn_layers"] = rnn_layers
        kwargs["dropout"] = dropout
        kwargs["learning_rate"] = learning_rate
        kwargs["weight_decay"] = weight_decay
        kwargs["optimizer"] = optimizer
        kwargs["loss"] = loss
        super().__init__(RecurrentNetwork, **kwargs) 
    
    def instantiate_kwargs(self):
        kwargs = self.kwargs 

        loss = self.kwargs["loss"]

        if loss == "rmse":
            loss_metric = RMSE() 
        elif loss == "mae":
            loss_metric = MAE() 

        kwargs["loss"] = loss_metric 

        return kwargs 
