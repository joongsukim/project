from pytorch_forecasting.models.rnn import RecurrentNetwork
from pytorch_forecasting.metrics import RMSE, MAE

def get_model(
        dataset,
        cell_type="LSTM",
        hidden_size=10,
        rnn_layers=2,
        dropout=0.1,
        learning_rate=0.001,
        weight_decay=0.0,
        optimizer="ranger",
        loss="default"
):
    # parse loss
    # TODO: modify loss function
    loss = None
    
    return RecurrentNetwork.from_dataset(
        dataset,
        cell_type=cell_type,
        hidden_size=hidden_size,
        rnn_layers=rnn_layers,
        dropout=dropout,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        optimizer=optimizer,
        loss=loss
    )
