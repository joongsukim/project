from pytorch_forecasting.models.temporal_fusion_transformer import TemporalFusionTransformer
from pytorch_forecasting.metrics import RMSE, MAE

def get_model(
        dataset,
        hidden_size=16,
        lstm_layers=2,
        dropout=0.1,
        attention_head_size=4,
        learning_rate=0.001,
        weight_decay=0.0,
        optimizer="ranger",
        loss="default"
):
    # parse loss
    # TODO: modify loss function
    loss = None
    
    return TemporalFusionTransformer.from_dataset(
        dataset,
        hidden_size=hidden_size,
        lstm_layers=lstm_layers,
        dropout=dropout,
        attention_head_size=attention_head_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        optimizer=optimizer,
        loss=loss
    )
