#30, 5
#30, 1
#15, 15
#15, 5
#15, 1
#5, 1
#5, 15
##5, 30
#1, 1
#1, 5

#!/bin/bash

python src/train_identification.py -m datamodule.context_length=30 datamodule.prediction_horizon=5 hparams_search=lstm_optuna model=identification_lstm
python src/train_identification.py -m datamodule.context_length=30 datamodule.prediction_horizon=1 hparams_search=lstm_optuna model=identification_lstm
python src/train_identification.py -m datamodule.context_length=15 datamodule.prediction_horizon=15 hparams_search=lstm_optuna model=identification_lstm
python src/train_identification.py -m datamodule.context_length=15 datamodule.prediction_horizon=5 hparams_search=lstm_optuna model=identification_lstm
python src/train_identification.py -m datamodule.context_length=15 datamodule.prediction_horizon=1 hparams_search=lstm_optuna model=identification_lstm
python src/train_identification.py -m datamodule.context_length=5 datamodule.prediction_horizon=1 hparams_search=lstm_optuna model=identification_lstm
python src/train_identification.py -m datamodule.context_length=5 datamodule.prediction_horizon=15 hparams_search=lstm_optuna model=identification_lstm
python src/train_identification.py -m datamodule.context_length=5 datamodule.prediction_horizon=30 hparams_search=lstm_optuna model=identification_lstm
python src/train_identification.py -m datamodule.context_length=1 datamodule.prediction_horizon=1 hparams_search=lstm_optuna model=identification_lstm
python src/train_identification.py -m datamodule.context_length=1 datamodule.prediction_horizon=5 hparams_search=lstm_optuna model=identification_lstm
python src/train_identification.py -m datamodule.context_length=1 datamodule.prediction_horizon=15 hparams_search=lstm_optuna model=identification_lstm
python src/train_identification.py -m datamodule.context_length=1 datamodule.prediction_horizon=30 hparams_search=lstm_optuna model=identification_lstm


python src/train_identification.py -m datamodule.context_length=30 datamodule.prediction_horizon=5 hparams_search=transformer_optuna model=identification_transformer
python src/train_identification.py -m datamodule.context_length=30 datamodule.prediction_horizon=1 hparams_search=transformer_optuna model=identification_transformer
python src/train_identification.py -m datamodule.context_length=15 datamodule.prediction_horizon=15 hparams_search=transformer_optuna model=identification_transformer
python src/train_identification.py -m datamodule.context_length=15 datamodule.prediction_horizon=5 hparams_search=transformer_optuna model=identification_transformer
python src/train_identification.py -m datamodule.context_length=15 datamodule.prediction_horizon=1 hparams_search=transformer_optuna model=identification_transformer
python src/train_identification.py -m datamodule.context_length=5 datamodule.prediction_horizon=1 hparams_search=transformer_optuna model=identification_transformer
python src/train_identification.py -m datamodule.context_length=5 datamodule.prediction_horizon=15 hparams_search=transformer_optuna model=identification_transformer
python src/train_identification.py -m datamodule.context_length=5 datamodule.prediction_horizon=30 hparams_search=transformer_optuna model=identification_transformer
python src/train_identification.py -m datamodule.context_length=1 datamodule.prediction_horizon=1 hparams_search=transformer_optuna model=identification_transformer
python src/train_identification.py -m datamodule.context_length=1 datamodule.prediction_horizon=5 hparams_search=transformer_optuna model=identification_transformer
python src/train_identification.py -m datamodule.context_length=1 datamodule.prediction_horizon=15 hparams_search=transformer_optuna model=identification_transformer
python src/train_identification.py -m datamodule.context_length=1 datamodule.prediction_horizon=30 hparams_search=transformer_optuna model=identification_transformer


      #datamodule.context_length: choice(1,10,30)
      #datamodule.prediction_horizon: choice(1,5)
