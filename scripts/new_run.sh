#!/bin/bash

# 30/1 (보험)
# 30/10 (최우선)
# 10/10 (최하순위)

# transformer_30_10_0.9_0.2_512_2048
#python src/train_new_identification.py -m datamodule=identification_new_CO2 datamodule.context_length=30 datamodule.prediction_horizon=10 datamodule.scaler=standard datamodule.alpha=0.9 model.net.d_model=512 model.net.dim_feedforward=2048 model.net.dropout=0.2 trainer.max_epochs=500
# transformer_30_1_0.9_0.2_512_2048
#python src/train_new_identification.py -m datamodule=identification_new_CO2 datamodule.context_length=30 datamodule.prediction_horizon=1 datamodule.scaler=standard datamodule.alpha=0.9 model.net.d_model=512 model.net.dim_feedforward=2048 model.net.dropout=0.2 trainer.max_epochs=500

# nonratio_transformer_30_10_0.9_0.2_512_2048
#python src/train_new_identification.py -m datamodule=identification_new_CO2_nonratio datamodule.context_length=30 datamodule.prediction_horizon=10 datamodule.scaler=minmax datamodule.alpha=0.9 model.net.d_model=512 model.net.dim_feedforward=2048 model.net.dropout=0.2 trainer.max_epochs=500
# nonratio_transformer_30_1_0.9_0.2_512_2048
#python src/train_new_identification.py -m datamodule=identification_new_CO2_nonratio datamodule.context_length=30 datamodule.prediction_horizon=1 datamodule.scaler=minmax datamodule.alpha=0.9 model.net.d_model=512 model.net.dim_feedforward=2048 model.net.dropout=0.2 trainer.max_epochs=500

# transformer_30_10_0.9_0.1_256_1024
#python src/train_new_identification.py -m datamodule=identification_new_CO2 datamodule.context_length=30 datamodule.prediction_horizon=10 datamodule.scaler=standard datamodule.alpha=0.9 model.net.d_model=256 model.net.dim_feedforward=1024 trainer.max_epochs=500
# transformer_30_1_0.9_0.1_256_1024
#python src/train_new_identification.py -m datamodule=identification_new_CO2 datamodule.context_length=30 datamodule.prediction_horizon=1 datamodule.scaler=standard datamodule.alpha=0.9 model.net.d_model=256 model.net.dim_feedforward=1024 trainer.max_epochs=500

# nonratio_transformer_30_10_0.9_0.1_256_1024
#python src/train_new_identification.py -m datamodule=identification_new_CO2_nonratio datamodule.context_length=30 datamodule.prediction_horizon=10 datamodule.scaler=minmax datamodule.alpha=0.9 model.net.d_model=256 model.net.dim_feedforward=1024 trainer.max_epochs=500
# nonratio_transformer_30_1_0.9_0.1_256_1024
#python src/train_new_identification.py -m datamodule=identification_new_CO2_nonratio datamodule.context_length=30 datamodule.prediction_horizon=1 datamodule.scaler=minmax datamodule.alpha=0.9 model.net.d_model=256 model.net.dim_feedforward=1024 trainer.max_epochs=500


# transformer_10_10_0.9_0.1_256_1024
#python src/train_new_identification.py -m datamodule=identification_new_CO2 datamodule.context_length=10 datamodule.prediction_horizon=10 datamodule.scaler=standard datamodule.alpha=0.9 model.net.d_model=256 model.net.dim_feedforward=1024 trainer.max_epochs=500
# transformer_10_10_0.9_0.2_512_2048
#python src/train_new_identification.py -m datamodule=identification_new_CO2 datamodule.context_length=10 datamodule.prediction_horizon=10 datamodule.scaler=standard datamodule.alpha=0.9 model.net.d_model=512 model.net.dim_feedforward=2048 model.net.dropout=0.2 trainer.max_epochs=500

# nonratio_transformer_10_10_0.9_0.1_256_1024
#python src/train_new_identification.py -m datamodule=identification_new_CO2_nonratio datamodule.context_length=10 datamodule.prediction_horizon=10 datamodule.scaler=minmax datamodule.alpha=0.9 model.net.d_model=256 model.net.dim_feedforward=1024 trainer.max_epochs=500
# nonratio_transformer_10_10_0.9_0.2_512_2048
#python src/train_new_identification.py -m datamodule=identification_new_CO2_nonratio datamodule.context_length=10 datamodule.prediction_horizon=10 datamodule.scaler=minmax datamodule.alpha=0.9 model.net.d_model=512 model.net.dim_feedforward=2048 model.net.dropout=0.2 trainer.max_epochs=500


#################################################


# nonratio_transformer_30_10_0.9_0.1_256_1024
#python src/train_new_identification.py -m datamodule=identification_new_CO2_nonratio datamodule.context_length=30 datamodule.prediction_horizon=10 datamodule.scaler=minmax datamodule.alpha=0.7 model.net.d_model=128 model.net.dim_feedforward=512 model.net.n_layers=2 model.net.dropout=0.0 trainer.max_epochs=500
# nonratio_transformer_30_1_0.9_0.1_256_1024
#python src/train_new_identification.py -m datamodule=identification_new_CO2_nonratio datamodule.context_length=30 datamodule.prediction_horizon=1 datamodule.scaler=minmax datamodule.alpha=0.9 model.net.d_model=256 model.net.dim_feedforward=1024 trainer.max_epochs=500


#python src/train_new_identification.py -m datamodule=identification_new_CO2_nonratio datamodule.context_length=30 datamodule.prediction_horizon=10 datamodule.scaler=minmax datamodule.alpha=0.9 model=identification_new_transformer_sep model.net.d_model=64 model.net.encoder_dim_feedforward=256 model.net.encoder_n_heads=4 model.net.encoder_n_layers=3 model.net.decoder_dim_feedforward=64 model.net.decoder_n_heads=4 model.net.decoder_n_layers=1 model.net.dropout=0.0 trainer.max_epochs=500


# dropout 0.0
# context 30
# prediction 10
# alpha 0.7
# d_model 128
# d_feedforward 512
# n_layers 2
# d_0.0_c_30_p_10_a_0.7_d_m_128_d_f_512_n_l_2
#python src/train_new_identification.py -m datamodule=identification_new_CO2_nonratio datamodule.context_length=30 datamodule.prediction_horizon=10 datamodule.scaler=minmax datamodule.alpha=0.7 model.net.d_model=128 model.net.dim_feedforward=512 model.net.n_layers=2 model.net.dropout=0.0 trainer.max_epochs=500

# dropout 0.0
# context 10
# prediction 10
# alpha 0.7
# d_model 128
# d_feedforward 512
# n_layers 2
# d_0.0_c_10_p_10_a_0.7_d_m_128_d_f_512_n_l_2
#python src/train_new_identification.py -m datamodule=identification_new_CO2_nonratio datamodule.context_length=10 datamodule.prediction_horizon=10 datamodule.scaler=minmax datamodule.alpha=0.7 model.net.d_model=128 model.net.dim_feedforward=512 model.net.n_layers=2 model.net.dropout=0.0 trainer.max_epochs=500

# dropout 0.0
# context 60
# prediction 10
# alpha 0.7
# d_model 128
# d_feedforward 512
# n_layers 2
# d_0.0_c_60_p_10_a_0.7_d_m_128_d_f_512_n_l_2
#python src/train_new_identification.py -m datamodule=identification_new_CO2_nonratio datamodule.context_length=60 datamodule.prediction_horizon=10 datamodule.scaler=minmax datamodule.alpha=0.7 model.net.d_model=128 model.net.dim_feedforward=512 model.net.n_layers=2 model.net.dropout=0.0 trainer.max_epochs=500


# dropout 0.2
# context 30
# prediction 10
# alpha 0.7
# d_model 128
# d_feedforward 512
# n_layers 2
# d_0.2_c_30_p_10_a_0.7_d_m_128_d_f_512_n_l_2
#python src/train_new_identification.py -m datamodule=identification_new_CO2_nonratio datamodule.context_length=30 datamodule.prediction_horizon=10 datamodule.scaler=minmax datamodule.alpha=0.7 model.net.d_model=128 model.net.dim_feedforward=512 model.net.n_layers=2 model.net.dropout=0.2 trainer.max_epochs=500

# dropout 0.2
# context 10
# prediction 10
# alpha 0.7
# d_model 128
# d_feedforward 512
# n_layers 2
# d_0.2_c_10_p_10_a_0.7_d_m_128_d_f_512_n_l_2
#python src/train_new_identification.py -m datamodule=identification_new_CO2_nonratio datamodule.context_length=10 datamodule.prediction_horizon=10 datamodule.scaler=minmax datamodule.alpha=0.7 model.net.d_model=128 model.net.dim_feedforward=512 model.net.n_layers=2 model.net.dropout=0.2 trainer.max_epochs=500

# dropout 0.2
# context 60
# prediction 10
# alpha 0.7
# d_model 128
# d_feedforward 512
# n_layers 2
# d_0.2_c_60_p_10_a_0.7_d_m_128_d_f_512_n_l_2
#python src/train_new_identification.py -m datamodule=identification_new_CO2_nonratio datamodule.context_length=60 datamodule.prediction_horizon=10 datamodule.scaler=minmax datamodule.alpha=0.7 model.net.d_model=128 model.net.dim_feedforward=512 model.net.n_layers=2 model.net.dropout=0.2 trainer.max_epochs=500



# dropout 0.4
# context 30
# prediction 10
# alpha 0.7
# d_model 128
# d_feedforward 512
# n_layers 2
# d_0.4_c_30_p_10_a_0.7_d_m_128_d_f_512_n_l_2
#python src/train_new_identification.py -m datamodule=identification_new_CO2_nonratio datamodule.context_length=30 datamodule.prediction_horizon=10 datamodule.scaler=minmax datamodule.alpha=0.7 model.net.d_model=128 model.net.dim_feedforward=512 model.net.n_layers=2 model.net.dropout=0.4 trainer.max_epochs=500

# dropout 0.4
# context 10
# prediction 10
# alpha 0.7
# d_model 128
# d_feedforward 512
# n_layers 2
# d_0.4_c_10_p_10_a_0.7_d_m_128_d_f_512_n_l_2
#python src/train_new_identification.py -m datamodule=identification_new_CO2_nonratio datamodule.context_length=10 datamodule.prediction_horizon=10 datamodule.scaler=minmax datamodule.alpha=0.7 model.net.d_model=128 model.net.dim_feedforward=512 model.net.n_layers=2 model.net.dropout=0.4 trainer.max_epochs=500

# dropout 0.4
# context 60
# prediction 10
# alpha 0.7
# d_model 128
# d_feedforward 512
# n_layers 2
# d_0.4_c_60_p_10_a_0.7_d_m_128_d_f_512_n_l_2
#python src/train_new_identification.py -m datamodule=identification_new_CO2_nonratio datamodule.context_length=60 datamodule.prediction_horizon=10 datamodule.scaler=minmax datamodule.alpha=0.7 model.net.d_model=128 model.net.dim_feedforward=512 model.net.n_layers=2 model.net.dropout=0.4 trainer.max_epochs=500



# dropout 0.0
# context 30
# prediction 1
# alpha 0.7
# d_model 128
# d_feedforward 512
# n_layers 2
# d_0.0_c_30_p_1_a_0.7_d_m_128_d_f_512_n_l_2
#python src/train_new_identification.py -m datamodule=identification_new_CO2_nonratio datamodule.context_length=30 datamodule.prediction_horizon=1 datamodule.scaler=minmax datamodule.alpha=0.7 model.net.d_model=128 model.net.dim_feedforward=512 model.net.n_layers=2 model.net.dropout=0.0 trainer.max_epochs=500

# dropout 0.0
# context 10
# prediction 1
# alpha 0.7
# d_model 128
# d_feedforward 512
# n_layers 2
# d_0.0_c_10_p_1_a_0.7_d_m_128_d_f_512_n_l_2
#python src/train_new_identification.py -m datamodule=identification_new_CO2_nonratio datamodule.context_length=10 datamodule.prediction_horizon=1 datamodule.scaler=minmax datamodule.alpha=0.7 model.net.d_model=128 model.net.dim_feedforward=512 model.net.n_layers=2 model.net.dropout=0.0 trainer.max_epochs=500

# dropout 0.0
# context 60
# prediction 1
# alpha 0.7
# d_model 128
# d_feedforward 512
# n_layers 2
##python src/train_new_identification.py -m datamodule=identification_new_CO2_nonratio datamodule.context_length=60 datamodule.prediction_horizon=1 datamodule.scaler=minmax datamodule.alpha=0.7 model.net.d_model=128 model.net.dim_feedforward=512 model.net.n_layers=2 model.net.dropout=0.0 trainer.max_epochs=500



# dropout 0.2
# context 30
# prediction 1
# alpha 0.7
# d_model 128
# d_feedforward 512
# n_layers 2
##python src/train_new_identification.py -m datamodule=identification_new_CO2_nonratio datamodule.context_length=30 datamodule.prediction_horizon=1 datamodule.scaler=minmax datamodule.alpha=0.7 model.net.d_model=128 model.net.dim_feedforward=512 model.net.n_layers=2 model.net.dropout=0.2 trainer.max_epochs=500

# dropout 0.2
# context 10
# prediction 1
# alpha 0.7
# d_model 128
# d_feedforward 512
# n_layers 2
##python src/train_new_identification.py -m datamodule=identification_new_CO2_nonratio datamodule.context_length=10 datamodule.prediction_horizon=1 datamodule.scaler=minmax datamodule.alpha=0.7 model.net.d_model=128 model.net.dim_feedforward=512 model.net.n_layers=2 model.net.dropout=0.2 trainer.max_epochs=500

# dropout 0.2
# context 60
# prediction 1
# alpha 0.7
# d_model 128
# d_feedforward 512
# n_layers 2
##python src/train_new_identification.py -m datamodule=identification_new_CO2_nonratio datamodule.context_length=60 datamodule.prediction_horizon=1 datamodule.scaler=minmax datamodule.alpha=0.7 model.net.d_model=128 model.net.dim_feedforward=512 model.net.n_layers=2 model.net.dropout=0.2 trainer.max_epochs=500


# dropout 0.0
# context 10
# prediction 1
# alpha 0.7
# d_model 64
# d_feedforward 256
# n_layers 2
# d_0.0_c_10_p_1_a_0.7_d_m_64_d_f_256_n_l_2
#python src/train_new_identification.py -m datamodule=identification_new_CO2_nonratio datamodule.context_length=10 datamodule.prediction_horizon=1 datamodule.scaler=minmax datamodule.alpha=0.7 model.net.d_model=64 model.net.dim_feedforward=256 model.net.n_layers=2 model.net.dropout=0.0 trainer.max_epochs=500


# 1. network size 줄이기
# 2. 그 머냐 lstm 테스트
# 3. no smoothing 테스트
# 30, 10

# dropout 0.0
# context 30
# prediction 10
# alpha 0.7
# d_model 64
# d_feedforward 256
# n_layers 2
# d_0.0_c_30_p_10_a_0.7_d_m_64_d_f_256_n_l_2
python src/train_new_identification.py -m datamodule=identification_new_CO2_nonratio datamodule.context_length=30 datamodule.prediction_horizon=10 datamodule.scaler=minmax datamodule.alpha=0.7 model.net.d_model=64 model.net.dim_feedforward=256 model.net.n_layers=2 model.net.dropout=0.0 trainer.max_epochs=500

# dropout 0.0
# context 30
# prediction 10
# alpha 0.7
# d_model 32
# d_feedforward 128
# n_layers 2
# d_0.0_c_30_p_10_a_0.7_d_m_32_d_f_128_n_l_2
python src/train_new_identification.py -m datamodule=identification_new_CO2_nonratio datamodule.context_length=30 datamodule.prediction_horizon=10 datamodule.scaler=minmax datamodule.alpha=0.7 model.net.d_model=32 model.net.dim_feedforward=128 model.net.n_layers=2 model.net.dropout=0.0 trainer.max_epochs=500

# dropout 0.0
# context 10
# prediction 1
# alpha 0.7
# d_model 32
# d_feedforward 128
# n_layers 2
# d_0.0_c_10_p_1_a_0.7_d_m_32_d_f_128_n_l_2
python src/train_new_identification.py -m datamodule=identification_new_CO2_nonratio datamodule.context_length=10 datamodule.prediction_horizon=1 datamodule.scaler=minmax datamodule.alpha=0.7 model.net.d_model=32 model.net.dim_feedforward=128 model.net.n_layers=2 model.net.dropout=0.0 trainer.max_epochs=500

# dropout 0.0
# context 30
# prediction 10
# alpha 1.0
# d_model 64
# d_feedforward 256
# n_layers 2
# d_0.0_c_30_p_10_a_1.0_d_m_64_d_f_256_n_l_2
python src/train_new_identification.py -m datamodule=identification_new_CO2_nonratio datamodule.context_length=30 datamodule.prediction_horizon=10 datamodule.scaler=minmax datamodule.alpha=1.0 model.net.d_model=64 model.net.dim_feedforward=256 model.net.n_layers=2 model.net.dropout=0.0 trainer.max_epochs=500

# dropout 0.0
# context 30
# prediction 10
# alpha 1.0
# d_model 32
# d_feedforward 128
# n_layers 2
# d_0.0_c_30_p_10_a_1.0_d_m_32_d_f_128_n_l_2
python src/train_new_identification.py -m datamodule=identification_new_CO2_nonratio datamodule.context_length=30 datamodule.prediction_horizon=10 datamodule.scaler=minmax datamodule.alpha=1.0 model.net.d_model=32 model.net.dim_feedforward=128 model.net.n_layers=2 model.net.dropout=0.0 trainer.max_epochs=500

# dropout 0.0
# context 10
# prediction 1
# alpha 1.0
# d_model 32
# d_feedforward 128
# n_layers 2
# d_0.0_c_10_p_1_a_1.0_d_m_32_d_f_128_n_l_2
python src/train_new_identification.py -m datamodule=identification_new_CO2_nonratio datamodule.context_length=10 datamodule.prediction_horizon=1 datamodule.scaler=minmax datamodule.alpha=1.0 model.net.d_model=32 model.net.dim_feedforward=128 model.net.n_layers=2 model.net.dropout=0.0 trainer.max_epochs=500

# dropout 0.0
# context 30
# prediction 10
# alpha 1.0
# d_model 256
# d_feedforward 1024
# n_layers 2
# d_0.0_c_30_p_10_a_1.0_d_m_256_d_f_1024_n_l_2
python src/train_new_identification.py -m datamodule=identification_new_CO2_nonratio datamodule.context_length=30 datamodule.prediction_horizon=10 datamodule.scaler=minmax datamodule.alpha=1.0 model.net.d_model=256 model.net.dim_feedforward=1024 model.net.n_layers=2 model.net.dropout=0.0 trainer.max_epochs=500

# dropout 0.0
# context 10
# prediction 1
# alpha 1.0
# d_model 256
# d_feedforward 1024
# n_layers 2
# d_0.0_c_10_p_1_a_1.0_d_m_256_d_f_1024_n_l_2
python src/train_new_identification.py -m datamodule=identification_new_CO2_nonratio datamodule.context_length=10 datamodule.prediction_horizon=1 datamodule.scaler=minmax datamodule.alpha=1.0 model.net.d_model=256 model.net.dim_feedforward=1024 model.net.n_layers=2 model.net.dropout=0.0 trainer.max_epochs=500


## LSTM LSTM LSTM
# dropout 0.0
# context 10
# prediction 1
# alpha 0.7
# hidden_dim 32
# n_layers 2
# d_0.0_c_10_p_1_a_0.7_h_32_n_l_2
python src/train_new_identification.py -m task_name=new_CO2_LSTM datamodule=identification_new_CO2_nonratio datamodule.context_length=10 datamodule.prediction_horizon=1 datamodule.scaler=minmax datamodule.alpha=0.7 model=identification_new_lstm model.net.hidden_dim=32 model.net.n_layers=2 model.net.dropout=0.0 trainer.max_epochs=500

# dropout 0.2
# context 10
# prediction 1
# alpha 0.7
# hidden_dim 32
# n_layers 2
# d_0.2_c_10_p_1_a_0.7_h_32_n_l_2
python src/train_new_identification.py -m task_name=new_CO2_LSTM datamodule=identification_new_CO2_nonratio datamodule.context_length=10 datamodule.prediction_horizon=1 datamodule.scaler=minmax datamodule.alpha=0.7 model=identification_new_lstm model.net.hidden_dim=32 model.net.n_layers=2 model.net.dropout=0.2 trainer.max_epochs=500

# dropout 0.4
# context 10
# prediction 1
# alpha 0.7
# hidden_dim 32
# n_layers 2
# d_0.4_c_10_p_1_a_0.7_h_32_n_l_2
python src/train_new_identification.py -m task_name=new_CO2_LSTM datamodule=identification_new_CO2_nonratio datamodule.context_length=10 datamodule.prediction_horizon=1 datamodule.scaler=minmax datamodule.alpha=0.7 model=identification_new_lstm model.net.hidden_dim=32 model.net.n_layers=2 model.net.dropout=0.4 trainer.max_epochs=500

# dropout 0.0
# context 30
# prediction 10
# alpha 0.7
# hidden_dim 32
# n_layers 2
# d_0.0_c_30_p_10_a_0.7_h_32_n_l_2
python src/train_new_identification.py -m task_name=new_CO2_LSTM datamodule=identification_new_CO2_nonratio datamodule.context_length=30 datamodule.prediction_horizon=10 datamodule.scaler=minmax datamodule.alpha=0.7 model=identification_new_lstm model.net.hidden_dim=32 model.net.n_layers=2 model.net.dropout=0.0 trainer.max_epochs=500

# dropout 0.2
# context 30
# prediction 10
# alpha 0.7
# hidden_dim 32
# n_layers 2
# d_0.2_c_30_p_10_a_0.7_h_32_n_l_2
python src/train_new_identification.py -m task_name=new_CO2_LSTM datamodule=identification_new_CO2_nonratio datamodule.context_length=30 datamodule.prediction_horizon=10 datamodule.scaler=minmax datamodule.alpha=0.7 model=identification_new_lstm model.net.hidden_dim=32 model.net.n_layers=2 model.net.dropout=0.2 trainer.max_epochs=500

# dropout 0.4
# context 30
# prediction 10
# alpha 0.7
# hidden_dim 32
# n_layers 2
# d_0.4_c_30_p_10_a_0.7_h_32_n_l_2
python src/train_new_identification.py -m task_name=new_CO2_LSTM datamodule=identification_new_CO2_nonratio datamodule.context_length=30 datamodule.prediction_horizon=10 datamodule.scaler=minmax datamodule.alpha=0.7 model=identification_new_lstm model.net.hidden_dim=32 model.net.n_layers=2 model.net.dropout=0.4 trainer.max_epochs=500


#####
# dropout 0.0
# context 10
# prediction 1
# alpha 0.7
# hidden_dim 64
# n_layers 2
# d_0.0_c_10_p_1_a_0.7_h_64_n_l_2
python src/train_new_identification.py -m task_name=new_CO2_LSTM datamodule=identification_new_CO2_nonratio datamodule.context_length=10 datamodule.prediction_horizon=1 datamodule.scaler=minmax datamodule.alpha=0.7 model=identification_new_lstm model.net.hidden_dim=64 model.net.n_layers=2 model.net.dropout=0.0 trainer.max_epochs=500

# dropout 0.2
# context 10
# prediction 1
# alpha 0.7
# hidden_dim 64
# n_layers 2
# d_0.2_c_10_p_1_a_0.7_h_64_n_l_2
python src/train_new_identification.py -m task_name=new_CO2_LSTM datamodule=identification_new_CO2_nonratio datamodule.context_length=10 datamodule.prediction_horizon=1 datamodule.scaler=minmax datamodule.alpha=0.7 model=identification_new_lstm model.net.hidden_dim=64 model.net.n_layers=2 model.net.dropout=0.2 trainer.max_epochs=500

# dropout 0.4
# context 10
# prediction 1
# alpha 0.7
# hidden_dim 64
# n_layers 2
# d_0.4_c_10_p_1_a_0.7_h_64_n_l_2
python src/train_new_identification.py -m task_name=new_CO2_LSTM datamodule=identification_new_CO2_nonratio datamodule.context_length=10 datamodule.prediction_horizon=1 datamodule.scaler=minmax datamodule.alpha=0.7 model=identification_new_lstm model.net.hidden_dim=64 model.net.n_layers=2 model.net.dropout=0.4 trainer.max_epochs=500

# dropout 0.0
# context 30
# prediction 10
# alpha 0.7
# hidden_dim 64
# n_layers 2
# d_0.0_c_30_p_10_a_0.7_h_64_n_l_2
python src/train_new_identification.py -m task_name=new_CO2_LSTM datamodule=identification_new_CO2_nonratio datamodule.context_length=30 datamodule.prediction_horizon=10 datamodule.scaler=minmax datamodule.alpha=0.7 model=identification_new_lstm model.net.hidden_dim=64 model.net.n_layers=2 model.net.dropout=0.0 trainer.max_epochs=500

# dropout 0.2
# context 30
# prediction 10
# alpha 0.7
# hidden_dim 64
# n_layers 2
# d_0.2_c_30_p_10_a_0.7_h_64_n_l_2
python src/train_new_identification.py -m task_name=new_CO2_LSTM datamodule=identification_new_CO2_nonratio datamodule.context_length=30 datamodule.prediction_horizon=10 datamodule.scaler=minmax datamodule.alpha=0.7 model=identification_new_lstm model.net.hidden_dim=64 model.net.n_layers=2 model.net.dropout=0.2 trainer.max_epochs=500

# dropout 0.4
# context 30
# prediction 10
# alpha 0.7
# hidden_dim 64
# n_layers 2
# d_0.4_c_30_p_10_a_0.7_h_64_n_l_2
python src/train_new_identification.py -m task_name=new_CO2_LSTM datamodule=identification_new_CO2_nonratio datamodule.context_length=30 datamodule.prediction_horizon=10 datamodule.scaler=minmax datamodule.alpha=0.7 model=identification_new_lstm model.net.hidden_dim=64 model.net.n_layers=2 model.net.dropout=0.4 trainer.max_epochs=500


##############
# dropout 0.0
# context 10
# prediction 1
# alpha 1.0
# hidden_dim 32
# n_layers 2
python src/train_new_identification.py -m task_name=new_CO2_LSTM datamodule=identification_new_CO2_nonratio datamodule.context_length=10 datamodule.prediction_horizon=1 datamodule.scaler=minmax datamodule.alpha=1.0 model=identification_new_lstm model.net.hidden_dim=32 model.net.n_layers=2 model.net.dropout=0.0 trainer.max_epochs=500

# dropout 0.2
# context 10
# prediction 1
# alpha 1.0
# hidden_dim 32
# n_layers 2
python src/train_new_identification.py -m task_name=new_CO2_LSTM datamodule=identification_new_CO2_nonratio datamodule.context_length=10 datamodule.prediction_horizon=1 datamodule.scaler=minmax datamodule.alpha=1.0 model=identification_new_lstm model.net.hidden_dim=32 model.net.n_layers=2 model.net.dropout=0.2 trainer.max_epochs=500

# dropout 0.4
# context 10
# prediction 1
# alpha 1.0
# hidden_dim 32
# n_layers 2
python src/train_new_identification.py -m task_name=new_CO2_LSTM datamodule=identification_new_CO2_nonratio datamodule.context_length=10 datamodule.prediction_horizon=1 datamodule.scaler=minmax datamodule.alpha=1.0 model=identification_new_lstm model.net.hidden_dim=32 model.net.n_layers=2 model.net.dropout=0.4 trainer.max_epochs=500

# dropout 0.0
# context 30
# prediction 10
# alpha 1.0
# hidden_dim 32
# n_layers 2
python src/train_new_identification.py -m task_name=new_CO2_LSTM datamodule=identification_new_CO2_nonratio datamodule.context_length=30 datamodule.prediction_horizon=10 datamodule.scaler=minmax datamodule.alpha=1.0 model=identification_new_lstm model.net.hidden_dim=32 model.net.n_layers=2 model.net.dropout=0.0 trainer.max_epochs=500

# dropout 0.2
# context 30
# prediction 10
# alpha 1.0
# hidden_dim 32
# n_layers 2
python src/train_new_identification.py -m task_name=new_CO2_LSTM datamodule=identification_new_CO2_nonratio datamodule.context_length=30 datamodule.prediction_horizon=10 datamodule.scaler=minmax datamodule.alpha=1.0 model=identification_new_lstm model.net.hidden_dim=32 model.net.n_layers=2 model.net.dropout=0.2 trainer.max_epochs=500

# dropout 0.4
# context 30
# prediction 10
# alpha 1.0
# hidden_dim 32
# n_layers 2
python src/train_new_identification.py -m task_name=new_CO2_LSTM datamodule=identification_new_CO2_nonratio datamodule.context_length=30 datamodule.prediction_horizon=10 datamodule.scaler=minmax datamodule.alpha=1.0 model=identification_new_lstm model.net.hidden_dim=32 model.net.n_layers=2 model.net.dropout=0.4 trainer.max_epochs=500

