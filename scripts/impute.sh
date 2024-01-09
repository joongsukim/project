#!/bin/bash

python src/train_impute.py -m datamodule.n_steps=20,30,60,105,210 model.rnn_hidden_size=16,32,64,128,256 model.learning_rate=0.001,0.0005,0.0001
