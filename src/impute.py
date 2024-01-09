# pip install --trusted-host pypi.python.org --trusted-host files.pythonhosted.org --trusted-host pypi.org 설치할패키지이름
# or
import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml", "requirements.txt"],
    pythonpath=True,
    dotenv=True,
)

# ------------------------------------------------------------------------------------ #
# `pyrootutils.setup_root(...)` is an optional line at the top of each entry file
# that helps to make the environment more robust and convenient
#
# the main advantages are:
# - allows you to keep all entry files in "src/" without installing project as a package
# - makes paths and scripts always work no matter where is your current work dir
# - automatically loads environment variables from ".env" file if exists
#
# how it works:
# - the line above recursively searches for either ".git" or "pyproject.toml" in present
#   and parent dirs, to determine the project root dir
# - adds root dir to the PYTHONPATH (if `pythonpath=True`), so this file can be run from
#   any place without installing project as a package
# - sets PROJECT_ROOT environment variable which is used in "configs/paths/default.yaml"
#   to make all paths always relative to the project root
# - loads environment variables from ".env" file in root dir (if `dotenv=True`)
#
# you can remove `pyrootutils.setup_root(...)` if you:
# 1. either install project as a package or move each entry file to the project root dir
# 2. simply remove PROJECT_ROOT variable from paths in "configs/paths/default.yaml"
# 3. always run entry files from the project root dir
#
# https://github.com/ashleve/pyrootutils
# ------------------------------------------------------------------------------------ #

from typing import List, Optional, Tuple

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
#from pytorch_lightning.loggers import LightningLoggerBase
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import src.impute.saits as saits
import torch

from src import utils
import os
import math
import time


log = utils.get_pylogger(__name__)


column_list = [
        ( 'COR (촉매반응로)', '2차 Air'),
        (    'Acid Gas',     '유량'),
        (    'Acid Gas',     '압력'),
        (    'Acid Gas',     '온도'),
        ( 'COR (촉매반응로)', '1차 Air'),
        ( 'COR (촉매반응로)', 'Air 압력'),
        ( 'COR (촉매반응로)', '연소 Air'),
        ( 'COR (촉매반응로)',  'COG유량'),
        ( 'COR (촉매반응로)',   '상부온도'),
        ( 'COR (촉매반응로)',   '중부온도'),
        (    '탈황 고압보일러',   '출구온도'),
        (    '탈황 고압보일러',   '스팀유량'),
        (    '탈황 고압보일러',     '압력'),
        (    '탈황 고압보일러',   '급수유량'),
        (    '탈황 고압보일러',     '레벨'),
        (       '저압보일러',     '압력'),
        (       '저압보일러',   '급수유량'),
        (       '저압보일러',   '스팀유량'),
        (     '반응기#1온도',     '입구'),
        (     '반응기#1온도',     '출구'),
        (     '반응기#2온도',     '입구'),
        (     '반응기#2온도',     '출구'),
        (         '콘덴서',     '압력'),
        (         '콘덴서',     '레벨'),
        (         '콘덴서',   '급수유량'),
        (         '유입량',   '스팀유량'),
        (         '유입량',  '순수유입량'),
        (   'PG Heat온도',   '#1출구'),
        (    'Tail Gas',    '유량1'),
        (    'Tail Gas',    '온도1'),
        (  'Drain Tank',     '레벨'),
        (  'Drain Tank',     '온도'),
        (   '저장Tank 온도',   '#100'),
        ('Desuper Heat',     '압력'),
        ('Desuper Heat',     '레벨'),
        ('TAIL GAS 분석기',    'H2S'),
        ('TAIL GAS 분석기',    'SO2')
]

"""
n_layers = 2
d_model = 256
d_inner = 128
n_head = 4
d_k = 64
d_v = 64
dropout = 0.1
diagonal_attention_mask = True
ORT_weight = 1
MIT_weight = 1
learning_rate = 1e-3
epochs = 100
patience = 10
batch_size = 32
weight_decay = 1e-5
#device=None
window_size = 60
stride = 1

data_file_path = "C:\\Users\\monet\\PycharmProjects\\posco-project-final-github\\data\\CO2_final\\missing\\missing_data.pkl"
out_file_dir = "C:\\Users\\monet\\PycharmProjects\\posco-project-final-github\\data\\CO2_final\\impute"
"""

@utils.task_wrapper
def train(cfg: DictConfig) -> Tuple[dict, dict]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator which applies extra utilities
    before and after the call.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    # 1.

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    log.info("Instantiating loggers...")
    logger = utils.instantiate_loggers(cfg.get("logger"))

    object_dict = {
        "cfg": cfg,
    #    "datamodule": datamodule,
    #    "model": model,
    #    "callbacks": callbacks,
        "logger": logger,
    #    "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        logger[0].log_hyperparams(cfg)

    # 1. df load
    df = pd.read_pickle(cfg.get("data_file_path"))

    # 2. numpy 형태로 만들기
    X = df[column_list].to_numpy()

    # 3. train / val set division
    window_size = cfg.get("window_size")
    stride = cfg.get("stride")
    num_data = len(X)
    idx_list_len = int((num_data - window_size) / stride + 1)

    # 8:2 split
    rng = np.random.default_rng(seed=1557)
    permutation = rng.permutation(idx_list_len)
    num_train = int(idx_list_len * 0.8)
    num_val = idx_list_len - num_train
    train_idx_list = permutation[:num_train]
    val_idx_list = permutation[num_train:]

    # 4. scaling
    scaler = MinMaxScaler()
    scaled_X = scaler.fit_transform(X)

    # 5. fitting (train/loss, val/loss)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_steps = window_size
    n_features = scaled_X.shape[1]

    n_layers = cfg.get("n_layers")
    d_model = cfg.get("d_model")
    d_inner = cfg.get("d_inner")
    n_head = cfg.get("n_head")
    d_k = cfg.get("d_k")
    d_v = cfg.get("d_v")
    dropout = cfg.get("dropout")
    diagonal_attention_mask = cfg.get("diagonal_attention_mask")
    ORT_weight = cfg.get("ORT_weight")
    MIT_weight = cfg.get("MIT_weight")
    learning_rate = cfg.get("learning_rate")
    epochs = cfg.get("epochs")
    patience = cfg.get("patience")
    batch_size = cfg.get("batch_size")
    weight_decay = cfg.get("weight_decay")
    saits_model = saits.SAITS(n_steps=n_steps, n_features=n_features, n_layers=n_layers, d_model=d_model, d_inner=d_inner, n_head=n_head,
                  d_k=d_k, d_v=d_v, dropout=dropout, diagonal_attention_mask=diagonal_attention_mask, ORT_weight=ORT_weight,
                  MIT_weight=MIT_weight, learning_rate=learning_rate, epochs=epochs, patience=patience, batch_size=batch_size,
                  weight_decay=weight_decay, device=device, stride=stride, train_idx_list=train_idx_list, val_idx_list=val_idx_list)

    start = time.time()

    saits_model.fit(torch.tensor(scaled_X, device=device))

    # 6. impute with whole data (test/loss)
    # [num_data, num_features]
    # [???, ???, num_featuers * num_window]
    trimmed_scaled_X_len = int(np.floor(num_data / window_size) * window_size)
    trimmed_scaled_X = scaled_X[:trimmed_scaled_X_len]
    trimmed_scaled_X = trimmed_scaled_X.reshape(int(trimmed_scaled_X_len/window_size), window_size, n_features)

    imputed_scaled_X = saits_model.impute(torch.tensor(trimmed_scaled_X, device=device))  # impute the originally-missing values and artificially-missing values
    imputed_scaled_X = imputed_scaled_X.reshape(trimmed_scaled_X_len, n_features)

    # 7. unscaling
    imputed_X = scaler.inverse_transform(imputed_scaled_X)

    # 8. finishing
    trimmed_df = df.iloc[:trimmed_scaled_X_len].copy()
    trimmed_missing_mask = trimmed_df.isna()
    trimmed_df[column_list] = imputed_X

    out_dir = "C:\\Users\\monet\\PycharmProjects\\posco-project-final-github\\data\\CO2_final\\impute"
    data_path = os.path.join(out_dir, "impute_data.pkl")
    mask_path = os.path.join(out_dir, "missing_mask.pkl")

    trimmed_df.to_pickle(data_path)
    trimmed_missing_mask.to_pickle(mask_path)

    end = time.time()

    print(saits_model.logger["training_loss"])
    logger[0].log_metrics({"train_loss": saits_model.logger["training_loss"], "val_loss": saits_model.logger["validating_loss"]})

    print(f"{end - start:.5f} sec")

    # 8. compute mae, mape per each columns

    # 9. numpy를 다시 df로 변환하기

    # 10. numpy 저장 / 각종 metric 저장

    #train_metrics = trainer.callback_metrics

    #if cfg.get("test"):
    #    log.info("Starting testing!")
    #    ckpt_path = trainer.checkpoint_callback.best_model_path if cfg.get("ckpt_path") is None else cfg.get("ckpt_path")
    #    if ckpt_path == "":
    #        log.warning("Best ckpt not found! Using current weights for testing...")
    #        ckpt_path = None
    #    trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
    #    # make snapshots
    #    utils.plot_snapshots(datamodule, model, ckpt_path, cfg.num_test_snapshots, trainer)
    #    #model_helper.load_checkpoint(ckpt_path)
    #    #model_helper.test(datamodule, trainer.logger)
    #    log.info(f"Best ckpt path: {ckpt_path}")

    #test_metrics = trainer.callback_metrics

    #if cfg.get("importance"):
    #    if not datamodule.is_setup:
    #        datamodule.setup()

    #    ckpt_path = trainer.checkpoint_callback.best_model_path if cfg.get("ckpt_path") is None else cfg.get("ckpt_path")
    #    if ckpt_path == "":
    #        log.warning("Best ckpt not found! Using current weights for testing...")
    #        ckpt_path = None
    #    model = model.load_from_checkpoint(ckpt_path)
    #    model.eval()

    #    dfs = model.permutation_importance(datamodule, cfg.get("importance_iter"), cfg.get("importance_batch_size"))
    #    for i, df in enumerate(dfs):
    #        df.to_excel(f"importance_{datamodule.decoder_Y_columns_title[i]}.xlsx")

    # merge train and test metrics
    #metric_dict = {**train_metrics, **test_metrics}

    #trainer.logger.log_hyperparams(metric_dict)

    return {}, object_dict


@hydra.main(config_path="../configs", config_name="impute_saits.yaml")
def main(cfg: DictConfig) -> Optional[float]:

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    #metric_value = utils.get_metric_value(
    #    metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    #)

    # return optimized metric
    #return metric_value
    return None


if __name__ == "__main__":
    main()

