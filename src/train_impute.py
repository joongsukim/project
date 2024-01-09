import pyrootutils


root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml", "requirements.txt"],
    pythonpath=True,
    dotenv=True
)


import hydra 
from omegaconf import DictConfig 
import pytorch_lightning as pl

from src import utils
from src.dataframe.dataframe import DataFrameUtils 
import torch 
import numpy as np 
from src.models.impute_model import ImputeModelUtils


log = utils.get_pylogger(__name__) 

@utils.task_wrapper 
def impute(cfg: DictConfig):
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    # 1. load DataFrame from pickle 
    df_loader = hydra.utils.instantiate(cfg.dataframe) 
    df = df_loader.load() 

    # 2. DataModule
    datamodule = hydra.utils.instantiate(cfg.datamodule, df=df)

    # 3. Model
    if cfg.trainer.accelerator == "gpu":
        device = torch.device('cuda')
    else:
        device = torch.device('cpu') 

    model = hydra.utils.instantiate(cfg.model, n_features=datamodule.n_features, device=device) 

    # train
    model.fit(datamodule.train_X, datamodule.val_X) 

    # test 
    loggers = utils.instantiate_loggers(cfg.get("logger"))
    logger = loggers[0] 
    df = datamodule.df 
    column_names = datamodule.column_names 
    n_strides = datamodule.n_strides 
    n_steps = datamodule.n_steps 
    test_missing_mask = datamodule.test_missing_mask_unprocessed
    test_X_array = datamodule.test_X_unprocessed
    intact_test_X_array = datamodule.test_intact_X_unprocessed
    scaler = datamodule.scaler 

    result_df, test_metrics = model.test(test_X_array, intact_test_X_array, test_missing_mask, n_steps, n_strides, column_names, df, scaler, logger)

    # 4. save to pickle
    result_df.to_pickle(cfg.get("save_path"))

    #

    return {}, test_metrics 


@hydra.main(config_path=root / "configs", config_name="train_impute.yaml")
def main(cfg: DictConfig):
    impute(cfg)  

    return None 


if __name__ == "__main__":
    main()  
    


## needed
#'train_X'
#'val_X'
#'test_X'
#'test_X_intact'
#'test_X_indicating_mask'
#'n_steps'
#'n_features'

# value and shapes
#####'n_classes': 2
#'n_steps': 24
#'n_features': 10
#'train_X': (1280, 24, 10) (NaN included)
#####'train_y': (1280,) np.ndarray
#'val_X': (320, 24, 10)
#####'val_y': (320,)
#'test_X': (400, 24, 10)
#####'test_y': (400,)
#'test_X_intact': (400, 24, 10)
#'test_X_indicating_mask': (400, 24, 10)

