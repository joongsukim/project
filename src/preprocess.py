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
from src.data.dataframe import DataFrameUtils 


log = utils.get_pylogger(__name__) 

@utils.task_wrapper 
def preprocess(cfg: DictConfig):
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    # 1. load Raw DataFrame (csv, excel, ...)
    df_loader = hydra.utils.instantiate(cfg.dataframe) 
    raw_df = df_loader.load() 
    # time_idx를 제외한 나머지 columns를 추출하는 작업
    column_names = DataFrameUtils.get_column_names(raw_df) 
    data_df, missing_mask_df, corrupt_mask_df, intact_df = DataFrameUtils.decompose_df(raw_df, column_names) 

    # (optional)
    # 2. corrupt time-series data 
    if cfg.get("corrupt"):
        target_columns = cfg.get("target_columns") 
        corrupter = hydra.utils.instantiate(cfg.corrupt) 
        corrupted_df, missing_mask_df, corrupt_mask_df = corrupter(data_df, target_columns) 
    else:
        corrupted_df = data_df.copy() 

    df = DataFrameUtils.join_dfs(corrupted_df, missing_mask_df, corrupt_mask_df, intact_df)

    # TODO: implement this for posco data 
    # 3. remove missing values 
    #missing_df, missing_mask = remove_missing(missing_df) 
    #missing_mask = missing_mask | corrupted_mask 

    # 4. save to pickle
    df.to_pickle(cfg.get("save_path"))

    return {}, {}



@hydra.main(config_path=root / "configs", config_name="preprocess.yaml")
def main(cfg: DictConfig):
    preprocess(cfg)  

    return None 


if __name__ == "__main__":
    main()  
    