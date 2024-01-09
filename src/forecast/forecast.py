from abc import ABCMeta, abstractmethod
from src.data.dataframe import DataFrameUtils
from src.utils import plot_dataframes
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
import numpy as np


class BaseForecastingModule(metaclass=ABCMeta):
    def __init__(self):
        pass

    def train(self):
        pass

    def forecast(self):
        pass
