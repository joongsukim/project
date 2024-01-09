"""
Dataset class for self-attention models trained with MIT (masked imputation task) task.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GLP-v3

from pycorruptor import mcar


"""
Utilities for data manipulation
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GPL-v3

from torch.utils.data import Dataset
import torch


class BaseDataset(Dataset):
    """Base dataset class in PyPOTS.
    Parameters
    ----------
    X : tensor, shape of [n_samples, n_steps, n_features]
        Time-series feature vector.
    y : tensor, shape of [n_samples], optional, default=None,
        Classification labels of according time-series samples.
    """

    def __init__(self, X, y=None):
        super().__init__()
        # types and shapes had been checked after X and y input into the model
        # So they are safe to use here. No need to check again.
        self.X = X
        self.y = y
        self.n_steps = self.X.shape[0]
        self.n_features = self.X.shape[1]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        """Fetch data according to index.
        Parameters
        ----------
        idx : int,
            The index to fetch the specified sample.
        """
        X = self.X[idx]
        missing_mask = ~torch.isnan(X)
        X = torch.nan_to_num(X)

        sample = [
            torch.tensor(idx),
            X.to(torch.float32),
            missing_mask.to(torch.float32),
        ]

        if self.y is not None:
            sample.append(self.y[idx].to(torch.long))

        return sample



class DatasetForMIT(BaseDataset):
    """Dataset for models that need MIT (masked imputation task) in their training, such as SAITS.
    For more information about MIT, please refer to :cite:`du2022SAITS`.
    Parameters
    ----------
    (old)X : tensor, shape of [n_samples, n_steps, n_features]
    (new)X : tensor, shape of [n_samples, n_features]
        Time-series feature vector.
    y : tensor, shape of [n_samples], optional, default=None,
        Classification labels of according time-series samples.
    rate : float, in (0,1),
        Artificially missing rate, rate of the observed values which will be artificially masked as missing.
        Note that,
        `rate` = (number of artificially missing values) / np.sum(~np.isnan(self.data)),
        not (number of artificially missing values) / np.product(self.data.shape),
        considering that the given data may already contain missing values,
        the latter way may be confusing because if the original missing rate >= `rate`,
        the function will do nothing, i.e. it won't play the role it has to be.
    window_size:
    stride:
    """

    def __init__(self, X, y=None, rate=0.2, idx_list=None, window_size=60, stride=1):
        super().__init__(X, y)
        self.rate = rate
        self.idx_list = idx_list
        self.window_size = window_size
        self.stride = stride
        self.n_steps = window_size
        self.n_features = X.shape[1]

    def get_window(self, target, idx):
        return target[idx*self.stride: self.window_size + idx*self.stride]

    def __len__(self):
        #
        # idx가 0일 떄
        # [0, window_size-1]
        # idx=1, stride=2일때
        # [2, window_size+2-1]
        # 일반화하면
        # [idx * stride, window_size + idx * stride - 1]
        # window_size + idx * stride -1 == num_data - 1
        # idx * stride = num_data - window_size
        # idx = (num_data - window_size) / stride
        # 데이터 개수는 idx+1 = (num_data - window_size) / stride + 1
        return len(self.idx_list)

    def __getitem__(self, idx):
        """Fetch data according to index.
        Parameters
        ----------
        idx : int,
            The index to fetch the specified sample.
        Returns
        -------
        dict,
            A dict contains
            index : int tensor,
                The index of the sample.
            X_intact : tensor,
                Original time-series for calculating mask imputation loss.
            X : tensor,
                Time-series data with artificially missing values for model input.
            missing_mask : tensor,
                The mask records all missing values in X.
            indicating_mask : tensor.
                The mask indicates artificially missing values in X.
        """
        real_idx = self.idx_list[idx]
        X = self.get_window(self.X, real_idx)
        #X = self.X[idx]
        X_intact, X, missing_mask, indicating_mask = mcar(X, rate=self.rate)

        sample = [
            torch.tensor(idx),
            X_intact.to(torch.float32),
            X.to(torch.float32),
            missing_mask.to(torch.float32),
            indicating_mask.to(torch.float32),
        ]

        if self.y is not None:
            sample.append(self.get_window(self.y, idx).to(torch.long))

        return sample
