from typing import Optional, List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pytorch_lightning.utilities import rank_zero_only
import seaborn as sns 


@rank_zero_only
def plot_dataframes(column: str, title: str, x_column: Optional[str]=None, show=False, *df_tuples, **plot_params):
    figsize = plot_params.get('figsize', None)
    linewidth = plot_params.get('linewidth', 3)
    shade = plot_params.get('shade', None)

    fig = plt.figure(figsize=figsize)

    for label, df in df_tuples:
        if x_column is not None:
            x = df[x_column].to_numpy()
        else:
            x = df.index.to_numpy()

        y = df[column].to_numpy()

        plt.plot(x, y, linewidth=linewidth, label=label)
    
    if shade is not None:
        for a, b in shade:
            plt.axvspan(a, b, color='y', alpha=0.5)# , lw=0)

    plt.title(title)
    #plt.xlabel('time_idx')
    plt.legend()

    if show:
        plt.show()

    return fig


@rank_zero_only
def plot_missing(df: pd.DataFrame, exception_columns: List[str], show=False, **plot_params):
    figsize = plot_params.get('figsize', None)

    fig = plt.figure(figsize=figsize)

    # df를 numpy array로 변환
    mat = df.to_numpy()

    # mat에서 NaN인 값은 0, NaN이 아닌 값은 1로 변환
    mat[~np.isnan(mat)] = 1
    mat[np.isnan(mat)] = 0

    #
    ylabels = list(df.columns)
    ax = sns.heatmap(mat.T, cmap='autumn', yticklabels=ylabels, cbar=False)

    if show:
        plt.show()

    return fig
