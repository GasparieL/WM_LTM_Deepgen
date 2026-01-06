# Import required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os.path as op
import glob
from datetime import timedelta
from sklearn.preprocessing import StandardScaler


def remove_unit_variance(df, col, unit, group=None, suffix="_within"):
    """Remove variance between sampling units.

    This is useful for plotting repeated-measures data using within-unit
    error bars.

    Parameters
    ----------
    df : DataFrame
        Input data. Will have a new column added.
    col : column name
        Column in dataframe with quantitative measure to modify.
    unit : column name
        Column in dataframe defining sampling units (e.g., subjects).
    group : column name(s), optional
        Columns defining groups to remove unit variance within.
    suffix : string, optional
        Suffix appended to ``col`` name to create new column.

    Returns
    -------
    df : DataFrame
        Returns modified dataframe.
    """
    new_col = col + suffix
    df_copy = df.copy()

    def demean(x):
        return x - x.mean()

    if group is None:
        new = df_copy.groupby(unit)[col].transform(demean)
        new += df_copy[col].mean()
        df_copy[new_col] = new
    else:
        df_copy[new_col] = np.nan
        for level, df_level in df_copy.groupby(group):
            new = df_level.groupby(unit)[col].transform(demean)
            new += df_level[col].mean()
            df_copy.loc[new.index, new_col] = new

    return df_copy
