import pandas as pd
import numpy as np
import functools
from sklearn.model_selection import KFold, train_test_split as sklearn_train_test_split
from utils import *


def kfold_train_test_split(df, filters, balanced=False):
    """
    KFold cross-validation for one-class problems, including the filter conjuction.
    :param df: The dataframe to be split.
    :param filters: The filters that need to be applied.
    :param balanced: TO-BE REMOVED
    :return: A list containing lists of dataframes, one for each fold.
    """
    X_target_cl = df[conjunction(filters)]
    X_target_cl = X_target_cl.iloc[:, 2:]
    X_target_cl.insert(loc=0, column='Class', value=1)
    y_target_cl = X_target_cl['Class']

    X_other = df[~(conjunction(filters))]
    X_other = X_other.iloc[:, 2:]
    X_other.insert(loc=0, column='Class', value=0)
    y_other = X_other['Class']

    if balanced:
        _, X_test_other_cl, _, y_test_other_cl = sklearn_train_test_split(X_other, y_other, test_size=0.2)
    else:
        X_test_other_cl = X_other
        y_test_other_cl = y_other

    dfs = []
    kf = KFold(n_splits=5, shuffle=True)
    for X_train_indices, X_test_target_cl_indices in kf.split(X_target_cl):
        X_train = X_target_cl.iloc[X_train_indices, :]
        y_train = X_train['Class']
        X_test_target_cl = X_target_cl.iloc[X_test_target_cl_indices, :]
        y_test_target_cl = X_test_target_cl['Class']

        X_test = pd.concat([X_test_target_cl, X_test_other_cl])
        y_test = pd.concat([y_test_target_cl, y_test_other_cl])

        dfs.append([X_train, y_train, X_test, y_test])
    return dfs


def train_test_split(df, filters):
    """
    Normal train-test split, including filter conjuction.
    :param df: The dataframe to be split.
    :param filters: The filters that need to be applied.
    :return: Four dataframes (X_train, X_test, y_train, y_test).
    """
    X_target_cl = df[conjunction(filters)]
    X_target_cl = X_target_cl.iloc[:, 2:]
    X_target_cl.insert(loc=0, column='Class', value=1)
    y_target_cl = X_target_cl['Class']

    X_other_cl = df[~(conjunction(filters))]
    X_other_cl = X_other_cl.iloc[:, 2:]
    X_other_cl.insert(loc=0, column='Class', value=0)
    y_other_cl = X_other_cl['Class']

    X_train, X_test_target_cl, y_train, y_test_target_cl = sklearn_train_test_split(X_target_cl, y_target_cl, test_size=0.2,
                                                                            random_state=41)
    _, X_test_other_cl, _, y_test_other_cl = sklearn_train_test_split(X_other_cl, y_other_cl, test_size=0.2,
                                                              random_state=41)
    X_test = pd.concat([X_test_target_cl, X_test_other_cl])
    y_test = pd.concat([y_test_target_cl, y_test_other_cl])

    return X_train, X_test, y_train, y_test
