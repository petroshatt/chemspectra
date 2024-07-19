import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler as SkMinMaxScaler
from sklearn.preprocessing import StandardScaler as SkStandardScaler


class MinMaxScaler(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        mmscaler = SkMinMaxScaler().set_output(transform="pandas")
        output = mmscaler.fit_transform(X)
        return output


class StandardScaler(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        scaler = SkStandardScaler()
        X.values[:] = scaler.fit_transform(X)
        return X


class AreaScaler(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        areas = X.sum(axis=1)
        normalized_df = X.div(areas, axis=0)
        return normalized_df


class MeanCentering(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        output = X.apply(lambda x: x - x.mean())
        return output


class SNV(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        output = X.copy()
        for i in range(X.shape[0]):
            output.iloc[i, :] = (X.iloc[i, :] - np.mean(X.iloc[i, :])) / np.std(X.iloc[i, :])
        return output


class MSC(BaseEstimator, TransformerMixin):

    def __init__(self, reference=None):
        self.reference = reference

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.reference is None:
            self.reference = np.mean(X, axis=0)

        for i in range(X.shape[0]):
            X.iloc[i, :] -= X.iloc[i, :].mean()

        output = X.copy()
        for i in range(X.shape[0]):
            fit = np.polyfit(self.reference, X.iloc[i, :], 1, full=True)
            output.iloc[i, :] = (X.iloc[i, :] - fit[0][1]) / fit[0][0]

        return output


class Savgol(BaseEstimator, TransformerMixin):

    def __init__(self, window_length, poly_order, deriv=0):
        self.window_length = window_length
        self.poly_order = poly_order
        self.deriv = deriv

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        output = X.apply(self.savgol, axis=1)
        return output

    def savgol(self, row):
        filtered_row = savgol_filter(row, window_length=self.window_length, polyorder=self.poly_order, deriv=self.deriv)
        return pd.Series(filtered_row, index=row.index)
