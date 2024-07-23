import numpy as np
import pandas as pd
import pybaselines
import scipy.signal as signal
from sklearn.base import BaseEstimator, TransformerMixin


class ALS(BaseEstimator, TransformerMixin):

    def __init__(self, lam=100):
        self.lam = lam

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        output = X.apply(self.als, axis=1)
        return output

    def als(self, row):
        array_1D = row.values
        x_baseline, _ = pybaselines.whittaker.asls(array_1D, lam=self.lam)
        output = array_1D - x_baseline
        return pd.Series(output, index=row.index)


class Derivative(BaseEstimator, TransformerMixin):

    def __init__(self, deriv=1, d=2):
        self.deriv = deriv
        self.d = d

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        CAN BE DONE WITHOUT LOSING SHAPE
        """
        output = X.copy()
        for _ in range(self.deriv):
            output = output.diff(self.d, axis=1)
            output = output.iloc[:, 2:]
        return output


class LinearBaseline(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        index = X.index
        columns = X.columns

        X_np = X.to_numpy()
        output_np = signal.detrend(X_np, type='linear')

        output = pd.DataFrame(output_np, index=index, columns=columns)
        return output


class SecondOrderBaseline(BaseEstimator, TransformerMixin):

    def __init__(self, degree=2):
        self.degree = degree

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        t = np.arange(0, X.shape[1])
        output = X.apply(lambda x: x - np.polyval(np.polyfit(t, x, self.degree), t), axis=1)
        return output
