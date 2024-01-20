import numpy as np
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")


class Scaler:

    def __init__(self):
        pass

    def fit_transform(self, X):
        mmscaler = MinMaxScaler().set_output(transform="pandas")
        output = mmscaler.fit_transform(X)
        return output


class Center:

    def __init__(self):
        pass

    def fit_transform(self, X):
        output = X.apply(lambda x: x - x.mean())
        return output


class Snv:

    def __init__(self):
        pass

    def fit_transform(self, X):
        output = X.copy()
        for i in range(X.shape[0]):
            output.iloc[i, :] = (X.iloc[i, :] - np.mean(X.iloc[i, :])) / np.std(X.iloc[i, :])
        return output


class Msc:

    def __init__(self, reference=None):
        self.reference = reference

    def fit_transform(self, X):
        if self.reference is None:
            self.reference = np.mean(X, axis=0)

        for i in range(X.shape[0]):
            X.iloc[i, :] -= X.iloc[i, :].mean()

        output = X.copy()
        for i in range(X.shape[0]):
            fit = np.polyfit(self.reference, X.iloc[i, :], 1, full=True)
            output.iloc[i, :] = (X.iloc[i, :] - fit[0][1]) / fit[0][0]

        return output


class Savgol:

    def __init__(self, window_length, poly_order, deriv):
        self.window_length = window_length
        self.poly_order = poly_order
        self.deriv = deriv

    def savgol(self, col):
        return savgol_filter(col, window_length=self.window_length, polyorder=self.poly_order, deriv=self.deriv)

    def fit_transform(self, X):
        output = X.apply(self.savgol)
        return output
