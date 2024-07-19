import warnings
warnings.filterwarnings('ignore')

import functools
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC

from chemspectra.preprocessing import *
from decomposition import *
from plot import *
from baselines import *


def conjunction(conditions):
    return functools.reduce(np.logical_and, conditions)


def calculate_mean_spectra(df, filt):
    mean_spectra = []
    types = df[filt].unique()

    for t in types:
        type_df = df[df[filt] == t]
        type_df = type_df.drop(filt, axis=1)
        type_mean = type_df.mean(axis=0)
        mean_spectra.append(type_mean)

    means_df = pd.concat(mean_spectra, axis=1).T
    return means_df


if __name__ == '__main__':
    metrics = pd.DataFrame(columns=['Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'F1_score'])

    df = pd.read_csv("../data/honey_ftir.csv")
    df = df.drop(df.loc[:, '499.96':'748.25'].columns, axis=1)
    df = df.drop(df.loc[:, '1311.36':'4000.12'].columns, axis=1)
    df.set_index('Sample', inplace=True)

    filters = ['Botanical']
    # filters = ['Geographical']

    y = df[filters]
    X = df.iloc[:, 2:]

    pipeline = Pipeline([
        ('baseline', SecondOrderBaseline()),
        ('savgol', Savgol(25, 5, 0)),
        ('scaler', StandardScaler())
    ])
    X = pipeline.fit_transform(X)

    # pca = PCA(n_components=5)
    # X = pca.fit_transform(X)

    clf = KNeighborsClassifier(n_neighbors=5)

    cv = KFold(n_splits=5, random_state=67, shuffle=True)
    y_pred = cross_val_predict(clf, X, y.values.ravel(), cv=cv)
    print(classification_report(y, y_pred))
    print(confusion_matrix(y, y_pred))
