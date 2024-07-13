import functools
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC

from chemspectra.preprocessing import *
from decomposition import *


def conjunction(conditions):
    return functools.reduce(np.logical_and, conditions)


if __name__ == '__main__':
    metrics = pd.DataFrame(columns=['Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'F1_score'])

    # df = pd.read_csv("../data/honey_ftir.csv")
    # df = df.drop(df.loc[:, '499.96':'748.25'].columns, axis=1)
    # df = df.drop(df.loc[:, '1802.15':'4000.12'].columns, axis=1)
    df = pd.read_csv("../data/honey_uvvis.csv")
    df = df.drop(df.loc[:, '190':'219.5'].columns, axis=1)
    df = df.drop(df.loc[:, '551':'900'].columns, axis=1)
    df.set_index('Sample', inplace=True)

    # filters = ['Botanical']
    filters = ['Geographical']

    y = df[filters]
    X = df.iloc[:, 2:]

    pipeline = Pipeline([
        ('snv', Snv()),
        ('savgol', Savgol(window_length=25, poly_order=5, deriv=0))
    ])
    X = pipeline.fit_transform(X)

    pca = PCA(n_components=7)
    X = pca.fit_transform(X)

    clf = SVC(gamma='auto', kernel='linear')
    y_pred = cross_val_predict(clf, X, y.values.ravel(), cv=5)
    print(classification_report(y, y_pred))
    print(confusion_matrix(y, y_pred))
