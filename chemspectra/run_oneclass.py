import warnings
warnings.filterwarnings('ignore')

from predict import *
from oneclass import *
from preprocessing import *
from plot import *
from tt_split import *
from decomposition import *
from baselines import *

import pandas as pd
from bokeh.layouts import column, gridplot
from bokeh.plotting import show
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict
import time
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


if __name__ == '__main__':

    plots = []
    metrics = pd.DataFrame(columns=['Accuracy', 'Balanced Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'F1_score'])

    df = pd.read_csv("../data/honey_ftir.csv")
    df = df.drop(df.loc[:, '499.96':'748.25'].columns, axis=1)
    df = df.drop(df.loc[:, '1802.15':'4000.12'].columns, axis=1)
    df.set_index('Sample', inplace=True)

    # filters = [(df['Botanical'] == 3)]
    filters = [(df['Geographical'] == 5)]
    # filters = [(df['Botanical'] == 3), (df['Geographical'] == 2)]

    y = df.iloc[:, :2]
    X = df.iloc[:, 2:]

    methods = {
        'Baseline': [None, LinearBaseline(), SecondOrderBaseline(), Derivative(deriv=1), Derivative(deriv=2)],
        'Scattering': [None, SNV(), MSC()],
        'Smoothing': [None, Savgol(25, 5, 0)],
        'Scaling': [None, StandardScaler(), MinMaxScaler()],
        'Classification': [DDSimca(n_comps=5, alpha=0.3, gamma=0.3), DDSimca(n_comps=3, alpha=0.3, gamma=0.3),
                           IsolationForest(n_estimators=500), OneClassSVM(nu=0.1, gamma='auto'),
                           OneClassSVM(nu=0.8, gamma='auto'), OneClassSVM(nu=0.8, gamma='auto')
                           ]
    }

    df = pd.concat([y, X], axis=1)

    results = predict_oneclass(X, y, filters, methods)
    results.to_csv('../data/results/oneclass/turkish_vs_all.csv')
