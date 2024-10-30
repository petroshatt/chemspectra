import warnings
warnings.filterwarnings('ignore')

import functools
import numpy as np
import pandas as pd
from bokeh.plotting import show

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from preprocessing import *
from decomposition import *
from plot import *
from baselines import *
from predict import *
from utils import *



if __name__ == '__main__':
    metrics = pd.DataFrame(columns=['Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'F1_score'])

    # FTIR - ADULTERATED LABELS
    # df = pd.read_csv("../data/honey_adulterated_ftir.csv")
    # df = df.drop(df.loc[:, '499.96':'748.25'].columns, axis=1)
    # df = df.drop(df.loc[:, '1802.15':'4000.12'].columns, axis=1)
    # df.set_index('Sample', inplace=True)

    # # FTIR
    # df = pd.read_csv("../data/honey_ftir.csv")
    # df = df.drop(df.loc[:, '499.96':'748.25'].columns, axis=1)
    # df = df.drop(df.loc[:, '1802.15':'4000.12'].columns, axis=1)
    # df.set_index('Sample', inplace=True)

    # UVVIS
    # df = pd.read_csv("../data/honey_uvvis.csv")
    # df = df.drop(df.loc[:, '190':'219.5'].columns, axis=1)
    # df = df.drop(df.loc[:, '500':'900'].columns, axis=1)
    # df.set_index('Sample', inplace=True)

    # Data Fusion (FTIR & UVVIS)
    ftir = pd.read_csv("../data/honey_ftir.csv")
    ftir = ftir.drop(ftir.loc[:, '499.96':'748.25'].columns, axis=1)
    ftir = ftir.drop(ftir.loc[:, '1311.36':'4000.12'].columns, axis=1)
    ftir.set_index('Sample', inplace=True)
    uvvis = pd.read_csv("../data/honey_uvvis.csv")
    uvvis = uvvis.drop(uvvis.loc[:, '190':'219.5'].columns, axis=1)
    uvvis = uvvis.drop(uvvis.loc[:, '500':'900'].columns, axis=1)
    uvvis.set_index('Sample', inplace=True)
    uvvis.drop(['Geographical', 'Botanical'], axis=1, inplace=True)
    df = ftir.join(uvvis, how='inner')

    # filters = ['Syrup']
    # filters = ['Botanical']
    filters = ['Geographical']

    y = df[filters]
    X = df.iloc[:, 2:]
    # FOR ADULTERATION LABELS
    # X = df.iloc[:, 5:]

    methods = {
        'Baseline': [None, LinearBaseline(), PolyfitBaseline(degree=2), Derivative(deriv=1), Derivative(deriv=2)],
        'Scattering': [None, SNV(), MSC()],
        'Smoothing': [None, Savgol(25, 5, 0)],
        'Scaling': [None, StandardScaler(), MinMaxScaler()],
        'DimRed': [None, PCA(n_components=5), PCA(n_components=11)],
        'Classification': [KNeighborsClassifier(n_neighbors=7),
                           # RandomForestClassifier(),
                           # MLPClassifier(),
                           # SVC(gamma='auto', kernel='linear')
                           ]
    }

    cv = KFold(n_splits=5, random_state=67, shuffle=True)
    results = predict(X, y, cv, methods)
    results.to_csv('../results/dummy1.csv')
