import warnings
warnings.filterwarnings('ignore')

import functools
import numpy as np
import pandas as pd
from astartes import train_test_split as astartes_tt_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
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

    # FTIR
    # df = pd.read_csv("../data/honey_ftir.csv")
    # df = df.drop(df.loc[:, '499.96':'748.25'].columns, axis=1)
    # df = df.drop(df.loc[:, '1311.36':'4000.12'].columns, axis=1)
    # df.set_index('Sample', inplace=True)

    # UVVIS
    # df = pd.read_csv("../data/honey_uvvis.csv")
    # df = df.drop(df.loc[:, '190':'219.5'].columns, axis=1)
    # df = df.drop(df.loc[:, '350':'900'].columns, axis=1)
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

    filters = ['Botanical']
    # filters = ['Geographical']

    y = df[filters]
    X = df.iloc[:, 2:]
    X = X.iloc[:, ::2]

    methods = {
        'Baseline': [None, LinearBaseline(), SecondOrderBaseline(), Derivative(deriv=1), Derivative(deriv=2)],
        'Scattering': [None, SNV(), MSC()],
        'Smoothing': [None, Savgol(25, 5, 0)],
        'Scaling': [None, StandardScaler(), MinMaxScaler()],
        'Classification': [KNeighborsClassifier(n_neighbors=7), RandomForestClassifier(),
                           MLPClassifier(), SVC(gamma='auto'), DecisionTreeClassifier()]
    }

    cv = KFold(n_splits=5, random_state=67, shuffle=True)
    results = predict(X, y, cv, methods)
    results.to_csv('../data/results/fusion_bot_results.csv')
