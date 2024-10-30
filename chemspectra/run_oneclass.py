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
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM


if __name__ == '__main__':

    plots = []
    metrics = pd.DataFrame(columns=['Accuracy', 'Balanced Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'F1_score'])

    # FTIR - ADULTERATED LABELS
    # df = pd.read_csv("../data/honey_adulterated_ftir.csv")
    # df = df.drop(df.loc[:, '499.96':'748.25'].columns, axis=1)
    # df = df.drop(df.loc[:, '1802.15':'4000.12'].columns, axis=1)
    # df.set_index('Sample', inplace=True)

    # FTIR
    df = pd.read_csv("../data/honey_ftir.csv")
    df = df.drop(df.loc[:, '499.96':'748.25'].columns, axis=1)
    df = df.drop(df.loc[:, '1802.15':'4000.12'].columns, axis=1)
    df.set_index('Sample', inplace=True)

    filters = [(df['Botanical'] == 1)]
    # filters = [(df['Geographical'] == 5)]
    # filters = [(df['Botanical'] == 1), (df['Geographical'] == 2)]
    # filters = [(df['Adulterated'] == 0)]

    y = df.iloc[:, 1]
    # FOR GEOGRAPHICAL LABELS
    # X =  df.iloc[:, 2]

    X = df.iloc[:, 3:]
    # FOR ADULTERATION LABELS
    # X =  df.iloc[:, 6:]

    df = pd.concat([y, X], axis=1)

    methods = {
        'Baseline': [None, LinearBaseline(), PolyfitBaseline(degree=2), Derivative(deriv=1), Derivative(deriv=2)],
        'Scattering': [None, SNV(), MSC()],
        'Smoothing': [None, Savgol(25, 5, 0)],
        'Scaling': [None, StandardScaler(), MinMaxScaler()],
        # 'DimReduction': [PCA(n_components=5)],
        'Classification': [DDSimca(n_comps=5, alpha=0.10, gamma=0.1),
                           IsolationForest(n_estimators=500),
                           OneClassSVM(nu=0.4, gamma='auto')
                           ]
    }

    results = predict_oneclass(X, y, filters, methods)
    results.to_csv('../results/thyme_or_not.csv')
