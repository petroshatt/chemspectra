from oneclass import *
from preprocessing import *
from plot import *
from tt_split import *
from decomposition import *

import pandas as pd
from bokeh.layouts import column, gridplot
from bokeh.plotting import show
from sklearn.pipeline import Pipeline
import time
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


if __name__ == '__main__':

    start_time = time.time()

    plots = []
    metrics = pd.DataFrame(columns=['Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'F1_score'])

    df = pd.read_csv("../data/honey_ftir.csv")
    df = df.drop(df.loc[:, '499.96':'748.25'].columns, axis=1)
    df = df.drop(df.loc[:, '1802.15':'4000.12'].columns, axis=1)
    df.set_index('Sample', inplace=True)

    filters = [(df['Botanical'] == 3)]
    # filters = [(df['Geographical'] == 1)]
    # filters = [(df['Botanical'] == 3), (df['Geographical'] == 2)]

    y = df.iloc[:, :2]
    X = df.iloc[:, 2:]

    p1 = plot_spectra(X)

    # p2 = plot(X)

    pipeline = Pipeline([
        ('scaler', Scaler()),
        ('center', Center()),
        ('snv', Snv()),
        ('savgol', Savgol(window_length=25, poly_order=5, deriv=0))
    ])
    X = pipeline.fit_transform(X)

    p2 = plot_spectra(X)

    # rffs = RFFeatureSelection(n_features=30)
    # X = rffs.fit_transform(X, df['Botanical'])
    # pca = PCA(n_components=5)
    # X = pca.fit_transform(X)

    df = pd.concat([y, X], axis=1)

    ddsimca = DDSimca(ncomps=5, alpha=0.05, gamma=0.05)
    dfs = kfold_train_test_split(df, filters, balanced=False)
    for fold in dfs:
        X_train, y_train, X_test, y_test = fold
        ddsimca.fit(X_train)
        ddsimca.predict(X_test)
        metrics.loc[len(metrics)] = ddsimca.confusion_matrix(print_metrics='off')

    p3 = ddsimca.acceptance_plot()
    p4 = ddsimca.pred_acceptance_plot()

    print(metrics.mean())
    grid = gridplot([[p1, p2], [p3, p4]], width=900, height=450)
    show(grid)

    # isfr = IsolationForest(n_estimators=500)
    # dfs = kfold_train_test_split(df, filters)
    # for fold in dfs:
    #     X_train, y_train, X_test, y_test = fold
    #     X_train = X_train.drop(columns=['Class'])
    #     X_test = X_test.drop(columns=['Class'])
    #     y_test[~(y_test == 1)] = -1
    #
    #     isfr.fit(X_train.values)
    #     y_pred_test = isfr.predict(X_test.values)
    #     metrics.loc[len(metrics)] = [accuracy_score(y_test, y_pred_test),
    #                                            precision_score(y_test, y_pred_test),
    #                                            recall_score(y_test, y_pred_test),
    #                                            recall_score(y_test, y_pred_test, pos_label=-1),
    #                                            f1_score(y_test, y_pred_test)
    #                                            ]
    # print(metrics.mean())

    print("\n\n--- %s seconds ---" % (time.time() - start_time))
