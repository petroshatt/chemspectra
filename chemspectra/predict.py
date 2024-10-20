import itertools
from tqdm import tqdm
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, recall_score, f1_score, balanced_accuracy_score, confusion_matrix, \
    precision_score

from chemspectra.preprocessing import *
from chemspectra.tt_split import kfold_train_test_split
from decomposition import *
from plot import *
from baselines import *
from utils import *


def predict(X, y, cv, methods):
    """
    Generates all possible permutations on the given methods and runs each pipeline.
    :param X: A dataframe with the spectra samples
    :param y: A dataframe with the classes of the spectra samples
    :param cv: Cross-validation method
    :param methods: A dictionary with the methods. Name of step as key and a list of the methods as value.
    :return: A dataframe with the metrics of each pipeline run.
             Sorted by Balanced Accuracy and Method column as index
    """
    permutations_dicts = generate_permutations(methods)
    results = pd.DataFrame(columns=['Pre-Processing', 'Classifier', 'Balanced Accuracy', 'Accuracy', 'Precision',
                                    'Sensitivity', 'F1 Score'])

    for perm in tqdm(permutations_dicts, desc='Processing permutations'):
        pipeline, clf, clf_name = build_pipeline(perm)
        X_transformed = pipeline.fit_transform(X)
        y_pred = cross_val_predict(clf, X_transformed, y.values.ravel(), cv=cv)

        accuracy, precision, recall, f1, balanced_acc = compute_metrics(y, y_pred)
        perm = get_methods_name(perm)

        results = pd.concat([results, pd.DataFrame({
            'Pre-Processing': [perm],
            'Classifier': [clf],
            'Balanced Accuracy': [balanced_acc],
            'Accuracy': [accuracy],
            'Precision': [precision],
            'Sensitivity': [recall],
            'F1 Score': [f1]
        })], ignore_index=True)

    results = results.sort_values(by='Balanced Accuracy', ascending=False)
    # results.set_index('Method', inplace=True)

    return results


def predict_oneclass(X, y, filters, methods):
    """
    Generates all possible permutations on the given methods and runs each pipeline.
    :param X: A dataframe with the spectra samples
    :param y: A dataframe with the classes of the spectra samples
    :param filters: Filters to apply during k-fold cross-validation
    :param methods: A dictionary with the methods. Name of step as key and a list of the methods as value.
    :return: A dataframe with the metrics of each pipeline run.
             Sorted by Balanced Accuracy and Method column as index
    """
    permutations_dicts = generate_permutations(methods)
    results = pd.DataFrame(columns=['Pre-Processing', 'Classifier', 'Balanced Accuracy', 'Accuracy', 'Sensitivity',
                                    'Specificity', 'F1 Score'])

    for perm in tqdm(permutations_dicts, desc='Processing permutations'):
        pipeline, clf, clf_name = build_pipeline(perm)
        X_transformed = pipeline.fit_transform(X)

        df = pd.concat([y, X_transformed], axis=1)

        metrics_list = []

        dfs = kfold_train_test_split(df, filters)

        for fold in dfs:
            X_train, y_train, X_test, y_test = fold

            X_train = X_train.drop(columns=['Class'])
            X_test = X_test.drop(columns=['Class'])
            y_test = y_test.apply(lambda x: 1 if x == 1 else -1)

            clf.fit(X_train.values)
            y_pred_test = clf.predict(X_test.values)

            fold_metrics = {
                'Balanced Accuracy': balanced_accuracy_score(y_test, y_pred_test),
                'Accuracy': accuracy_score(y_test, y_pred_test),
                'Precision': precision_score(y_test, y_pred_test, pos_label=1),
                'Sensitivity': recall_score(y_test, y_pred_test),
                'Specificity': recall_score(y_test, y_pred_test, pos_label=-1),
                'F1 Score': f1_score(y_test, y_pred_test, pos_label=1)
            }

            metrics_list.append(fold_metrics)

        metrics_df = pd.DataFrame(metrics_list)
        mean_metrics = metrics_df.mean()

        perm = get_methods_name(perm)

        results = pd.concat([results, pd.DataFrame({
            'Pre-Processing': [perm],
            'Classifier': [clf],
            'Balanced Accuracy': [mean_metrics['Balanced Accuracy']],
            'Accuracy': [mean_metrics['Accuracy']],
            'Precision': [mean_metrics['Precision']],
            'Sensitivity': [mean_metrics['Sensitivity']],
            'Specificity': [mean_metrics['Specificity']],
            'F1 Score': [mean_metrics['F1 Score']]
        })], ignore_index=True)

    # results = results.sort_values(by='Balanced Accuracy', ascending=False)
    # results = results.round(3)
    # results.set_index('Method', inplace=True)

    return results
