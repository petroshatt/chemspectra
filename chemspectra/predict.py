import itertools
from tqdm import tqdm
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, recall_score, f1_score, balanced_accuracy_score, confusion_matrix


from chemspectra.preprocessing import *
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
    results = pd.DataFrame(columns=['Method', 'Balanced Accuracy', 'Accuracy', 'Recall', 'F1 Score'])

    for perm in tqdm(permutations_dicts, desc='Processing permutations'):
        pipeline, clf, clf_name = build_pipeline(perm)
        X_transformed = pipeline.fit_transform(X)
        y_pred = cross_val_predict(clf, X_transformed, y.values.ravel(), cv=cv)

        accuracy, recall, f1, balanced_acc = compute_metrics(y, y_pred)
        methods_name = get_methods_name(perm, clf)

        results = pd.concat([results, pd.DataFrame({
            'Method': [methods_name],
            'Balanced Accuracy': [balanced_acc],
            'Accuracy': [accuracy],
            'Recall': [recall],
            'F1 Score': [f1]
        })], ignore_index=True)

    results = results.sort_values(by='Balanced Accuracy', ascending=False)
    results.set_index('Method', inplace=True)

    return results


