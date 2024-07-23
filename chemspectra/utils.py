import functools
import itertools

import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score, balanced_accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline


def conjunction(conditions):
    return functools.reduce(np.logical_and, conditions)


def generate_permutations(methods):
    keys, values = zip(*methods.items())
    return [dict(zip(keys, v)) for v in itertools.product(*values)]


def build_pipeline(perm):
    pipeline_steps = []
    clf_name, clf = perm.popitem()
    for method_name, method in perm.items():
        pipeline_steps.append((str(method_name), method))
    pipeline = Pipeline(pipeline_steps)
    return pipeline, clf, clf_name


def compute_metrics(y_true, y_pred):
    accuracy = round(accuracy_score(y_true, y_pred), 2)
    recall = round(recall_score(y_true, y_pred, average='macro'), 2)
    f1 = round(f1_score(y_true, y_pred, average='macro'), 2)
    balanced_acc = round(balanced_accuracy_score(y_true, y_pred), 2)

    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = round(tn / (tn + fp), 2)
    else:
        specificity = round(np.mean(
            [cm[i, i] / (np.sum(cm[i, :]) - cm[i, i] + np.sum(cm[:, i]) - cm[i, i]) for i in range(cm.shape[0])]), 2)

    return accuracy, recall, f1, balanced_acc, specificity


def get_methods_name(perm, clf):
    return ' + '.join([str(method) for method in perm.values() if method is not None] + [str(clf)])
