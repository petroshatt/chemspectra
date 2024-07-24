import functools
import itertools

import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score, balanced_accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline


def generate_permutations(methods):
    """
    Generates permutations of all methods
    :param methods: A dictionary with the methods. Name of step as key and a list of the methods as value.
    :return: A dictionary with the permutations
    """
    keys, values = zip(*methods.items())
    return [dict(zip(keys, v)) for v in itertools.product(*values)]


def build_pipeline(perm):
    """
    Builds a pipeline of the proprocessing
    :param perm: A dictionary with the permutations of the methods (pre-processing and classification model)
    :return: The pre-processing methods into a Pipeline object
             The classifier key and value popped from the dictionary
    """
    pipeline_steps = []
    clf_name, clf = perm.popitem()
    for method_name, method in perm.items():
        pipeline_steps.append((str(method_name), method))
    pipeline = Pipeline(pipeline_steps)
    return pipeline, clf, clf_name


def compute_metrics(y_true, y_pred):
    """
    Computes the metrics of a prediction
    :param y_true: Array of true values
    :param y_pred: Array of predicted values
    :return: Four floats for each metric (Accuracy, Recall, F1 score, Balanced Accuracy)
    """
    accuracy = round(accuracy_score(y_true, y_pred), 2)
    recall = round(recall_score(y_true, y_pred, average='macro'), 2)
    f1 = round(f1_score(y_true, y_pred, average='macro'), 2)
    balanced_acc = round(balanced_accuracy_score(y_true, y_pred), 2)

    return accuracy, recall, f1, balanced_acc


def get_methods_name(perm, clf):
    """
    Creates the name for each permutation
    :param perm: Dictionary of permutations
    :param clf: Classifier object
    :return: A string with the name of the pipeline by joining the methods name,
             excluding None, and joining the classifier name at the end
    """
    return ' + '.join([str(method) for method in perm.values() if method is not None] + [str(clf)])


def conjunction(conditions):
    """
    Conjuction of filters for one-class problems with combination of filters
    :param conditions: The filters as conditions
    :return: The conjunction of the filters
    """
    return functools.reduce(np.logical_and, conditions)

