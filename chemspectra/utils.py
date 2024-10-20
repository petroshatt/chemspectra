import functools
import itertools

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, f1_score, balanced_accuracy_score, confusion_matrix, \
    precision_score
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
    precision = round(precision_score(y_true, y_pred, average='macro'), 2)
    f1 = round(f1_score(y_true, y_pred, average='macro'), 2)
    balanced_acc = round(balanced_accuracy_score(y_true, y_pred), 2)

    return accuracy, precision, recall, f1, balanced_acc


def get_methods_name(perm):
    """
    Creates the name for each permutation
    :param perm: Dictionary of permutations
    :return: A string with the name of the pipeline by joining the methods name,
             excluding None
    """
    return ' + '.join([str(method) for method in perm.values() if method is not None])


def conjunction(conditions):
    """
    Conjuction of filters for one-class problems with combination of filters
    :param conditions: The filters as conditions
    :return: The conjunction of the filters
    """
    return functools.reduce(np.logical_and, conditions)


import numpy as np
import pandas as pd


def signal_to_noise_ratio(df, ddof=0):
    """
    Calculates the signal-to-noise ratio for a spectrum.

    :param df: A 1D DataFrame containing the spectrum data.
    :param ddof: Degrees of freedom correction for standard deviation. Default is 0.
    :return: The signal-to-noise ratio or 0 where the standard deviation is 0.
    """
    # Ensure input is a 1D DataFrame
    if not isinstance(df, pd.DataFrame) or df.shape[0] != 1:
        raise ValueError("Input must be a 1D DataFrame.")

    a = df.squeeze().to_numpy()
    mean_signal = np.mean(a)
    std_signal = np.std(a, ddof=ddof)

    # Compute signal-to-noise ratio (SNR)
    return 0 if std_signal == 0 else mean_signal / std_signal

def add_artificial_noise(df, noise_level=0.01):
    """
    Adds artificial Gaussian noise to a 1D spectrum.

    :param df: A 1D DataFrame containing the spectrum data (shape: 1, x).
    :param noise_level: The standard deviation of the noise as a fraction of the
                        signal's standard deviation. Default is 1% of the signal's std.
    :return: A new DataFrame with noise added to the original spectrum.
    """
    # Ensure input is a 1D DataFrame (either a single row or single column)
    if not isinstance(df, pd.DataFrame) or (df.shape[0] != 1 and df.shape[1] != 1):
        raise ValueError("Input must be a 1D DataFrame with shape (1, x) or (x, 1).")

    # Convert the DataFrame to a NumPy array and flatten it
    spectrum = df.values.flatten()  # This handles both row and column formats

    # Calculate the standard deviation of the original spectrum
    spectrum_std = np.std(spectrum)

    # Generate Gaussian noise
    noise = np.random.normal(0, noise_level * spectrum_std, size=spectrum.shape)

    # Add the noise to the original spectrum
    noisy_spectrum = spectrum + noise

    # Reshape the noisy spectrum back to the original shape and return as DataFrame
    return pd.DataFrame([noisy_spectrum], index=df.index)
