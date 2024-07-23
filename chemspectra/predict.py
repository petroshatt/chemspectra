import itertools

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
    permutations_dicts = generate_permutations(methods)
    results = pd.DataFrame(columns=['Method', 'Balanced Accuracy', 'Accuracy', 'Recall', 'Specificity', 'F1 Score'])

    for perm in permutations_dicts:
        pipeline, clf, clf_name = build_pipeline(perm)
        X_transformed = pipeline.fit_transform(X)
        y_pred = cross_val_predict(clf, X_transformed, y.values.ravel(), cv=cv)

        accuracy, recall, f1, balanced_acc, specificity = compute_metrics(y, y_pred)
        methods_name = get_methods_name(perm, clf)

        results = pd.concat([results, pd.DataFrame({
            'Method': [methods_name],
            'Balanced Accuracy': [balanced_acc],
            'Accuracy': [accuracy],
            'Recall': [recall],
            'Specificity': [specificity],
            'F1 Score': [f1]
        })], ignore_index=True)

    results = results.sort_values(by='Balanced Accuracy', ascending=False)
    results.set_index('Method', inplace=True)

    return results


