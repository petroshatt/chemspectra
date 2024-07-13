import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA as SkPCA


class RFFeatureSelection(BaseEstimator, TransformerMixin):

    def __init__(self, n_features=20):
        self.n_features = n_features

    def fit(self, X, y):
        return self

    def transform(self, X, y):
        rf = RandomForestClassifier(n_estimators=500)
        rf.fit(X, y)
        feature_scores = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
        top_features = feature_scores.head(self.n_features).index
        X = X[top_features]
        return X


class PCA(BaseEstimator, TransformerMixin):

    def __init__(self, n_components=5):
        self.n_components = n_components

    def fit(self, X):
        return self

    def transform(self, X):
        index = X.index
        pca = SkPCA(n_components=self.n_components)
        pca_result = pca.fit_transform(X)
        columns = [f"PC{i + 1}" for i in range(pca_result.shape[1])]
        X = pd.DataFrame(data=pca_result, columns=columns, index=index)
        return X

