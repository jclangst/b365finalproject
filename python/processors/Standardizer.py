import pandas as pd

from sklearn.base import TransformerMixin


class Standardizer(TransformerMixin):
    """Fit all attributes into the range [0, 1]"""

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        cols = [x for x in X.columns if x != 'id']  # remove 'id' column
        cols = pd.Index(cols)  # convert from list to pandas.Index

        return (X[cols] - X[cols].min()) / (X[cols].max() - X[cols].min())
