import numpy as np
import pandas as pd

from sklearn.base import TransformerMixin
from sklearn.preprocessing import Imputer

from python.Data import getCategoricalColumns, getIntervalColumns


class NAImputer_Categorical(TransformerMixin):
    """Give NAs in categorical attributes their own category"""

    def fit(self, X, y=None):
        categoricalColumns = getCategoricalColumns(X)
        self.newNACodes = np.max(X[categoricalColumns]) + 1
        return self

    def transform(self, X, y=None):
        for index in self.newNACodes.index:
            X[index] = X[index].replace(np.nan, self.newNACodes[index]).astype(np.int64)
        return X


class NAImputer_Interval(TransformerMixin):
    """Give NAs in interval attributes their own category"""

    def __init__(self, dropThreshold, strategy="median"):
        self.dropThreshold = dropThreshold
        self.strategy = strategy

    def fit(self, X, y=None):
        # select the interval columns
        intervalColumns = getIntervalColumns(X)

        # drop columns with too much missing data
        colsToDrop = []
        for col in intervalColumns:
            if X[col].isnull().sum() / X.shape[0] >= self.dropThreshold:
                colsToDrop.append(col)

        # calculate the impending transformations
        self.colsToIgnore = X.columns.difference(intervalColumns).difference(colsToDrop)
        self.colsToTransform = intervalColumns.difference(colsToDrop)
        self.imputer = Imputer(strategy=self.strategy).fit(X[self.colsToTransform])

        return self

    def transform(self, X, y=None):
        return pd.concat((X[self.colsToIgnore],
                          pd.DataFrame(self.imputer.transform(X[self.colsToTransform]), columns=self.colsToTransform)),
                         axis=1)
