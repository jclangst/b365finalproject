import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from python.Data import Data, getCatoricalColumns, getIntervalColumns

#Replace the -1s with NAs
class NAAdder(TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.replace(to_replace=-1, value=np.nan)



#Give NAs in the categorical variable section their own category
class ImputerNAs_Categorical(TransformerMixin):

    def fit(self, X, y=None):
        categoricalColumns = getCatoricalColumns(X)
        self.newNACodes = np.max(X[categoricalColumns]) + 1
        return self

    def transform(self, X):
        for index in self.newNACodes.index:
            X[index] = X[index].replace(np.nan, self.newNACodes[index]).astype(np.int64)
        return X


class ImputerNAs_Interval(TransformerMixin):

    def __init__(self, dropThreshold, strategy="median"):
        self.dropThreshold = dropThreshold
        self.strategy = strategy

    def fit(self, X):

        #select the interval columns
        intervalColumns = getIntervalColumns(X)

        #drop columns with too much missing data
        colsToDrop = []
        for col in intervalColumns:
            if X[col].isnull().sum() / X.shape[0] >= self.dropThreshold:
                colsToDrop.append(col)

        #calculate the impending transformations
        self.colsToIgnore = X.columns.difference(intervalColumns).difference(colsToDrop)
        self.colsToTransform = intervalColumns.difference(colsToDrop)
        self.imputer = Imputer(strategy=self.strategy).fit(X[self.colsToTransform])

        return self

    def transform(self, X):
        return  pd.concat((X[self.colsToIgnore],
                pd.DataFrame(self.imputer.transform(X[self.colsToTransform]), columns=self.colsToTransform)), axis=1)


NAPipeline = Pipeline([
    ('Transform the -1 data to NA', NAAdder()),
    ('Add an additional category for NAs in the categorical data', ImputerNAs_Categorical()),
    ('Fix up the interval variables', ImputerNAs_Interval(0.05))
])


if __name__ == "__main__":
    data = Data()
    transformed = NAPipeline.fit_transform(data.trainX)
    print(transformed.head())
    print(transformed.info())
