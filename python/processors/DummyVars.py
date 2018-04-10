import pandas as pd

from sklearn.base import TransformerMixin

from python.Data import getCategoricalColumns


class DummyVars(TransformerMixin):
    """Replaces categorical attributes with dummy variables"""

    def fit(self, X, y=None):
        self.catsToDrop = getCategoricalColumns(X)
        self.catDummies = []
        for col in self.catsToDrop:
            self.catDummies.append(pd.get_dummies(X[col], prefix=col).iloc[:, 1:])


    def transform(self, X, y=None):
        for col in range(len(self.catsToDrop)):
            X.drop(labels=self.catsToDrop[col], axis="columns", inplace=True)
            X = pd.concat([X, self.catDummies[col]], axis=1)
        return X