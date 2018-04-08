from sklearn.base import TransformerMixin
from sklearn.ensemble import RandomForestRegressor


class RandomForestTest(TransformerMixin):
    def fit(self, X, y=None):
        # number of CPU cores not auto-detected on my computer, set jobs=4
        self.rf = RandomForestRegressor(n_jobs=4)
        print("RF fit starting")
        self.rf.fit(X, y)
        print("RF fit done")
        return self

    def transform(self, X, y=None):
        print("RF prediction starting")
        self.predicted = self.rf.predict(X)
        print("RF prediction done")
        return X
