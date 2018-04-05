from sklearn.base import TransformerMixin


class CalcRemover(TransformerMixin):
    """Remove calculated attributes"""

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        cols_to_keep = [col for col in X.columns if 'calc' not in col]
        return X[cols_to_keep]
