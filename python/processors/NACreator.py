import numpy as np
from sklearn.base import TransformerMixin


class NAAdder(TransformerMixin):
    """Replaces -1 with NA"""

    def fit(self, X, y=None):
        return self

    @staticmethod
    def transform(X, y=None):
        return X.replace(to_replace=-1, value=np.nan)
