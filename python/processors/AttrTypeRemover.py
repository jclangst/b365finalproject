from numpy.core.multiarray import dtype
from sklearn.base import TransformerMixin


class AttrTypeRemover(TransformerMixin):
    """Remove certain types of attributes"""

    def __init__(self, ints=False, floats=False, cats=False, bins=False, calcs=False):
        """
        Removing ints will not remove categorical/binary variables.
        However, removing ints/floats/cats/bins will remove calculated attributes of that type.
        """
        self.remove_ints = ints
        self.remove_floats = floats
        self.remove_cats = cats
        self.remove_bins = bins
        self.remove_calcs = calcs

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        cols_to_keep = X.columns

        # remove ints: keep cats, bins, and floats
        if self.remove_ints:
            cols_to_keep = [col for col in X.columns if ('cat' in col or 'bin' in col
                                                         or X.dtypes[col] == dtype('float64'))]
        if self.remove_floats:
            cols_to_keep = [col for col in X.columns if X.dtypes[col] != dtype('float64')]
        if self.remove_cats:
            cols_to_keep = [col for col in X.columns if 'cat' not in col]
        if self.remove_bins:
            cols_to_keep = [col for col in X.columns if 'bin' not in col]
        if self.remove_calcs:
            cols_to_keep = [col for col in X.columns if 'calc' not in col]

        return X[cols_to_keep]
