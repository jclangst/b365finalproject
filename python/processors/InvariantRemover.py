from sklearn.base import TransformerMixin

from python.Data import getVariances


class InvariantRemover(TransformerMixin):
    """Removes attributes with low variance"""

    # stores variance for each attribute
    # in transform step, iterate through this to find attributes with sufficient variance

    def __init__(self, min_variance=0, attrs_to_drop=0):
        """
        Choose one argument:

        :param min_variance: attributes with less variance than this will be removed
        :param attrs_to_drop: the number of attributes to remove from data; least varying attributes deleted first
        """
        self.threshold = min_variance
        self.attrs_to_drop = attrs_to_drop

    def fit(self, X, y=None):
        self.variances = getVariances(X)
        return self

    def transform(self, X, y=None):
        # no conditions specified
        if self.threshold == 0 and self.attrs_to_drop == 0: return X

        if self.threshold != 0:
            # for every key with value > variance threshold, append it to cols_to_keep
            cols_to_keep = []
            for col, var in self.variances.items():
                if var > self.threshold:
                    cols_to_keep.append(col)
            return X[cols_to_keep]

        elif self.attrs_to_drop != 0:
            # add the key with minimum variance to cols_to_drop; do this attrs_to_drop number of times
            cols_to_drop = []
            for i in range(self.attrs_to_drop):
                col_with_min_variance = min(self.variances, key=self.variances.get)
                cols_to_drop.append(col_with_min_variance)
                del self.variances[col_with_min_variance]

            # and make cols_to_keep every column name that's not also in cols_to_drop
            cols_to_keep = [col for col in X.columns if col not in cols_to_drop]
            return X[cols_to_keep]
