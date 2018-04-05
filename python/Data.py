import pandas as pd
import os.path as path
from sklearn.model_selection import StratifiedShuffleSplit

DATA_DIR = "../data/"


class Data:
    """
    Data loader that creates a train and test set (if they don't already exist)

    Data.train: all of the training data
    Data.trainX: predictor variables only
    Data.trainY: class variable only
    Same for test/testX/testY
    """

    def __init__(self, dataDir=DATA_DIR, force_regenerate=False):
        """
        :param dataDir: directory containing csv/feather files
        :param force_regenerate: regenerate test/train partitions even if they already exist
        """
        trainPath = path.join(dataDir, 'train.feather')
        testPath = path.join(dataDir, 'test.feather')
        origPath = path.join(dataDir, 'orig.csv')

        # if we already have the train and test data in feather format, load it
        if path.isfile(testPath) and path.isfile(trainPath) and not force_regenerate:
            self.train = pd.read_feather(trainPath, 4)
            self.test = pd.read_feather(testPath, 4)
        # otherwise reconstruct them from the original csv and save them in feather format
        elif path.isfile(origPath):
            self.train, self.test = self.split_train_test(pd.read_csv(origPath), 0.2)
            self.train.to_feather(trainPath)
            self.test.to_feather(testPath)
        else:
            raise Exception("No data detected in directory!")

        # separate predictors and targets from each set
        self.trainX = self.train.drop(columns="target")
        self.testX = self.test.drop(columns='target')
        self.trainY = self.train['target'].copy()
        self.testY = self.test['target'].copy()

    @staticmethod
    def split_train_test(data, test_ratio):
        """split the data into a training and test set with the same ratio of claimants vs non-claimants"""
        split = StratifiedShuffleSplit(test_size=test_ratio, n_splits=1)
        for train_index, test_index in split.split(data, data["target"]):
            return data.iloc[train_index].reset_index(drop=True), data.iloc[test_index].reset_index(drop=True)


def getCategoricalColumns(data):
    """Returns list of column names that have categorical attributes"""
    cols = []
    for col in data.columns:
        if 'bin' in col or 'cat' in col:
            cols.append(col)
    return cols


def getIntervalColumns(data):
    """Returns list of column names that have interval attributes"""
    return data.columns.difference(getCategoricalColumns(data))


def getVariances(data):
    """Returns dict of column name --> variance"""
    variances = {}
    for col in data.columns:
        variances[col] = data[col].var()
    return variances


if __name__ == "__main__":
    data = Data()
