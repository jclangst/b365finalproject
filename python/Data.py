import pandas as pd
import os.path as path
from sklearn.model_selection import StratifiedShuffleSplit


DATA_DIR = "../data/"

#Simple loader for the data; creates a test set and train set if they do not already exist

# Data.train - all of the training data
# Data.trainX - predictor variables only
# Data.trainY - class variables

# Data.test - all of the test data
# Data.testX - predictor variables only
# Data.trainY - class variables

class Data():

    def __init__(self, dataDir):

        trainPath = path.join(dataDir, 'train.feather')
        testPath = path.join(dataDir, 'test.feather')
        origPath = path.join(dataDir, 'orig.csv')

        #if we already have the train and test data in feather format, save load it
        if path.isfile(testPath) and path.isfile(trainPath):
            self.train = pd.read_feather(trainPath, 4)
            self.test = pd.read_feather(testPath, 4)

        #otherwise reconstruct them from the original csv and save them in feather format
        elif path.isfile(origPath):
            self.train, self.test = self.split_train_test(pd.read_csv(origPath), 0.2)
            self.train.to_feather(trainPath)
            self.test.to_feather(testPath)

        else:
            raise Exception("No data detected in directory!")


        #sepearate predictors and targets from each set
        self.trainX = self.train.drop(columns="target")
        self.testX = self.test.drop(columns='target')
        self.trainY = self.train[['target']]
        self.testY = self.test[['target']]

    #splits the data into a training and test set with the same ratio of claimants vs non-claimants
    def split_train_test(self, data, test_ratio):
        split = StratifiedShuffleSplit(test_size=test_ratio, n_splits=1)
        for train_index, test_index in split.split(data, data["target"]):
            return data.iloc[train_index].reset_index(drop=True), data.iloc[test_index].reset_index(drop=True)




if __name__ == "__main__":
    data = Data("../data")

    print(data.trainX.head())
    print(data.trainY.head())
