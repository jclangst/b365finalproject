from sklearn.pipeline import Pipeline

from python.Data import Data

from python.processors.NACreator import NAAdder
from python.processors.NAImputers import NAImputer_Categorical, NAImputer_Interval
from python.processors.InvariantRemover import InvariantRemover
from python.processors.CalcRemover import CalcRemover
from python.processors.Standardizer import Standardizer
from python.processors.RandomForestTest import RandomForestTest

# variables declared here so they can be accessed in console after running code
rf = RandomForestTest()

pipeline = Pipeline([
    # comment out any lines with non-desired pipeline parts
    ('Transform -1 to NA', NAAdder()),
    ('Add additional category for NAs in categorical data', NAImputer_Categorical()),
    ('Fix interval variables', NAImputer_Interval(0.05)),
    ('Remove attributes with low variance', InvariantRemover(attrs_to_drop=5)),  # or use min_variance=x instead
    ('Remove calculated attributes', CalcRemover()),
    ('Fit all attributes into [0, 1]', Standardizer()),
    ('Random forest test', rf),
])

if __name__ == "__main__":
    data = Data()
    fitted = pipeline.fit(data.trainX, data.trainY)
    transformed = pipeline.transform(data.testX)

print()
print("Total datapoints: {}".format(len(rf.predicted)))
print("Predicted claims: {}".format(len(rf.predicted[rf.predicted == 1])))
print("Correct predictions: {}".format(len(rf.predicted[(rf.predicted == 1) & (data.testY == 1)])))
