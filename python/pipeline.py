from sklearn.pipeline import Pipeline

from python.Data import Data

from python.processors.AttrTypeRemover import AttrTypeRemover
from python.processors.NACreator import NAAdder
from python.processors.NAImputers import NAImputer_Categorical, NAImputer_Interval
from python.processors.InvariantRemover import InvariantRemover
from python.processors.Standardizer import Standardizer
from python.processors.RandomForestTest import RandomForestTest

# variables declared here so they can be accessed in console after running code
rf = RandomForestTest()

pipeline = Pipeline([
    # comment out any lines with non-desired pipeline parts
    ('Remove certain types of attributes', AttrTypeRemover(cats=True)),
    ('Transform -1 to NA', NAAdder()),
    ('Add additional category for NAs in categorical data', NAImputer_Categorical()),
    ('Fix interval variables', NAImputer_Interval(0.05)),
    ('Remove attributes with low variance', InvariantRemover(attrs_to_drop=5)),  # or use min_variance=x instead
    ('Fit all attributes into [0, 1]', Standardizer()),
    ('Random forest test', rf),
])

if __name__ == "__main__":
    data = Data()
    fitted = pipeline.fit(data.trainX, data.trainY)
    transformed = pipeline.transform(data.testX)

# output model correctness
predictions = rf.predicted

print()
print("Total datapoints:      {}".format(len(predictions)))
print("Actual claims:         {}".format(len(data.testY[data.testY == 1])))
print("Predicted claims:      {}".format(len(predictions[predictions == 1])))
print()
print("Correct predictions:   {}".format(len(predictions[(predictions == 1) & (data.testY == 1)])))
print("Incorrect predictions: {}".format(len(predictions[(predictions == 1) & (data.testY == 0)])))
print("Unpredicted claims:    {}".format(len(predictions[(predictions == 0) & (data.testY == 1)])))
