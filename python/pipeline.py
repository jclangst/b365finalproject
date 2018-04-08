from sklearn.pipeline import Pipeline

from python.Data import Data
from python.gini import gini_normalized

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

    pipeline.fit(data.trainX, data.trainY)
    pipeline.transform(data.testX)

    predictions = rf.predicted
    print("\nGini: " + str(gini_normalized(data.testY, predictions)) + '\n')
