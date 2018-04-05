import pandas as pd
from sklearn.pipeline import Pipeline

from python.Data import Data, getVariances
from python.processors.InvariantRemover import InvariantRemover

from python.processors.NACreator import NAAdder
from python.processors.NAImputers import NAImputer_Categorical, NAImputer_Interval

pipeline = Pipeline([
    ('Transform -1 to NA', NAAdder()),
    ('Add additional category for NAs in categorical data', NAImputer_Categorical()),
    ('Fix interval variables', NAImputer_Interval(0.05)),
    ('Remove attributes with low variance', InvariantRemover(attrs_to_drop=54)),  # or use min_variance=x instead
])

if __name__ == "__main__":
    data = Data()

    transformed = pipeline.fit_transform(data.trainX)
    print(pd.get_dummies(data.trainX['ps_ind_01'], prefix='ps_ind_01', drop_first=True).info())
