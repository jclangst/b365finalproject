import feather
import pandas as pd

# load the train.csv, and convert it to the Feather format - https://github.com/wesm/feather
# reduces time to load data from 2.5s to 0.15s on my computer
# pip install feather-format

# import time
# t = time.time()
# train_csv_dataframe = pd.read_csv('../input/train.csv') # load csv
# print('csv: ' + str(time.time() - t))
# feather.write_dataframe(train_csv_dataframe, '../input/train.feather') # csv -> feather
# t = time.time()
# train = feather.read_dataframe('../input/train.feather') # load feather
# print('feather: ' + str(time.time() - t))

train = feather.read_dataframe('../input/train.feather')
# the class of train is pandas.core.frame.DataFrame
# print(train)
