## Input files:
`train.csv` and `test.csv`

## Scripts
Requires the [Kaggle API](https://github.com/kaggle/kaggle-api) to be configured & for the default competition to be set.

`./check`: return a list of submissions & scores

`./process`: do whatever is required to generate the submission file `out.csv`

`./submit "message"`: calls `process` and then submits `out.csv`
