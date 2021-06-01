# run-predictor

Linear regression on Garmin Connect activities history to predict how long a run will take.

# Setup

```
python3 -m venv env
. env/bin/activate
pip install statsmodels matplotlib
```

# Usage

1. Go to https://connect.garmin.com/modern/activities and click on 'Export CSV' to download Activities.csv.
2. Run the following:

```
    . env/bin/activate
    PYTHONPATH=. python3 main.py ~/Downloads/Activities.csv 12 300 2021-01-01 2021-03-31
    PYTHONPATH=. python3 main.py ~/Downloads/Activities.csv 12 300 2021-05-01 2021-05-31
```
