import csv
from datetime import date, datetime, timedelta
import sys

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def import_activities(activities_path):
    return pd.read_csv(activities_path)

def filter_activities(df, from_date, to_date):
    filters = [
        lambda df: df['Activity Type'].apply(lambda at: not ('Bik' in at)), # exclude Activity Type of biking
        lambda df: df['Date'] != '2021-03-06 07:39:33', # exclude TDCC (was Other)
        lambda df: df['Activity Type'].apply(lambda at: at != 'Hiking'), # exclude Activity Type of Hiking
        lambda df: df['Distance'].apply(lambda d: int(d) >= 10), # distance > 10 km
        lambda df: df['Elev Gain'].apply(lambda v: int(str(v).replace(',','')) >= 500), # vert > 500 m
        lambda df: df['Time'].apply(lambda t: int(str(t).split(':')[0]) >= 2), # time > 2 hrs
        #lambda df: df['Date'].apply(lambda d: datetime.fromisoformat(d) > datetime.today() - timedelta(days=30)), # in the last 30 days
        lambda df: df['Date'].apply(lambda d: datetime.fromisoformat(d) > from_date),
        lambda df: df['Date'].apply(lambda d: datetime.fromisoformat(d) <= to_date),
    ]
    for f in filters:
        # e.g.
        #not_biking = df['Activity Type'].apply(lambda at: not ('Bik' in at))
        #df = df[not_biking]
        #not_tdcc = df['Date'] != '2021-03-06 07:39:33'
        #df = df[not_tdcc]
        boolean_indices = f(df)
        df = df[boolean_indices]
    return df

def extract_distance_vert_time(df):
    df['distance_km'] = pd.to_numeric(df['Distance'])
    df['vert_m'] = pd.to_numeric(df['Elev Gain'].str.replace(',',''))
    df['timedelta'] = pd.to_timedelta(df['Time'])
    df['time_min'] = df['timedelta'].apply(lambda x: int(x.seconds / 60))
    return df[['distance_km', 'vert_m', 'time_min']]

def main(activities_path, predicted_distance_km, predicted_vert_m, from_date, to_date):
    # prepare the data
    df = import_activities(activities_path)
    df = filter_activities(df, from_date, to_date)
    print(df)
    df = extract_distance_vert_time(df)

    # compute the model
    X = df[['distance_km', 'vert_m']]
    y = df['time_min']
    X = sm.add_constant(X)
    est = sm.OLS(y, X).fit()
    #print(est.summary())
    print('model summary:')
    per_km_minutes, per_km_seconds = divmod(timedelta(minutes=est.params[1]).seconds, 60)
    print('  each km: {:02}:{:02}'.format(int(per_km_minutes), int(per_km_seconds)))
    per_100m_minutes, per_100m_seconds = divmod(timedelta(minutes=est.params[2] * 100).seconds, 60)
    print('  each 100m: {:02}:{:02}'.format(int(per_100m_minutes), int(per_100m_seconds)))
    print()

    # predict the result
    result_df = est.predict(exog=pd.DataFrame({
        'const': [1.0],
        'distance_km': [predicted_distance_km],
        'vert_m': [predicted_vert_m]
    }))
    predicted_time_min = int(result_df[0])
    hours, minutes = divmod(predicted_time_min, 60)
    print('predicted time: {:02}:{:02}'.format(int(hours), int(minutes)))


if __name__ == '__main__':
    activities_path, predicted_distance_km, predicted_vert_m, from_date, to_date = sys.argv[1:]
    predicted_distance_km = float(predicted_distance_km)
    predicted_vert_m = int(predicted_vert_m)
    from_date = datetime.fromisoformat(from_date)
    to_date = datetime.fromisoformat(to_date)
    print(f'Garmin activity history: {activities_path}')
    print(f'predicting for distance_km: {predicted_distance_km}')
    print(f'predicting for vert_m: {predicted_vert_m}')
    print(f'start training date: {from_date}')
    print(f'until training date: {to_date}')
    print()
    main(activities_path, predicted_distance_km, predicted_vert_m, from_date, to_date)
