import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

INSTALLED_POWER = 17500

target = pd.read_csv('target_stv_2018.csv', header=None, names=['target'])

# normalize target 
target['target'] = target['target'].apply(lambda x: x/17500)

# create displaced target 
target['target_displaced'] = target['target'].shift(1).fillna(target['target'].mean())

# compute MAE between target and displaced target
mae = mean_absolute_error(target['target'], target['target_displaced'])

print('MAE between target and displaced target: {}'.format(np.round(mae,4)))

# compute MAE of predictions Nh into the future
N_HOURS = 1
if N_HOURS > 1:
    target_into_future = target['target'].iloc[N_HOURS-1:].values
    # predictions
    predictions = target['target_displaced'].values[:-N_HOURS+1]
else:
    target_into_future = target['target'].values
    # predictions
    predictions = target['target_displaced'].values

# check if the length of the two arrays is the same
assert len(target_into_future) == len(predictions)

# compute MAE
mae = mean_absolute_error(target_into_future, predictions)

print('MAE of predictions {}h into the future: {}'.format(N_HOURS, np.round(mae,8)))

# compute MSE error
mse = np.mean((target_into_future - predictions)**2)

print('MSE of predictions {}h into the future: {}'.format(N_HOURS, np.round(mse,8)))