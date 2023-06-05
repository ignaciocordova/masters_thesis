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

# plot target every 1000 points
plt.figure(figsize=(15,5))
plt.plot(target['target'].values)
plt.title('Target')
plt.savefig('./figures/target.png')


