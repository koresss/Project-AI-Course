import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import mean_absolute_error

# Naive model simply predicts value of previous day

with open('split_data.pkl', 'rb') as f:
	train, val, test = pickle.load(f)
train = train[:, 0]
val = val[:, 0]
test = test[:, 0]
train = np.append(train, val)

forecast_train = np.roll(train, 1)
forecast_test = np.roll(test, 1)

# Delete 1st elem since it cannot be forecasted
forecast_train = np.delete(forecast_train, 0)
train = np.delete(train, 0)

forecast_test = np.delete(forecast_test, 0)
test = np.delete(test, 0)

# Finally print MAE
print('Train MAE:', mean_absolute_error(train, forecast_train))
print('Test MAE:', mean_absolute_error(test, forecast_test))


# Probably no sense to use val and test for naive method, tbh
# Might as well do it on the entire set