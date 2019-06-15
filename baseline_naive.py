import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import mean_absolute_error

# Naive model simply predicts value of previous day

def naive_forecast():
	with open('synthetic_data.pkl', 'rb') as f:
		train, val, test = pickle.load(f)
	train = train[:, 0]
	val = val[:, 0]
	test = test[:, 0]
	train = np.append(train, val)

	forecast_train = np.roll(train, 1)
	forecast_val = np.roll(val, 1)
	forecast_test = np.roll(test, 1)
	# Delete 1st elem since it cannot be forecasted
	forecast_train = np.delete(forecast_train, 0)
	train = np.delete(train, 0)

	forecast_test = np.delete(forecast_test, 0)
	test = np.delete(test, 0)

	forecast_val = np.delete(forecast_val, 0)
	val = np.delete(val, 0)

	train_mae = mean_absolute_error(train, forecast_train)
	test_mae = mean_absolute_error(test, forecast_test)
	val_mae = mean_absolute_error(val, forecast_val)

	return train_mae, val_mae, test_mae


if __name__ == "__main__":
	train_mae, val_mae, test_mae = naive_forecast()

	print('Train MAE: {}\nVal MAE: {}\nTest MAE: {}'.format(train_mae, val_mae, test_mae))
	# Probably no sense to use val and test for naive method, tbh
	# Might as well do it on the entire set