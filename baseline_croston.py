import pickle
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error
from baseline_naive import naive_forecast

fname = 'synthetic_data_4.pkl'
with open(fname, 'rb') as f:
	train, val, test = pickle.load(f)

train = train[:,0]
val = val[:,0]
test = test[:,0]

def crostons(train, alpha):
	a = int(train[0])
	p = 1
	# q is time elapsed since previous demand occurence
	q = 0

	forecasts = []

	for idx,elem in enumerate(train[1:], start=1):
		# If we have a demand occurence, update a and p
		if elem != 0:
			a = alpha*elem + (1-alpha)*a
			p = alpha*q + (1-alpha)*p
			q = 1
		# If not, no need to update a and p
		# but need to keep track of q
		if elem==0:
			q += 1
		try:
			forecast = a/p
		except:
			forecast = 0
		forecasts.append(forecast)

	return forecasts

train_temp = np.delete(train, 0)
train_temp = train_temp.tolist()

best_mae = 100

for alpha in np.linspace(0.01, 0.99, 30):
	forecasts = crostons(train, alpha)
	forecast_val = np.ones(len(val))*forecasts[-1]
	forecast_train = np.ones(len(train_temp))*forecasts[-1]

	mae_train = mean_absolute_error(forecasts, train_temp)
	mae_val = mean_absolute_error(forecast_val, val)

	if mae_val < best_mae:
		best_mae = mae_val
		best_alpha = alpha

	print('Alpha = {} train MAE: {} val MAE: {}'.format(alpha, mae_train, mae_val))
	print('-'*30)

print('Best alpha : ', best_alpha)

forecasts = crostons(train, best_alpha)
forecast_val = np.ones(len(val))*forecasts[-1]
forecast_val = np.ones(len(val))*forecasts[-1]
forecast_train = np.ones(len(train_temp))*forecasts[-1]
forecast_test = np.ones(len(test))*forecasts[-1]

plt.plot(forecasts, label='forecasts')
plt.plot(train, label='train')
plt.legend()
plt.show()

naive_mae_train, naive_mae_val, naive_mae_test = naive_forecast(fname)

print('Train')
train_mae = mean_absolute_error(forecast_train, train_temp)
print('MAE: {} Naive MAE: {} MASE: {}'.format(train_mae, naive_mae_train, train_mae/naive_mae_train))

print('Val')
val_mae = mean_absolute_error(forecast_val, val)
print('MAE: {} Naive MAE: {} MASE: {}'.format(val_mae, naive_mae_val, val_mae/naive_mae_val))


print('Test')
test_mae = mean_absolute_error(forecast_test, test)
print('MAE: {} Naive MAE: {} MASE: {}'.format(test_mae, naive_mae_test, test_mae/naive_mae_test))
