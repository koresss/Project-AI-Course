import pickle
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error

with open('split_data.pkl', 'rb') as f:
	train, val, test = pickle.load(f)

train = train[:,0]
val = val[:,0]

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

for alpha in np.linspace(0.1, 1, 15):
	forecasts = crostons(train, alpha)
	forecast_val = np.ones(len(val))*forecasts[-1]
	forecast_train = np.ones(len(train_temp))*forecasts[-1]
	print('Alpha = {} train MAE: {} val MAE: {}'.format(alpha, mean_absolute_error(forecast_train, train_temp),
		  mean_absolute_error(forecast_val, val)
			))
	print('-'*30)

plt.plot(forecast_train, label='forecasts')
plt.plot(train, label='train')
plt.legend()
plt.show()

forecast_val = np.ones(len(val))*forecasts[-1]
print('MAE val: ', mean_absolute_error(forecast_val, val))

