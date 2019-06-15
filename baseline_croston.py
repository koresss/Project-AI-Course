import pickle
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error

with open('split_data.pkl', 'rb') as f:
	train, val, test = pickle.load(f)

train = train[:,0]
val = val[:,0]

def crostons(train, alpha):
	a = 0
	p = 0
	# q is time elapsed since previous demand occurence
	q = 0

	forecasts = []

	for idx,elem in enumerate(train[1:], start=1):
		# If we have a demand occurence, update a and p
		if elem != 0:
			a = alpha*elem + (1-alpha)*a
			p = alpha*q + (1-alpha)*p
		# If not, no need to update a and p
		# but need to keep track of q
		if elem==0:
			if train[idx-1] == 0:
				# print('increment')
				q += 1
			if train[idx-1] != 0:
				q = 1
		try:
			forecast = a/p
		except:
			forecast = 0
		forecasts.append(forecast)

	return forecasts

train_temp = np.delete(train, 0)
train_temp = train_temp.tolist()

for alpha in np.linspace(0.0001, 1, 15):
	forecasts = crostons(train, alpha)
	print('Alpha = {} train MAE: {}'.format(alpha, mean_absolute_error(forecasts, train_temp)
			))
	print('-'*30)

plt.plot(forecasts, label='forecasts')
plt.plot(train, label='train')
plt.legend()
plt.show()
