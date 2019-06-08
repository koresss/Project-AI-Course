import pickle
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error

with open('split_data.pkl', 'rb') as f:
	train, val, test = pickle.load(f)

train = train[:,0]
val = val[:,0]

alpha = 0.8
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

train = np.delete(train, 0)
train = train.tolist()

print('Train MAE:', mean_absolute_error(forecasts, train))

plt.plot(forecasts, label='forecasts')
plt.plot(train, label='train')
plt.legend()
plt.show()