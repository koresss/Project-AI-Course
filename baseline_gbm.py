import numpy as np
import pickle
import itertools
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

with open('split_data.pkl', 'rb') as f:
	train, val, test = pickle.load(f)

# Split the x and y vars
x_train, y_train = train[:,1], train[:,0]
x_val, y_val = val[:, 1], val[:, 0]
x_test, y_test = test[:, 1], test[:, 0]

x_train = x_train.reshape(-1, 1)
x_val = x_val.reshape(-1, 1)
x_test = x_test.reshape(-1, 1)

model = GradientBoostingRegressor(loss='lad', learning_rate=0.01, n_estimators=100,
								 )
model.fit(x_train, y_train)

preds_train = model.predict(x_train)
preds_val = model.predict(x_val)
print('Base')
print('Train MAE:', mean_absolute_error(y_train, preds_train))
print('Val MAE:', mean_absolute_error(y_val, preds_val))


n_estims = [40, 100, 160, 240, 300]
subsamples = np.linspace(0, 1, 6)[1:]
lrs = [0.01, 0.03, 0.1]

for elem in itertools.product(n_estims, subsamples, lrs):
	n_estim = elem[0]
	subsample = elem[1]
	lr = elem[2]
	print('-'*30)
	
	model = GradientBoostingRegressor(loss='lad',
									  criterion='mae',
									  learning_rate=lr,
									  subsample=subsample,
									  n_estimators=n_estim)

	model.fit(x_train, y_train)
	preds_train = model.predict(x_train)
	preds_val = model.predict(x_val)

	mae_train = mean_absolute_error(y_train, preds_train)
	mae_val = mean_absolute_error(y_val, preds_val)

	print('n estim={}, subsample={}, lr={}\nTrain MAE: {}\nVal MAE: {}'.format(
		  n_estim, subsample, lr, mae_train, mae_val))
