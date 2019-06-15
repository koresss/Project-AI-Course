import numpy as np
import pickle
import itertools
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import scale

from baseline_naive import naive_forecast
import matplotlib.pyplot as plt

fname = 'synthetic_data.pkl'
with open(fname, 'rb') as f:
	train, val, test = pickle.load(f)

# Split the x and y vars
x_train, y_train = train[:,1:], train[:,0]
x_val, y_val = val[:, 1:], val[:, 0]
x_test, y_test = test[:, 1:], test[:, 0]
print(x_train.shape)
# x_train = x_train.reshape(-1, 1)
# x_val = x_val.reshape(-1, 1)
# x_test = x_test.reshape(-1, 1)

model = GradientBoostingRegressor(loss='lad', learning_rate=0.01, n_estimators=100,
								 )
model.fit(x_train, y_train)

preds_train = model.predict(x_train)
preds_val = model.predict(x_val)

print('Train MAE:', mean_absolute_error(y_train, preds_train))
print('Val MAE:', mean_absolute_error(y_val, preds_val))


n_estims = [40, 100, 160, 240, 300]
subsamples = np.linspace(0, 1, 6)[1:]
lrs = [0.01, 0.03, 0.1]
best_val = 100


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
	preds_test = model.predict(x_test)

	mae_train = mean_absolute_error(y_train, preds_train)
	mae_val = mean_absolute_error(y_val, preds_val)
	mae_test = mean_absolute_error(y_test, preds_test)

	if mae_val < best_val:
		best_val = mae_val
		best_train = mae_train
		best_test = mae_test
		best_params = elem

	print('n estim={}, subsample={}, lr={}\nTrain MAE: {}\nVal MAE: {}'.format(
		  n_estim, subsample, lr, mae_train, mae_val))

naive_mae_train, naive_mae_val, naive_mae_test = naive_forecast(fname)

print('-'*30)
print('Best params found: {} with val MAE of {}'.format(best_params, best_val))
print('Naive MAE val : ', naive_mae_val)
print('-'*30)
print('MASE train of best params: {}'.format(best_train/naive_mae_train))
print('MASE val of best params: {}'.format(best_val/naive_mae_val))
print('MASE test of best params: {}'.format(best_test/naive_mae_test))
print('-'*30)

n_estim, subsample, lr = best_params

model = GradientBoostingRegressor(loss='lad',
								  criterion='mae',
								  learning_rate=lr,
								  subsample=subsample,
								  n_estimators=n_estim)

model.fit(x_train, y_train)
preds_test = model.predict(x_test)
preds_train = model.predict(x_train)

plt.plot(preds_train, label='pred')
plt.plot(y_train, label='ground truth')
plt.legend()
plt.show()

plt.plot(preds_test, label='pred')
plt.plot(y_test, label='ground truth')
plt.legend()
plt.show()