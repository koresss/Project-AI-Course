import numpy as np
import pickle
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
print('Train MAE:', mean_absolute_error(y_train, preds_train))
print('Val MAE:', mean_absolute_error(y_val, preds_val))