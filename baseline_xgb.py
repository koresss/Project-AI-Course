import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import pickle

with open('split_data.pkl', 'rb') as f:
	train, val, test = pickle.load(f)

x_train, y_train = train[:,1], train[:,0]
x_val, y_val = val[:, 1], val[:, 0]

x_train = x_train.reshape(1, -1)
y_train = y_train.reshape(1, -1)
dtrain = xgb.DMatrix(x_train, label=y_train)
# dval = xgb.DMatrix(val)
num_round = 2
hparams = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'reg:squarederror' }
model = xgb.train(hparams, dtrain, num_boost_round=num_round)