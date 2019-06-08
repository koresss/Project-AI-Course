import pickle
import numpy as np

with open('data.pkl', 'rb') as f:
	data = pickle.load(f)

data['day_num'] = np.arange(0, len(data), 1)

# Convert to np array
data = data.values
# Split to train val test
train, val, test = np.split(data, [int(0.8*len(data)), int(0.9*len(data))], axis=0)

with open('split_data.pkl', 'wb') as f:
	pickle.dump((train,val,test), f)