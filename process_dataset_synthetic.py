import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import pickle
# Read csv, select a shop, drop date block num
df = pd.read_csv('Dataset/sales_train_v2.csv')
# df = df[df['shop_id']==31]
df.drop('date_block_num', axis=1, inplace=True)

# Select pink floyd dark side of the moon albums

# df = df[df['item_id'].isin([5466,5467,5468,5469,5470])]
df = df[df['shop_id'].isin([45])]
# df = df[df['item_id'].isin([8885])]
# Convert date to datetime obj, sort by it, sum item counts per day
df['date'] = pd.to_datetime(df['date'])#.apply(lambda x: x.toordinal())
df.sort_values(by=['date'], inplace=True, ascending=True)
df = df.groupby('date')['item_cnt_day'].agg('sum')



idx = pd.date_range('2013-01-02', '2015-10-31')
df.index = pd.DatetimeIndex(df.index)
df = df.reindex(idx, fill_value=0)


df = pd.DataFrame(df)

print(df[df['item_cnt_day'] == 0.0].count())



# Encode day of week as cyclical feature
df['day_num'] = np.arange(0, len(df), 1)
df.columns = ['item_cnt_day', 'day_num']
df['dow'] = (df['day_num']+3)%7
df['dow_sin'] = np.sin(2 * np.pi * df['dow']/6.0)
df['dow_cos'] = np.cos(2 * np.pi * df['dow']/6.0)

# Insert zeros depending on day number
vals = df['item_cnt_day'].values
dows = df['dow'].values
vals_to_zero_out = [0, 1]
for i,elem in enumerate(dows):
	if elem in vals_to_zero_out:
		vals[i] = 0

# df['item_cnt_day'] = vals

# Zero out an entire month per year
day_count = df['day_num'].values
vals = df['item_cnt_day'].values
for i,elem in enumerate(day_count):
	if elem%365 in range(90,120):
		vals[i] = 0
# Drop useless cols
df.drop(labels=['day_num', 'dow'], axis=1, inplace=True)

# Add lags up to 30
from pandas import Series
from pandas import DataFrame
from pandas import concat

num_lags = 30
for i in range(1,num_lags):
	df['lag'+str(i)] = df['item_cnt_day'].shift(i)

# Remove first rows as they would have NaNs
df = df.iloc[num_lags:]

# Add cumulative zeros column
# TODO

# Cut to first 1000 elems only
# df = df.head(1000)
plt.plot(df['item_cnt_day'])
plt.show()
# Convert to np array
print(df.columns)
print(df.head(10))
df = df.values
# Split to train val test
train, val, test = np.split(df, [int(0.8*len(df)), int(0.9*len(df))], axis=0)

with open('synthetic_data_combined_2days.pkl', 'wb') as f:
	pickle.dump((train,val,test), f)
# with open('synthetic_data_' + str(len(vals_to_zero_out)) + '.pkl', 'wb') as f:
# 	pickle.dump((train,val,test), f)