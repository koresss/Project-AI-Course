import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import pickle
# Read csv, select a shop, drop date block num
df = pd.read_csv('Dataset/sales_train_v2.csv')
# df = df[df['shop_id']==31]
print(df.head(5))
df.drop('date_block_num', axis=1, inplace=True)

# Select pink floyd dark side of the moon albums

df = df[df['item_id'].isin([5466,5467,5468,5469,5470])]
# df = df[df['shop_id'].isin([45])]
# df = df[df['item_id'].isin([8885])]

# df = df[df['item_id'].isin(range(1749, 1754))]
# Convert date to datetime obj, sort by it, sum item counts per day
df['date'] = pd.to_datetime(df['date'])#.apply(lambda x: x.toordinal())
df.sort_values(by=['date'], inplace=True, ascending=True)
df = df.groupby('date')['item_cnt_day'].agg('sum')


idx = pd.date_range('2013-01-02', '2015-10-31')
df.index = pd.DatetimeIndex(df.index)
df = df.reindex(idx, fill_value=0)


df = pd.DataFrame(df)

print(df.head(50))
print(df.count())
print(df[df['item_cnt_day'] == 0.0].count())


# Create synthetic intermittent data
# vals = df.values
# vals2 = np.copy(vals)
# for idx, elem in enumerate(vals):
# 	if idx%30==0 and idx!=0:
# 		vals2 = np.insert(vals2, idx, np.zeros(np.random.randint(7,15)))

# df = pd.DataFrame(vals2)


# Add col with day num
df['day_num'] = np.arange(0, len(df), 1)
df.columns = ['item_cnt_day', 'day_num']


# Add lags up to 30
from pandas import Series
from pandas import DataFrame
from pandas import concat


num_lags = 30
for i in range(1,num_lags):
	df['lag'+str(i)] = df['item_cnt_day'].shift(i)

# Remove first rows as they would have NaNs
df = df.iloc[num_lags:]
print(df.head(10))
# Add cumulative zeros column
# TODO

# Cut to first 1000 elems only
# df = df.head(1000)
plt.plot(df['item_cnt_day'])
plt.show()
# Convert to np array
df = df.values
# Split to train val test
train, val, test = np.split(df, [int(0.8*len(df)), int(0.9*len(df))], axis=0)


with open('split_data.pkl', 'wb') as f:
	pickle.dump((train,val,test), f)