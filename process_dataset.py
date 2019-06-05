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
df = df[df['item_id'].isin([5466,5467,5468,5469,5470])]
# df = df[df['item_id'].isin([8885])]

# Convert date to datetime obj, sort by it, sum item counts per day
df['date'] = pd.to_datetime(df['date'])#.apply(lambda x: x.toordinal())
df.sort_values(by=['date'], inplace=True, ascending=True)
df = df.groupby('date')['item_cnt_day'].agg('sum')


idx = pd.date_range('2013-01-02', '2015-10-31')
df.index = pd.DatetimeIndex(df.index)
df = df.reindex(idx, fill_value=0)


df = pd.DataFrame(df)
plt.plot(df)
plt.show()

print(df.head(50))
print(df.count())
print(df[df['item_cnt_day'] == 0.0].count())
with open('data.pkl', 'wb') as f:
	pickle.dump(df, f)