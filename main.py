import pandas as pd
import matplotlib.pyplot as  plt

df=pd.read_csv('Dataset/sales_train_v2.csv')
print(df.describe())
tdf=df.loc[df['shop_id']==25]
tdf=tdf.loc[tdf['item_id']==2574]
print(tdf)
plt.plot(tdf.date,tdf.item_cnt_day)
plt.show()