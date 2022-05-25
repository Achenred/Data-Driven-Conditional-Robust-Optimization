import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scipy.stats
import sklearn.datasets
from sklearn import preprocessing

# def scaler(df):
#     x=df.values
#     min_max_scaler = preprocessing.MinMaxScaler()
#     x_scaled = min_max_scaler.fit_transform(x)
#     df = 100*pd.DataFrame(x_scaled)
#     return df


year = int(sys.argv[1])


data=pd.read_csv(r'path/data/final_port.csv')

data.DATE=pd.DatetimeIndex(data.DATE)

train=data[(data.DATE.dt.year>=year)&(data.DATE.dt.year<year+4)]
test=data[(data.DATE.dt.year==year+4)]

train.drop('DATE', axis=1, inplace=True)
test.drop('DATE', axis=1, inplace=True)

# train = scaler(train)
# test = scaler(test)

train.to_csv(r'path/data/train_'+str(year)+'.csv',index=False)
test.to_csv(r'path/data/test_'+str(year)+'.csv',index=False)
