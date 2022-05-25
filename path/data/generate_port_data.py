import yfinance as yf
from functools import reduce
import pandas as pd 
import numpy as np
from sklearn.cluster import KMeans
import sys
import random

seed = int(sys.argv[1])


df_open=pd.DataFrame()
df_volume=pd.DataFrame()
cnt=0
tickers_to_retry = []

ticker_list=['CSCO', 'BA', 'MDT', 'HSBC', 'MO',
        'NVS', 'BCH', 'CHTR', 'C', 'T', 'SNP', 'BAC', 'BP',
        'PEP', 'IEP', 'UL', 'D', 'MRK', 'TSM', 'CODI', 'ORCL',
        'PG', 'CAT', 'MCD', 'AMZN', 'INTC', 'MMM', 'KO', 'NEE', 'UPS', 'MSFT',
        'EXC', 'HD', 'SO', 'XOM', 'CVX', 'CMCSA', 'PCG', 'GOOG',
        'FB', 'NGG', 'BHP', 'WFC', 'GD', 'PM', 'DIS', 'GE', 'PTR',
        'BSAC', 'JPM', 'DHR', 'ABB', 'SRE', 'GOOG', 'PFE', 'DUK',
        'VZ', 'AMGN', 'SNY', 'UNH', 'MA', 'HON', 'SLB', 'AAPL', 'WMT',
        'LMT', 'AEP', 'JNJ', 'REX', 'PPL'] #, 'BRK-A', 'TM', 'V',

stocks=pd.read_csv(r'path/data/stocks.csv')

ticker_list_small=list(stocks.iloc[seed,:])
ticker_list_small.sort()

start="2012-01-01"
end="2021-12-31"

# Downloading data
data = yf.download(ticker_list_small, start = start, end = end).reset_index()


data_subset = data.loc[:,('Adj Close', slice(None))]
data_subset.columns = ticker_list_small
data_subset = data_subset[ticker_list_small].pct_change()  #.dropna()
returns = data_subset
data_subset = - data_subset
data_subset['DATE'] = data.loc[:,('Date', '')]
data_subset.dropna(inplace = True)

returns['DATE'] = data.loc[:,('Date', '')]
returns.dropna(inplace = True)

data_vol = data.loc[:,('Volume', slice(None))]
data_vol.columns = ticker_list_small
data_vol['DATE'] = data.loc[:,('Date', '')]
data_vol.dropna(inplace = True)
data_vol = data_vol.add_suffix('_SI')
data_vol = data_vol.rename({'DATE_SI': 'DATE'}, axis=1)
data_vol.dropna(inplace = True)


data_volume=data_vol[data_vol.DATE.isin(data_subset.DATE)]


market_features = ['^DJI', '^VIX', '^GSPC','CL=F', 'GC=F', '^TNX' ]
market_col_names = ['DJI_SI', 'VIX_SI', 'GSPC_SI','CLF_SI', 'GCF_SI', 'TNX_SI' ]
data_m = yf.download(market_features, start = start, end = end).reset_index()
data_market = data_m.loc[:,('Open', slice(None))]
data_market.columns = market_col_names
data_market['DATE'] = data_m.loc[:,('Date', '')]
data_market.dropna(inplace = True)

data_market=data_market[data_market.DATE.isin(data_subset.DATE)]


data_subset.DATE=data_subset.DATE.astype('datetime64[ns]')
data_volume.DATE  =data_volume.DATE.astype('datetime64[ns]')
data_market.DATE=data_market.DATE.astype('datetime64[ns]')

col_order = ticker_list_small
volume = True
if volume:
    data_side=pd.merge(data_volume,data_market,on='DATE', how = 'left')
else:
    data_side = data_market
col_order = col_order + list(data_side.columns)
col_order.remove('DATE')

data_final=pd.merge(data_subset,data_side,on='DATE', how = 'left')
data_final.fillna(data_final.mean(), inplace=True)
data_final[col_order] = data_final[col_order].apply(pd.to_numeric, errors='coerce')
col_order = ['DATE'] + col_order
data_final = data_final.reindex(columns=col_order)


data_subset.to_csv(r'path/data/expected_cost.csv',index=False)
returns.to_csv(r'path/data/expected_return.csv',index=False)
data_side.to_csv(r'path/data/side_info.csv',index=False)
data_final.to_csv(r'path/data/final_port.csv',index=False)

