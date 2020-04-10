import os
import math
import pandas as po
from datetime import datetime
from tqdm import tqdm_notebook
import matplotlib.pyplot as plt

df = po.read_csv('data/raw/kresit_main_2018-oct_to_2019-oct.csv')

df['Series'] = df['Series;Time;Value'].apply(lambda row: row.split(';')[0])
df['Date-Time'] = df['Series;Time;Value'].apply(lambda row: row.split(';')[1].split('+')[0])
df['Value'] = df['Series;Time;Value'].apply(lambda row: row.split(';')[2])
df.drop('Series;Time;Value', axis = 1, inplace=True)
df.head()

df['Date'] = df['Date-Time'].apply(lambda row: row.split('T')[0])
df['Time'] = df['Date-Time'].apply(lambda row: row.split('T')[1])
df.drop('Date-Time', axis = 1, inplace=True)
df.head()

df["Series"].value_counts()

df_1 = df[df['Series'] == 'power_k_m']
df_1['Value'] = df_1['Value'].astype(float)
df_1.head()

df_1 = df_1.sort_values(by = 'Date')
df_1.head()

df_1 = df_1.sort_values(by='Date').reset_index(drop = True)

df_f_19 = po.DataFrame(df_1['Value'])
df_f_19.columns = ['W']
#df_2 = (df_2 - df_2.mean())/df_2.std()
#df_2 = (df_2 - df_2.min())/(df_2.max() - df_2.min())
plt.figure(figsize=(15, 8))
plt.plot(df_f_19)

df_f_19.to_csv('data/processed/2019_processed.csv', index = False)

base_path = 'data/raw/2018/'

files = os.listdir(base_path)
files.remove('.DS_Store' )

df_f = po.DataFrame()
for file in tqdm_notebook(files):
    df_t = po.read_csv(base_path + file)
    df_f = po.concat([df_f, df_t], axis = 0).reset_index(drop = True)

df_f = df_f.dropna()

df_f['Date'] = df_f['TS'].apply(lambda row: datetime.utcfromtimestamp(math.trunc(row)).strftime('%Y-%m-%d %H:%M:%S').split(' ')[0])
df_f['Time'] = df_f['TS'].apply(lambda row: datetime.utcfromtimestamp(math.trunc(row)).strftime('%Y-%m-%d %H:%M:%S').split(' ')[1])
df_f['h'] = df_f['Time'].apply(lambda row: int(row.split(':')[0]))
df_f['m'] = df_f['Time'].apply(lambda row: int(row.split(':')[1]))
#df_f['h:m'] = df_f['Time'].apply(lambda row: row.split(':')[0]+':'+row.split(':')[1]) 
df_f.drop('TS', axis = 1, inplace=True)

df_f.head()

df_f = df_f.sort_values(by = 'Date').reset_index(drop=True)

df_f.to_csv('data/processed/2018_full_concated.csv')

df_f = po.read_csv('data/processed/2017_full_concated.csv')

df_f.head(10)

df_f_daily = po.DataFrame(df_f.groupby(by = 'Date')['W'].mean())
df_f_daily.to_csv('data/processed/2017_full_daily.csv')

df_f_17 = po.read_csv('data/processed/2017_full_daily.csv')
df_f_18 = po.read_csv('data/processed/2018_full_daily.csv')

df_f_daily = po.concat([df_f_17, df_f_18], axis = 0)

df_f_daily.to_csv('data/processed/energy_daily.csv', index = False)

gb_date = df_f.groupby(by = 'Date')
dates = df_f['Date'].unique()

df_f['Date'].min()

df_f['Date'].max()

for d in tqdm_notebook(dates):
    df_t = gb_date.get_group(d)
    df_t.to_csv('data/processed/2018/'+d+'.csv')

base_path = 'data/processed/2017/'

files = os.listdir(base_path)
#files.remove('.DS_Store' )

for f in tqdm_notebook(files):
    df_t = po.read_csv(base_path+f)
    df_t['m_b'] = po.cut(df_t['m'], bins = range(0, 61, 5), labels = range(0, 60, 5))
    df_t['h_b'] = po.cut(df_t['h'], bins = range(25), labels = range(24))
    df_t = df_t.sort_values(by = 'Time')
    df_t = df_t.drop(['Unnamed: 0', 'm', 'h', 'Date', 'Time'], axis = 1)
    df_t = df_t.sort_values(['h_b', 'm_b']).reset_index(drop = True)
    df_t = df_t.groupby(['h_b', 'm_b']).mean()['W']
    df_t.index = range(len(df_t))
    df_t = po.DataFrame(df_t)
    df_t = df_t.to_csv('data/processed/2017_binned/'+f, index = False)

base_path = 'data/processed/2018/'

files = os.listdir(base_path)
#files.remove('.DS_Store' )

for f in tqdm_notebook(files):
    df_t = po.read_csv(base_path+f)
    df_t['m_b'] = po.cut(df_t['m'], bins = range(0, 61, 5), labels = range(0, 60, 5))
    df_t['h_b'] = po.cut(df_t['h'], bins = range(25), labels = range(24))
    df_t = df_t.sort_values(by = 'Time')
    df_t = df_t.drop(['Unnamed: 0', 'm', 'h', 'Date', 'Time'], axis = 1)
    df_t = df_t.sort_values(['h_b', 'm_b']).reset_index(drop = True)
    df_t = df_t.groupby(['h_b', 'm_b']).mean()['W']
    df_t.index = range(len(df_t))
    df_t = po.DataFrame(df_t)
    df_t = df_t.to_csv('data/processed/2018_binned/'+f, index = False)

base_path = 'data/processed/2017_binned/'

files = os.listdir(base_path)
if os.path.exists(base_path+'.DS_Store'):
    files.remove(base_path+'.DS_Store')

df_f_17 = po.DataFrame()
for f in tqdm_notebook(files):
    df_t = po.read_csv(base_path+f)
    df_f_17 = po.concat([df_f_17, df_t], axis = 0).reset_index(drop = True)

df_f_17.to_csv('data/processed/2017_processed.csv', index = False)

plt.figure(figsize = (20, 10))
plt.plot(df_f_17)

base_path = 'data/processed/2018_binned/'
files = os.listdir(base_path)
#files.remove('.DS_Store' )

df_f_18 = po.DataFrame()
for f in tqdm_notebook(files):
    df_t = po.read_csv(base_path+f)
    df_f_18 = po.concat([df_f_18, df_t], axis = 0).reset_index(drop = True)

df_f_18.to_csv('data/processed/2018_processed.csv', index = False)

len(df_f_18)

plt.figure(figsize = (20, 10))
plt.plot(df_f_18)

plt.figure(figsize = (20, 10))
plt.plot(df_f_19)

df_f = po.concat([df_f_17, df_f_18, df_f_19], axis = 0).reset_index(drop = True)

plt.figure(figsize = (20, 10))
plt.plot(df_f)

df_f_17_l = po.read_csv('data/processed/2017_processed.csv')
df_f_18_l = po.read_csv('data/processed/2018_processed.csv')
df_f_19_l = po.read_csv('data/processed/2019_processed.csv')

df_f_l = po.concat([df_f_17_l, df_f_18_l], axis = 0).reset_index(drop = True)

plt.figure(figsize = (20, 10))
plt.plot(df_f_l[:5000])

df_f_l = df_f_l.fillna(df_f_l.median())

# plt.figure(figsize = (20, 10))
# plt.plot(df_f_l)

df_f_l

# +
#df_f_l = (df_f_l - df_f_l.min())/(df_f_l.max() - df_f_l.min())
# -

plt.figure(figsize = (20, 10))
plt.plot(df_f_l)

from statsmodels.tsa.seasonal import seasonal_decompose

freq = int(60/5)*24*30

seas_decomp = seasonal_decompose(df_f_l, freq = freq)
seas_decomp.plot()

plt.figure(figsize=(50,8))
plt.plot(seas_decomp.seasonal.dropna().reset_index(drop=True))

plt.figure(figsize=(12,8))
plt.plot(seas_decomp.trend.dropna().reset_index(drop=True))

df_f_l_t = seas_decomp.trend.dropna().reset_index(drop=True)

plt.figure(figsize=(12,8))
plt.plot(df_f_l_t)

df_f_l_t.min()

df_f_l_t.max()

df_f_l_t = (df_f_l_t - df_f_l_t.min())/(df_f_l_t.max() - df_f_l_t.min())

df_f_l_t = po.DataFrame(df_f_l_t)

df_f_l_t.to_csv('data/processed/energy_consumption_trend_only.csv', index = False)

















# print(datetime.utcfromtimestamp(1483554358).strftime('%Y-%m-%d %H:%M:%S'))
#
# df_t['TS'][0]
#
# df_t['Date'] = df_t['TS'].apply(lambda row: datetime.utcfromtimestamp(math.trunc(row)).strftime('%Y-%m-%d %H:%M:%S').split(' ')[0])
# df_t['Time'] = df_t['TS'].apply(lambda row: datetime.utcfromtimestamp(math.trunc(row)).strftime('%Y-%m-%d %H:%M:%S').split(' ')[1])
#
# df_t['m'] = df_t['Time'].apply(lambda row: int(row.split(':')[1]))
# df_t['h'] = df_t['Time'].apply(lambda row: int(row.split(':')[0]))
#
# df_t['h:m'] = df_t['Time'].apply(lambda row: row.split(':')[0]+':'+row.split(':')[1]) 
# df_t.head()
#
# df_t.drop('TS', axis = 1, inplace=True)
#
# df_t.head()
#
# df_t['Date'].value_counts()
#
# gb_date = df_t.groupby(by = 'Date')
# dates = list(df_t['Date'].unique()) 
#
# dates
#
# #temp[temp['m_b'] == 55]
#
# temp.groupby(by = 'm_b').sum()
#
# temp['h'].value_counts()
#
# len(temp)
#
# po.cut(temp['h'], bins = range(24), labels = range(23)).value_counts()
#
# for d in dates: 
#     temp = gb_date.get_group(d)
#     #temp['m'] = temp['m'].astype(int)
#     #tem
#     temp['m_b'] = po.cut(temp['m'], bins = range(0, 61, 5), labels = range(0, 60, 5))
#     temp['m_b'] = po.cut(temp['m'], bins = range(0, 61, 5), labels = range(0, 60, 5))
#     gb_m_b = temp.groupby(by = 'm_b')
#     m_bs = list(range(0, 60, 5))
#     #for b in m_bs:
#         
#     
#     break
#
# temp
#
# df_t.groupby(by = 'h:m').mean()
#
#
