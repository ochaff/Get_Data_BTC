from config import API_SECRET,API_KEY
from binance import Client
import datetime as dt
import pandas as pd
import numpy as np

def df_column_switch(df, column1, column2):
    i = list(df.columns)
    a, b = i.index(column1), i.index(column2)
    i[b], i[a] = i[a], i[b]
    df = df[i]
    return df

def getbinance(start = "18 Aug 2017", sym = "BTCUSDT"):
    client = Client(API_KEY, API_SECRET)
    klines = client.get_historical_klines(sym, Client.KLINE_INTERVAL_1HOUR, start)
    df = pd.DataFrame(klines)
    df = df.iloc[:,0:6]
    df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
    df.index = [dt.datetime.fromtimestamp(int(x)/1000) for x in df.date]

    L = df['date'].values
    L = list(range(L[0],L[-1]+1, 3600*1000))
    for i,a in enumerate(L) :
        L[i] = dt.datetime.fromtimestamp(a/1000)
    df = df.reindex(L,method="ffill")

    df = df.assign(date=L)
    df = df_column_switch(df, 'close', 'volume')
    df.drop(index=L[0], axis=0)
    df = df.iloc[1:]
    return df

def getbinancedaily(start = "18 Aug 2017", sym = "BTCUSDT"):
    client = Client(API_KEY, API_SECRET)
    klines = client.get_historical_klines(sym, Client.KLINE_INTERVAL_1DAY, start)
    df = pd.DataFrame(klines)
    df = df.iloc[:,0:6]
    df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
    df.index = [dt.datetime.fromtimestamp(int(x)/1000) for x in df.date]

    L = df['date'].values
    L = list(range(L[0],L[-1]+1, 24*3600*1000))
    for i,a in enumerate(L) :
        L[i] = dt.datetime.fromtimestamp(a/1000)
    df = df.reindex(L,method="ffill")

    df = df.assign(date=L)
    df = df_column_switch(df, 'close', 'volume')
    df.drop(index=L[0], axis=0)
    # df = df.iloc[1:]
    return df

def getbinanceweekly(start="18 Aug 2017", sym="BTCUSDT"):
    client = Client(API_KEY, API_SECRET)
    # Utiliser l'intervalle hebdomadaire ici
    klines = client.get_historical_klines(sym, Client.KLINE_INTERVAL_1WEEK, start)
    df = pd.DataFrame(klines)
    df = df.iloc[:, 0:6]
    df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
    df['date'] = pd.to_datetime(df['date'], unit='ms')

    return df

if __name__ == '__main__':
    df = getbinanceweekly()
    # df.to_csv('BTC_BinDaily.csv', index= False)
    df.to_csv('BTC_BinWeekly.csv', index= False)
