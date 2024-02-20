import requests
import json
import pandas as pd
import datetime as dt

def df_column_switch(df, column1, column2):
    i = list(df.columns)
    a, b = i.index(column1), i.index(column2)
    i[b], i[a] = i[a], i[b]
    df = df[i]
    return df


def getbitstampdaily(sym = 'btcusd', start = "2017-08-17"):

    end = dt.datetime.now()
    dates = pd.date_range(start, end, freq = "1000D", inclusive="both")
    dates  = [ int(x.value/10**9) for x in list(dates)] 
    dates.append(dates[-1]+24*3600*1000)

    url = f'https://www.bitstamp.net/api/v2/ohlc/{sym}/'
    Data = []

    for S,E in zip(dates,dates[1:]):
        params = {
            "step": 86400,
            "limit":1000,
            "start":S,
            "end":E,
        }
        data = requests.get(url, params=params)
        
        data = data.json()["data"]["ohlc"]

        Data += data

    df = pd.DataFrame(Data)
    df = df.drop_duplicates()

    df["timestamp"] = df["timestamp"].astype(int)
    df.index = [x for x in df.timestamp]

    # df = df.sort_values(by="timestamp")
    # df = df[ df["timestamp"] >= dates[0] ]
    # df = df[ df["timestamp"] < dates[-1] ]

    L = df["timestamp"].values
    L = list(range(L[0],L[-1]+1, 86400))
    df = df.reindex(L,method="ffill")
    for i,a in enumerate(L) :
        L[i] = dt.datetime.fromtimestamp(a)
    df = df.assign(timestamp=L)
    print(df.columns)
    df = df_column_switch(df, 'close', "timestamp")
    df = df_column_switch(df, 'volume', 'close')
    df = df_column_switch(df, 'open', 'low')
    df = df_column_switch(df, 'open', 'high')


    return df



def getbitstamp(sym = 'btcusd', start = "2017-08-17"):

    end = dt.datetime.now()
    dates = pd.date_range(start, end, freq = "1000H", inclusive="both")
    dates  = [ int(x.value/10**9) for x in list(dates)] 
    dates.append(dates[-1]+3600*1000)

    url = f'https://www.bitstamp.net/api/v2/ohlc/{sym}/'
    Data = []

    for S,E in zip(dates,dates[1:]):
        params = {
            "step":3600,
            "limit":1000,
            "start":S,
            "end":E,
        }
        data = requests.get(url, params=params)
        
        data = data.json()["data"]["ohlc"]

        Data += data

    df = pd.DataFrame(Data)
    df = df.drop_duplicates()

    df["timestamp"] = df["timestamp"].astype(int)
    df.index = [x for x in df.timestamp]

    # df = df.sort_values(by="timestamp")
    # df = df[ df["timestamp"] >= dates[0] ]
    # df = df[ df["timestamp"] < dates[-1] ]

    L = df["timestamp"].values.astype(dt.datetime)
    # L = list(range(L[0],L[-1]+1, 3600))
    df = df.reindex(L,method="ffill")
    for i,a in enumerate(L) :
        L[i] = dt.datetime.fromtimestamp(a)
    df = df.assign(timestamp=L)
    print(df.columns)
    df = df_column_switch(df, 'close', "timestamp")
    df = df_column_switch(df, 'volume', 'close')
    df = df_column_switch(df, 'open', 'low')
    df = df_column_switch(df, 'open', 'high')


    return df

if __name__ == '__main__':

    df = getbitstampdaily()
    df.to_csv("BTC_BitDaily.csv", index=False)