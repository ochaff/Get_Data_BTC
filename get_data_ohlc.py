import requests
import json
import pandas as pd
import datetime as dt
from config import API_SECRET,API_KEY
from binance import Client
from get_data_binance import getbinancedaily
from get_data_bitstamp import getbitstampdaily
from math import *
import numpy as np
from ts2vg import NaturalVG
import networkx as nx
import tsfracdiff
import pickle as pkl
from trend import fit_trendlines_high_low
from ta import add_all_ta_features
from ta.trend import ADXIndicator


if __name__ == '__main__' :
    dfBn = getbinancedaily()
    dfBt = getbitstampdaily()
    dfBn.to_csv('BTC_BinDaily.csv', index= False)
    dfBt.to_csv('BTC_BitDaily.csv', index= False)
    dfBn = pd.read_csv('BTC_BinDaily.csv')
    dfBt = pd.read_csv('BTC_BitDaily.csv')


    df = pd.DataFrame()

    df['open'] = (dfBn['open']*dfBn['volume']+dfBt['open']*dfBt['volume'])/(dfBt['volume']+dfBn['volume'])
    df['high'] = (dfBn['high']*dfBn['volume']+dfBt['high']*dfBt['volume'])/(dfBt['volume']+dfBn['volume'])
    df['low'] = (dfBn['low']*dfBn['volume']+dfBt['low']*dfBt['volume'])/(dfBt['volume']+dfBn['volume'])
    df['close'] = (dfBn['close']*dfBn['volume']+dfBt['close']*dfBt['volume'])/(dfBt['volume']+dfBn['volume'])
    df['date'] = dfBn['date']
    cols = ['date','open', 'high', 'low', 'close']
    df = df[cols]
    df.to_csv('BTC_Daily_ohlc.csv', index = False)