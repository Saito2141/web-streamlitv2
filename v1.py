import pandas as pd
import yfinance as yf
import altair as alt
import streamlit as st

def get_data(days, tickers):
    df = pd.DataFrame()
    for company in tickers.keys():
        tkr = yf.Ticker(tickers[company])
        hist = tkr.history(period=f'{days}d')
        hist = hist[['Close']]
        hist.columns = [company]
        hist = hist.T
        hist.index.name = 'Name'
        df = pd.concat([df, hist])
    return df

days = 20
tickers = {
    'apple': 'AAPL',
    'google' : 'GOOGL',
    'netflix' : 'NFLX',
    'amazon' : 'AMZN'
}

print(get_data(days,tickers))

