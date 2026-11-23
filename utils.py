import os
import pandas as pd
import numpy as np
import yfinance as yf

TW50_CACHE = "tw50_cache.csv"

def load_or_update_tw50(start_hist="2025-01-01"):
    if os.path.exists(TW50_CACHE):
        tw50 = pd.read_csv(TW50_CACHE)
        tw50['Date'] = pd.to_datetime(tw50['Date'])
        last_d = tw50['Date'].max()
        tw_start = (last_d + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        if pd.to_datetime(tw_start) <= pd.Timestamp.today():
            new_tw = yf.download("0050.TW", start=tw_start, progress=False, auto_adjust=True, group_by='ticker')
            if not new_tw.empty:
                new_tw = new_tw.reset_index()
                if isinstance(new_tw.columns, pd.MultiIndex):
                    new_tw.columns = [col[1] if col[1] != '' else col[0] for col in new_tw.columns]
                new_tw['Date'] = pd.to_datetime(new_tw['Date'])
                tw50 = pd.concat([tw50, new_tw], ignore_index=True).drop_duplicates(subset=['Date'], keep='last')
    else:
        tw50 = yf.download("0050.TW", start=start_hist, progress=False, auto_adjust=True, group_by='ticker').reset_index()
        if isinstance(tw50.columns, pd.MultiIndex):
            tw50.columns = [col[1] if col[1] != '' else col[0] for col in tw50.columns]
        tw50['Date'] = pd.to_datetime(tw50['Date'])

    if isinstance(tw50.columns, pd.MultiIndex):
        tw50.columns = [col[1] if col[1] != '' else col[0] for col in tw50.columns]
    tw50['Date'] = pd.to_datetime(tw50['Date'])
    tw50['大盤_20日均線'] = tw50['Close'].rolling(20).mean()
    tw50['大盤_20日乖離率'] = (tw50['Close'] - tw50['大盤_20日均線']) / (tw50['大盤_20日均線'] + 1e-8)
    tw50['大盤_5日漲跌幅'] = tw50['Close'].pct_change(5)
    tw50.to_csv(TW50_CACHE, index=False, encoding="utf-8-sig")
    return tw50

def build_features(df):
    g = df.groupby('stock_id')

    df['5日均量'] = g['Volume_Size'].transform(lambda x: x.rolling(5).mean())
    df['當日漲跌幅'] = (df['Close_Price'] - df['Open_Price']) / (df['Open_Price'] + 1e-8)
    df['當日振幅'] = (df['High_Price'] - df['Low_Price']) / (df['Open_Price'] + 1e-8)
    df['5日乖離率'] = (df['Close_Price'] - df['MA_5']) / (df['MA_5'] + 1e-8)
    df['20日乖離率'] = (df['Close_Price'] - df['MA_20']) / (df['MA_20'] + 1e-8)
    df['均線差比率'] = (df['MA_5'] - df['MA_20']) / (df['MA_20'] + 1e-8)
    df['量比'] = df['Volume_Size'] / (df['5日均量'] + 1e-8)

    df['過去5日漲跌幅'] = g['Close_Price'].pct_change(5)
    df['過去5日波動率'] = g['當日漲跌幅'].transform(lambda x: x.rolling(5).std())

    low_20 = g['Low_Price'].transform(lambda x: x.rolling(20).min())
    high_20 = g['High_Price'].transform(lambda x: x.rolling(20).max())
    df['20日價格位置'] = (df['Close_Price'] - low_20) / (high_20 - low_20 + 1e-8)

    def calc_rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
        rs = gain / (loss + 1e-8)
        return 100 - (100 / (1 + rs))

    df['RSI_14'] = g['Close_Price'].transform(lambda x: calc_rsi(x, 14))

    df['Next_Open'] = g['Open_Price'].shift(-1)
    df['Future_Close'] = g['Close_Price'].shift(-6)
    df['Real_Return'] = (df['Future_Close'] - df['Next_Open']) / (df['Next_Open'] + 1e-8)
    df['Target'] = np.where(df['Real_Return'] >= 0.03, 1, 0)
    return df