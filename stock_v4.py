import datetime

import pandas as pd
import yfinance
# import altair as alt
import streamlit as st
# import IPython.display
# import matplotlib.pyplot as plt
import numpy as np
import pandas_datareader
import sklearn
import sklearn.linear_model
import sklearn.model_selection

st.title('株価予測アプリ')

st.sidebar.write("""
# 
以下のオプションから表示日数を指定できます。
""")

st.sidebar.write("""
## 表示日数選択
""")

days = st.sidebar.slider('日数', 1, 3000, 1500)


tickers = {
            'SONY': '6758.T',
            'Skylark': '3197.T',
            'TOYOTA': '7203.T',
            'Toho': '9602.T',
            'Canon': '7751.T',
            'Panasonic': '6752.T'
}

def get_data(company):
    st.write(f"""
    # {company} 
    """)
    yfinance.pdr_override()
    df = pandas_datareader.data.get_data_yahoo(f'{tickers[company]}','2010-1-01')
    # df = pandas_datareader.data.get_data_yahoo('AAPL', '2010-1-01')

    # 機械学習(マシンラーニング)
    df['label'] = df['Close'].shift(-30)
    df['SMA'] = df['Close'].rolling(window=14).mean()
    df['Close'].plot(figsize=(15, 6), color="red")
    df['SMA'].plot(figsize=(15, 6), color="green")

    st.write(f"""
    ### ■　本日の株価
    """)
    st.write(df['Close'].tail(1))

    # st.write("### 株価 (USD)", df.tail(40))
    # ラベル行を削除したデーターをXに代入
    X = np.array(df.drop(['label', 'SMA'], axis=1))
    # 取りうる値の大小が著しく異なる特徴量を入れると結果が悪くなり、平均を引いて、標準偏差で割ってスケーリングする
    X = sklearn.preprocessing.scale(X)

    # 予測に使う過去30日間のデーター
    predict_data = X[-30:]
    # 過去30日を取り除いた入力データー
    X = X[:-30]
    y = np.array(df['label'])
    # 過去30日を取り除いた正解ラベル
    y = y[:-30]

    # 訓練データー80% 検証データー 20%に分ける
    # 第一引数に入力データー、第２引数に正解ラベルの配列
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, test_size=0.2)

    # 訓練データーを用いて学習する
    lr = sklearn.linear_model.LinearRegression()
    lr.fit(X_train, y_train)

    # 検証データーを用いて検証してみる
    accuracy = lr.score(X_test, y_test)
    st.write(accuracy)

    predicted_data = lr.predict(predict_data)


    df['Predict'] = np.nan

    last_date = df.iloc[-1].name

    one_day = 86400
    next_unix = last_date.timestamp() + one_day

    for data in predicted_data:
        next_date = datetime.datetime.fromtimestamp(next_unix)
        next_unix += one_day
        df.loc[next_date] = np.append([np.nan] * (len(df.columns) - 1), data)

    st.write(f"""
        ### ■　30日後日の予測株価
        """)
    st.write(df['Predict'].tail(1))

    times = ['Close', 'Predict']

    df = df[times]


    st.line_chart(
      df.tail(days)
    )

    return df


try:
    # st.sidebar.write("""
    # ## 株価の範囲指定
    # """)

    # ymin, ymax = st.sidebar.slider(
    #     '範囲を指定してください',
    #     0.0, 3500.0, (0.0, 3500.0)
    # )

    df = pd.DataFrame()

    companies = st.multiselect(
        '会社名を選択してください',
        list(tickers.keys()),
        ['Panasonic']
    )

    if not companies:
        st.error('少なくとも一社は選んでください')
    else:

        for company in companies:

             get_data(company)

except:
    st.error(
        "エラーが起きているようです．"
    )
