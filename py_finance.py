import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import pandas_datareader.data as web

# # Many methods to open data
# -------------------------
#
# style.use('ggplot')
# start = dt.datetime(2016,1,1)
# end = dt.datetime.now()
#
# df = web.DataReader('SWKS', 'yahoo', start, end) # opens the stock you want
# # print(df.head()) # first 5
# # print(df.tail()) # last five
#
# # As a CSV file
# df.to_csv('swks.csv') # converting dataframe into CSV or creating a CSV file
# df = pd.read_csv('swks.csv', parse_dates = True, index_col = 0) # turn CSV into data frame
# df.head()
#
# # -------------------------
#
# # Visualizing data
# # ----------------
#
# df['Adj Close'].plot() # specific data
# df.plot() # whole dataset
# plt.show()
#
# # ----------------
#
# # Stock Manipulation
# # ------------------
#
# df = pd.read_csv('asomy.csv', parse_dates=True, index_col=0)
# df['100ma'] = df['Adj Close'].rolling(window=100, min_periods = 0).mean()
# df.dropna(inplace=True) # removes missing values
# df.head()
#
# # graphing with matplotlib
# ax1 = plt.subplot2grid((6,1), (0,0), rowspan = 5, colspan = 1)
# ax2 = plt.subplot2grid((6,1), (5,0), rowspan = 1, colspan = 1, sharex=ax1)
#
# ax1.plot(df.index, df['Adj Close'])
# ax1.plot(df.index, df['100ma'])
# ax2.bar(df.index, df['Volume'])
#
# plt.show()

# Candle stick plot
from mpl_finance import candlestick_ohlc
import matplotlib.dates as mdates
df = pd.read_csv('uber.csv', parse_dates=True, index_col=0)

df_ohlc = df['Adj Close'].resample('10D').ohlc() # .resample('INPUT') e.g. INPUT = '10D' or '6Min' etc
df_volume = df['Volume'].resample('10D').sum()

df_ohlc.reset_index(inplace=True)
df_ohlc['Date'] = df_ohlc['Date'].map(mdates.date2num) # changes date format into python numbers i.e. 736312.0

ax1 = plt.subplot2grid((6,1), (0,0), rowspan = 5, colspan = 1)
ax2 = plt.subplot2grid((6,1), (5,0), rowspan = 1, colspan = 1, sharex=ax1)
ax1.xaxis_date()

candlestick_ohlc(ax1, df_ohlc.values, width=2, colorup='g')
ax2.fill_between(df_volume.index.map(mdates.date2num), df_volume.values, 0)

plt.show()

# Automating getting S&P 500 list
from collections import Counter
import bs4 as bs
import datetime as dt
import numpy as np
import os
import pickle
import requests

style.use('ggplot')

def save_sp500_tickers():
    resp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, "lxml")
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text.strip()
        tickers.append(ticker)

    with open("sp500tickers.pickle", "wb") as f:
        pickle.dump(tickers, f)
    return(tickers)
# save_sp500_tickers()

def get_data_from_yahoo(reload_sp500=False):

    if reload_sp500:
        tickers = save_sp500_tickers()
    else:
        with open("sp500tickers.pickle", "rb") as f:
            tickers = pickle.load(f)
            print('opening {}'.format('tickers'))

    if not os.path.exists('stock_dfs'):
        os.makedirs('stock_dfs')

    start = dt.datetime(2014, 1, 1)
    end = dt.datetime.now()

    failed_tickers = []

    for ticker in tickers: # tickers[:25] for first 25 tickers
        verify_ticker = os.path.exists('stock_dfs/{}.csv'.format(ticker))
        if not verify_ticker:
            try:
                df = web.DataReader(ticker, 'yahoo', start, end)
                df.to_csv('stock_dfs/{}.csv'.format(ticker))
            except KeyError: # can't find, ticker symbol is incorrect
                failed_tickers.append(ticker)
                print(ticker)
                print("Error: Cannot find, make sure ticker symbol is correct.")
        else:
            print('Already have {}'.format(ticker))
# get_data_from_yahoo()

def compile_data():
    with open("sp500tickers.pickle", "rb") as f:
        tickers = pickle.load(f)

    main_df = pd.DataFrame

    for count, ticker in enumerate(tickers):
        try:
            df = pd.read_csv('stock_dfs/{}.csv'.format(ticker))
            df.set_index('Date', inplace=True)
        except FileNotFoundError:  # can't find, ticker symbol is incorrect
            print(ticker)
            print("Error: File does not exist")
            continue

        df.rename(columns = {'Adj Close':ticker}, inplace=True)
        df.drop(['Open', 'High', 'Low', 'Close', 'Volume'], 1, inplace=True)

        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df, how='outer')

        if count % 10 == 0:
            print(count)

    print(main_df.head())
    main_df.to_csv('sp500_joined_closes.csv')
# compile_data()

def visualize_data():
    df = pd.read_csv('sp500_joined_closes.csv')
    # df['AAPL'].plot()
    # plt.show()
    df_corr = df.corr()
    print(df_corr.head())

    data = df_corr.values
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    heatmap = ax.pcolor(data, cmap=plt.cm.RdYlGn)
    fig.colorbar(heatmap)
    ax.set_xticks(np.arange(data.shape[0]) + 0.5, minor=False)
    ax.set_yticks(np.arange(data.shape[1]) + 0.5, minor=False)
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    column_labels = df_corr.columns
    row_labels = df_corr.index

    ax.set_xticklabels(column_labels)
    ax.set_yticklabels(row_labels)
    plt.xticks(rotation=90)
    heatmap.set_clim(-1, 1)
    plt.tight_layout()
    plt.show()
# visualize_data()

def process_data_for_labels(ticker):
    hm_days = 7
    df = pd.read_csv('sp500_joined_closes.csv', index_col=0)
    tickers = df.columns.values.tolist()
    df.fillna(0, inplace=True)

    for i in range(1, hm_days+1):
        df['{}_{}d'.format(ticker, i)] = (df[ticker].shift(-i) - df[ticker]) / df[ticker]

    df.fillna(0, inplace=True)
    return tickers, df
# process_data_for_labels('XOM')

def buy_sell_hold(*args):
    cols = [c for c in args]
    requirement = 0.02
    for col in cols:
        if col > 0.02685:
            return 1 # buy
        if col < -0.0265:
            return -1 # sell
    return 0 # hold

def extract_featuressets(ticker):
    hm_days = 7
    tickers, df = process_data_for_labels(ticker)

    # df['{}_target'.format(ticker)] = list(map(buy_sell_hold,
    #                                           df['{}_1d'.format(ticker)],
    #                                           df['{}_2d'.format(ticker)],
    #                                           df['{}_3d'.format(ticker)],
    #                                           df['{}_4d'.format(ticker)],
    #                                           df['{}_5d'.format(ticker)],
    #                                           df['{}_6d'.format(ticker)],
    #                                           df['{}_7d'.format(ticker)]
    #                                           ))
    df['{}_target'.format(ticker)] = list(map(buy_sell_hold,
                                              *[df['{}_{}d'.format(ticker, i)] for i in range(1, hm_days + 1)]))

    vals = df['{}_target'.format(ticker)].values.tolist()
    str_vals = [str(i) for i in vals]
    print('Data spread:', Counter(str_vals))

    df.fillna(0, inplace=True)
    df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)

    df_vals = df[[ticker for ticker in tickers]].pct_change()
    df_vals = df_vals.replace([np.inf, -np.inf], 0)
    df_vals.fillna(0, inplace=True)

    X = df_vals.values
    y = df['{}_target'.format(ticker)].values

    return X, y, df
# extract_featuressets('XOM')

# Machine learning
from sklearn import svm, neighbors
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier, RandomForestClassifier

def do_ml(ticker):
    X, y, df = extract_featuressets(ticker)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    #clf = neighbors.KNeighborsClassifier()
    clf = VotingClassifier([('lsvc', svm.LinearSVC()),
                            ('knn', neighbors.KNeighborsClassifier()),
                            ('rfor', RandomForestClassifier())])

    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)
    print('Accuracy: ', confidence)

    predictions = clf.predict(X_test)
    print('Predicted spread: ', Counter(predictions))

    return confidence
# do_ml('AMZN')
