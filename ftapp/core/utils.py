import pandas as pd
import numpy as np
import ftapp.core.constants as constants
import pickle
import time
import matplotlib.pyplot as plt
import io
import base64
timestamp = time.strftime('%Y%m%d%H%M')
import datetime as dt
import os

from flask import Flask

from ftapp import logger
import seaborn as sns
from ftapp import db
from factor_analyzer import FactorAnalyzer
import traceback
from sklearn.model_selection import GridSearchCV
from yellowbrick.classifier import ClassificationReport

#models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import scale
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

from sklearn.model_selection import train_test_split
#train test split
from definitions import ROOT_DIR

def split_data(x,y, split):
    #remove FB from the list
    fb_idx = x.index.get_loc('FB')
    fb_x = x.loc['FB']
    x.drop(['FB'])
    print("y",y)
    print("fb_idx", fb_idx)
    fb_y = y.iloc[fb_idx]
    y.drop([fb_idx])
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(x, y, x.index, test_size=split)

    X_test = X_test.append(fb_x)
    y_test = y_test.append(fb_y)
    print("idx_test", idx_test)
    idx_test = idx_test.append(pd.Index(['FB']))
    print("idx_test")
    for each in idx_test:
        if each =='FB':
            print("YES ADDED FB to idx")
    for each in X_test.index:
        if each =='FB':
            print("YES ADDED FB to test")


    return X_train, X_test, y_train, y_test, idx_train, idx_test

def get_data(symbols, dates, addSPY=True, colname = constants.CLOSE_COLUMN):
    '''

    :param symbols:
    :param dates:
    :param addSPY:
    :param colname:
    :return: data df
    '''

    df = pd.DataFrame(index=dates)
    # add SPY if not already in symbols
    if addSPY and 'SPY' not in symbols:
        symbols = ['SPY'] + symbols

        # loop over all the stocks in the list and add to the column
    for symbol in symbols:
        file_name = os.path.join(ROOT_DIR, constants.MARKETDATA_FOLDER, symbol+'.csv')

        df_temp = pd.read_csv(file_name, index_col='Date',
                              parse_dates=True, usecols=['Date', colname], na_values=['nan'])
        df_temp = df_temp.rename(columns={colname: symbol})
        df = df.join(df_temp)
        # filter out the days where SPY is not available - non-trading days
        if symbol == 'SPY':
            df = df.dropna(subset=["SPY"])

    return df


# helper function to compute the portfolio values
def compute_portvals(orders, start_date, end_date, start_val=constants.SV, commission=constants.COMMISSION, impact=constants.IMPACT, ):


    orders = orders.sort_index()
    stock_list = list(set(orders["Symbol"]))

    print("portval start, end", start_date, end_date)


    dates = pd.date_range(start_date, end_date)

    price = get_data(stock_list, dates)

    price.fillna(method='ffill', inplace=True)
    price.fillna(method='bfill', inplace=True)

    price["cash"] = 1.0

    trades = pd.DataFrame(np.zeros((price.shape)), price.index, price.columns)

    mask = (orders['Order'] == "SELL")
    orders['Shares'] = orders['Shares'].mask(mask, -1 * orders["Shares"])

    for i, row in orders.iterrows():
        try:
            trades.loc[i, row["Symbol"]] = trades.loc[i, row["Symbol"]] + row["Shares"]
            trades.loc[i, 'cash'] = trades.loc[i, 'cash'] - row["Shares"] * price.loc[
                i, row["Symbol"]] - commission - abs(row["Shares"]) * price.loc[i, row["Symbol"]] * impact
        except:
            pass

    holdings = pd.DataFrame(np.zeros((price.shape)), price.index, price.columns)
    print(holdings)

    holdings.iloc[0, :-1] = trades.iloc[0, :-1].copy()
    holdings.iloc[0, -1] = trades.iloc[0, -1] + start_val

    for i in range(1, len(trades.index.values)):
        holdings.iloc[i] = holdings.iloc[i - 1] + trades.iloc[i]
    values = holdings * price

    values['portval'] = values.sum(axis=1)
    portval = values['portval']

    return portval

def generate_data(df_trades, symbol, start, end, sv, commission, impact, mode):
    # run model and generate data for graph
    '''
    Return:
    - portvals
    - trades
    - benchmark

    '''
    print(df_trades)
    df_b = df_trades.loc[df_trades.iloc[:, 0] > 0]
    if len(df_b) > 0:
        df_b.loc[:, 'Order'] = "Buy"

    df_s = df_trades.loc[df_trades['Shares'] < 0]
    if len(df_s) > 0:
        df_s.loc[:, 'Order'] = "Sell"
    trades = pd.concat([df_b, df_s])
    trades = trades.sort_index()

    portvals = compute_portvals(trades, start, end, start_val=sv, commission=commission, impact=impact)
    print(portvals)

    if isinstance(portvals, pd.DataFrame):
        portvals_test = portvals[portvals.columns[0]]  # just get the first column
    else:
        "warning, code did not return a DataFrame"

    dates = pd.date_range(start, end)

    # print portvals
    price = get_data([symbol], dates)[symbol]

    benchmark = pd.DataFrame(0, portvals.index, columns=["Benchmark_Return"])
    cash = sv - price[0] * 1000
    benchmark = cash + price * 1000
    benchmark[0] = sv

    #

    # save portfolio value history to disk
    with open('{}/{}-{}.p'.format(constants.PORTVAL_FOLDER,timestamp, mode), 'wb') as fp:
        pickle.dump(portvals, fp)

    print(portvals)
    return portvals, benchmark





#helper function to get the stats
def compute_stat(prices, rfr = constants.RFR):
    daily_rets = (prices/prices.shift(1))-1
    #print portvals
    #print portvals.shift(1)
    daily_rets = daily_rets[1:]
    cr = (prices[-1]/prices[0])-1
    adr = daily_rets.mean()
    sddr = daily_rets.std()
    sr = np.sqrt(252) * (daily_rets - rfr).mean() / sddr

    return cr, adr, sddr, sr

def get_IPO_dates(ticker):
    '''

    :param ticker: given ticker
    :return: str dates
    '''
    df = pd.read_csv('{}/{}.csv'.format(constants.USERDATA_FOLDER, constants.IPO_DATA), index_col = 0)

    ticker_dates = df.loc[ticker]
    ipo = dt.datetime.strptime(ticker_dates[0], '%m/%d/%Y')
    lockup = dt.datetime.strptime(ticker_dates[1], '%m/%d/%Y')
    return ipo, lockup

def print_stats(portvals, benchmark,name,start, end, rfr = constants.RFR):

    #compute stats
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = compute_stat(portvals, rfr=0.0)
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = compute_stat(benchmark, rfr=0.0)

    #print out results
    print ("%s Dataset"%(name))
    print ("Date Range: {} to {}".format(start, end))
    print ()
    print ("Sharpe Ratio of Fund: {}".format(sharpe_ratio))
    print ("Sharpe Ratio of Buy & Hold : {}".format(sharpe_ratio_SPY))
    print()
    print ("Cumulative Return of Fund: {}".format(cum_ret))
    print ("Cumulative Return of Buy & Hold : {}".format(cum_ret_SPY))
    print()
    print ("Standard Deviation of Fund: {}".format(std_daily_ret))
    print ("Standard Deviation of Buy & Hold : {}".format(std_daily_ret_SPY))
    print()
    print ("Average Daily Return of Fund: {}".format(avg_daily_ret))
    print ("Average Daily Return of Buy & Hold : {}".format(avg_daily_ret_SPY))
    print()
    print ("Final Portfolio Value: {}".format(portvals[-1]))
    return None


app = Flask(__name__)
UPLOAD_FOLDER = './user_data'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
def load_screener_data( inputFile):

    log_info = f'Start to upload screening factors with {inputFile}'
    print(log_info)
    print(log_info)
    #filename = f'{constants.USERINPUT_FOLDER}/{inputFile}'
    try:
        filename =os.path.join(app.config['UPLOAD_FOLDER'],inputFile)
        print("filename", filename)
        input = pd.read_csv(filename, index_col=0)
        print("input", input.head())
        print("load data success")
        '''if y_col ==-1:
            y = input[y_col]
            buy_info = BuyInfo.query.filter(BuyInfo.indicator_id==indicator_id, BuyInfo.symbol==symbol, BuyInfo.buydate==dbDate).first()
            if (not buy_info and math.isnan(float(buyprice)) == False and buyprice > 0.0001):
                buyinfo = BuyInfo(buydate=dbDate, symbol=symbol, buyprice=buyprice, openprice=openprice, closeprice=closeprice, highprice=highprice, lowprice=lowprice, indicator_id=indicator_id)
                db.session.add(buyinfo)

        db.session.commit()
        logger.info('Finish to upload buy indicator')'''
        return input

    except Exception:
        print("Exception for initialing ")
        logger.exception(f'Exceptiion for initialing test')
        traceback.print_exc()


def clean_data(input, cat_cols, num_cols):

    #sring split
    print("cat cols", cat_cols)
    print("num_cols", num_cols)
    if cat_cols =='0':
        pass
    elif cat_cols == '-1':
        cat_cols = input.columns.drop(num_cols)
        for each in cat_cols:
            print(input[each])
            input[each] = pd.factorize(input[each])[0]
    else:
        x = cat_cols.split(',')
        print(x)
        for each in x:
            print(input[each])
            input[each] = pd.factorize(input[each])[0]

    if num_cols =='0':
        pass
    elif num_cols == '-1':
        num_cols = input.columns.drop(x)
        input[num_cols] = input[num_cols].apply(pd.to_numeric, errors='coerce')
    else:
        num_cols = num_cols.split(',')
        print(num_cols)
        input[num_cols] = input[num_cols].apply(pd.to_numeric, errors='coerce')
    return input


# loadings
def discretize_y(X_tensor, Y_matrix, upper_percentile=60, lower_percentile=40):
    img = io.BytesIO()

    plt.switch_backend('Agg')

    plt.style.use('ggplot')
    n_stocks, n_factors = np.shape(X_tensor)
    print("Y", Y_matrix)
    upper = np.percentile(Y_matrix, upper_percentile)
    lower = np.percentile(Y_matrix, lower_percentile)

    Y_binary = np.zeros(n_stocks)

    for i in range(n_stocks):
        print("value", Y_matrix.iloc[i])
        if Y_matrix.iloc[i] > upper:
            Y_binary[i] = 1
        elif Y_matrix.iloc[i] <= lower:
            Y_binary[i] = -1

    #print(Y_binary)
    return Y_binary





def create_graph(portvals, benchmark, trades_df, name, save = False, show = True):
    #print(benchmark)
    img = io.BytesIO()

    plt.switch_backend('Agg')

    plt.style.use('ggplot')


    b_temp = benchmark / benchmark.iloc[0]
    pv_temp = portvals / portvals.iloc[0]
    plt.plot(b_temp, label='Benchmark (Buy & Hold)', color="g")
    plt.plot(pv_temp, label='Portfolio', color="r")
    for i in range(len(trades_df)):
        if trades_df.iloc[i]["Shares"] > 0:
            plt.axvline(x=trades_df.index[i], color='blue')
        elif trades_df.iloc[i]["Shares"] < 0:
            plt.axvline(x=trades_df.index[i], color='black')
    name = name
    plt.xlabel('Date')
    plt.xticks(rotation=70)
    plt.ylabel('Normalized Daily Value')
    plt.title('Q-Learner vs Benchmark Performance - ' + str(name))
    plt.legend()
    #if show:
    #    plt.show()
    #if save:
        #plt.savefig('%s Q-Learner Performance.png'%(name))

    plt.savefig(img, format='png')

    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()
    plt.close()

    return 'data:image/png;base64,{}'.format(graph_url)

def save(x,y, screener_name):
    x.to_csv('./user_data/{}-x.csv'.format(screener_name))
    pd.DataFrame(y).to_csv('./user_data/{}-y.csv'.format(screener_name))

def eigenvalues_plt(data):
    img = io.BytesIO()

    plt.switch_backend('Agg')

    plt.style.use('ggplot')
    fa = FactorAnalyzer()
    fa.fit(data)
    eigen_values, vectors = fa.get_eigenvalues()
    plt.figure(figsize=(10, 10))
    plt.scatter(range(1, data.shape[1] + 1), eigen_values)
    plt.plot(range(1, data.shape[1] + 1), eigen_values)
    plt.title('Factor Importance by Eigenvalues')

    plt.xlabel('Factors')
    plt.ylabel('Eigenvalue')
    plt.grid()

    plt.savefig(img, format='png')

    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()
    plt.close()

    return 'data:image/png;base64,{}'.format(graph_url)

def corr_matrix(data):
    img = io.BytesIO()

    plt.switch_backend('Agg')

    plt.style.use('ggplot')
    data.columns = ["V"+str(i) for i in range(1, len(data.columns)+1)]  # rename column names to be similar to R naming convention
    # financials.V1 = financials.V1.astype(str)
    # X = financials  # independent variables data

    corrmat = data.corr()
    plt.figure(figsize=(10, 10))
    sns.heatmap(corrmat, vmax=1., square=False, cmap="YlGnBu").xaxis.tick_top()
    plt.title('Correlation Matrix')
    plt.savefig(img, format='png')

    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()
    plt.close()

    return 'data:image/png;base64,{}'.format(graph_url)

def mosthighlycorrelated(mydataframe, numtoreport):
    # find the correlations
    cormatrix = mydataframe.corr()
    # set the correlations on the diagonal or lower triangle to zero,
    # so they will not be reported as the highest ones:
    cormatrix *= np.tri(*cormatrix.values.shape, k=-1).T
    # find the top n correlations
    cormatrix = cormatrix.stack()
    cormatrix = cormatrix.reindex(cormatrix.abs().sort_values(ascending=False).index).reset_index()
    # assign human-friendly names
    cormatrix.columns = ["FirstVariable", "SecondVariable", "Correlation"]
    return cormatrix.head(numtoreport)

def standardize(x):
    standardisedX = scale(x)
    standardisedX = pd.DataFrame(standardisedX, index=x.index, columns=x.columns)
    standardisedX.apply(np.mean)
    standardisedX.apply(np.std)
    return standardisedX

def pca_summary(pca, standardised_data):
    names = ["PC"+str(i) for i in range(1, len(pca.explained_variance_ratio_)+1)]
    a = list(np.std(pca.transform(standardised_data), axis=0))
    b = list(pca.explained_variance_ratio_)
    c = [np.sum(pca.explained_variance_ratio_[:i]) for i in range(1, len(pca.explained_variance_ratio_)+1)]
    columns = pd.MultiIndex.from_tuples([("sdev", "Standard deviation"), ("varprop", "Proportion of Variance"), ("cumprop", "Cumulative Proportion")])
    summary = pd.DataFrame(zip(a, b, c), index=names, columns=columns)
    return summary

param_grid = {"Decision Tree": [
  {'n_estimators': [7,8,9,10,11,100,400,1000],
   'algorithm':['SAMME.R','SAMME' ],
   'learning_rate': [0.05,0.1,0.2,0.3, 0.4,0.8,1.0]}
 ],
    "Random Forest": [{'n_estimators': [700, 800, 900],
    'max_features': ['auto', 'sqrt', 'log2'],
    'criterion': ['gini', 'entropy'],
    'min_samples_split':[2,3,4],
    'max_depth':[None,10,20]
}],
    "Ada Boost": [
  {'n_estimators': [7,8,9,10,11,100,400,1000],'algorithm':['SAMME.R','SAMME' ],'learning_rate': [0.05,0.1,0.2,0.3, 0.4,0.8,1.0]}
 ],
    "Neural Network": [
  {'fit_intercept':[True, False],'normalize':[True, False]}
 ]
}

choices=[(1,'Decision Tree'),(2,'Ada Boost'), (3,'Neural Network'), (4,'Random Forest')]

models = [DecisionTreeClassifier(),\
          AdaBoostClassifier(n_estimators=100, learning_rate=0.6, random_state=0),\
          LinearRegression(), \
          RandomForestClassifier(max_features= 'sqrt' ,n_estimators=700, oob_score = True) ]


def evaluate(model, X_train, X_test, y_train, y_test, optimize =False, cv = 5):
    print("model", int(model))
    clf = models[int(model)-1]
    if optimize:
        clf = GridSearchCV(clf, param_grid[model], cv=cv)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    score = clf.score(X_test, y_test)

    return score, y_pred, clf


def confusion_matrix(classifier, x,y):
    img = io.BytesIO()

    plt.switch_backend('Agg')

    plt.style.use('ggplot')
    plt.figure(figsize=(10, 10))
    print(classifier)

    plot_confusion_matrix(classifier, x, y, cmap='YlGnBu')

    plt.title('Correlation Matrix')
    plt.savefig(img, format='png')

    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()
    plt.close()

    return 'data:image/png;base64,{}'.format(graph_url)


def classificationreport(clf, classes, X_train, y_train, X_test, y_test):
    #classes = ['increase','little change', 'decrease']
    img = io.BytesIO()

    #plt.switch_backend('Agg')

    #plt.style.use('ggplot')


    visualizer = ClassificationReport(clf, classes=classes, support=True)

    visualizer.fit(X_train, y_train)        # Fit the visualizer and the model
    visualizer.score(X_test, y_test)        # Evaluate the model on the test data
    visualizer.show(outpath = img)                       # Finalize and show the figure
    plt.figure(figsize=(8, 8))

    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()
    return 'data:image/png;base64,{}'.format(graph_url)


def get_stocks_classification(y, index):
    increase = []
    stay = []
    decrease = []
    for i in range(len(y)):
        if y[i]==-1:
            decrease.append(index[i])
        elif y[i]==1:
            increase.append(index[i])
        else:
            stay.append(index[i])
    return increase, stay, decrease