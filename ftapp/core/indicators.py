import numpy as np
import pandas as pd

#helper functions to calculate the indicator
def sma(prices, days):
    sma = prices.rolling(window = days).mean()
    return sma

def stdev(prices, days):
    std = prices.rolling(window = days).std()
    return std

def eva(prices, days):
    weights = np.exp(np.linspace(-1, 0., days))
    weights /= weights.sum()

    eva = np.convolve(prices, weights)
    return eva


def indicators(prices):
    days = 20
    width = 2
    days_long = 26
    days_short = 12

    ind = pd.DataFrame(0, prices.index,
                       columns=['sma', 'sma_ratio', 'std', 'bollinger_v', 'eva_long', 'eva_short', 'MACD', 'signal'])

    ind['sma'] = sma(prices, days)
    # indicator #1: sma_ratio
    ind['sma_ratio'] = prices / ind['sma']
    ind['std'] = stdev(prices, days)

    # how many std away is price from sma
    ind['bollinger_v'] = (prices - ind['sma']) / ind['std']

    ind['eva_long'] = prices.ewm(days_long, adjust=False).mean()
    ind['eva_short'] = prices.ewm(days_short, adjust=False).mean()
    # indicator 3: MACD
    ind['MACD'] = ind['eva_long'] - ind['eva_short']
    # indicator 4: signal
    ind['signal'] = ind['MACD'].ewm(span=9).mean()

    ind = ind.drop(['sma', 'std', 'eva_short', 'eva_long'], axis=1)

    return ind