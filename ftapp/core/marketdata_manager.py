import os
import pandas as pd
pd.core.common.is_list_like = pd.api.types.is_list_like
import datetime as dt
from pandas_datareader import data as pdr

import ftapp.core.constants as constants
from ftapp import logger

from ftapp import db

INDEX_SYMBOLS = ['^GSPC', '^DJI', '^IXIC', 'HKHSI']

MARKETDATA_COLUMN_NAMES = [constants.OPEN_COLUMN, constants.HIGH_COLUMN, constants.LOW_COLUMN, constants.CLOSE_COLUMN, constants.VOLUME_COLUMN]

INPUT_DATE_FORMAT = '%m/%d/%Y'

def get_market_data(ticker, startDate, endDate):
    print('from {} to {}'.format(startDate, endDate))
    data = pdr.get_data_yahoo(ticker, start=startDate, end=endDate)
    return data

def get_yahoo_market_data(ticker, startDate, present):
    logger.info(f'get_market_data:{ticker} from {startDate} to {present}')
    print(startDate, present)

    all_data = pd.DataFrame()
    while startDate < present:
        endDate = startDate + dt.timedelta(days=150)
        if present < endDate:
            endDate = present
        start = startDate.strftime(constants.DATE_FORMAT)
        end = endDate.strftime(constants.DATE_FORMAT)

        try:
            data = get_market_data(ticker, start, end)
            if data.empty == False:
                del data['Adj Close']
                data = data[[constants.OPEN_COLUMN, constants.HIGH_COLUMN, constants.LOW_COLUMN, constants.CLOSE_COLUMN, constants.VOLUME_COLUMN]]
                if all_data.empty:
                    all_data = data
                else:
                    all_data = all_data.append(data)
        except ValueError as error:
            print(error)
            logger.info(f'get_market_data {ticker}: {error}')
        except KeyError as error:
            print(error)
            logger.info(f'get_market_data {ticker}: {error}')
        except Exception:
            print('Exception during getting data for ', ticker)
            logger.exception(f'Exceptiion for get_market_data {ticker}')
        startDate = endDate + dt.timedelta(days=1)

    if all_data.empty:
        log_info = f'Failed to get data for {ticker}'
        print(log_info)
        logger.info(log_info)
        return None
    else:
        filename = '{}/{}.csv'.format(constants.MARKETDATA_FOLDER, ticker)
        log_info = f'Save {ticker} data in {filename}'
        print(log_info)
        logger.info(log_info)
        all_data.to_csv(filename)
        return ticker

def get_yahoo_index_return(start, end):
    tickers = ['^GSPC', '^DJI', '^IXIC']
    price_data = pdr.get_data_yahoo(tickers, start=start, end=end)
    price_data = price_data['Close']
    ret_data = price_data.pct_change()[1:]
    print(ret_data.head(1))
    return ret_data


def download_marketdata( ticker, fromdate, todate):
    print(fromdate)
    print(todate)
    get_yahoo_market_data(constants.BENCHMARK, fromdate, todate)
    get_yahoo_market_data(ticker, fromdate, todate)