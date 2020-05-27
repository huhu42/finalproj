
import pandas as pd
import numpy as np
import datetime as dt
from sklearn.preprocessing import StandardScaler

import time

import ftapp.core.constants as constants
import ftapp.core.indicators as get_indicator
import ftapp.core.learner as learner
import ftapp.core.utils as utils


timestamp = time.strftime('%Y%m%d%H%M')

class Learner_Strat(object):

    def __init__(self, verbose=False, impact=constants.IMPACT, bins=constants.BINS):
        self.verbose = verbose
        self.impact = impact
        self.bins = bins
        self.batch_size = 10
        self.scaler = None
        self.num_state = 0
        # initialize learner
        self.learner = learner.DQLearner(num_states=self.num_state, \
                                         num_actions=constants.NUM_ACTIONS, alpha=constants.ALPHA, rar=constants.RAR,
                                         radr=constants.RADR, gamma=constants.GAMMA)


    def get_scaler(self, ind):
        """ Takes a env and returns a scaler for its observation space """

        low = np.min(ind, axis=0)
        high = np.max(ind, axis=0)
        # print("low, high", low, high)

        scaler = StandardScaler()
        scaler.fit([low, high])
        return scaler

    def addEvidence(self, symbol,
                    sd,
                    ed,
                    sv=constants.SV):

        # get data first
        sd_e = sd - dt.timedelta(days=30)

        dates = pd.date_range(sd_e, ed)
        prices = utils.get_data([symbol], dates)
        prices.fillna(method='ffill', inplace=True)
        prices.fillna(method='bfill', inplace=True)

        # print("prices",prices)
        # compute indicator values
        prices_norm = prices / prices.iloc[0,]

        ind = get_indicator.indicators(prices[symbol])
        # bins = self.bins
        ind_disc = ind.loc[ind.index >= sd].dropna(axis=0, how='any')
        ind_disc = ind_disc.replace([np.inf, -np.inf], 0)

        print("final indicators table", ind_disc)

        # change indicators into array
        ind_array = ind_disc.to_numpy()

        # scale and transform indicators
        self.scaler = self.get_scaler(ind_array)

        self.num_state = len(ind_array[0])


        # table for holdings
        # action will be orders to place
        df_trades = pd.DataFrame(0, ind_disc.index, columns=["Prev", "Shares"])

        min_iteration = constants.MIN_ITER
        max_iteration = constants.MAX_ITER
        prices_sym = prices[symbol]
        daily_ret = prices_sym / prices_sym.shift(1) - 1
        port_val = pd.DataFrame(0, ind_disc.index, columns=['Daily_val'])

        iter = 0
        converged = False
        cum_rets = []

        while iter < max_iteration:

            for i in range(len(df_trades.index.values) - 1):
                # on the first day
                date = df_trades.index[i]

                # get new state
                state_ori = np.array(ind_disc.loc[date, :])
                print("or state", state_ori)
                state = self.scaler.transform([state_ori])
                # print ("scaled state", state, state.shape)
                state = np.reshape(state, [1, self.num_state])

                if i == 0:
                    order = 0
                    net_holdings = 0
                    # set initial state
                    self.learner.s = state
                    # print('shape',state.shape, state)
                    action = self.learner.act(state)
                    r = 0
                else:
                    # reward = daily return * the last action-1
                    print(daily_ret)
                    r = daily_ret.loc[date] * (net_holdings)
                    # print r
                    # get a new action
                    # print('shape', state.shape, state)
                    action = self.learner.act(state)

                # remember
                self.learner.remember(self.learner.s, action, r, state, converged)

                self.learner.s = state
                self.learner.a = action

                # experience replay
                if len(self.learner.memory) > constants.BATCH_SIZE:
                    self.learner.replay(constants.BATCH_SIZE)
                    # TODO: freeze target? only update every x iterations

                # epsilon decay
                if self.learner.rar > self.learner.rarm:
                    self.learner.rar = self.learner.rar * self.learner.radr

                # implement action the learner returned and update portfolio value
                order = action - 1
                # actions 0 = short, 1 = hold, 2 = long
                if net_holdings == -1000 and action == 2:
                    df_trades.iloc[i]["Shares"] = 2000
                    net_holdings += 2000
                elif net_holdings == 0 and action == 2:
                    df_trades.iloc[i]["Shares"] = 1000
                    net_holdings += 1000
                elif net_holdings == 0 and action == 0:
                    df_trades.iloc[i]["Shares"] = -1000
                    net_holdings += -1000
                elif net_holdings == 1000 and action == 0:
                    df_trades.iloc[i]["Shares"] = -2000
                    net_holdings += -2000

            trades_copy = df_trades.copy()
            trades_copy = trades_copy.filter(['Shares'], axis=1)
            trades_copy = trades_copy[(trades_copy.T != 0).any()]

            trades_copy.loc[:, 'Symbol'] = symbol
            trades_copy_b = trades_copy[trades_copy['Shares'] > 0]
            if len(trades_copy_b) > 0:
                print("trades copy b", trades_copy_b)
                trades_copy_b.loc[:, 'Order'] = "Buy"
            trades_copy_s = trades_copy[trades_copy['Shares'] < 0]
            if len(trades_copy_s) > 0:
                trades_copy_s.loc[:, 'Order'] = "Sell"
            trades = pd.concat([trades_copy_b, trades_copy_s])
            port_val = utils.compute_portvals(trades, sd, ed, impact=self.impact, commission=0.0, start_val=sv)

            cum_ret = port_val[-1] / port_val[0] - 1
            cum_rets.append(cum_ret)

            equal_check = df_trades['Prev'].equals(df_trades['Shares'])

            # if len(self.learner.memory) > batch_size:
            #    self.learner.replay(batch_size)

            if (iter + 1) % 10 == 0:  # checkpoint weights
                self.learner.save('{}/{}-dqn.h5'.format(constants.WEIGHTS_FOLDER, timestamp))

            if equal_check == True and iter >= min_iteration and cum_rets[iter] == cum_rets[iter - 1]:
                print("Converged in {} iterations".format(iter))
                converged = True
                self.learner.save('{}/{}.h5'.format(constants.WEIGHTS_FOLDER, symbol))
                self.scalar.save('{}/{}.h5'.format(constants.SCALER_FOLDER, symbol))
                break
            else:
                iter += 1
                # print df_trades[:5]
                df_trades['Prev'] = df_trades['Shares']
                # print df_trades[:5]
                df_trades['Shares'] = 0
                # print df_trades[:5]
                # print df_trades
        # print df_trades

        df_trades = df_trades.filter(['Shares'], axis=1)
        df_trades = df_trades[(df_trades.T != 0).any()]
        # print df_trades

        return df_trades

    def testPolicy(self, symbol, \
                   sd, \
                   ed, \
                   sv=constants.SV):

        sd_e = sd - dt.timedelta(days=30)
        dates = pd.date_range(sd_e, ed)
        prices = utils.get_data([symbol], dates)
        prices.fillna(method='ffill', inplace=True)
        prices.fillna(method='bfill', inplace=True)

        # compute indicator values
        prices_norm = prices / prices.iloc[0]
        ind = get_indicator.indicators(prices[symbol])

        # delete the early dates
        ind = ind.loc[ind.index >= sd]

        net_holdings = 0
        df_trades = pd.DataFrame(0, ind.index, columns=["Shares"])

        #if no scaler, get scaler, for when running pre-trained models
        if self.scaler == None:
            ind_array = ind.to_numpy()
            self.scaler = self.get_scaler(ind_array)


        for i in range(len(df_trades.index.values) - 1):
            date = df_trades.index[i]

            state_ori = np.array(ind.loc[date, :])
            # print("or state", state_ori)
            state = self.scaler.transform([state_ori])
            # print ("scaled state", state, state.shape)
            state = np.reshape(state, [1, self.num_state])

            # if state not NaN:
            # set state
            self.learner.s = state
            # take an action
            action = self.learner.act(state)
            if net_holdings == -1000 and action == 2:
                df_trades.iloc[i]["Shares"] = 2000
                net_holdings += 2000
            elif net_holdings == 0 and action == 2:
                df_trades.iloc[i]["Shares"] = 1000
                net_holdings += 1000
            elif net_holdings == 0 and action == 0:
                df_trades.iloc[i]["Shares"] = -1000
                net_holdings += -1000
            elif net_holdings == 1000 and action == 0:
                df_trades.iloc[i]["Shares"] = -2000
                net_holdings += -2000

        df_trades = df_trades.loc[(df_trades != 0).all(axis=1), :]
        # print df_trades

        return df_trades