# https://www.pythonforfinance.net/2018/07/04/mean-reversion-pairs-trading-with-inclusion-of-a-kalman-filter/
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl

mpl.style.use('bmh')
import pandas_datareader.data as web
import matplotlib.pylab as plt
from datetime import datetime
import statsmodels.api as sm
from pykalman import KalmanFilter
from math import sqrt
import functions.functions as base_fx
import functions.portfolio_functions as portfolio_fx


def KalmanFilterAverage(x):
    # Construct a Kalman filter
    kf = KalmanFilter(transition_matrices=[1],
                      observation_matrices=[1],
                      initial_state_mean=0,
                      initial_state_covariance=1,
                      observation_covariance=1,
                      transition_covariance=.01)
    # Use the observed values of the price to get a rolling mean
    state_means, _ = kf.filter(x.values)
    state_means = pd.Series(state_means.flatten(), index=x.index)
    return state_means


# Kalman filter regression
def KalmanFilterRegression(x, y):
    delta = 1e-3
    trans_cov = delta / (1 - delta) * np.eye(2)  # How much random walk wiggles
    obs_mat = np.expand_dims(np.vstack([[x], [np.ones(len(x))]]).T, axis=1)
    kf = KalmanFilter(n_dim_obs=1, n_dim_state=2,  # y is 1-dimensional, (alpha, beta) is 2-dimensional
                      initial_state_mean=[0, 0],
                      initial_state_covariance=np.ones((2, 2)),
                      transition_matrices=np.eye(2),
                      observation_matrices=obs_mat,
                      observation_covariance=2,
                      transition_covariance=trans_cov)
    # Use the observations y to get running estimates and errors for the state parameters
    state_means, state_covs = kf.filter(y.values)
    return state_means


def half_life(spread):
    spread_lag = spread.shift(1)
    spread_lag.iloc[0] = spread_lag.iloc[1]
    spread_ret = spread - spread_lag
    spread_ret.iloc[0] = spread_ret.iloc[1]
    spread_lag2 = sm.add_constant(spread_lag)
    model = sm.OLS(spread_ret, spread_lag2)
    res = model.fit()
    halflife = int(round(-np.log(2) / res.params[1], 0))
    if halflife <= 0:
        halflife = 1
    return halflife


def backtest(df, symbol1, symbol2):
    #############################################################
    # INPUT:
    # DataFrame of prices
    # s1: the symbol of contract one
    # s2: the symbol of contract two
    # x: the price series of contract one
    # y: the price series of contract two
    # OUTPUT:
    # df1['cum rets']: cumulative returns in pandas data frame
    # sharpe: Sharpe ratio
    # CAGR: Compound Annual Growth Rate
    portfolio = portfolio_fx.init_portfolio()
    val_df = pd.DataFrame(index=df.index, columns=['hr','spread','zscore'])
    for day in df.index:
        quotes = df[:day]
        if len(quotes) < 100:
            continue

        x = quotes[symbol1]
        y = quotes[symbol2]

        # run regression (including Kalaman Filter) to find hedge ratio and then create spread series
        df1 = pd.DataFrame({'y': y, 'x': x})
        df1.index = pd.to_datetime(df1.index)

        state_means = KalmanFilterRegression(KalmanFilterAverage(x), KalmanFilterAverage(y))
        df1['hr'] = - state_means[:, 0]  ## Hedge Ratio
        val_df.loc[day]['hr'] = df1['hr'][-1]

        df1['spread'] = df1.y + (df1.x * df1.hr)  ## Spread
        val_df.loc[day]['spread'] = df1['spread'][-1]

        halflife = half_life(df1['spread'])  ## Half life

        # calculate z-score with window = half life period
        meanSpread = df1.spread.rolling(window=halflife).mean()
        stdSpread = df1.spread.rolling(window=halflife).std()
        df1['zScore'] = (df1.spread - meanSpread) / stdSpread
        val_df.loc[day]['zscore'] = df1['zScore'][-1]

        ###### Trade ######
        entryZscore = 2
        exitZscore = 0

        # set up num units long
        if (df1.loc[day]['zScore'] > entryZscore):
            if portfolio['holdings']['s1'] is None and portfolio['holdings']['s2'] is None:
                #print(f'sell {day}')
                ## sell
                portfolio = portfolio_fx.buy(day, symbol1, x.loc[day], symbol2, y.loc[day],
                                             df1.zScore[day],
                                             1, 1, portfolio)

        elif (df1.loc[day]['zScore'] < - entryZscore):
            if portfolio['holdings']['s1'] is None and portfolio['holdings']['s2'] is None:
               # print(f'buy {day}')

                ## buy
               portfolio = portfolio_fx.sell(day, symbol1, x.loc[day], symbol2, y.loc[day],
                                             df1.zScore[day],
                                             1, 1, portfolio)


       # ((df1.zScore > - exitZscore) & (df1.zScore.shift(1) < - exitZscore))
        elif (df1.loc[day]['zScore'] < abs(exitZscore)):
            if portfolio['holdings']['s1'] != None and portfolio['holdings']['s2'] != None:
                ## exit
               # print(f'exit {day}')
                portfolio = portfolio_fx.exit(day, symbol1, x.loc[day], symbol2, y.loc[day],
                                                df1.zScore[day],
                                                1, 1, portfolio)

        portfolio = portfolio_fx.calculate_nav(day, portfolio, x.loc[day], y.loc[day])
      #  print(portfolio['nav'].tail(1))
    return portfolio, df1, val_df

        # # df1['long entry'] = ((df1.zScore < - entryZscore) & (df1.zScore.shift(1) > - entryZscore))
        # # df1['long exit'] = ((df1.zScore > - exitZscore) & (df1.zScore.shift(1) < - exitZscore))
        # #
        # # # print(df1)
        # # df1['num units long'] = np.nan
        # # df1.loc[df1['long entry'], 'num units long'] = 1
        # # df1.loc[df1['long exit'], 'num units long'] = 0
        # # df1['num units long'][0] = 0
        # # df1['num units long'] = df1['num units long'].fillna(method='pad')
        # # # set up num units short
        # # df1['short entry'] = ((df1.zScore > entryZscore) & (df1.zScore.shift(1) < entryZscore))
        # # df1['short exit'] = ((df1.zScore < exitZscore) & (df1.zScore.shift(1) > exitZscore))
        # # df1.loc[df1['short entry'], 'num units short'] = -1
        # # df1.loc[df1['short exit'], 'num units short'] = 0
        # # df1['num units short'][0] = 0
        # # df1['num units short'] = df1['num units short'].fillna(method='pad')
        # # df1['numUnits'] = df1['num units long'] + df1['num units short']
        # # df1['spread pct ch'] = (df1['spread'] - df1['spread'].shift(1)) / ((df1['x'] * abs(df1['hr'])) + df1['y'])
        # # df1['port rets'] = df1['spread pct ch'] * df1['numUnits'].shift(1)
        # # df1['cum rets'] = df1['port rets'].cumsum()
        # # df1['cum rets'] = df1['cum rets'] + 1
        #
        # ##############################################################
        # try:
        #     sharpe = ((df1['port rets'].mean() / df1['port rets'].std()) * sqrt(252))
        # except ZeroDivisionError:
        #     sharpe = 0.0
        # ##############################################################
        # start_val = 1
        # end_val = df1['cum rets'].iat[-1]
        # start_date = df1.iloc[0].name
        # end_date = df1.iloc[-1].name
        # days = (end_date - start_date).days
        # CAGR = round(((float(end_val) / float(start_val)) ** (252.0 / days)) - 1, 4)
        # df1[s1 + " " + s2] = df1['cum rets']
        # print(df1)  # df1[s1 + " " + s2], sharpe, CAGR



def fake_backtest(k_df, quotes, symbol1, symbol2):
    portfolio = portfolio_fx.init_portfolio()
    val_df = pd.DataFrame(index=k_df.index, columns=['hr','spread','zscore'])
    for day in k_df.index:
        quotes = k_df[:day]
        if len(quotes) < 100:
            continue

        x = quotes[symbol1]
        y = quotes[symbol2]

        # run regression (including Kalaman Filter) to find hedge ratio and then create spread series
        df1 = pd.DataFrame({'y': y, 'x': x})
        df1.index = pd.to_datetime(df1.index)

        state_means = KalmanFilterRegression(KalmanFilterAverage(x), KalmanFilterAverage(y))
        df1['hr'] = - state_means[:, 0]  ## Hedge Ratio
        val_df.loc[day]['hr'] = df1['hr'][-1]

        df1['spread'] = df1.y + (df1.x * df1.hr)  ## Spread
        val_df.loc[day]['spread'] = df1['spread'][-1]

        halflife = half_life(df1['spread'])  ## Half life

        # calculate z-score with window = half life period
        meanSpread = df1.spread.rolling(window=halflife).mean()
        stdSpread = df1.spread.rolling(window=halflife).std()
        df1['zScore'] = (df1.spread - meanSpread) / stdSpread
        val_df.loc[day]['zscore'] = df1['zScore'][-1]

        ###### Trade ######
        entryZscore = 2
        exitZscore = 0

        # set up num units long
        if (df1.loc[day]['zScore'] > entryZscore):
            if portfolio['holdings']['s1'] is None and portfolio['holdings']['s2'] is None:
                #print(f'sell {day}')
                ## sell
                portfolio = portfolio_fx.buy(day, symbol1, x.loc[day], symbol2, y.loc[day],
                                             df1.zScore[day],
                                             1, 1, portfolio)

        elif (df1.loc[day]['zScore'] < - entryZscore):
            if portfolio['holdings']['s1'] is None and portfolio['holdings']['s2'] is None:
               # print(f'buy {day}')

                ## buy
               portfolio = portfolio_fx.sell(day, symbol1, x.loc[day], symbol2, y.loc[day],
                                             df1.zScore[day],
                                             1, 1, portfolio)


       # ((df1.zScore > - exitZscore) & (df1.zScore.shift(1) < - exitZscore))
        elif (df1.loc[day]['zScore'] < abs(exitZscore)):
            if portfolio['holdings']['s1'] != None and portfolio['holdings']['s2'] != None:
                ## exit
               # print(f'exit {day}')
                portfolio = portfolio_fx.exit(day, symbol1, x.loc[day], symbol2, y.loc[day],
                                                df1.zScore[day],
                                                1, 1, portfolio)

        portfolio = portfolio_fx.calculate_nav(day, portfolio, x.loc[day], y.loc[day])
      #  print(portfolio['nav'].tail(1))
    return portfolio, df1, val_df

        # # df1['long entry'] = ((df1.zScore < - entryZscore) & (df1.zScore.shift(1) > - entryZscore))
        # # df1['long exit'] = ((df1.zScore > - exitZscore) & (df1.zScore.shift(1) < - exitZscore))
        # #
        # # # print(df1)
        # # df1['num units long'] = np.nan
        # # df1.loc[df1['long entry'], 'num units long'] = 1
        # # df1.loc[df1['long exit'], 'num units long'] = 0
        # # df1['num units long'][0] = 0
        # # df1['num units long'] = df1['num units long'].fillna(method='pad')
        # # # set up num units short
        # # df1['short entry'] = ((df1.zScore > entryZscore) & (df1.zScore.shift(1) < entryZscore))
        # # df1['short exit'] = ((df1.zScore < exitZscore) & (df1.zScore.shift(1) > exitZscore))
        # # df1.loc[df1['short entry'], 'num units short'] = -1
        # # df1.loc[df1['short exit'], 'num units short'] = 0
        # # df1['num units short'][0] = 0
        # # df1['num units short'] = df1['num units short'].fillna(method='pad')
        # # df1['numUnits'] = df1['num units long'] + df1['num units short']
        # # df1['spread pct ch'] = (df1['spread'] - df1['spread'].shift(1)) / ((df1['x'] * abs(df1['hr'])) + df1['y'])
        # # df1['port rets'] = df1['spread pct ch'] * df1['numUnits'].shift(1)
        # # df1['cum rets'] = df1['port rets'].cumsum()
        # # df1['cum rets'] = df1['cum rets'] + 1
        #
        # ##############################################################
        # try:
        #     sharpe = ((df1['port rets'].mean() / df1['port rets'].std()) * sqrt(252))
        # except ZeroDivisionError:
        #     sharpe = 0.0
        # ##############################################################
        # start_val = 1
        # end_val = df1['cum rets'].iat[-1]
        # start_date = df1.iloc[0].name
        # end_date = df1.iloc[-1].name
        # days = (end_date - start_date).days
        # CAGR = round(((float(end_val) / float(start_val)) ** (252.0 / days)) - 1, 4)
        # df1[s1 + " " + s2] = df1['cum rets']
        # print(df1)  # df1[s1 + " " + s2], sharpe, CAGR
