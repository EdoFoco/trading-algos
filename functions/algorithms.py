import functions.functions as base_fx
import functions.pair_portfolio_functions as portfolio_fx
from statsmodels.tsa.stattools import coint
import pandas as pd
import numpy as np
import math
from joblib import Parallel, delayed
import multiprocessing
import statsmodels.api as sm
from pykalman import KalmanFilter


def pair_trade_with_zscore_limit(settings, quotes):
    portfolio = portfolio_fx.init_portfolio()
    vol_limit = pd.Series()
    correlations = pd.Series()
    zscore_correlations = pd.Series()

    for day, row in quotes.iterrows():
        ## Calculate indicators
        filtered_quotes = quotes.loc[:day]
        s1 = filtered_quotes[settings['symbol1']]
        s2 = filtered_quotes[settings['symbol2']]

        ratios = base_fx.get_ratios(s1, s2)

        mavg_fast_r = base_fx.get_mavg(ratios, settings['mavg_1'])
        mavg_slow_r = base_fx.get_mavg(ratios, settings['mavg_2'])

        std_r = base_fx.get_std(ratios, settings['mavg_2'])

        if mavg_fast_r is None or mavg_slow_r is None or std_r is None:
            continue

        mavg_zscore = base_fx.get_zscore(mavg_fast_r, mavg_slow_r, std_r)

        macd_fast = base_fx.get_macd(mavg_zscore, settings['macd_fast_points'])
        macd_slow = base_fx.get_macd(mavg_zscore, settings['macd_slow_points'])

        vol = ratios.rolling(settings['vol_interval']).std(ddof=0)

        ## Understand is volaltility is within limits for the period
        ## Get the the last 30 peaks, then find the mean of those peaks. That's the limit
        sorted_vol = vol.tail(settings['vol_peak_period']).sort_values(ascending=False).head(
            settings['vol_mean_peak_period'])

        vol_mean = sorted_vol.mean()
        if len(vol_limit) == 0:
            vol_limit = pd.Series(index=vol.index)

        vol_limit.loc[day] = vol_mean
        within_vol = not math.isnan(vol.loc[day]) and vol.loc[day] < vol_limit[day]

        c = coint(s1.tail(settings['coint_period']), s2.tail(settings['coint_period']))
        correlations.loc[day] = c[1]

        # print(c[1])
        correlation_exists = c[1] < settings['coint_limit']
        ## Trade

        if macd_slow[day] <= macd_fast[day]:
            new_trend = "UP"
        else:
            new_trend = "DOWN"

        if mavg_zscore[day] > settings['limit'] and new_trend == "DOWN" and within_vol and correlation_exists:
            if portfolio['holdings']['s1'] == None and portfolio['holdings']['s2'] == None:
                ## sell
                portfolio = portfolio_fx.sell_pair(day, settings['symbol1'], s1.loc[day], settings['symbol2'], s2.loc[day],
                                                   mavg_zscore[day],
                                                   settings['leverage_limit'], settings['max_leverage'], portfolio)

        elif mavg_zscore[day] < -settings['limit'] and new_trend == "UP" and within_vol and correlation_exists:
            if portfolio['holdings']['s1'] == None and portfolio['holdings']['s2'] == None:
                ## buy
                portfolio = portfolio_fx.buy_pair(day, settings['symbol1'], s1.loc[day], settings['symbol2'], s2.loc[day],
                                                  mavg_zscore[day],
                                                  settings['leverage_limit'], settings['max_leverage'], portfolio)

        elif abs(mavg_zscore[day]) < settings['exit_limit'] or not within_vol or not correlation_exists:
            if portfolio['holdings']['s1'] != None and portfolio['holdings']['s2'] != None:
                ## exit
                portfolio = portfolio_fx.exit_pairs(day, settings['symbol1'], s1.loc[day], settings['symbol2'], s2.loc[day],
                                                    mavg_zscore[day],
                                                    settings['leverage_limit'], settings['max_leverage'], portfolio)

        portfolio = portfolio_fx.calculate_nav(day, portfolio, s1.loc[day], s2.loc[day])

    portfolio['nav'] = portfolio['nav'].set_index('date')

    portfolio_fx.plot_correlations(correlations, zscore_correlations)

    portfolio_fx.plot_normal_zscore(base_fx.get_zscore(ratios, ratios.mean(), np.std(ratios)))
    portfolio_fx.plot_indicators(mavg_zscore, macd_fast, macd_slow, vol, vol_mean, vol_limit)
    portfolio_fx.plot_normalized_returns(settings['symbol1'], s1, settings['symbol2'], s2, portfolio)
    portfolio_fx.print_kpis(portfolio['nav'])

    return portfolio, mavg_zscore


def pair_trade_with_zscore_limit_and_corr(settings, quotes):
    portfolio = portfolio_fx.init_portfolio()
    vol_limit = pd.Series()
    correlations = pd.Series()
    zscore_correlations = pd.Series()
    zscore_vol = pd.Series()
    mavg_zscore = None
    ratios = pd.Series()
    vol = pd.Series()

    for day, row in quotes.iterrows():
        ## Calculate indicators
        filtered_quotes = quotes.loc[:day]
        s1 = filtered_quotes[settings['symbol1']].tail(settings['analysis_period'])
        s2 = filtered_quotes[settings['symbol2']].tail(settings['analysis_period'])

        ratios.loc[day] = base_fx.get_ratios(s1, s2)[-1]

        mavg_fast_r = base_fx.get_mavg(ratios, settings['mavg_1'])
        mavg_slow_r = base_fx.get_mavg(ratios, settings['mavg_2'])

        std_r = base_fx.get_std(ratios, settings['mavg_2'])

        if mavg_fast_r is None or mavg_slow_r is None or std_r is None:
            continue

        if mavg_zscore is None:
            mavg_zscore = base_fx.get_zscore(mavg_fast_r, mavg_slow_r, std_r)

        mavg_zscore.loc[day] = base_fx.get_zscore(mavg_fast_r, mavg_slow_r, std_r)[-1]

        macd_fast = base_fx.get_macd(mavg_zscore, settings['macd_fast_points'])
        macd_slow = base_fx.get_macd(mavg_zscore, settings['macd_slow_points'])

        vol.loc[day] = ratios.tail(settings['analysis_period']).rolling(settings['mavg_2']).std(ddof=0)[-1]  # ratios.rolling(settings['vol_interval']).std(ddof=0)[-1]

        vol_mean = (vol.mean() * settings['vol_tolerance']) + vol.mean()  # sorted_vol.mean()
        if len(vol_limit) == 0:
            vol_limit = pd.Series(index=vol.index)

        vol_limit.loc[day] = vol_mean

        vol_mavg_slow_r = base_fx.get_mavg(vol, settings['mavg_2'])
        if vol_mavg_slow_r is None:
            continue

        zscore_vol.loc[day] = base_fx.get_zscore(vol_mavg_slow_r, vol_mavg_slow_r.mean(), vol_mavg_slow_r.std())[-1]

        within_vol = abs(zscore_vol.loc[day]) <= settings['coint_limit'] and vol.loc[day] < vol_limit[day]

        c = coint(s1.tail(settings['analysis_period']), s2.tail(settings['analysis_period']))
        correlations.loc[day] = c[1]
        zscore_correlations.loc[day] = base_fx.get_zscore(correlations,
                                                          correlations.mean(),
                                                          correlations.std())[-1]  # base_fx.get_mavg(correlations, 450)

        correlation_exists = abs(zscore_correlations.loc[day]) < settings['coint_limit']  # c[1] < settings['coint_limit'] # abs(zscore_correlations.loc[day]) < settings['coint_limit']  # c[1] < zscore_correlations.loc[day] #settings['coint_limit']

        ## Trade

        if macd_slow[day] <= macd_fast[day]:
            new_trend = "UP"
        else:
            new_trend = "DOWN"

        if mavg_zscore[day] > settings['limit'] and new_trend == "DOWN" and within_vol and correlation_exists:
            if portfolio['holdings']['s1'] == None and portfolio['holdings']['s2'] == None:
                ## sell
                portfolio = portfolio_fx.sell_pair(day, settings['symbol1'], s1.loc[day], settings['symbol2'], s2.loc[day],
                                                   mavg_zscore[day],
                                                   settings['leverage_limit'], settings['max_leverage'], portfolio)

        elif mavg_zscore[day] < -settings['limit'] and new_trend == "UP" and within_vol and correlation_exists:
            if portfolio['holdings']['s1'] == None and portfolio['holdings']['s2'] == None:
                ## buy
                portfolio = portfolio_fx.buy_pair(day, settings['symbol1'], s1.loc[day], settings['symbol2'], s2.loc[day],
                                                  mavg_zscore[day],
                                                  settings['leverage_limit'], settings['max_leverage'], portfolio)

        elif abs(mavg_zscore[day]) < settings['exit_limit'] or not within_vol or not correlation_exists:
            if portfolio['holdings']['s1'] != None and portfolio['holdings']['s2'] != None:
                ## exit
                portfolio = portfolio_fx.exit_pairs(day, settings['symbol1'], s1.loc[day], settings['symbol2'], s2.loc[day],
                                                    mavg_zscore[day],
                                                    settings['leverage_limit'], settings['max_leverage'], portfolio)

        portfolio = portfolio_fx.calculate_nav(day, portfolio, s1.loc[day], s2.loc[day])
        # portfolio = portfolio_fx.stop_loss(day, s1.loc[day], s2.loc[day], portfolio, settings['sl_limit'])

    portfolio['nav'] = portfolio['nav'].set_index('date')

    print('zscore correlation')
    portfolio_fx.plot_objects(
        [{'label': 'zscore_corr', 'value': zscore_correlations}, {'label': 'corr', 'value': correlations}])

    print('corr volatility ')
    # portfolio_fx.plot_objects([{'label': 'zscore_vol', 'value': zscore_vol}])
    portfolio_fx.plot_indicators(mavg_zscore, macd_fast, macd_slow, vol, vol_mean, vol_limit)

    print('corr vol zscore')
    portfolio_fx.plot_objects(
        [{'label': 'vol', 'value': vol}, {'label': 'zscore', 'value': zscore_vol},
         {'label': 'vol limit', 'value': vol_limit}])

    print('ratios')
    portfolio_fx.plot_normal_zscore(base_fx.get_zscore(ratios, ratios.mean(), np.std(ratios)))
    # portfolio_fx.plot_indicators(mavg_zscore, macd_fast, macd_slow, vol, vol_mean, vol_limit)
    print('returns')
    portfolio_fx.plot_normalized_returns(settings['symbol1'], quotes[settings['symbol1']], settings['symbol2'],
                                         quotes[settings['symbol2']], portfolio)
    portfolio_fx.print_kpis(portfolio['nav'])

    return portfolio, mavg_zscore


def pair_trade_with_zscore_limit_and_corr_and_sl(settings, quotes):
    portfolio = portfolio_fx.init_portfolio()
    vol_limit = pd.Series()
    correlations = pd.Series()
    zscore_correlations = pd.Series()
    zscore_vol = pd.Series()
    mavg_zscore = None
    ratios = pd.Series()
    vol = pd.Series()

    for day, row in quotes.iterrows():
        ## Calculate indicators
        filtered_quotes = quotes.loc[:day]
        s1 = filtered_quotes[settings['symbol1']].tail(settings['analysis_period'])
        s2 = filtered_quotes[settings['symbol2']].tail(settings['analysis_period'])

        ratios.loc[day] = base_fx.get_ratios(s1, s2)[-1]

        mavg_fast_r = base_fx.get_mavg(ratios, settings['mavg_1'])
        mavg_slow_r = base_fx.get_mavg(ratios, settings['mavg_2'])

        std_r = base_fx.get_std(ratios, settings['mavg_2'])

        if mavg_fast_r is None or mavg_slow_r is None or std_r is None:
            continue

        if mavg_zscore is None:
            mavg_zscore = base_fx.get_zscore(mavg_fast_r, mavg_slow_r, std_r)

        mavg_zscore.loc[day] = base_fx.get_zscore(mavg_fast_r, mavg_slow_r, std_r)[-1]

        macd_fast = base_fx.get_macd(mavg_zscore, settings['macd_fast_points'])
        macd_slow = base_fx.get_macd(mavg_zscore, settings['macd_slow_points'])

        vol.loc[day] = ratios.tail(settings['analysis_period']).rolling(settings['mavg_2']).std(ddof=0)[
            -1]  # ratios.rolling(settings['vol_interval']).std(ddof=0)[-1]

        vol_mean = (vol.mean() * settings['vol_tolerance']) + vol.mean()  # sorted_vol.mean()
        if len(vol_limit) == 0:
            vol_limit = pd.Series(index=vol.index)

        vol_limit.loc[day] = vol_mean

        vol_mavg_slow_r = base_fx.get_mavg(vol, settings['mavg_2'])
        if vol_mavg_slow_r is None:
            continue

        zscore_vol.loc[day] = base_fx.get_zscore(vol_mavg_slow_r, vol_mavg_slow_r.mean(), vol_mavg_slow_r.std())[-1]

        within_vol = abs(zscore_vol.loc[day]) <= 2 and vol.loc[day] < vol_limit[day]

        c = coint(s1.tail(settings['analysis_period']), s2.tail(settings['analysis_period']))
        correlations.loc[day] = c[1]
        zscore_correlations.loc[day] = base_fx.get_zscore(correlations,
                                                          correlations.mean(),
                                                          correlations.std())[-1]  # base_fx.get_mavg(correlations, 450)

        correlation_exists = abs(zscore_correlations.loc[day]) < settings[
            'coint_limit']  # c[1] < settings['coint_limit'] # abs(zscore_correlations.loc[day]) < settings['coint_limit']  # c[1] < zscore_correlations.loc[day] #settings['coint_limit']

        ## Trade

        if macd_slow[day] <= macd_fast[day]:
            new_trend = "UP"
        else:
            new_trend = "DOWN"

        if mavg_zscore[day] > settings['limit'] and new_trend == "DOWN" and within_vol and correlation_exists:
            if portfolio['holdings']['s1'] == None and portfolio['holdings']['s2'] == None:
                ## sell
                portfolio = portfolio_fx.sell_pair(day, settings['symbol1'], s1.loc[day], settings['symbol2'], s2.loc[day],
                                                   mavg_zscore[day],
                                                   settings['leverage_limit'], settings['max_leverage'], portfolio)

        elif mavg_zscore[day] < -settings['limit'] and new_trend == "UP" and within_vol and correlation_exists:
            if portfolio['holdings']['s1'] == None and portfolio['holdings']['s2'] == None:
                ## buy
                portfolio = portfolio_fx.buy_pair(day, settings['symbol1'], s1.loc[day], settings['symbol2'], s2.loc[day],
                                                  mavg_zscore[day],
                                                  settings['leverage_limit'], settings['max_leverage'], portfolio)

        elif abs(mavg_zscore[day]) < settings['exit_limit'] or not within_vol or not correlation_exists:
            #if portfolio['holdings']['s1'] != None and portfolio['holdings']['s2'] != None:
                ## exit
            portfolio = portfolio_fx.exit_pairs(day, settings['symbol1'], s1.loc[day], settings['symbol2'], s2.loc[day],
                                                mavg_zscore[day],
                                                settings['leverage_limit'], settings['max_leverage'], portfolio)

        portfolio = portfolio_fx.calculate_nav(day, portfolio, s1.loc[day], s2.loc[day])
        portfolio = portfolio_fx.stop_loss_pair(day, s1.loc[day], s2.loc[day], portfolio, settings['sl_limit'])

    portfolio['nav'] = portfolio['nav'].set_index('date')

    print('zscore correlation')
    portfolio_fx.plot_objects(
        [{'label': 'zscore_corr', 'value': zscore_correlations}, {'label': 'corr', 'value': correlations}])

    print('corr volatility ')
    # portfolio_fx.plot_objects([{'label': 'zscore_vol', 'value': zscore_vol}])
    portfolio_fx.plot_indicators(mavg_zscore, macd_fast, macd_slow, vol, vol_mean, vol_limit)

    print('corr vol zscore')
    portfolio_fx.plot_objects(
        [{'label': 'vol', 'value': vol}, {'label': 'zscore', 'value': zscore_vol},
         {'label': 'vol limit', 'value': vol_limit}])

    print('ratios')
    portfolio_fx.plot_normal_zscore(base_fx.get_zscore(ratios, ratios.mean(), np.std(ratios)))
    # portfolio_fx.plot_indicators(mavg_zscore, macd_fast, macd_slow, vol, vol_mean, vol_limit)
    print('returns')
    portfolio_fx.plot_normalized_returns(settings['symbol1'], quotes[settings['symbol1']], settings['symbol2'],
                                         quotes[settings['symbol2']], portfolio)
    portfolio_fx.print_kpis(portfolio['nav'])

    return portfolio, mavg_zscore


def get_hedge_ratio(symbol1, symbol2, quotes):
    df = pd.DataFrame(index=quotes.index)
    df[symbol1] = quotes[symbol1]
    df[symbol2] = quotes[symbol2]

    est = sm.OLS(df[symbol1], df[symbol2])
    est = est.fit()
    df['hr'] = -est.params[0] / 100
    df['spread'] = df[symbol1] + (df[symbol2] * df.hr)
    #print(df)

    return df


def get_hedge_ratio_with_kalman(symbol1, symbol2, quotes):
    state_means = KalmanFilterRegression(KalmanFilterAverage(quotes[symbol1]), KalmanFilterAverage(quotes[symbol2]))
    df = pd.DataFrame(index=quotes.index)
    df['hr'] = - state_means[:, 0]
    df['spread'] = quotes[symbol1] + (quotes[symbol2] * df.hr)
    return df


def get_half_life(spread):
    #print(spread)
    spread_lag = spread.shift(1)
    spread_lag.iloc[0] = spread_lag.iloc[1]
    spread_ret = spread - spread_lag
    spread_ret.iloc[0] = spread_ret.iloc[1]
    spread_lag2 = sm.add_constant(spread_lag)
    model = sm.OLS(spread_ret,spread_lag2)
    res = model.fit()
    halflife = int(round(-np.log(2) / res.params[1],0))
    if halflife <= 0:
        halflife = 1
    return halflife


def KalmanFilterAverage(x):
  # Construct a Kalman filter
    kf = KalmanFilter(transition_matrices = [1],
    observation_matrices = [1],
    initial_state_mean = 0,
    initial_state_covariance = 1,
    observation_covariance=1,
    transition_covariance=.01)
  # Use the observed values of the price to get a rolling mean
    state_means, _ = kf.filter(x.values)
    state_means = pd.Series(state_means.flatten(), index=x.index)
    return state_means

# Kalman filter regression
def KalmanFilterRegression(x,y):
    delta = 1e-3
    trans_cov = delta / (1 - delta) * np.eye(2) # How much random walk wiggles
    obs_mat = np.expand_dims(np.vstack([[x], [np.ones(len(x))]]).T, axis=1)
    kf = KalmanFilter(n_dim_obs=1, n_dim_state=2, # y is 1-dimensional, (alpha, beta) is 2-dimensional
    initial_state_mean=[0,0],
    initial_state_covariance=np.ones((2, 2)),
    transition_matrices=np.eye(2),
    observation_matrices=obs_mat,
    observation_covariance=2,
    transition_covariance=trans_cov)
    # Use the observations y to get running estimates and errors for the state parameters
    state_means, state_covs = kf.filter(y.values)
    return state_means


def pair_trade_with_zscore_limit_and_corr_and_hr_and_half_life(settings, quotes):
    portfolio = portfolio_fx.init_portfolio()
    vol_limit = pd.Series()
    correlations = pd.Series()
    zscore_correlations = pd.Series()
    zscore_vol = pd.Series()
    daily_hr = pd.DataFrame(index=quotes.index, columns=['hr', 'spread', 'half_life'])

    for day, row in quotes.iterrows():
        ## Calculate indicators
        filtered_quotes = quotes.loc[:day]

        if len(filtered_quotes) <= settings['mavg_2']:
            continue

        s1 = filtered_quotes[settings['symbol1']]
        s2 = filtered_quotes[settings['symbol2']]

        ratios = base_fx.get_ratios(s1, s2)

## Start old part
        # mavg_fast_r = base_fx.get_mavg(ratios, settings['mavg_1'])
        # mavg_slow_r = base_fx.get_mavg(ratios, settings['mavg_2'])
        #
        # std_r = base_fx.get_std(ratios, settings['mavg_2'])
        #
        # if mavg_fast_r is None or mavg_slow_r is None or std_r is None:
        #    continue
        #
        # mavg_zscore = base_fx.get_zscore(mavg_fast_r, mavg_slow_r, std_r)
        # hr = pd.DataFrame(columns=['hr'])
        # hr.loc[day] = None
        # hr.loc[day]['hr'] = 0
## End old part
## Start New

        #print(state_means)

        #hr = get_hedge_ratio_with_kalman(settings['symbol1'], settings['symbol2'], filtered_quotes)
        hr = get_hedge_ratio(settings['symbol1'], settings['symbol2'], filtered_quotes.tail(252))
        #print(hr)
        half_life = get_half_life(hr['spread'])
        if len(filtered_quotes) < half_life:
            continue

        daily_hr.loc[day] = None
        daily_hr.loc[day]['hr'] = hr.iloc[-1]['hr']
        daily_hr.loc[day]['spread'] = hr.iloc[-1]['spread']
        daily_hr.loc[day]['half_life'] = half_life
        #
        # meanSpread = daily_hr['spread'].rolling(window=half_life).mean()
        # stdSpread = daily_hr['spread'].rolling(window=half_life).std()
        # mavg_zscore = (daily_hr['spread'] - meanSpread) / stdSpread
       # print(half_life)
        #print(half_life)
        mavg_fast_r = base_fx.get_mavg(ratios, settings['mavg_1'])
        mavg_slow_r = base_fx.get_mavg(ratios, settings['mavg_2'])

        std_r = base_fx.get_std(ratios, settings['mavg_2'])

        if mavg_fast_r is None or mavg_slow_r is None or std_r is None:
            continue

        mavg_zscore = base_fx.get_zscore(mavg_fast_r, mavg_slow_r, std_r)
## End New

        macd_fast = base_fx.get_macd(mavg_zscore, settings['macd_fast_points'])
        macd_slow = base_fx.get_macd(mavg_zscore, settings['macd_slow_points'])

        vol = ratios.rolling(settings['vol_interval']).std(ddof=0)

        ## Understand is volaltility is within limits for the period
        ## Get the the last 30 peaks, then find the mean of those peaks. That's the limit
        sorted_vol = vol.tail(settings['vol_peak_period']).sort_values(ascending=False).head(
            settings['vol_mean_peak_period'])

        vol_mean = sorted_vol.mean()
        if len(vol_limit) == 0:
            vol_limit = pd.Series(index=vol.index)

        vol_limit.loc[day] = vol_mean
        within_vol = not math.isnan(vol.loc[day]) and vol.loc[day] < vol_limit[day]

        c = coint(s1, s2)
        correlations.loc[day] = c[1]
        zscore_correlations.loc[day] = base_fx.get_zscore(correlations,
                                                          correlations.mean(),
                                                          correlations.std())[-1]  # base_fx.get_mavg(correlations, 450)

        zscore_vol.loc[day] = zscore_correlations.std()

        # print(c[1])
        correlation_exists = c[1] < 0.4 and abs(zscore_correlations.loc[day]) < settings['coint_limit']  # c[1] < zscore_correlations.loc[day] #settings['coint_limit']
        ## Trade

        if macd_slow[day] <= macd_fast[day]:
            new_trend = "UP"
        else:
            new_trend = "DOWN"

        if mavg_zscore[day] > settings['limit'] and new_trend == "DOWN" and within_vol and correlation_exists:
            if portfolio['holdings']['s1'] == None and portfolio['holdings']['s2'] == None:
                ## sell
                portfolio = portfolio_fx.sell_pair_with_hedge(day, settings['symbol1'], s1.loc[day], settings['symbol2'], s2.loc[day],
                                                              mavg_zscore[day],
                                                              settings['leverage_limit'], settings['max_leverage'], portfolio, hr.iloc[-1]['hr'])

        elif mavg_zscore[day] < -settings['limit'] and new_trend == "UP" and within_vol and correlation_exists:
            if portfolio['holdings']['s1'] == None and portfolio['holdings']['s2'] == None:
                ## buy
                portfolio = portfolio_fx.buy_pair_with_hedge(day, settings['symbol1'], s1.loc[day], settings['symbol2'], s2.loc[day],
                                                             mavg_zscore[day],
                                                             settings['leverage_limit'], settings['max_leverage'], portfolio, hr.iloc[-1]['hr'])

        elif abs(mavg_zscore[day]) < settings['exit_limit'] or not within_vol or not correlation_exists:
            if portfolio['holdings']['s1'] != None and portfolio['holdings']['s2'] != None:
                ## exit
                portfolio = portfolio_fx.exit_pairs(day, settings['symbol1'], s1.loc[day], settings['symbol2'], s2.loc[day],
                                                    mavg_zscore[day],
                                                    settings['leverage_limit'], settings['max_leverage'], portfolio)

        portfolio = portfolio_fx.calculate_nav(day, portfolio, s1.loc[day], s2.loc[day])

    portfolio['nav'] = portfolio['nav'].set_index('date')

    portfolio_fx.plot_objects([{'label': 'zscore_corr', 'value': zscore_correlations}, {'label': 'corr', 'value': correlations}])
    portfolio_fx.plot_objects([{'label': 'zscore_vol', 'value': zscore_vol}])

    portfolio_fx.plot_normal_zscore(base_fx.get_zscore(ratios, ratios.mean(), np.std(ratios)))
    #portfolio_fx.plot_indicators(mavg_zscore, macd_fast, macd_slow, vol, vol_mean, vol_limit)
    portfolio_fx.plot_normalized_returns(settings['symbol1'], s1, settings['symbol2'], s2, portfolio)
    portfolio_fx.print_kpis(portfolio['nav'])

    return portfolio, mavg_zscore, daily_hr


def pair_trade_with_zscore_limit_and_corr_and_hr(settings, quotes):
    portfolio = portfolio_fx.init_portfolio()
    vol_limit = pd.Series()
    correlations = pd.Series()
    zscore_correlations = pd.Series()
    zscore_vol = pd.Series()
    daily_hr = pd.DataFrame(index=quotes.index, columns=['hr', 'spread', 'half_life'])

    for day, row in quotes.iterrows():
        ## Calculate indicators
        filtered_quotes = quotes.loc[:day]

        if len(filtered_quotes) <= settings['mavg_2']:
            continue
        s1 = filtered_quotes[settings['symbol1']]
        s2 = filtered_quotes[settings['symbol2']]

        ratios = base_fx.get_ratios(s1, s2)

## Start old part
        # mavg_fast_r = base_fx.get_mavg(ratios, settings['mavg_1'])
        # mavg_slow_r = base_fx.get_mavg(ratios, settings['mavg_2'])
        #
        # std_r = base_fx.get_std(ratios, settings['mavg_2'])
        #
        # if mavg_fast_r is None or mavg_slow_r is None or std_r is None:
        #    continue
        #
        # mavg_zscore = base_fx.get_zscore(mavg_fast_r, mavg_slow_r, std_r)
        # hr = pd.DataFrame(columns=['hr'])
        # hr.loc[day] = None
        # hr.loc[day]['hr'] = 0
## End old part
## Start New

        #state_means = KalmanFilterRegression(KalmanFilterAverage(s1), KalmanFilterAverage(s2))
        #print(state_means)

        hr = get_hedge_ratio(settings['symbol1'], settings['symbol2'], filtered_quotes)
        #print(hr)
        #half_life = get_half_life(hr['spread'])
        #if len(filtered_quotes) < half_life:
        #    continue

        mavg_fast_r = base_fx.get_mavg(ratios, settings['mavg_1'])
        mavg_slow_r = base_fx.get_mavg(ratios, settings['mavg_2'])
        std_r = base_fx.get_std(ratios, settings['mavg_2'])

        if mavg_fast_r is None or mavg_slow_r is None or std_r is None:
           continue

        mavg_zscore = base_fx.get_zscore(mavg_fast_r, mavg_slow_r, std_r)

        daily_hr.loc[day] = None
        daily_hr.loc[day]['hr'] = hr.iloc[-1]['hr']
        daily_hr.loc[day]['spread'] = hr.iloc[-1]['spread']
        #daily_hr.loc[day]['half_life'] = half_life

## End New

        macd_fast = base_fx.get_macd(mavg_zscore, settings['macd_fast_points'])
        macd_slow = base_fx.get_macd(mavg_zscore, settings['macd_slow_points'])

        vol = ratios.rolling(settings['vol_interval']).std(ddof=0)

        ## Understand is volaltility is within limits for the period
        ## Get the the last 30 peaks, then find the mean of those peaks. That's the limit
        sorted_vol = vol.tail(settings['vol_peak_period']).sort_values(ascending=False).head(
            settings['vol_mean_peak_period'])

        vol_mean = sorted_vol.mean()
        if len(vol_limit) == 0:
            vol_limit = pd.Series(index=vol.index)

        vol_limit.loc[day] = vol_mean
        within_vol = not math.isnan(vol.loc[day]) and vol.loc[day] < vol_limit[day]

        c = coint(s1, s2)
        correlations.loc[day] = c[1]
        zscore_correlations.loc[day] = base_fx.get_zscore(correlations,
                                                          correlations.mean(),
                                                          correlations.std())[-1]  # base_fx.get_mavg(correlations, 450)

        zscore_vol.loc[day] = zscore_correlations.std()

        # print(c[1])
        correlation_exists = c[1] < settings['coint_limit'] # abs(zscore_correlations.loc[day]) < settings['coint_limit']  # c[1] < zscore_correlations.loc[day] #settings['coint_limit']
        ## Trade

        if macd_slow[day] <= macd_fast[day]:
            new_trend = "UP"
        else:
            new_trend = "DOWN"

        if mavg_zscore[day] > settings['limit'] and new_trend == "DOWN" and within_vol and correlation_exists:
            if portfolio['holdings']['s1'] == None and portfolio['holdings']['s2'] == None:
                ## sell
                portfolio = portfolio_fx.sell_pair_with_hedge(day, settings['symbol1'], s1.loc[day], settings['symbol2'], s2.loc[day],
                                                              mavg_zscore[day],
                                                              settings['leverage_limit'], settings['max_leverage'], portfolio, hr.iloc[-1]['hr'])

        elif mavg_zscore[day] < -settings['limit'] and new_trend == "UP" and within_vol and correlation_exists:
            if portfolio['holdings']['s1'] == None and portfolio['holdings']['s2'] == None:
                ## buy
                portfolio = portfolio_fx.buy_pair_with_hedge(day, settings['symbol1'], s1.loc[day], settings['symbol2'], s2.loc[day],
                                                             mavg_zscore[day],
                                                             settings['leverage_limit'], settings['max_leverage'], portfolio, hr.iloc[-1]['hr'])

        elif abs(mavg_zscore[day]) < settings['exit_limit'] or not within_vol or not correlation_exists:
            if portfolio['holdings']['s1'] != None and portfolio['holdings']['s2'] != None:
                ## exit
                portfolio = portfolio_fx.exit_pairs(day, settings['symbol1'], s1.loc[day], settings['symbol2'], s2.loc[day],
                                                    mavg_zscore[day],
                                                    settings['leverage_limit'], settings['max_leverage'], portfolio)

        portfolio = portfolio_fx.calculate_nav(day, portfolio, s1.loc[day], s2.loc[day])

    portfolio['nav'] = portfolio['nav'].set_index('date')

    portfolio_fx.plot_objects([{'label': 'zscore_corr', 'value': zscore_correlations}, {'label': 'corr', 'value': correlations}])
    portfolio_fx.plot_objects([{'label': 'zscore_vol', 'value': zscore_vol}])

    portfolio_fx.plot_normal_zscore(base_fx.get_zscore(ratios, ratios.mean(), np.std(ratios)))
    #portfolio_fx.plot_indicators(mavg_zscore, macd_fast, macd_slow, vol, vol_mean, vol_limit)
    portfolio_fx.plot_normalized_returns(settings['symbol1'], s1, settings['symbol2'], s2, portfolio)
    portfolio_fx.print_kpis(portfolio['nav'])

    return portfolio, mavg_zscore, daily_hr


def pair_trade_with_zscore_limit_and_corr_and_bruteforce(settings, quotes, is_backtest=False):
    portfolio = portfolio_fx.init_portfolio()
    vol_limit = pd.Series()
    correlations = pd.Series()
    zscore_correlations = pd.Series()

    for day, row in quotes.iterrows():

        if is_last_day_of_month(day) and not is_backtest:
            print(day)
            settings = optimize_settings_through_bruteforce(settings, quotes[quotes.index < day])

        ## Calculate indicators
        filtered_quotes = quotes.loc[:day]
        s1 = filtered_quotes[settings['symbol1']]
        s2 = filtered_quotes[settings['symbol2']]

        ratios = base_fx.get_ratios(s1, s2)

        mavg_fast_r = base_fx.get_mavg(ratios, settings['mavg_1'])
        mavg_slow_r = base_fx.get_mavg(ratios, settings['mavg_2'])

        std_r = base_fx.get_std(ratios, settings['mavg_2'])

        if mavg_fast_r is None or mavg_slow_r is None or std_r is None:
            continue

        mavg_zscore = base_fx.get_zscore(mavg_fast_r, mavg_slow_r, std_r)

        macd_fast = base_fx.get_macd(mavg_zscore, settings['macd_fast_points'])
        macd_slow = base_fx.get_macd(mavg_zscore, settings['macd_slow_points'])

        vol = ratios.rolling(settings['vol_interval']).std(ddof=0)

        ## Understand is volaltility is within limits for the period
        ## Get the the last 30 peaks, then find the mean of those peaks. That's the limit
        sorted_vol = vol.tail(settings['vol_peak_period']).sort_values(ascending=False).head(
            settings['vol_mean_peak_period'])

        vol_mean = sorted_vol.mean()
        if len(vol_limit) == 0:
            vol_limit = pd.Series(index=vol.index)

        vol_limit.loc[day] = vol_mean
        within_vol = not math.isnan(vol.loc[day]) and vol.loc[day] < vol_limit[day]

        c = coint(s1.tail(settings['coint_period']), s2.tail(settings['coint_period']))
        correlations.loc[day] = c[1]
        zscore_correlations.loc[day] = base_fx.get_zscore(correlations,
                                                          correlations.mean(),
                                                          correlations.std())[-1]  # base_fx.get_mavg(correlations, 450)

        # print(c[1])
        correlation_exists = abs(zscore_correlations.loc[day]) < settings['coint_limit']  # c[1] < zscore_correlations.loc[day] #settings['coint_limit']
        ## Trade

        if macd_slow[day] <= macd_fast[day]:
            new_trend = "UP"
        else:
            new_trend = "DOWN"

        if mavg_zscore[day] > settings['limit'] and new_trend == "DOWN" and within_vol and correlation_exists:
            if portfolio['holdings']['s1'] == None and portfolio['holdings']['s2'] == None:
                ## sell
                portfolio = portfolio_fx.sell_pair(day, settings['symbol1'], s1.loc[day], settings['symbol2'], s2.loc[day],
                                                   mavg_zscore[day],
                                                   settings['leverage_limit'], settings['max_leverage'], portfolio)

        elif mavg_zscore[day] < -settings['limit'] and new_trend == "UP" and within_vol and correlation_exists:
            if portfolio['holdings']['s1'] == None and portfolio['holdings']['s2'] == None:
                ## buy
                portfolio = portfolio_fx.buy_pair(day, settings['symbol1'], s1.loc[day], settings['symbol2'], s2.loc[day],
                                                  mavg_zscore[day],
                                                  settings['leverage_limit'], settings['max_leverage'], portfolio)

        elif abs(mavg_zscore[day]) < settings['exit_limit'] or not within_vol or not correlation_exists:
            if portfolio['holdings']['s1'] != None and portfolio['holdings']['s2'] != None:
                ## exit
                portfolio = portfolio_fx.exit_pairs(day, settings['symbol1'], s1.loc[day], settings['symbol2'], s2.loc[day],
                                                    mavg_zscore[day],
                                                    settings['leverage_limit'], settings['max_leverage'], portfolio)

        portfolio = portfolio_fx.calculate_nav(day, portfolio, s1.loc[day], s2.loc[day])

    portfolio['nav'] = portfolio['nav'].set_index('date')

    if not is_backtest:

        portfolio_fx.plot_objects(
            [{'label': 'zscore', 'value': zscore_correlations}, {'label': 'corr', 'value': correlations}])

        portfolio_fx.plot_normal_zscore(base_fx.get_zscore(ratios, ratios.mean(), np.std(ratios)))
        portfolio_fx.plot_indicators(mavg_zscore, macd_fast, macd_slow, vol, vol_mean, vol_limit)
        portfolio_fx.plot_normalized_returns(settings['symbol1'], s1, settings['symbol2'], s2, portfolio)
        portfolio_fx.print_kpis(portfolio['nav'])

    return portfolio#, mavg_zscore


def is_last_day_of_month(date):
    if date.day == 28:
        return True
    return False


def get_bactest_results(settings, quotes):
    backtest_port = pair_trade_with_zscore_limit_and_corr_and_bruteforce(settings, quotes, True)
    ret, vol, sharpe = portfolio_fx.get_monthly_kpis(backtest_port['nav'])
    return {"Return": ret, "Vol": vol, "Sharpe": sharpe, "Vol_Interval": settings['vol_interval'],
     "Vol_Mean_Peak_Period": settings['vol_mean_peak_period']}


def optimize_settings_through_bruteforce(settings, quotes):
    original_settings = settings.copy()
    print('Running backtes')
    print(f"Prev vol_interval: {settings['vol_interval']}. Prev vol_mean_peak: {settings['vol_mean_peak_period']}")
    last_6_months = quotes.tail(6*30)

    num_cores = multiprocessing.cpu_count()



    kpis = pd.DataFrame(columns=["Return", "Vol", "Sharpe", "Vol_Interval", "Vol_Mean_Peak_Period"])

    jobs = []
    for vol_interval in range(20, 50, 3):
        for vol_mean_peak_period in range(30, 60, 5):
            settings['vol_interval'] = vol_interval
            settings['vol_mean_peak_period'] = vol_mean_peak_period
            jobs.append(delayed(get_bactest_results)(settings.copy(),last_6_months))


            #print(f'Return: {str(ret)}, Vol: {str(vol)}, Sharpe: {str(sharpe)}')

    results = Parallel(n_jobs=num_cores)(jobs)
    for res in results:
        kpis = kpis.append(res, ignore_index=True)

    #print(kpis)
    kpis = kpis.dropna()
    if len(kpis) == 0:
        return original_settings

    best = kpis.sort_values(by="Sharpe", ascending=False)
    #print(best)

    settings["vol_interval"] = int(best.iloc[0]["Vol_Interval"])
    settings["vol_mean_peak_period"] = int(best.iloc[0]["Vol_Mean_Peak_Period"])
    print(settings['vol_interval'])
    return settings
