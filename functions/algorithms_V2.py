import functions.functions as base_fx
import functions.pair_portfolio_functions as portfolio_fx
from statsmodels.tsa.stattools import coint
import pandas as pd
import numpy as np
import math


def pair_trade_v2_with_hurst(settings, quotes):
    portfolio = portfolio_fx.init_portfolio()
    vol_limit = pd.Series()
    hursts = pd.Series()

    for day, row in quotes.iterrows():
        ## Calculate indicators
        filtered_quotes = quotes.loc[:day]
        s1 = filtered_quotes[settings['symbol1']]
        s2 = filtered_quotes[settings['symbol2']]

        ratios = base_fx.get_ratios(s1, s2)
        ratios = ratios.astype(float)

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

        # c = coint(s1, s2)
        # correlations.loc[day] = c[1]
        # zscore_correlations.loc[day] = base_fx.get_zscore(correlations,
        #                                                   correlations.mean(),
        #                                                   correlations.std())[-1]  # base_fx.get_mavg(correlations, 450)
        #
        # zscore_vol.loc[day] = zscore_correlations.std()
        #
        # # print(c[1])
        # correlation_exists = c[1] < settings['coint_limit'] # abs(zscore_correlations.loc[day]) < settings['coint_limit']  # c[1] < zscore_correlations.loc[day] #settings['coint_limit']

        hurst = get_hurst_exp(settings, ratios.tail(365))
        hursts.loc[day] = hurst


        correlation_exists = hurst < 0.5
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

    portfolio_fx.plot_objects([{'label': 'hursts', 'value': hursts}])

    portfolio_fx.plot_normal_zscore(base_fx.get_zscore(ratios, ratios.mean(), np.std(ratios)))
    portfolio_fx.plot_indicators(mavg_zscore, macd_fast, macd_slow, vol, vol_mean, vol_limit)
    portfolio_fx.plot_normalized_returns(settings['symbol1'], s1, settings['symbol2'], s2, portfolio)
    portfolio_fx.print_kpis(portfolio['nav'])

    return portfolio, mavg_zscore



def pair_trade_v2_with_hurst_and_hedge(settings, quotes, spread):
    portfolio = portfolio_fx.init_portfolio()
    vol_limit = pd.Series()
    hursts = pd.Series()

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

        # c = coint(s1, s2)
        # correlations.loc[day] = c[1]
        # zscore_correlations.loc[day] = base_fx.get_zscore(correlations,
        #                                                   correlations.mean(),
        #                                                   correlations.std())[-1]  # base_fx.get_mavg(correlations, 450)
        #
        # zscore_vol.loc[day] = zscore_correlations.std()
        #
        # # print(c[1])
        # correlation_exists = c[1] < settings['coint_limit'] # abs(zscore_correlations.loc[day]) < settings['coint_limit']  # c[1] < zscore_correlations.loc[day] #settings['coint_limit']

        hurst = get_hurst_exp(settings, ratios.tail(365))
        hursts.loc[day] = hurst


        correlation_exists = hurst < 0.5
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
                                                              settings['leverage_limit'], settings['max_leverage'], portfolio, spread.loc[day])

        elif mavg_zscore[day] < -settings['limit'] and new_trend == "UP" and within_vol and correlation_exists:
            if portfolio['holdings']['s1'] == None and portfolio['holdings']['s2'] == None:
                ## buy
                portfolio = portfolio_fx.buy_pair_with_hedge(day, settings['symbol1'], s1.loc[day], settings['symbol2'], s2.loc[day],
                                                             mavg_zscore[day],
                                                             settings['leverage_limit'], settings['max_leverage'], portfolio, spread.loc[day])

        elif abs(mavg_zscore[day]) < settings['exit_limit'] or not within_vol or not correlation_exists:
            if portfolio['holdings']['s1'] != None and portfolio['holdings']['s2'] != None:
                ## exit
                portfolio = portfolio_fx.exit_pairs(day, settings['symbol1'], s1.loc[day], settings['symbol2'], s2.loc[day],
                                                    mavg_zscore[day],
                                                    settings['leverage_limit'], settings['max_leverage'], portfolio)

        portfolio = portfolio_fx.calculate_nav(day, portfolio, s1.loc[day], s2.loc[day])

    portfolio['nav'] = portfolio['nav'].set_index('date')

    portfolio_fx.plot_objects([{'label': 'hursts', 'value': hursts}])

    portfolio_fx.plot_normal_zscore(base_fx.get_zscore(ratios, ratios.mean(), np.std(ratios)))
    portfolio_fx.plot_indicators(mavg_zscore, macd_fast, macd_slow, vol, vol_mean, vol_limit)
    portfolio_fx.plot_normalized_returns(settings['symbol1'], s1, settings['symbol2'], s2, portfolio)
    portfolio_fx.print_kpis(portfolio['nav'])

    return portfolio, mavg_zscore



def get_hurst_exp(settings, ratios):
    if len(ratios) < settings['hurst_period']:
        return False

    lag1 = 2
    lags = range(lag1, settings['hurst_period'])
    tau = [np.sqrt(np.std(np.subtract(ratios[lag:], ratios[:-lag]))) for lag in lags]
    m = np.polyfit(np.log(lags), np.log(tau), 1)
    hurst = m[0]*2
    return hurst
