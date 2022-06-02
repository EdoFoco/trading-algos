import functions.functions as base_fx
import functions.portfolio_functions as portfolio_fx
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def mean_reversion_zscore(settings, quotes):
    portfolio = portfolio_fx.init_portfolio()
    portfolio['holdings'][settings['symbol']] = None

    for day, row in quotes.iterrows():
        ## Calculate indicators
        filtered_quotes = quotes.loc[:day]
        s1 = filtered_quotes[settings['symbol']]

        mavg_fast_r = base_fx.get_mavg(s1, settings['mavg_1'])
        mavg_slow_r = base_fx.get_mavg(s1, settings['mavg_2'])

        std_r = base_fx.get_std(s1, settings['mavg_2'])

        if mavg_fast_r is None or mavg_slow_r is None or std_r is None:
            continue

        mavg_zscore = base_fx.get_zscore(mavg_fast_r, mavg_slow_r, std_r)

        macd_fast = base_fx.get_macd(mavg_zscore, settings['macd_fast_points'])
        macd_slow = base_fx.get_macd(mavg_zscore, settings['macd_slow_points'])

        # vol = ratios.rolling(settings['vol_interval']).std(ddof=0)

        # Get top 30 vol peaks
        # sorted_vol = vol.tail(settings['vol_peak_period']).sort_values(ascending=False).head(
        #    settings['vol_mean_peak_period'])

        # vol_mean = sorted_vol.mean()
        # if len(vol_limit) == 0:
        #      vol_limit = pd.Series(index=vol.index)
        #
        #  vol_limit.loc[day] = vol_mean
        #
        #  ## Trade
        #  within_vol = not math.isnan(vol.loc[day]) and vol.loc[day] < vol_limit[day]
        within_vol = True

        if macd_slow[day] <= macd_fast[day]:
            new_trend = "UP"
        else:
            new_trend = "DOWN"

        if mavg_zscore[day] > settings['limit'] and new_trend == "DOWN" and within_vol:
            if portfolio['holdings'][settings['symbol']] is None:
                ## sell
                portfolio = portfolio_fx.sell(day, settings['symbol'], s1.loc[day],
                                              mavg_zscore[day],
                                              settings['leverage_limit'], settings['max_leverage'], portfolio)

        elif mavg_zscore[day] < -settings['limit'] and new_trend == "UP" and within_vol:
            if portfolio['holdings'][settings['symbol']] is None:
                ## buy
                portfolio = portfolio_fx.buy(day, settings['symbol'], s1.loc[day],
                                             mavg_zscore[day],
                                             settings['leverage_limit'], settings['max_leverage'], portfolio)

        elif abs(mavg_zscore[day]) < settings['exit_limit'] or not within_vol:
            if portfolio['holdings'][settings['symbol']] is not None:
                ## exit
                portfolio = portfolio_fx.exit(day, settings['symbol'], s1.loc[day], portfolio)

        portfolio = portfolio_fx.calculate_nav(day, portfolio, settings['symbol'], s1.loc[day])

    portfolio['nav'] = portfolio['nav'].set_index('date')

    # portfolio_fx.plot_ratios(ratios)
    # portfolio_fx.plot_normal_zscore(base_fx.get_zscore(ratios, ratios.mean(), np.std(ratios)))
    # portfolio_fx.plot_indicators(mavg_zscore, macd_fast, macd_slow, vol, vol_mean, vol_limit)
    # portfolio_fx.plot_normalized_returns(settings['symbol1'], s1, settings['symbol2'], s2, portfolio)
    # portfolio_fx.print_kpis(portfolio['nav'])

    return portfolio, mavg_zscore


def get_bollinger_bands(window, no_of_std, ts):
    # Calculate rolling mean and standard deviation using number of days set above
    rolling_mean = ts.rolling(window).mean()
    rolling_std = ts.rolling(window).std()
    # create two new DataFrame columns to hold values of upper and lower Bollinger bands
    df = pd.DataFrame(columns=['mean', 'high', 'low'])
    df['mean'] = rolling_mean
    df['high'] = rolling_mean + (rolling_std * no_of_std)
    df['low'] = rolling_mean - (rolling_std * no_of_std)

    return df


def mean_reversion_bbands(settings, quotes):
    portfolio = portfolio_fx.init_portfolio()
    portfolio['holdings'][settings['symbol']] = None
    mavg_fast_hist = pd.Series()
    mavg_slow_hist = pd.Series()
    bbands_hist = None

    for day, row in quotes.iterrows():
        ## Calculate indicators
        # d = day - timedelta(days=1)
        filtered_quotes = quotes.loc[:day]

        s1 = filtered_quotes[settings['symbol']]

        mavg_fast = base_fx.get_mavg(s1, settings['mavg_1'])
        mavg_slow = base_fx.get_mavg(s1, settings['mavg_2'])

        if mavg_fast is None or mavg_slow is None:
            continue

        mavg_fast_hist[day] = mavg_fast[day]
        mavg_slow_hist[day] = mavg_slow[day]

        bbands = get_bollinger_bands(settings['bband_period'], settings['bband_std'], s1)
        if bbands_hist is None:
            bbands_hist = bbands
        else:
            bbands_hist.loc[day] = bbands.loc[day]

        if mavg_fast[day] >= mavg_slow[day]:
            new_trend = "UP"
        else:
            new_trend = "DOWN"

        if s1[day] >= bbands['high'][day] and new_trend == "DOWN":
            # how distant the actual price is from the signal
            diff = 1 - s1[day] / mavg_slow[day]
            diff = diff * 100

            # mavgs diff -> to check if this is a false signal.
            # e.g. the signal was triggered some time ago, the mavgs have diverted in the meanwhile
            # but still in range (this could happen when the trend has inverted and but mavgs haven't detected it yet
            mavg_diff = 1 - mavg_fast[day] / mavg_slow[day]
            mavg_diff = mavg_diff * 100
            # print(f'{day} - {mavg_diff}')

            if portfolio['holdings'][settings['symbol']] is None and abs(diff) < 0.1 and abs(mavg_diff) < 0.1:
                ## sell
                portfolio = portfolio_fx.sell(day, settings['symbol'], s1.loc[day],
                                              2, settings['leverage_limit'], settings['max_leverage'], portfolio,
                                              settings['fee'])

        elif s1[day] <= bbands['low'][day] and new_trend == "UP":
            diff = 1 - s1[day] / mavg_slow[day]
            diff = diff * 100
            #  print(abs(diff))

            mavg_diff = 1 - mavg_fast[day] / mavg_slow[day]
            mavg_diff = mavg_diff * 100
            # print(f'{day} - {mavg_diff}')
            if portfolio['holdings'][settings['symbol']] is None and abs(diff) < 0.1 and abs(mavg_diff) < 0.1:
                ## buy
                portfolio = portfolio_fx.buy(day, settings['symbol'], s1.loc[day],
                                             2, settings['leverage_limit'], settings['max_leverage'], portfolio,
                                             settings['fee'])

        elif portfolio['holdings'][settings['symbol']] is not None:
            ## if direction is up and (s1 > bband high or s1 > mean)
            ## if direction is down and (s1 < bband low or s1 < mean)
            ## if direction is inverted
            if new_trend == "UP" and (s1[day] > bbands['high'][day] or s1[day] > bbands['mean'][day]):
                portfolio = portfolio_fx.exit(day, settings['symbol'], s1.loc[day], portfolio)
            elif new_trend == "Down" and (s1[day] < bbands['low'][day] or s1[day] < bbands['mean'][day]):
                portfolio = portfolio_fx.exit(day, settings['symbol'], s1.loc[day], portfolio)
            elif old_trend != new_trend:
                portfolio = portfolio_fx.exit(day, settings['symbol'], s1.loc[day], portfolio)

        portfolio = portfolio_fx.calculate_nav(day, portfolio, settings['symbol'], s1.loc[day])
        portfolio = portfolio_fx.stop_loss(day, settings['symbol'], s1.loc[day], portfolio, settings['stop_loss'])

        old_trend = new_trend

    portfolio['nav'] = portfolio['nav'].set_index('date')

    # portfolio_fx.plot_ratios(ratios)
    # portfolio_fx.plot_normal_zscore(base_fx.get_zscore(ratios, ratios.mean(), np.std(ratios)))
    # portfolio_fx.plot_indicators(mavg_zscore, macd_fast, macd_slow, vol, vol_mean, vol_limit)
    # portfolio_fx.plot_normalized_returns(settings['symbol1'], s1, settings['symbol2'], s2, portfolio)
    # portfolio_fx.print_kpis(portfolio['nav'])

    return portfolio, {'bbands': bbands_hist, 'mavg_fast': mavg_fast_hist, 'mavg_slow': mavg_slow_hist}


def scalping_macd(settings, quotes):
    portfolio = portfolio_fx.init_portfolio()
    portfolio['holdings'][settings['symbol']] = None
    macd_13 = pd.Series()
    macd_21 = pd.Series()
    macd_34 = pd.Series()
    ema_21 = pd.Series()
    ema_34 = pd.Series
    ema_144 = pd.Series()

    for point, row in quotes.iterrows():
        ## Calculate indicators
        # d = day - timedelta(days=1)
        filtered_quotes = quotes.loc[:point]

        s1 = filtered_quotes[settings['symbol']]

        macd_13 = base_fx.get_real_macd(13, 21, 1, s1)
        macd_21 = base_fx.get_real_macd(21, 34, 1, s1)
        macd_34 = base_fx.get_real_macd(34, 144, 1, s1)

        ema_21 = base_fx.get_ema(21, s1)
        ema_34 = base_fx.get_ema(34, s1)
        ema_144 = base_fx.get_ema(144, s1)

        if len(macd_13) == 0 or len(macd_21) == 0 or len(macd_34) == 0 or len(ema_21) == 0 or len(ema_34) == 0 or len(
                ema_144) == 0:
            continue

        if macd_13[point] >= macd_21[point] and macd_13[point] >= macd_34[point] \
                and ema_21[point] >= ema_34[point] and ema_21[point] >= ema_144[point]:
            ## buy
            if portfolio['holdings'][settings['symbol']] is None:
                portfolio = portfolio_fx.buy(point, settings['symbol'], s1.loc[point],
                                              2, settings['leverage_limit'], settings['max_leverage'], portfolio,
                                              settings['fee'])

        elif macd_13[point]<= macd_21[point] and macd_13[point] <= macd_34[point] \
                and ema_21[point] <= ema_34[point] and ema_21[point] <= ema_144[point]:
            if portfolio['holdings'][settings['symbol']] is None:
                portfolio = portfolio_fx.sell(point, settings['symbol'], s1.loc[point],
                                             2, settings['leverage_limit'], settings['max_leverage'], portfolio,
                                             settings['fee'])

        portfolio = portfolio_fx.stop_loss(point, settings['symbol'], s1.loc[point], portfolio, settings['stop_loss'])
        portfolio = portfolio_fx.take_profit(point,  settings['symbol'], s1.loc[point], portfolio, settings['take_profit'])

        portfolio = portfolio_fx.calculate_nav(point, portfolio, settings['symbol'], s1.loc[point])

    portfolio['nav'] = portfolio['nav'].set_index('date')

    return portfolio, {'macd_13': macd_13, 'macd_21': macd_21, 'macd_34': macd_34, 'ema_21': ema_21, 'ema_34': ema_34, 'ema_144': ema_144 }
