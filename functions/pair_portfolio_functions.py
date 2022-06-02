import pandas as pd
import plotly.graph_objects as go
import functions.functions as base_fx
import numpy as np
from math import sqrt

def init_portfolio():
    return {
        'cash': 10000,
        'nav': 10000,
        'holdings': {
            's1': None,
            's2': None
        },
        'history': pd.DataFrame(columns=["symbol", "type", "amount", "bought_at", "bought_on", "status", "trigger_sl"]),
        'nav': pd.DataFrame(columns=['date', 'nav'])
    }


def plot_normal_zscore(zscore):
    fig = go.Figure()

    # print(macd_low)
    fig.add_trace(go.Scatter(x=zscore.index, y=zscore,
                             mode='lines',
                             name='zscore'))
    fig.show()



def plot_objects(objs):
    fig = go.Figure()

    for o in objs:
        # print(macd_low)
        fig.add_trace(go.Scatter(x=o['value'].index, y=o['value'],
                                 mode='lines',
                                 name=o['label']))

    fig.show()



def plot_correlations(correlations, mavg):
    fig = go.Figure()

    # print(macd_low)
    fig.add_trace(go.Scatter(x=correlations.index, y=correlations,
                             mode='lines',
                             name='corr'))

    fig.add_trace(go.Scatter(x=mavg.index, y=mavg,
                             mode='lines',
                             name='mavg'))
    fig.show()


def plot_ratios(ratios):
    fig = go.Figure()

    # print(macd_low)
    fig.add_trace(go.Scatter(x=ratios.index, y=ratios,
                             mode='lines',
                             name='ratios'))
    fig.show()


def plot_indicators(mavg_zscore, macd_fast, macd_slow, vol, vol_mean, vol_limit):
    fig = go.Figure()

    # print(macd_low)
    # fig.add_trace(go.Scatter(x=mavg_zscore.index, y=mavg_zscore,
    #                          mode='lines',
    #                          name='mavg_zscore'))
    #
    # fig.add_trace(go.Scatter(x=macd_fast.index, y=macd_fast,
    #                          mode='lines',
    #                          name='macd_fast'))
    #
    # fig.add_trace(go.Scatter(x=macd_slow.index, y=macd_slow,
    #                          mode='lines',
    #                          name='macd_slow'))

    fig.add_trace(go.Scatter(x=vol.index, y=vol,
                             mode='lines',
                             name='vol'))

    # fig.add_trace(go.Scatter(x=vol_mean.index, y=vol_mean,
    #                          mode='lines',
    #                          name='vol_mean'))

    fig.add_trace(go.Scatter(x=vol_limit.index, y=vol_limit,
                             mode='lines',
                             name='vol_limit'))

    fig.show()


def get_open_holding_nav(holding, current_price):
    if holding['type'] == 'BUY':
        holding['return'] = (current_price / holding['bought_at'] -1) * holding['leverage']
    else:
        holding['return'] = (1 - current_price / holding['bought_at']) * holding['leverage']
    holding['nav'] = holding['amount'] + (holding['return'] * holding['amount'])
    return holding['nav']


def get_holding_return_on_close(holding):
    if holding['type'] == 'BUY':
        holding['return'] = (holding['sold_at'] / holding['bought_at'] -1) * holding['leverage']
    else:
        holding['return'] = (1 - holding['sold_at'] / holding['bought_at'])  * holding['leverage']
    holding['nav'] = holding['amount'] + (holding['return'] * holding['amount'])
    return holding


def sell_pair(date, symbol1, s1_price, symbol2, s2_price, indicator_val, leverage_limit, max_leverage, portfolio):
    amount = portfolio['cash'] * 0.5
    leverage = 1
    if indicator_val > leverage_limit:
        leverage = max_leverage

    portfolio['holdings']['s1'] = {'amount': amount, 'bought_at': s1_price, 'type': 'SELL', 'bought_on': date, 'symbol': symbol1, 'status': 'OPEN', 'leverage': leverage}
    portfolio['holdings']['s2'] = {'amount': amount, 'bought_at': s2_price, 'type': 'BUY', 'bought_on': date, 'symbol': symbol2, 'status': 'OPEN', 'leverage': leverage}
    portfolio['cash'] = portfolio['cash'] - amount*2
    return portfolio


def buy_pair(date, symbol1, s1_price, symbol2, s2_price, indicator_val, leverage_limit, max_leverage, portfolio):
    amount = portfolio['cash'] * 0.5
    leverage = 1
    if indicator_val < -leverage_limit:
        leverage = max_leverage

    portfolio['holdings']['s1'] = {'amount': amount, 'bought_at': s1_price, 'type': 'BUY', 'bought_on': date, 'symbol': symbol1, 'status': 'OPEN', 'leverage': leverage}
    portfolio['holdings']['s2'] = {'amount': amount, 'bought_at': s2_price, 'type': 'SELL', 'bought_on': date, 'symbol': symbol2, 'status': 'OPEN', 'leverage': leverage}
    portfolio['cash'] = portfolio['cash'] - amount*2
    #print(portfolio['cash'])

    return portfolio


def sell_pair_with_hedge(date, symbol1, s1_price, symbol2, s2_price, indicator_val, leverage_limit, max_leverage, portfolio, hr):
    #hedge_amount = abs(spread) / 100
    hedge_amount = portfolio['cash'] * -hr
    print(f'HedgeRatio: {str(hr)}, HedgeAmount:{str(hedge_amount)}')
    equal_parts = portfolio['cash'] / 2
    s1_amount = equal_parts + hedge_amount
    s2_amount = portfolio['cash'] - s1_amount



    leverage = 1
    if indicator_val > leverage_limit:
        leverage = max_leverage

    portfolio['holdings']['s1'] = {'amount': s1_amount, 'bought_at': s1_price, 'type': 'SELL', 'bought_on': date, 'symbol': symbol1, 'status': 'OPEN', 'leverage': leverage}
    portfolio['holdings']['s2'] = {'amount': s2_amount, 'bought_at': s2_price, 'type': 'BUY', 'bought_on': date, 'symbol': symbol2, 'status': 'OPEN', 'leverage': leverage}
    portfolio['cash'] = portfolio['cash'] - s1_amount - s2_amount

    return portfolio


def buy_pair_with_hedge(date, symbol1, s1_price, symbol2, s2_price, indicator_val, leverage_limit, max_leverage, portfolio, hr):
    #hedge_amount = abs(spread) / 100
    hedge_amount = portfolio['cash'] * -hr
    print(f'HedgeRatio: {str(hr)}, HedgeAmount:{str(hedge_amount)}')
    equal_parts = portfolio['cash'] / 2
    s2_amount = equal_parts + hedge_amount
    s1_amount = portfolio['cash'] - s2_amount

    leverage = 1
    if indicator_val < -leverage_limit:
        leverage = max_leverage

    portfolio['holdings']['s1'] = {'amount': s1_amount, 'bought_at': s1_price, 'type': 'BUY', 'bought_on': date, 'symbol': symbol1, 'status': 'OPEN', 'leverage': leverage}
    portfolio['holdings']['s2'] = {'amount': s2_amount, 'bought_at': s2_price, 'type': 'SELL', 'bought_on': date, 'symbol': symbol2, 'status': 'OPEN', 'leverage': leverage}
    portfolio['cash'] = portfolio['cash'] - s1_amount - s2_amount

    return portfolio


def exit_pairs(date, symbol1, s1_price, symbol2, s2_price, indicator_val, leverage_limit, max_leverage, portfolio):
    if portfolio['holdings']['s1'] is not None:
        s1_res = portfolio['holdings']['s1']
        s1_res['sold_on'] = date
        s1_res['sold_at'] = s1_price
        s1_res['status'] = 'CLOSE'
        s1_res = get_holding_return_on_close(s1_res)
        #print(s1_res)
        portfolio['cash'] += s1_res['nav']
        portfolio['history'] = portfolio['history'].append(s1_res, ignore_index=True)
        portfolio['holdings']['s1'] = None

    if portfolio['holdings']['s2'] is not None:
        s2_res = portfolio['holdings']['s2']
        s2_res['sold_on'] = date
        s2_res['sold_at'] = s2_price
        s2_res['status'] = 'CLOSE'
        s2_res = get_holding_return_on_close(s2_res)
        portfolio['cash'] += s2_res['nav']
        portfolio['history'] = portfolio['history'].append(s2_res, ignore_index=True)
        portfolio['holdings']['s2'] = None

    return portfolio


def stop_loss_pair(date, s1_price, s2_price, portfolio, sl_value):
    if portfolio['holdings']['s1'] is not None:
       # print('hi')
        #print(portfolio['holdings']['s1'])
        s1_res = portfolio['holdings']['s1'].copy()
        s1_res['sold_on'] = date
        s1_res['sold_at'] = s1_price
        s1_res['status'] = 'CLOSE'
        s1_res['trigger_sl'] = True
        s1_res = get_holding_return_on_close(s1_res)
        #print(s1_res['return'])
        if s1_res['return'] < sl_value:
            print(f"Trigger SL on s1: {date} {s1_res['return']}")
            portfolio['cash'] += s1_res['nav']
            portfolio['history'] = portfolio['history'].append(s1_res, ignore_index=True)
            portfolio['holdings']['s1'] = None

        #print('after')
        #print(portfolio['holdings']['s1'])

    if portfolio['holdings']['s2'] is not None:
        s2_res = portfolio['holdings']['s2'].copy()
        s2_res['sold_on'] = date
        s2_res['sold_at'] = s2_price
        s2_res['status'] = 'CLOSE'
        s2_res['trigger_sl'] = True
        s2_res = get_holding_return_on_close(s2_res)

        if s2_res['return'] < sl_value:
            print(f"Trigger SL on  s2: {date} {s2_res['return']}")
            portfolio['cash'] += s2_res['nav']
            portfolio['history'] = portfolio['history'].append(s2_res, ignore_index=True)
            portfolio['holdings']['s2'] = None



    return portfolio


def calculate_nav(date, portfolio, s1_price, s2_price):
    s1_nav = 0
    s2_nav = 0
    if portfolio['holdings']['s1'] is not None:
        s1_nav = get_open_holding_nav(portfolio['holdings']['s1'], s1_price)

    if portfolio['holdings']['s2'] is not None:
        s2_nav = get_open_holding_nav(portfolio['holdings']['s2'], s2_price)

    holdings_nav = s1_nav + s2_nav
    portfolio['nav'] = portfolio['nav'].append({'date': date, 'nav': portfolio['cash'] + holdings_nav}, ignore_index=True)
    return portfolio


def plot_normalized_returns(s1_symbol, s1, s2_symbol, s2, portfolio):
    normalized_S1 = base_fx.normalize(s1)
    normalized_S2 = base_fx.normalize(s2)
    normalized_nav = base_fx.normalize(portfolio['nav'])

    benchmark = normalized_S1 * 0.5 + normalized_S2 * 0.5

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=benchmark.index, y=benchmark,
                        mode='lines',
                        name='benchmark'))

    #fig.add_trace(go.Scatter(x=normalized_S1.index, y=normalized_S1,
                       # mode='lines',
                       # name=chosen[0]))

    #fig.add_trace(go.Scatter(x=normalized_S2.index, y=normalized_S2,
                       # mode='lines',
                       # name=chosen[1]))
    fig.add_trace(go.Scatter(x=normalized_nav.index, y=normalized_nav['nav'],
                        mode='lines',
                        name='strategy'))

    fig.show()


def print_kpis(data):
    #data = data.astype({'nav': 'float'})
    normalized = base_fx.normalize(data)

    returns = np.log(normalized / normalized.shift(1))
    vol = returns.std() * 250 ** 0.5
    vol.columns = ['Vol']
    print('Return: ')
    print(str((normalized.iloc[-1]['nav'] - 1) * 100) + '%')

    print('Vol:')
    print(vol.iloc[-1])

    cum_ret = returns.cumsum() + 1
    print('Cum Sum:')
    print(cum_ret)

    sharpe = ((returns.mean() / returns.std()) * sqrt(252))
    print('Sharpe:')
    print(sharpe)

def get_monthly_kpis(data):
    if len(data) == 0:
        return None, None, None

    normalized = base_fx.normalize(data)
    returns = np.log(normalized / normalized.shift(1))
    vol = returns.std()
    vol.columns = ['Vol']
    ret = normalized.iloc[-1]['nav'] - 1
    sharpe = ret / vol.iloc[-1]
    #print(f'Return: {str(ret)}, Vol: {str(vol)}, Sharpe: {str(sharpe)}')

    return ret, vol.iloc[-1], sharpe


def plot_trades_on_zscore_for_symbol(mavg_zscore, portfolio, symbol):
    new_ratios = mavg_zscore.to_frame()
    # print(new_zscores)
    history = portfolio['history']

    history = history[history['symbol'] == symbol]
    trades = pd.DataFrame(columns=["Type" "BoughtOn", "SoldOn", "Return"])
    trades['Type'] = history["type"]
    trades['BoughtOn'] = history['bought_on']
    trades['SoldOn'] = history['sold_on']
    trades['Return'] = history['return']

    bought_on = trades['BoughtOn'].to_frame()
    bought_on['Type'] = trades['Type']
    bought_on.columns = ['Date', 'Type']
    bought_on.set_index('Date', inplace=True)
    new_ratios = new_ratios.join(bought_on)

    sold_on = trades['SoldOn'].to_frame()
    sold_on['exit'] = "Exit"
    sold_on.columns = ['Date', 'exit']
    sold_on['Exit'] = "EXIT"
    sold_on.set_index('Date', inplace=True)
    new_ratios = new_ratios.join(sold_on)

    bought_zscore = new_ratios[new_ratios['Type'] == 'BUY']
    sold_zscore = new_ratios[new_ratios['Type'] == 'SELL']
    exit_zscore = new_ratios[new_ratios['Exit'] == 'EXIT']

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=new_ratios.index, y=new_ratios[0],
                             mode='lines',
                             name='ratios'))

    fig.add_trace(go.Scatter(x=bought_zscore.index, y=bought_zscore[0],
                             mode='markers',
                             name='sell'))

    fig.add_trace(go.Scatter(x=sold_zscore.index, y=sold_zscore[0],
                             mode='markers',
                             name='buy'))

    fig.add_trace(go.Scatter(x=exit_zscore.index, y=exit_zscore[0],
                             mode='markers',
                             name='exit'))
    fig.show()


def plot_trades_on_series(symbol1, symbol2, s1, s2, portfolio):
    #print(s1)
    s1 = s1.to_frame()
    history = portfolio['history']
    print(portfolio['history'])

    s1_history = portfolio['history'][portfolio['history']['symbol'] == symbol1]
    s1_sell = s1_history[s1_history['type'] == "SELL"]
    s1_buy = s1_history[s1_history['type'] == "BUY"]

    s2_history = portfolio['history'][portfolio['history']['symbol'] == symbol2]
    s2_sell = s2_history[s2_history['type'] == "SELL"]
    s2_buy = s2_history[s2_history['type'] == "BUY"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s1.index, y=s1[symbol1],
                             mode='lines',
                             name='s1'))

    fig.add_trace(go.Scatter(x=s1_sell['bought_on'], y=s1_sell['bought_at'],
                             mode='markers',
                             name='sell_in'))

    fig.add_trace(go.Scatter(x=s1_sell['sold_on'], y=s1_sell['sold_at'],
                             mode='markers',
                             #color='orange',
                             name='sell_out'))

    fig.add_trace(go.Scatter(x=s1_buy['bought_on'], y=s1_buy['bought_at'],
                             mode='markers',
                             name='buy_in'))

    fig.add_trace(go.Scatter(x=s1_buy['sold_on'], y=s1_buy['sold_at'],
                             mode='markers',
                             # color='orange',
                             name='buy_out'))

    fig.show()

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=s1_history['sold_on'], y=s1_history['nav'],
                             mode='lines',
                             # color='orange',
                             name='s1_nav'))

    fig.add_trace(go.Scatter(x=s2_history['sold_on'], y=s2_history['nav'],
                             mode='lines',
                             # color='orange',
                             name='s2_nav'))

    fig.show()
