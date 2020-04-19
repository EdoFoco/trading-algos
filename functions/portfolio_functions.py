import pandas as pd
import plotly.graph_objects as go
import functions.functions as base_fx
import numpy as np


def init_portfolio():
    return {
        'cash': 10000,
        'nav': 10000,
        'holdings': {
            's1': None,
            's2': None
        },
        'history': pd.DataFrame(columns=["symbol", "type", "amount", "bought_at", "bought_on", "status"]),
        'nav': pd.DataFrame(columns=['date', 'nav'])
    }


def plot_normal_zscore(zscore):
    fig = go.Figure()

    # print(macd_low)
    fig.add_trace(go.Scatter(x=zscore.index, y=zscore,
                             mode='lines',
                             name='zscore'))
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


def sell(date, symbol1, s1_price, symbol2, s2_price, indicator_val, leverage_limit, max_leverage, portfolio):
    amount = portfolio['cash'] * 0.5
    leverage = 1
    if indicator_val > leverage_limit:
        leverage = max_leverage

    portfolio['holdings']['s1'] = {'amount': amount, 'bought_at': s1_price, 'type': 'SELL', 'bought_on': date, 'symbol': symbol1, 'status': 'OPEN', 'leverage': leverage}
    portfolio['holdings']['s2'] = {'amount': amount, 'bought_at': s2_price, 'type': 'BUY', 'bought_on': date, 'symbol': symbol2, 'status': 'OPEN', 'leverage': leverage}
    portfolio['cash'] = portfolio['cash'] - amount*2
    return portfolio


def buy(date, symbol1, s1_price, symbol2, s2_price, indicator_val, leverage_limit, max_leverage, portfolio):
    amount = portfolio['cash'] * 0.5
    leverage = 1
    if indicator_val < -leverage_limit:
        leverage = max_leverage

    portfolio['holdings']['s1'] = {'amount': amount, 'bought_at': s1_price, 'type': 'BUY', 'bought_on': date, 'symbol': symbol1, 'status': 'OPEN', 'leverage': leverage}
    portfolio['holdings']['s2'] = {'amount': amount, 'bought_at': s2_price, 'type': 'SELL', 'bought_on': date, 'symbol': symbol2, 'status': 'OPEN', 'leverage': leverage}
    portfolio['cash'] = portfolio['cash'] - amount*2
    return portfolio


def exit(date, symbol1, s1_price, symbol2, s2_price, indicator_val, leverage_limit, max_leverage, portfolio):
    s1_res = portfolio['holdings']['s1']
    s1_res['sold_on'] = date
    s1_res['sold_at'] = s1_price
    s1_res['status'] = 'CLOSE'
    s1_res = get_holding_return_on_close(s1_res)

    s2_res = portfolio['holdings']['s2']
    s2_res['sold_on'] = date
    s2_res['sold_at'] = s2_price
    s2_res['status'] = 'CLOSE'
    s2_res = get_holding_return_on_close(s2_res)

    portfolio['cash'] = s1_res['nav'] + s2_res['nav']

    portfolio['nav'] = portfolio['nav'].append({'date': date, 'nav': portfolio['cash']}, ignore_index=True)

    portfolio['history'] = portfolio['history'].append(s1_res, ignore_index=True)
    portfolio['history'] = portfolio['history'].append(s2_res, ignore_index=True)

    portfolio['holdings']['s1'] = None
    portfolio['holdings']['s2'] = None

    return portfolio


def calculate_nav(date, portfolio, s1_price, s2_price):
    holdings_nav = 0
    if portfolio['holdings']['s1'] != None and portfolio['holdings']['s2'] != None:
        s1_nav = get_open_holding_nav(portfolio['holdings']['s1'], s1_price)
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
    normalized = base_fx.normalize(data)
    returns = np.log(normalized / normalized.shift(1))
    vol = returns.std() * 250 ** 0.5
    vol.columns = ['Vol']
    print('Return: ')
    print(str((normalized.iloc[-1]['nav'] - 1) * 100) + '%')

    print('Vol:')
    print(vol.iloc[-1])


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