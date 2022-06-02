import pandas as pd
import plotly.graph_objects as go
import functions.functions as base_fx
import numpy as np
from math import sqrt

def init_portfolio():
    return {
        'cash': 100,
        'nav': 100,
        'holdings': {
        },
        'history': pd.DataFrame(columns=["symbol", "type", "amount", "bought_at", "bought_on", "status", "trigger_sl"]),
        'nav': pd.DataFrame(columns=['date', 'nav'])
    }


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


def sell(date, symbol, price, indicator_val, leverage_limit, max_leverage, portfolio, fee):
    amount = portfolio['cash']
    fee_amount = amount * fee
    amount = amount - fee_amount
    leverage = 1
    if indicator_val > leverage_limit:
        leverage = max_leverage

    portfolio['holdings'][symbol] = {'amount': amount, 'bought_at': price, 'type': 'SELL', 'bought_on': date, 'symbol': symbol, 'status': 'OPEN', 'leverage': leverage}
    portfolio['cash'] = portfolio['cash'] - amount - fee_amount
    return portfolio


def buy(date, symbol, price, indicator_val, leverage_limit, max_leverage, portfolio, fee):
    amount = portfolio['cash']
    fee_amount = amount * fee
    leverage = 1
    if indicator_val > leverage_limit:
        leverage = max_leverage

    portfolio['holdings'][symbol] = {'amount': amount, 'bought_at': price, 'type': 'BUY', 'bought_on': date, 'symbol': symbol, 'status': 'OPEN', 'leverage': leverage}
    portfolio['cash'] = portfolio['cash'] - amount - fee_amount
    return portfolio


def exit(date, symbol, price, portfolio):

    if portfolio['holdings'][symbol] is not None:
        s1_res = portfolio['holdings'][symbol]
        s1_res['sold_on'] = date
        s1_res['sold_at'] = price
        s1_res['status'] = 'CLOSE'
        s1_res = get_holding_return_on_close(s1_res)
        #print(s1_res)
        portfolio['cash'] += s1_res['nav']
        portfolio['history'] = portfolio['history'].append(s1_res, ignore_index=True)
        portfolio['holdings'][symbol] = None

    return portfolio


def stop_loss(date, symbol, price, portfolio, sl_value):
    if portfolio['holdings'][symbol] is not None:
       # print('hi')
        #print(portfolio['holdings']['s1'])
        s1_res = portfolio['holdings'][symbol].copy()
        s1_res['sold_on'] = date
        s1_res['sold_at'] = price
        s1_res['status'] = 'CLOSE'
        s1_res['trigger_sl'] = True
        s1_res = get_holding_return_on_close(s1_res)
        if s1_res['return'] < sl_value:
            print(f"Trigger SL on s1: {date} {s1_res['return']}")
            portfolio['cash'] += s1_res['nav']
            portfolio['history'] = portfolio['history'].append(s1_res, ignore_index=True)
            portfolio['holdings'][symbol] = None

    return portfolio


def take_profit(date, symbol, price, portfolio, tp_value):
    if portfolio['holdings'][symbol] is not None:
       # print('hi')
        #print(portfolio['holdings']['s1'])
        s1_res = portfolio['holdings'][symbol].copy()
        s1_res['sold_on'] = date
        s1_res['sold_at'] = price
        s1_res['status'] = 'CLOSE'
        s1_res['trigger_sl'] = True
        s1_res = get_holding_return_on_close(s1_res)
        #print(s1_res['return'])
        if s1_res['return'] >= tp_value:
            print(f"Trigger TP on s1: {date} {s1_res['return']}")
            portfolio['cash'] += s1_res['nav']
            portfolio['history'] = portfolio['history'].append(s1_res, ignore_index=True)
            portfolio['holdings'][symbol] = None

    return portfolio


def calculate_nav(date, portfolio, symbol, price):
    nav = 0
    if portfolio['holdings'][symbol] is not None:
        nav = get_open_holding_nav(portfolio['holdings'][symbol], price)

    portfolio['nav'] = portfolio['nav'].append({'date': date, 'nav': portfolio['cash'] + nav}, ignore_index=True)
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
