import pandas as pd

def get_ratios(s1, s2):
    return s1 / s2


def get_mavg(data, points):
    if points > len(data):
        return None

    return data.rolling(window=points, center=False).mean()


def get_std(data, points):
    if points > len(data):
        return None

    return data.rolling(window=points, center=False).std()


def get_rolling_vol(data, period):
    ratios_vol = pd.DataFrame(index=data.index, columns=['Vol'])
    for i in range(period, len(data)):
        date = data.index[i]
        r = data[i - period: i]
        vol = r.std()
        ratios_vol.loc[date]['Vol'] = vol

    return ratios_vol['Vol']


def get_zscore(val1, val2, std):
    return (val1-val2) / std


def get_mavg_zscore(ratios, fast_mavg_limit, slow_mavg_limit):
    mavg_fast = get_mavg(ratios, fast_mavg_limit)
    if mavg_fast is None:
        return None

    mavg_slow = get_mavg(ratios, slow_mavg_limit)
    if mavg_slow is None:
        return None

    std = get_std(ratios, slow_mavg_limit)
    if std is None:
        return None

    return get_zscore(mavg_fast, mavg_slow, std)


def get_macd(data, points):
    return data.ewm(span=points, adjust=False).mean()


def get_real_macd(interval1, interval2, signal_interval, data):
    mavg1 = data.ewm(span=interval1, adjust=False).mean()
    mavg2 = data.ewm(span=interval2, adjust=False).mean()
    macd = mavg1-mavg2
    signal = macd.ewm(span=signal_interval, adjust=False).mean()
    return signal


def get_ema(interval, data):
    return data.ewm(span=interval, adjust=False).mean()


def normalize(data):
    return data / data.iloc[0]