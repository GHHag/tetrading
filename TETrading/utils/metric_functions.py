import math
from decimal import Decimal

import numpy as np


def calculate_max_drawdown(price_series):
    """
    Calculates and returns the maximum drawdown of the
    given price series member.

    Parameters
    ----------
    :param price_series:
        'list' : A collection with price time series data.

    :return:
        'float'
    """

    max_drawdown = 0
    peak_value = (price_series[0] - 1)

    for index, equity in enumerate(price_series):
        if equity > peak_value:
            peak_value = equity
            trough_value = min(price_series[index:])
            drawdown = (trough_value - peak_value) / peak_value

            if drawdown < max_drawdown:
                max_drawdown = drawdown

    return abs(max_drawdown * 100)


def calculate_cagr(initial_value, final_value, num_of_periods, yearly_periods=251):
    """
    Calculates and returns the compound annual growth rate.

    Parameters
    ----------
    :param initial_value:
        'float/Decimal' : The initial value.
    :param final_value:
        'float/Decimal' : The final value.
    :param num_of_periods:
        'int' : The number of periods in the dataset which
        has been used to get the values of initial_value and
        final_value.
    :param yearly_periods:
        Keyword arg 'int' : The number of periods in a trading year
        for the time frame in the dataset. Default value=251

    :return:
        'float'
    """

    years = num_of_periods / yearly_periods

    if final_value < 0:
        final_value += abs(final_value)
        initial_value += abs(final_value)

    try:
        cagr = math.pow((final_value / initial_value), (1 / years)) - 1
    except (ValueError, ZeroDivisionError):
        return 0

    return cagr * 100


def calculate_sharpe_ratio(
    returns_list, risk_free_rate=0.05, yearly_periods=251
):
    """
    Calculates and returns the annualized sharpe ratio.

    Parameters
    ----------
    :param returns_list:
        'list' : A list containing a sequence of returns.
    :param risk_free_rate:
        Keyword arg 'Decimal' : The yearly return of a risk free asset.
        Default value=0.05
    :param yearly periods:
        Keyword arg 'int' : The number of periods in a trading year
        for the time frame of the dataset. Default value=251

    :return:
        'Decimal'
    """

    excess_returns = returns_list - risk_free_rate / yearly_periods
    if not len(excess_returns) > 0:
        return np.nan
    else:
        return np.sqrt(yearly_periods) * np.mean(excess_returns) / \
            np.std(excess_returns)
