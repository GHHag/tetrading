import math


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
        'int' : The number of periods in a trading year
        for the time frame in the dataset.

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
