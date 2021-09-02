import random

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mtick

from TETrading.position.position_manager import PositionManager
from TETrading.utils.metric_functions import calculate_cagr


def monte_carlo_simulate_returns(positions, symbol, num_testing_periods, start_capital=10000, capital_fraction=1.0,
                                 num_of_sims=1000, data_amount_used=0.25, print_dataframe=True,
                                 plot_fig=False, save_fig_to_path=None):
    """
    Simulates equity curves from a given sequence of Position objects.

    Parameters
    ----------
    :param positions:
        'List' : A collection of Position objects.
    :param symbol:
        'str' : The symbol/ticker of an asset.
    :param num_testing_periods:
        'int' : The number of periods in the dataset used
        to generate positions.
    :param start_capital:
        Keyword arg 'int/float' : The amount of starting capital.
        Default value=10000
    :param capital_fraction:
        Keyword arg 'float' : The fraction of capital to be used
        to purchase assets. Default value=1.0
    :param num_of_sims:
        'Keyword arg 'int' : The number of simulations to run.
        Default value=1000
    :param data_amount_used:
        Keyword arg 'float' : The fraction of historic positions to
        use in the simulation output. Default value=0.5
    :param print_dataframe:
        Keyword arg 'Boolean' : True/False decides whether to print
        the dataframe to console or not. Default value=True
    :param plot_fig:
        Keyword arg 'Boolean' : True/False decides whether to plot
        the figure or not. Default value=True
    :param save_fig_to_path:
        Keyword arg 'None/str' : Provide a file path as a string
        to save the plot as a file. Default value=None
    :return:
        'List'
    """

    monte_carlo_sims_data = []

    equity_curves_list = []
    eq_curve_final_equity = []
    max_drawdowns = []
    sim_positions = None

    def generate_pos_sequence(position_list, **kwargs):
        """
        Generates positions from given list of objects of type Position.
        The list will be sliced at a percentage of the total amount of
        positions, determined by 'data_amount_used'.

        Parameters
        ----------
        :param position_list:
            'List' : A list of Position objects.
        :param kwargs:
            'Dict' : A dict with additional keyword arguments which
            are never used, but might be provided depending on how the
            trading system logic and parameters are structured.
        """

        for x in position_list[:int(len(position_list) * data_amount_used)]:
            yield x

    for i in range(num_of_sims):
        sim_positions = PositionManager(symbol, (num_testing_periods * data_amount_used), start_capital,
                                        capital_fraction)

        tr_list = random.sample(positions, len(positions))
        sim_positions.generate_positions(generate_pos_sequence, tr_list)
        monte_carlo_sims_data.append(sim_positions.metrics.summary_data_dict)
        eq_curve_final_equity.append(float(sim_positions.metrics.equity_list[-1]))

        max_drawdowns.append(sim_positions.metrics.max_drawdown)

        equity_curves_list.append(sim_positions.metrics.equity_list)

    eq_curve_final_equity = sorted(eq_curve_final_equity)

    car25 = calculate_cagr(sim_positions.metrics.start_capital,
                           eq_curve_final_equity[(int(len(eq_curve_final_equity) * 0.25))],
                           sim_positions.metrics.num_testing_periods)
    car75 = calculate_cagr(sim_positions.metrics.start_capital,
                           eq_curve_final_equity[(int(len(eq_curve_final_equity) * 0.75))],
                           sim_positions.metrics.num_testing_periods)

    monte_carlo_sims_data[-1]['CAR25'] = car25
    monte_carlo_sims_data[-1]['CAR75'] = car75

    if print_dataframe:
        sim_data = pd.DataFrame(columns=list(monte_carlo_sims_data[-1].keys()))
        for data_dict in monte_carlo_sims_data:
            sim_data = sim_data.append(data_dict, ignore_index=True)
        print(sim_data.to_string())

    if plot_fig:
        monte_carlo_simulations_plot(symbol, equity_curves_list, max_drawdowns, eq_curve_final_equity,
                                     capital_fraction, car25, car75, save_fig_to_path=save_fig_to_path)

    return monte_carlo_sims_data


def monte_carlo_simulations_plot(symbol, simulated_equity_curves_list, max_drawdowns, eq_curve_final_equity,
                                 capital_fraction, car25, car75, save_fig_to_path=None):
    """
    Plots equity curves generated from a Monte Carlo simulation.
    Also plots inverse CDF of maximum drawdown and final equity
    of the simulated equity curves.

    Parameters
    ----------
    :param symbol:
        'str' : The symbol/ticker of an asset.
    :param simulated_equity_curves_list:
        'List' : A list containing simulated equity curves.
    :param max_drawdowns:
        'List' : A list containing the maximum drawdowns of
        the simulated equity curves.
    :param eq_curve_final_equity:
        'List' : A list containing the final equity of the
        simulated equity curves.
    :param capital_fraction:
        'float' : The fraction of capital that was used
        by the trading system to purchase assets.
    :param car25:
        'float' : The compound annual rate of return at the
        25th percentile of the CDF of simulated equity curves
    :param car75:
        'float' : The compound annual rate of return at the
        75th percentile of the CDF of simulated equity curves
    :param save_fig_to_path:
        Keyword arg 'None/str' : Provide a file path as a string
        to save the plot as a file. The file name is hard coded
        to be of the format 'path\\symbol_mc_sims.jpg'.
        Default value=None
    """

    plt.style.use('seaborn')
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(9, 9.5))
    fig.tight_layout()

    num_of_sims = len(simulated_equity_curves_list)

    for eq in simulated_equity_curves_list:
        axs[0].plot(eq, linewidth=0.5)
        axs[0].grid(True)
        axs[0].set_title(f'{str(num_of_sims)} Monte Carlo simulations of returns ({symbol})')
        axs[0].set_xlabel('Periods')
        axs[0].set_ylabel('Equity')

    hmdd_patch = mpatches.Patch(label='Highest max drawdown ' + str(round(max(max_drawdowns), 2)) + '%')
    lmdd_patch = mpatches.Patch(label='Lowest max drawdown ' + str(round(min(max_drawdowns), 2)) + '%')
    axs[0].legend(handles=[hmdd_patch, lmdd_patch])

    axs[1].plot(eq_curve_final_equity, color='royalblue')
    axs[1].xaxis.set_major_formatter(mtick.PercentFormatter(xmax=num_of_sims))
    axs[1].set_xticks(np.arange(0, num_of_sims + 1, num_of_sims * 0.25))
    axs[1].set_title('Inverse CDF of final equity when traded at f=' + str(capital_fraction))
    axs[1].set_xlabel('Percentile')
    axs[1].set_ylabel('Final equity')

    min_equity = min(eq_curve_final_equity)
    twtyfifth_pctl = num_of_sims * 0.25
    twtyfifth_pctl_ret = eq_curve_final_equity[(int(len(eq_curve_final_equity) * 0.25))]
    axs[1].plot([twtyfifth_pctl, twtyfifth_pctl], [min_equity, twtyfifth_pctl_ret],
                color='purple', linestyle=(0, (3, 1, 1, 1)))
    axs[1].plot([0, twtyfifth_pctl], [twtyfifth_pctl_ret, twtyfifth_pctl_ret],
                color='purple', linestyle=(0, (3, 1, 1, 1)))

    svntyfifth_pctl = num_of_sims * 0.75
    svntyfifth_pctl_ret = eq_curve_final_equity[(int(len(eq_curve_final_equity) * 0.75))]
    axs[1].plot([svntyfifth_pctl, svntyfifth_pctl], [min_equity, svntyfifth_pctl_ret],
                color='black', linestyle=(0, (3, 1, 1, 1)))
    axs[1].plot([0, svntyfifth_pctl], [svntyfifth_pctl_ret, svntyfifth_pctl_ret],
                color='black', linestyle=(0, (3, 1, 1, 1)))

    car25_patch = mpatches.Patch(label='CAR25 ' + str(round(car25, 2)), color='purple')
    car75_patch = mpatches.Patch(label='CAR75 ' + str(round(car75, 2)), color='black')
    axs[1].legend(handles=[car25_patch, car75_patch])

    axs[2].plot(sorted(max_drawdowns), color='royalblue')
    axs[2].xaxis.set_major_formatter(mtick.PercentFormatter(xmax=num_of_sims))
    axs[2].set_xticks(np.arange(0, num_of_sims + 1, num_of_sims * 0.25))
    axs[2].set_title('Inverse CDF of max drawdown when traded at f=' + str(capital_fraction))
    axs[2].set_xlabel('Percentile')
    axs[2].set_ylabel('Max drawdown (%)')

    plt.tight_layout()

    if save_fig_to_path:
        plt.savefig(save_fig_to_path + f'\\{symbol}_mc_sims.jpg')

    plt.show()


def monte_carlo_simulation_series_write(data_dicts_list):
    """
    A function that receives data from multiple Monte Carlo
    simulations, summarizes and returns the data in a dictionary.

    Parameters
    ----------
    :param data_dicts_list:
        'List' : A list with dictionaries containing data from
        Monte Carlo simulations.
    :return:
        'Dict'
    """

    summarized_data = {k: [] for k in data_dicts_list[-1].keys()}
    for metrics_dict in data_dicts_list:
        for metric, value in metrics_dict.items():
            summarized_data[metric].append(value)

    monte_carlo_summmary_data_dict = dict()
    monte_carlo_summmary_data_dict['Ticker'] = summarized_data['Ticker'][0]
    monte_carlo_summmary_data_dict['Start capital'] = summarized_data['Start capital'][0]
    monte_carlo_summmary_data_dict['Median gross profit'] = round(np.median(summarized_data['Total gross profit']), 2)
    monte_carlo_summmary_data_dict['Median profit factor'] = round(np.median(summarized_data['Profit factor']), 3)
    monte_carlo_summmary_data_dict['Median expectancy'] = round(np.median(summarized_data['Expectancy']), 3)
    monte_carlo_summmary_data_dict['Avg RoR'] = str(round(np.mean(summarized_data['Rate of return']), 2))
    monte_carlo_summmary_data_dict['Median RoR'] = str(round(np.median(summarized_data['Rate of return']), 2))
    monte_carlo_summmary_data_dict['Median max drawdown (%)'] = \
        str(round(np.median(summarized_data['Max drawdown (%)']), 2))
    monte_carlo_summmary_data_dict['Avg RoMad'] = str(round(np.mean(summarized_data['RoMad']), 2))
    monte_carlo_summmary_data_dict['Median RoMad'] = str(round(np.median(summarized_data['RoMad']), 2))
    monte_carlo_summmary_data_dict['Avg CAGR (%)'] = str(round(np.mean(summarized_data['CAGR (%)']), 2))
    monte_carlo_summmary_data_dict['Median CAGR (%)'] = str(round(np.median(summarized_data['CAGR (%)']), 2))
    monte_carlo_summmary_data_dict['Mean % wins'] = np.mean(summarized_data['% wins'])
    monte_carlo_summmary_data_dict['CAR25'] = summarized_data['CAR25'][-1]
    monte_carlo_summmary_data_dict['CAR75'] = summarized_data['CAR75'][-1]

    return monte_carlo_summmary_data_dict


def monte_carlo_simulate_best_estimated_trades(best_estimate_trades, per_len, num_of_trades=500,
                                               capital=10000, safe_f=1.0, num_of_sims=1000,
                                               plot_fig=False, save_fig_to_path=None, print_dataframe=False):
    """
    Simulates randomized sequences of given trades and
    returns data generated from the simulations.

    Parameters
    ----------
    :param best_estimate_trades:
        'List' : A collection of trades.
    :param per_len:
        'int' : The number of periods in the data
        used to generate the collection of trades.
    :param num_of_trades:
        Keyword arg 'int' : The number of trades to
        use in the Monte Carlo simulation.
        Default value=500
    :param capital:
        Keyword arg 'int/float' : The amount of capital
        to purchase assets for. Default value=10000
    :param safe_f:
        Keyword arg 'float' : The fraction of capital to be
        used to purchase assets. Default value=1.0
    :param num_of_sims:
        Keyword arg 'int' : The number of simulation to run.
        Default value=1000
    :param plot_fig:
        Keyword arg 'Boolean' : True/False decides whether
        to plot a figure with data from the Monte Carlo
        simulations or not. Default value=False
    :param save_fig_to_path:
        Keyword arg 'None/str' : Provide a file path as a
        string to save the plot as a file. Default value=None
    :param print_dataframe:
        Keyword arg 'Boolean' : True/False decides whether to print
        the dataframe to console or not. Default value=False
    :return:
        'Pandas DataFrame'
    """

    # data amount for monte carlo simulations
    data_used = 1.0
    if len(best_estimate_trades) >= num_of_trades:
        data_fraction = len(best_estimate_trades) / num_of_trades
        data_used = data_used / data_fraction

    period_len = int(per_len * data_used)
    # sort trades on date
    best_estimate_trades.sort(key=lambda x: x.entry_date)
    monte_carlo_sims_df = monte_carlo_simulate_returns(best_estimate_trades
                                                       [-(int(len(best_estimate_trades) * data_used)):],
                                                       '', period_len, capital, safe_f,
                                                       plot_fig=plot_fig, num_of_sims=num_of_sims,
                                                       data_amount_used=0.5,
                                                       save_fig_to_path=save_fig_to_path,
                                                       print_dataframe=print_dataframe)
    return monte_carlo_sims_df


def monte_carlo_simulate_trade_sequence(trades, num_testing_periods, start_capital, capital_fraction=1.0,
                                        num_of_sims=1000, data_amount_used=0.5, symbol='', print_dataframe=True):
    """
    Randomizes the order of a sequence of trades, calculates
    metrics for system evaluation and stores them in a Pandas
    DataFrame which is returned by the function.

    Parameters
    ----------
    :param trades:
        'List' : A collection of trades.
    :param num_testing_periods:
        'int' : The number of periods in the dataset.
    :param start_capital:
        'int/float' : The amount of starting capital.
    :param capital_fraction:
        Keyword arg 'float' : The fraction of capital to be used
        to purchase assets. Default value=1.0
    :param num_of_sims:
        Keyword arg 'int' : The number of simulations to run.
        Default value=1000
    :param data_amount_used:
        Keyword arg 'float' : The fraction of historic trades to
        use in the simulation output. Default value=0.5
    :param symbol:
        Keyword arg 'str' : The symbol/ticker of an asset.
        Default value='', empty string
    :param print_dataframe:
        Keyword arg 'Boolean' : True/False decides if the DataFrame
        with metrics and statistics should be printed to console.
        Default value=True
    :return:
        'Pandas DataFrame'
    """

    monte_carlo_sims_df = pd.DataFrame()

    final_equity = []
    max_drawdowns = []
    sim_trades = None

    def generate_trade_sequence(trades_list, **kwargs):
        """
        Generates positions from given list of objects of type Position.
        The list will be sliced at a percentage of the total amount of
        positions, determined by 'data_amount_used'.

        Parameters
        ----------
        :param trades_list:
            'List' : A list of Position objects.
        :param kwargs:
            'Dict' : A dict with additional keyword arguments which
            are never used, but might be provided depending on how the
            trading system logic and parameters are structured.
        """

        for trade in trades_list[:int(len(trades_list) * data_amount_used)]:
            yield trade

    for i in range(num_of_sims):
        sim_trades = PositionManager(symbol, (num_testing_periods * data_amount_used), start_capital, capital_fraction)
        tr_list = random.sample(trades, len(trades))
        sim_trades.generate_positions(generate_trade_sequence, tr_list)

        monte_carlo_sims_df = monte_carlo_sims_df.append(sim_trades.metrics.summary_data_dict, ignore_index=True)

        final_equity.append(sim_trades.metrics.equity_list[-1])
        max_drawdowns.append(sim_trades.metrics.max_drawdown)

    final_equity = sorted(final_equity)
    car25 = calculate_cagr(sim_trades.metrics.start_capital, final_equity[(int(len(final_equity) * 0.25))],
                           sim_trades.metrics.num_testing_periods)
    car75 = calculate_cagr(sim_trades.metrics.start_capital, final_equity[(int(len(final_equity) * 0.75))],
                           sim_trades.metrics.num_testing_periods)

    car_series = pd.Series()
    car_series['CAR25'] = car25
    car_series['CAR75'] = car75
    monte_carlo_sims_df = monte_carlo_sims_df.append(car_series, ignore_index=True)

    if print_dataframe:
        print(monte_carlo_sims_df.to_string())

    return monte_carlo_sims_df


def calculate_safe_f(best_estimate_trades, period_len, tolerated_pct_dd, tolerated_dd_percentile,
                     forecast_trades=500, forecast_data_fraction=0.5, capital=10000, num_of_sims=2500,
                     symbol='', print_dataframe=False):
    """
    Calls method to simulate given sequence of trades and
    calculates and returns the safe-F metric.

    Parameters
    ----------
    :param best_estimate_trades:
        'List' : A sequence of trades.
    :param period_len:
        'int' : The number or periods in the dataset.
    :param tolerated_pct_dd:
        'float/int' : The percentage amount of drawdown that
        will be tolerated.
    :param tolerated_dd_percentile:
        'float' : The percentile of the distribution of maximum
        drawdowns to act as a threshold for the tolerated maximum
        drawdown, e.g. when declaring the variables with the
        following values:
        tolerated_pct_max_drawdown = 15
        max_drawdown_percentile_threshold = 0.8
        The Safe-F will be held at a level so that 80% of the
        distribution of maximum drawdown values will be 15% or less.
    :param forecast_trades:
        Keyword arg 'int' : The maximum number of trades to use in the
        simulation. Default value=500
    :param forecast_data_fraction:
        Keyword arg 'float' : The fraction of data to use in the
        simulation. Default value=0.5
    :param capital:
        Keyword arg 'int/float' : The amount of capital to purchase
        assets for. Default value=10000
    :param num_of_sims:
        Keyword arg 'int' : The number of simulations to run.
        Default value=1000
    :param symbol:
        Keyword arg 'str' : The symbol/ticker of an asset.
        Default value='', empty string
    :param print_dataframe:
        Keyword arg 'Boolean' : True/False decides if the DataFrame
        with metrics and statistics should be printed to console.
        Default value=True
    :return:
        'float'
    """

    split_data_fraction = 1.0
    if len(best_estimate_trades) >= forecast_trades:
        split_data_fraction = forecast_trades / len(best_estimate_trades)

    period_len = int(period_len * split_data_fraction)
    # sort trades on date
    best_estimate_trades.sort(key=lambda tr: tr.entry_date)
    monte_carlo_sims_df = monte_carlo_simulate_trade_sequence(
        best_estimate_trades[-(int(len(best_estimate_trades) * split_data_fraction)):], period_len, capital,
        num_of_sims=num_of_sims, data_amount_used=forecast_data_fraction, symbol=symbol,
        print_dataframe=print_dataframe)

    max_dds = sorted(monte_carlo_sims_df['Max drawdown (%)'].to_list())
    dd_at_tolerated_percentile = max_dds[int(len(max_dds) * tolerated_dd_percentile)]
    safe_f = (dd_at_tolerated_percentile - tolerated_pct_dd) / dd_at_tolerated_percentile

    return safe_f
