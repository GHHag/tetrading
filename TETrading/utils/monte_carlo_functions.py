import random
from typing import List

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mtick

from TETrading.data.metadata.trading_system_metrics import TradingSystemMetrics
from TETrading.data.metadata.trading_system_simulation_attributes import \
    TradingSystemSimulationAttributes
from TETrading.position.position import Position
from TETrading.position.position_manager import PositionManager
from TETrading.utils.metric_functions import calculate_cagr


def monte_carlo_simulate_returns(
    positions, symbol, num_testing_periods, start_capital=10000, 
    capital_fraction=1.0, num_of_sims=1000, data_amount_used=0.25, 
    print_dataframe=True, plot_fig=False, save_fig_to_path=None
):
    """
    Simulates equity curves from a given sequence of Position objects.

    Parameters
    ----------
    :param positions:
        'list' : A collection of Position objects.
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
        when simulating asset purchases. Default value=1.0
    :param num_of_sims:
        'Keyword arg 'int' : The number of simulations to run.
        Default value=1000
    :param data_amount_used:
        Keyword arg 'float' : The fraction of historic positions to
        use in the simulation output. Default value=0.25
    :param print_dataframe:
        Keyword arg 'bool' : True/False decides whether to print
        the dataframe to console or not. Default value=True
    :param plot_fig:
        Keyword arg 'bool' : True/False decides whether to plot
        the figure or not. Default value=True
    :param save_fig_to_path:
        Keyword arg 'None/str' : Provide a file path as a string
        to save the plot as a file. Default value=None

    :return:
        'list'
    """

    monte_carlo_sims_data = []
    equity_curves_list = []
    final_equity_list = []
    max_drawdowns_list = []
    sim_positions = None

    def generate_pos_sequence(position_list, **kwargs):
        """
        Generates positions from given list of objects of type Position.
        The list will be sliced at a percentage of the total amount of
        positions, determined by 'data_amount_used'.

        Parameters
        ----------
        :param position_list:
            'list' : A list of Position objects.
        :param kwargs:
            'dict' : A dict with additional keyword arguments which
            are never used, but might be provided depending on how the
            trading system logic and parameters are structured.
        """

        for pos in position_list[:int(len(position_list) * data_amount_used)]:
            yield pos

    for _ in range(num_of_sims):
        sim_positions = PositionManager(
            symbol, (num_testing_periods * data_amount_used), start_capital,
            capital_fraction
        )

        pos_list = random.sample(positions, len(positions))
        sim_positions.generate_positions(generate_pos_sequence, pos_list)
        monte_carlo_sims_data.append(sim_positions.metrics.summary_data_dict)
        final_equity_list.append(float(sim_positions.metrics.equity_list[-1]))

        max_drawdowns_list.append(sim_positions.metrics.max_drawdown)

        equity_curves_list.append(sim_positions.metrics.equity_list)

    final_equity_list = sorted(final_equity_list)

    car25 = calculate_cagr(
        sim_positions.metrics.start_capital, 
        final_equity_list[(int(len(final_equity_list) * 0.25))],
        sim_positions.metrics.num_testing_periods
    )
    car75 = calculate_cagr(
        sim_positions.metrics.start_capital,
        final_equity_list[(int(len(final_equity_list) * 0.75))],
        sim_positions.metrics.num_testing_periods
    )

    monte_carlo_sims_data[-1]['CAR25'] = round(car25, 3)
    monte_carlo_sims_data[-1]['CAR75'] = round(car75, 3)

    if print_dataframe:
        sim_data = pd.DataFrame(columns=list(monte_carlo_sims_data[-1].keys()))
        for data_dict in monte_carlo_sims_data:
            sim_data = sim_data.append(data_dict, ignore_index=True)
        print(sim_data.to_string())

    if plot_fig:
        monte_carlo_simulations_plot(
            symbol, equity_curves_list, max_drawdowns_list, final_equity_list,
            capital_fraction, car25, car75, save_fig_to_path=save_fig_to_path
        )

    return monte_carlo_sims_data


def monte_carlo_simulations_plot(
    symbol, simulated_equity_curves_list, max_drawdowns_list, final_equity_list,
    capital_fraction, car25, car75, save_fig_to_path=None
):
    """
    Plots equity curves generated from a Monte Carlo simulation.
    Also plots inverse CDF of maximum drawdown and final equity
    of the simulated equity curves.

    Parameters
    ----------
    :param symbol:
        'str' : The symbol/ticker of an asset.
    :param simulated_equity_curves_list:
        'list' : A list containing simulated equity curves.
    :param max_drawdowns_list:
        'list' : A list containing the maximum drawdowns of
        the simulated equity curves.
    :param final_equity_list:
        'list' : A list containing the final equity of the
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
        axs[0].set_title(f'{str(num_of_sims)} Monte Carlo simulations of returns {symbol}')
        axs[0].set_xlabel('Periods')
        axs[0].set_ylabel('Equity')

    hmdd_patch = mpatches.Patch(
        label=f'Highest max drawdown {np.max(max_drawdowns_list):.2f}%'
    )
    lmdd_patch = mpatches.Patch(
        label=f'Lowest max drawdown {np.min(max_drawdowns_list):.2f}%'
    )
    axs[0].legend(handles=[hmdd_patch, lmdd_patch])

    axs[1].plot(final_equity_list, color='royalblue')
    axs[1].xaxis.set_major_formatter(mtick.PercentFormatter(xmax=num_of_sims))
    axs[1].set_xticks(np.arange(0, num_of_sims + 1, num_of_sims * 0.25))
    axs[1].set_title('Inverse CDF of final equity when traded at f=' + str(capital_fraction))
    axs[1].set_xlabel('Percentile')
    axs[1].set_ylabel('Final equity')

    min_equity = np.min(final_equity_list)
    twentyfifth_pctl = num_of_sims * 0.25
    twentyfifth_pctl_ret = final_equity_list[(int(len(final_equity_list) * 0.25))]
    axs[1].plot(
        [twentyfifth_pctl, twentyfifth_pctl], [min_equity, twentyfifth_pctl_ret],
        color='purple', linestyle=(0, (3, 1, 1, 1))
    )
    axs[1].plot(
        [0, twentyfifth_pctl], [twentyfifth_pctl_ret, twentyfifth_pctl_ret],
        color='purple', linestyle=(0, (3, 1, 1, 1))
    )

    seventyfifth_pctl = num_of_sims * 0.75
    seventyfifth_pctl_ret = final_equity_list[(int(len(final_equity_list) * 0.75))]
    axs[1].plot(
        [seventyfifth_pctl, seventyfifth_pctl], [min_equity, seventyfifth_pctl_ret],
        color='black', linestyle=(0, (3, 1, 1, 1))
    )
    axs[1].plot(
        [0, seventyfifth_pctl], [seventyfifth_pctl_ret, seventyfifth_pctl_ret],
        color='black', linestyle=(0, (3, 1, 1, 1))
    )

    car25_patch = mpatches.Patch(label=f'CAR25 {car25:.2f}', color='purple')
    car75_patch = mpatches.Patch(label=f'CAR75 {car75:.2f}', color='black')
    axs[1].legend(handles=[car25_patch, car75_patch])

    axs[2].plot(sorted(max_drawdowns_list), color='royalblue')
    axs[2].xaxis.set_major_formatter(mtick.PercentFormatter(xmax=num_of_sims))
    axs[2].set_xticks(np.arange(0, num_of_sims + 1, num_of_sims * 0.25))
    axs[2].set_title('Inverse CDF of max drawdown when traded at f=' + str(capital_fraction))
    axs[2].set_xlabel('Percentile')
    axs[2].set_ylabel('Max drawdown (%)')

    plt.tight_layout()

    if save_fig_to_path:
        plt.savefig(save_fig_to_path + f'\\{symbol}_mc_sims.jpg')

    plt.show()


def monte_carlo_simulation_summary_data(data_dicts_list):
    """
    A function that receives data from multiple Monte Carlo
    simulations, summarizes and returns the data as a dictionary.

    Parameters
    ----------
    :param data_dicts_list:
        'list' : A list with dictionaries containing data from
        Monte Carlo simulations.

    :return:
        'dict'
    """

    summarized_data = {k: [] for k in data_dicts_list[-1].keys()}
    for metrics_dict in data_dicts_list:
        for metric, value in metrics_dict.items():
            summarized_data[metric].append(value)
            if metric == TradingSystemMetrics.EXPECTANCY and np.isnan(value):
                summarized_data[metric].remove(value)

    monte_carlo_summmary_data_dict = dict()
    monte_carlo_summmary_data_dict[TradingSystemSimulationAttributes.SYMBOL] = \
        summarized_data[TradingSystemSimulationAttributes.SYMBOL][0]
    monte_carlo_summmary_data_dict[TradingSystemSimulationAttributes.START_CAPITAL] = \
        summarized_data[TradingSystemSimulationAttributes.START_CAPITAL][0]
    monte_carlo_summmary_data_dict[TradingSystemSimulationAttributes.MEDIAN_GROSS_PROFIT] = \
        round(np.median(summarized_data[TradingSystemMetrics.TOTAL_GROSS_PROFIT]), 2)
    monte_carlo_summmary_data_dict[TradingSystemSimulationAttributes.MEDIAN_PROFIT_FACTOR] = \
        round(np.median(summarized_data[TradingSystemMetrics.PROFIT_FACTOR]), 3)
    monte_carlo_summmary_data_dict[TradingSystemSimulationAttributes.MEDIAN_EXPECTANCY] = \
        round(np.median(summarized_data[TradingSystemMetrics.EXPECTANCY]), 3)
    monte_carlo_summmary_data_dict[TradingSystemSimulationAttributes.AVG_RATE_OF_RETURN] = \
        str(round(np.mean(summarized_data[TradingSystemMetrics.RATE_OF_RETURN]), 2))
    monte_carlo_summmary_data_dict[TradingSystemSimulationAttributes.MEDIAN_RATE_OF_RETURN] = \
        str(round(np.median(summarized_data[TradingSystemMetrics.RATE_OF_RETURN]), 2))
    monte_carlo_summmary_data_dict[TradingSystemSimulationAttributes.MEDIAN_MAX_DRAWDOWN] = \
        str(round(np.median(summarized_data[TradingSystemMetrics.MAX_DRAWDOWN]), 2))
    monte_carlo_summmary_data_dict[TradingSystemSimulationAttributes.AVG_ROMAD] = \
        str(round(np.mean(summarized_data[TradingSystemMetrics.ROMAD]), 2))
    monte_carlo_summmary_data_dict[TradingSystemSimulationAttributes.MEDIAN_ROMAD] = \
        str(round(np.median(summarized_data[TradingSystemMetrics.ROMAD]), 2))
    monte_carlo_summmary_data_dict[TradingSystemSimulationAttributes.AVG_CAGR] = \
        str(round(np.mean(summarized_data[TradingSystemMetrics.CAGR]), 2))
    monte_carlo_summmary_data_dict[TradingSystemSimulationAttributes.MEDIAN_CAGR] = \
        str(round(np.median(summarized_data[TradingSystemMetrics.CAGR]), 2))
    monte_carlo_summmary_data_dict[TradingSystemSimulationAttributes.MEAN_PCT_WINS] = \
        np.mean(summarized_data[TradingSystemMetrics.PCT_WINS])

    return monte_carlo_summmary_data_dict


def monte_carlo_simulate_positions(
    positions, period_len, safe_f=1.0, forecast_positions=500, 
    forecast_data_fraction=0.5, capital=10000, num_of_sims=1000,
    plot_fig=False, save_fig_to_path=None, print_dataframe=False
):
    """
    Simulates randomized sequences of given positions and
    returns data generated from the simulations.

    Parameters
    ----------
    :param positions:
        'list' : A collection of Position objects.
    :param period_len:
        'int' : The number of periods in the data
        used to generate the collection of positions.
    :param safe_f:
        Keyword arg 'float' : The fraction of capital to be
        used to purchase assets. Default value=1.0
    :param forecast_positions:
        Keyword arg 'int' : The number of positions to
        use in the Monte Carlo simulation. Default value=500
    :param forecast_data_fraction:
        Keyword arg 'float' : The fraction of data to use in the
        simulation. Default value=0.5
    :param capital:
        Keyword arg 'int/float' : The amount of capital
        to purchase assets for. Default value=10000
    :param num_of_sims:
        Keyword arg 'int' : The number of simulations to run.
        Default value=1000
    :param plot_fig:
        Keyword arg 'bool' : True/False decides whether
        to plot a figure with data from the Monte Carlo
        simulations or not. Default value=False
    :param save_fig_to_path:
        Keyword arg 'None/str' : Provide a file path as a
        string to save the plot as a file. Default value=None
    :param print_dataframe:
        Keyword arg 'bool' : True/False decides whether to print
        the dataframe to console or not. Default value=False

    :return:
        'Pandas DataFrame'
    """

    # data amount for monte carlo simulations
    split_data_fraction = 1.0
    if len(positions) >= forecast_positions:
        split_data_fraction = forecast_positions / len(positions)

    period_len = int(period_len * split_data_fraction)
    # sort positions on date
    positions.sort(key=lambda x: x.entry_dt)

    monte_carlo_sims_dicts_list = monte_carlo_simulate_returns(
        positions[-(int(len(positions) * split_data_fraction)):], '', period_len, capital, safe_f,
        plot_fig=plot_fig, num_of_sims=num_of_sims, data_amount_used=forecast_data_fraction,
        save_fig_to_path=save_fig_to_path, print_dataframe=print_dataframe
    )

    return monte_carlo_sims_dicts_list


def calculate_safe_f(
    positions: List[Position], period_len, tolerated_pct_max_dd, 
    max_dd_pctl_threshold,
    forecast_data_fraction=0.5, capital=10000, num_of_sims=2500, 
    symbol='', print_dataframe=False
):
    """
    Calls method to simulate given sequence of positions and
    calculates and returns the safe-F metric.

    Parameters
    ----------
    :param positions:
        'list' : A list of Position objects.
    :param period_len:
        'int' : The number or periods in the dataset.
    :param tolerated_pct_max_dd:
        'float/int' : The percentage amount of drawdown that
        will be tolerated.
    :param max_dd_pctl_threshold:
        'float' : The percentile of the distribution of maximum
        drawdowns to act as a threshold for the tolerated maximum
        drawdown, e.g. when declaring the variables with the
        following values:
        tolerated_pct_max_drawdown = 15
        max_drawdown_percentile_threshold = 0.8
        The Safe-F will be held at a level so that 80% of the
        distribution of maximum drawdown values will be 15% or less.
    :param forecast_data_fraction:
        Keyword arg 'float' : The fraction of data to use in the
        simulation. Default value=0.5
    :param capital:
        Keyword arg 'int/float' : The amount of capital to purchase
        assets for. Default value=10000
    :param num_of_sims:
        Keyword arg 'int' : The number of simulations to run.
        Default value=2500
    :param symbol:
        Keyword arg 'str' : The symbol/ticker of an asset.
        Default value=''
    :param print_dataframe:
        Keyword arg 'bool' : True/False decides if the DataFrame
        with metrics and statistics should be printed to console.
        Default value=False

    :return:
        'float'
    """

    period_len = int(period_len * forecast_data_fraction)
    # sort positions on date
    positions.sort(key=lambda tr: tr.entry_dt)

    monte_carlo_sims_dicts_list = monte_carlo_simulate_returns(
        positions[-(int(len(positions) * forecast_data_fraction)):], symbol, period_len,
        start_capital=capital, num_of_sims=num_of_sims, data_amount_used=forecast_data_fraction,
        print_dataframe=print_dataframe
    )

    max_dds = np.sort([dd[TradingSystemMetrics.MAX_DRAWDOWN] for dd in monte_carlo_sims_dicts_list])
    dd_at_tolerated_threshold = max_dds[int(len(max_dds) * max_dd_pctl_threshold)]

    if dd_at_tolerated_threshold <= 0: dd_at_tolerated_threshold = 1
    safe_f = tolerated_pct_max_dd / dd_at_tolerated_threshold

    return safe_f
