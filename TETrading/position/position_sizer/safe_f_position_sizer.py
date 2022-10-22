import random
from typing import List

import pandas as pd

from TETrading.position.position import Position
from TETrading.position.position_sizer.position_sizer import IPositionSizer
from TETrading.position.position_manager import PositionManager
from TETrading.utils.metric_functions import calculate_cagr
from TETrading.utils.monte_carlo_functions import monte_carlo_simulations_plot


class SafeFPositionSizer(IPositionSizer):
    """
    Based on Monte Carlo simulations of a distribution of recent
    returns, risk and profit potential, levels are calculated and
    the trading systems health can be evaluated. Safe-F is the
    fraction of capital that can be safely traded at any point in
    time without passing the given threshold for how much drawdown,
    and the chance to reach that drawdown, the trader want to take,
    assuming the distribution of returns has the same characteristics
    going forward.

    Parameters
    ----------
    tolerated_pct_max_drawdown : 'float/int'
        The percentage amount of drawdown that will be tolerated.
    max_drawdown_percentile_threshold : 'float'
        The percentile of the distribution of maximum drawdowns to
        act as a threshold for the tolerated maximum drawdown, e.g.
        when declaring the variables with the following values:
        tolerated_pct_max_drawdown = 15
        max_drawdown_percentile_threshold = 0.8
        The Safe-F will be held at a level so that 80% of the
        distribution of maximum drawdown values will be 15% or less.
    """

    def __init__(
        self, objective_function_str, tolerated_pct_max_drawdown, 
        max_drawdown_percentile_threshold
    ):
        self.__position_size_metric_str = 'safe-f'
        self.__objective_function_str = objective_function_str
        self.__tol_pct_max_dd = tolerated_pct_max_drawdown
        self.__max_dd_pctl_threshold = max_drawdown_percentile_threshold

    @property
    def position_size_metric_str(self):
        return self.__position_size_metric_str

    @property
    def objective_function_str(self):
        return self.__objective_function_str

    def _monte_carlo_simulate_pos_sequence(
        self, positions, num_testing_periods, start_capital, capital_fraction=1.0,
        num_of_sims=1000, data_amount_used=0.5, symbol='', print_dataframe=False, 
        plot_monte_carlo_sims=False, **kwargs
    ):
        """
        Randomizes the order of a sequence of positions, calculates
        metrics for system evaluation and stores them in a Pandas
        DataFrame which is returned by the method.

        Parameters
        ----------
        :param positions:
            'list' : A collection of positions.
        :param num_testing_periods:
            'int' : The number of periods used to generate the given
            sequence of positions.
        :param start_capital:
            'int/float' : The amount of starting capital.
        :param capital_fraction:
            Keyword arg 'float' : The fraction of capital to be used
            to purchase assets. Default value=1.0
        :param num_of_sims:
            Keyword arg 'int' : The number of simulations to run.
            Default value=1000
        :param data_amount_used:
            Keyword arg 'float' : The fraction of historic positions to
            use in the simulation output. Default value=0.5
        :param symbol:
            Keyword arg 'str' : The symbol/ticker of an asset.
            Default value=''
        :param print_dataframe:
            Keyword arg 'bool' : True/False decides if the DataFrame
            with metrics and statistics should be printed to console.
            Default value=True
        :param plot_monte_carlo_sims:
            Keyword arg 'bool' : True/False decides whether to plot
            a chart with data from the simulations or not.

        :return:
            'Pandas DataFrame'
        """

        monte_carlo_sims_df: pd.DataFrame = pd.DataFrame()

        equity_curves_list = []
        final_equity_list = []
        max_drawdowns_list = []
        sim_positions = None

        def generate_position_sequence(position_list, **kw):
            """
            Generates positions from given list of objects of type Position.
            The list will be sliced at a percentage of the total amount of
            positions, determined by 'data_amount_used'.

            Parameters
            ----------
            :param position_list:
                'list' : A list of Position objects.
            :param kw:
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
            sim_positions.generate_positions(generate_position_sequence, pos_list)
            monte_carlo_sims_df: pd.DataFrame = monte_carlo_sims_df.append(
                sim_positions.metrics.summary_data_dict, ignore_index=True
            )
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

        car_series = pd.Series()
        car_series['CAR25'] = car25
        car_series['CAR75'] = car75
        monte_carlo_sims_df = monte_carlo_sims_df.append(car_series, ignore_index=True)

        if print_dataframe:
            print(monte_carlo_sims_df.to_string())
        if plot_monte_carlo_sims:
            monte_carlo_simulations_plot(
                symbol, equity_curves_list, max_drawdowns_list, final_equity_list,
                capital_fraction, car25, car75
            )

        return monte_carlo_sims_df

    def __call__(
        self, positions: List[Position], period_len, 
        forecast_positions=100, forecast_data_fraction=0.5, persistant_safe_f=None,
        capital=10000, num_of_sims=2500, symbol='', **kwargs
    ):
        """
        Calls method to simulate given sequence of positions and gets the
        drawdown at the percentile that is the tolerated threshold from
        the distribution of maximum drawdowns. Calculates the safe-F
        metric and returns a dict with data for system evaluation.

        Parameters
        ----------
        :param positions:
            'list' : A sequence of Position objects.
        :param period_len:
            'int' : The number of periods used to generate the given
            sequence of positions.
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
            Default value=''
        :param kwargs:
            'dict' : A dict with additional keyword arguments.

        :return:
            'dict'
        """

        split_data_fraction = 1.0
        if len(positions) >= forecast_positions:
            split_data_fraction = forecast_positions / len(positions)

        period_len = int(period_len * split_data_fraction)
        # sort positions on date
        positions.sort(key=lambda tr: tr.entry_dt)

        # simulate sequences of given Position objects
        monte_carlo_sims_df = self._monte_carlo_simulate_pos_sequence(
            positions[-(int(len(positions) * split_data_fraction)):], 
            period_len, capital, 
            capital_fraction=persistant_safe_f if persistant_safe_f else 1.0, 
            num_of_sims=num_of_sims, data_amount_used=forecast_data_fraction, 
            symbol=symbol, **kwargs
        )

        # sort the 'Max drawdown (%)' column and convert to a list
        max_dds = sorted(monte_carlo_sims_df['Max drawdown (%)'].to_list())
        # get the drawdown value at the percentile set to be the threshold at which to limit the 
        # probability of getting a max drawdown of that magnitude at when simulating sequences 
        # of the best estimate positions
        dd_at_tolerated_threshold = max_dds[int(len(max_dds) * self.__max_dd_pctl_threshold)]
        if dd_at_tolerated_threshold < 1:
            dd_at_tolerated_threshold = 1
        # calculate the safe fraction of capital to be used to purchase assets with
        if not persistant_safe_f:
            safe_f = self.__tol_pct_max_dd / dd_at_tolerated_threshold
        else:
            safe_f = persistant_safe_f

        return {
            'symbol': symbol,
            'sharpe_ratio': kwargs['metrics_dict']['Sharpe ratio'],
            'profit_factor': kwargs['metrics_dict']['Profit factor'],
            'expectancy': kwargs['metrics_dict']['Expectancy'],
            'CAR25': round(monte_carlo_sims_df.iloc[-1]['CAR25'], 3),
            'CAR75': round(monte_carlo_sims_df.iloc[-1]['CAR75'], 3),
            self.__position_size_metric_str: round(safe_f, 3),
        }
