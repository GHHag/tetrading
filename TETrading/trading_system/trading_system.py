import os
from inspect import isfunction

import pandas as pd

from TETrading.position.position_manager import PositionManager
from TETrading.position.position_sizer.position_sizer import PositionSizer
from TETrading.signal_events.signal_handler import SignalHandler
from TETrading.utils.monte_carlo_functions import monte_carlo_simulate_returns, monte_carlo_simulation_summary_data
from TETrading.metrics.metrics_summary_plot import returns_distribution_plot


class TradingSystem:
    """
    Data together with logic forms the trading system. Objects of this class
    use fields with data and logic to generate historic positions, new signals
    and a position sizing/system health mechanism.

    Parameters
    ----------
    data_dict : 'Dict'
        A dict with key: symbol, value: Pandas DataFrame with data for the
        assets used in the system.
    strategy_function : 'Function'
        A function with logic to iterate over data and create and yield positions
    entry_logic_function : 'Function'
        The logic used for entering a position.
    exit_logic_function : 'Function'
        The logic used to exit a position.
    pos_sizer : Subclass object of 'PositionSizer'
        An object of a class with functionality for sizing positions.
    """

    def __init__(self, data_dict, strategy_function, entry_logic_function, exit_logic_function, pos_sizer):
        assert isinstance(data_dict, dict), \
            'Parameter \'data_dict\' must be a dict with format key "Ticker" (str): value (Pandas.DataFrame)'
        self.__data_dict = data_dict
        assert isfunction(strategy_function), \
            "Parameter 'strategy_function' must be a function."
        self.__strategy_function = strategy_function
        assert isfunction(entry_logic_function), \
            "Parameter 'entry_logic_function' must be a function."
        self.__entry_logic_function = entry_logic_function
        assert isfunction(exit_logic_function), \
            "Parameter 'exit_logic_function' must be a function."
        self.__exit_logic_function = exit_logic_function
        assert issubclass(type(pos_sizer), PositionSizer), \
            "Parameter 'pos_sizer' must be an object inheriting from the PositionSizer class."
        self.__pos_sizer = pos_sizer

        self.__total_period_len = 0
        self.__full_pos_list = []
        self.__full_market_to_market_returns_list = []
        self.__full_mae_list = []
        self.__full_mfe_list = []

        # Instantiate SignalHandler object
        self.__signal_handler = SignalHandler(self.__pos_sizer.objective_function_str)

        self.__metrics_df = self._create_metrics_df()
        self.__monte_carlo_simulations_df = self._create_monte_carlo_sims_df()

    @property
    def total_period_len(self):
        return self.__total_period_len

    @property
    def full_pos_list(self):
        return self.__full_pos_list

    def _create_metrics_df(self):
        return pd.DataFrame(columns=[
            'Ticker', 'Number of positions', 'Start capital', 'Final capital',
            'Total gross profit', 'Avg pos net profit', '% wins', 'Profit factor',
            'Sharpe ratio', 'Rate of return', 'Mean P/L', 'Median P/L', 'Std of P/L',
            'Mean return', 'Median return', 'Std of returns', 'Expectancy',
            'Avg MAE', 'Min MAE', 'Avg MFE', 'Max MFE',
            'Max drawdown (%)', 'RoMad', 'CAGR (%)'
        ])

    def _print_metrics_df(self):
        print('\nStrategy performance summary: \n', self.__metrics_df.to_string())

    def _create_monte_carlo_sims_df(self):
        return pd.DataFrame(columns=[
            'Ticker', 'Start capital', 'Median gross profit', 'Median profit factor', 'Median expectancy',
            'Avg RoR', 'Median RoR', 'Median max drawdown (%)', 'Avg RoMad', 'Median RoMad', 'Avg CAGR (%)',
            'Median CAGR (%)', 'Mean % wins'
        ])

    def _print_monte_carlo_sims_df(self, num_of_monte_carlo_sims):
        print('\nMonte carlo simulation stats (' + str(num_of_monte_carlo_sims) + ' simulations):\n',
              self.__monte_carlo_simulations_df.to_string())

    def run_monte_carlo_simulation(self, safe_f,
                                    forecast_trades=500, forecast_data_fraction=0.5,
                                    capital=10000, num_of_sims=1500,
                                    plot_fig=False, save_fig_to_path=None, print_dataframe=False):
        """
        Simulates randomized sequences of given trades and
        returns data generated from the simulations.

        Parameters
        ----------
        :param positions:
            'List' : A collection of Position objects.
        :param period_len:
            'int' : The number of periods in the data
            used to generate the collection of trades.
        :param forecast_trades:
            Keyword arg 'int' : The number of trades to
            use in the Monte Carlo simulation.
            Default value=500
        :param forecast_data_fraction:
            Keyword arg 'float' : The fraction of data to use in the
            simulation. Default value=0.5
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
        split_data_fraction = 1.0
        if len(self.__full_pos_list) >= forecast_trades:
            split_data_fraction = forecast_trades / len(self.__full_pos_list)

        period_len = int(self.__total_period_len * split_data_fraction)
        # sort trades on date
        self.__full_pos_list.sort(key=lambda x: x.entry_dt)

        monte_carlo_sims_dicts_list = monte_carlo_simulate_returns(
            self.__full_pos_list[-(int(len(self.__full_pos_list) * split_data_fraction)):], 
            '', period_len, capital, safe_f,
            plot_fig=plot_fig, num_of_sims=num_of_sims, data_amount_used=forecast_data_fraction,
            save_fig_to_path=save_fig_to_path, print_dataframe=print_dataframe
        )

        return monte_carlo_sims_dicts_list
        
    def __call__(
            self, *args, capital=10000, capital_fraction=1.0, yearly_periods=251,
            plot_performance_summary=False, save_summary_plot_to_path=None, system_analysis_to_csv_path=None,
            plot_returns_distribution=False, save_returns_distribution_plot_to_path=None,
            run_monte_carlo_sims=False, num_of_monte_carlo_sims=2500, monte_carlo_data_amount=0.4,
            plot_monte_carlo=False, print_monte_carlo_df=False, monte_carlo_analysis_to_csv_path=None,
            write_signals_to_file_path=None, insert_signal_data_to_db=False,
            **kwargs
    ):
        """
        Iterates over data, creates a PositionManager instance and generates
        positions.

        Parameters
        ----------
        :param args:
            'Tuple' : Args to pass along to PositionManager.generate_positions.
        :param capital:
            Keyword arg 'int/float' : The amount of capital to purchase assets for.
            Default value=10000
        :param capital_fraction:
            Keyword arg 'float' : The fraction of the capital that will be used.
            Default value=1.0
        :param yearly_periods:
            Keyword arg 'int' : The number of periods in a year for the time frame
            of the given datasets.
            Default value=251
        :param plot_performance_summary:
            Keyword arg 'Boolean' : True/False decides whether to plot summary
            statistics or not. Default value=False
        :param save_summary_plot_to_path:
            Keyword arg 'None/str' : Provide a file path as a str to save the
            summary plot as a file. Default value=None
        :param system_analysis_to_csv_path:
            Keyword arg 'None/str' : Provide a file path as a str to save the
            system analysis Pandas DataFrame as a CSV file. Default value=None
        :param plot_returns_distribution:
            Keyword arg 'Boolean' : True/False decides whether to plot charts with
            returns, MAE and MFE distributions for the system. Default value=False
        :param save_returns_distribution_plot_to_path:
            Keyword arg 'None/str' : Provide a file path as a str to save the
            returns distribution plot as a file. Default value=None
        :param run_monte_carlo_sims:
            Keyword arg 'Boolean' : True/False decides whether to run Monte Carlo
            return simulations on each assets return sequence. Default value=False
        :param num_of_monte_carlo_sims:
            Keyword arg 'int' : The number of Monte Carlo simulations to run.
            Default value=2500
        :param monte_carlo_data_amount:
            Keyword arg 'float' : The fraction of data to be used in the Monte Carlo
            simulations. Default value=0.4
        :param plot_monte_carlo:
            Keyword arg 'Boolean' : True/False decides whether to plot the results
            of the Monte Carlo simulations. Default value=False
        :param print_monte_carlo_df:
            Keyword arg 'Boolean' : True/False decides whether or not to print a Pandas
            DataFrame with stats generated from Monte Carlo simulations to the console.
            Default value=False
        :param monte_carlo_analysis_to_csv_path:
            Keyword arg 'None/str' : Provide a file path as a str to save the
            Monte Carlo simulations Pandas DataFrame as a CSV file. Default value=None
        :param write_signals_to_file_path:
            Keyword arg 'None/str': Provide a file path as a str to save any signals
            generated by the system. Default value=None
        :param insert_signal_data_to_db:
            Keyword arg 'Boolean' : True/False decides whether or not generated signal
            data should be inserted into database or not.
        :param kwargs:
            'Dict' : Dictionary with keyword arguments to pass along to
            PositionManager.generate_positions.
        """

        for instrument, data in self.__data_dict.items():
            try:
                if 'Close' in data:
                    asset_price_series = [float(close) for close in data['Close']]
                elif f'Close_{instrument}' in data:
                    asset_price_series = [float(close) for close in data[f'Close_{instrument}']]
                else:
                    raise Exception('Column missing in DataFrame')
            except TypeError:
                print('TypeError', instrument)
                x = input('Enter to proceed')
                continue

            pos_manager = PositionManager(instrument, data.shape[0], capital, capital_fraction,
                                          asset_price_series=asset_price_series)

            # call generate_positions and pass the trading system logic functions as arguments
            pos_manager.generate_positions(
                self.__strategy_function, data, self.__entry_logic_function, self.__exit_logic_function,
                *args, signal_handler=self.__signal_handler, symbol=instrument,
                **kwargs
            )

            # summary output of the trading system
            if not pos_manager.metrics:
                print(f'\nNo positions generated for {pos_manager.symbol}')
                continue
            else:
                try:
                    if save_summary_plot_to_path:
                        if not os.path.exists(save_summary_plot_to_path):
                            os.makedirs(save_summary_plot_to_path)
                    pos_manager.summarize_performance(plot_fig=plot_performance_summary,
                                                      save_fig_to_path=save_summary_plot_to_path)
                except ValueError:
                    print('ValueError')

            # write trading system data and stats to DataFrame
            self.__metrics_df = self.__metrics_df.append(pos_manager.metrics.summary_data_dict, ignore_index=True)

            # run Monte Carlo simulations, plot and write stats to DataFrame
            if run_monte_carlo_sims:
                print('\nRunning Monte Carlo simulations...')
                monte_carlo_sims_data_dicts_list = monte_carlo_simulate_returns(
                    pos_manager.metrics.positions, pos_manager.symbol, pos_manager.metrics.num_testing_periods,
                    start_capital=capital, capital_fraction=capital_fraction,
                    num_of_sims=num_of_monte_carlo_sims, data_amount_used=monte_carlo_data_amount,
                    print_dataframe=print_monte_carlo_df,
                    plot_fig=plot_monte_carlo, save_fig_to_path=save_summary_plot_to_path
                )
                monte_carlo_summary_data_dict = monte_carlo_simulation_summary_data(monte_carlo_sims_data_dicts_list)
                self.__monte_carlo_simulations_df = self.__monte_carlo_simulations_df.append(
                    monte_carlo_summary_data_dict, ignore_index=True
                )

            # add position sizing and system health data to the SignalHandler
            if len(pos_manager.metrics.positions) > 0:
                avg_yearly_positions = len(pos_manager.metrics.positions) / (len(data) / yearly_periods)
                self.__signal_handler.add_pos_sizing_evaluation_data(
                    self.__pos_sizer(
                        pos_manager.metrics.positions, len(data), forecast_positions=150,
                        forecast_data_fraction=(avg_yearly_positions / len(pos_manager.metrics.positions)) * 2,
                        capital=capital, num_of_sims=num_of_monte_carlo_sims, symbol=instrument,
                        print_dataframe=print_monte_carlo_df, plot_monte_carlo_sims=plot_monte_carlo,
                        metrics_dict=pos_manager.metrics.summary_data_dict
                    )
                )

            self.__full_pos_list += pos_manager.metrics.positions[:]
            self.__total_period_len += len(data)

            if pos_manager.metrics.market_to_market_returns_list:
                self.__full_market_to_market_returns_list += pos_manager.metrics.market_to_market_returns_list[:]
                self.__full_mae_list += pos_manager.metrics.w_mae_list[:]
                self.__full_mfe_list += pos_manager.metrics.mfe_list[:]

        self._print_metrics_df()

        if run_monte_carlo_sims:
            self._print_monte_carlo_sims_df(num_of_monte_carlo_sims)
            if monte_carlo_analysis_to_csv_path and monte_carlo_analysis_to_csv_path.endswith('.csv'):
                self.__monte_carlo_simulations_df.to_csv(monte_carlo_analysis_to_csv_path)

        if system_analysis_to_csv_path and system_analysis_to_csv_path.endswith('.csv'):
            self.__metrics_df.to_csv(system_analysis_to_csv_path)

        print(self.__signal_handler)
        if write_signals_to_file_path:
            self.__signal_handler.write_to_csv(write_signals_to_file_path, self.__strategy_function.__name__)
        if insert_signal_data_to_db:
            self.__signal_handler.insert_into_db(self.__strategy_function.__name__)

        returns_distribution_plot(self.__full_market_to_market_returns_list, self.__full_mae_list, self.__full_mfe_list,
                                  plot_fig=plot_returns_distribution,
                                  save_fig_to_path=save_returns_distribution_plot_to_path)
