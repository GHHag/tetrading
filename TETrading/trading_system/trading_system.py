import os
from inspect import isfunction
from typing import Callable, Dict

import pandas as pd
import numpy as np

from TETrading.data.metadata.trading_system_metrics import TradingSystemMetrics
from TETrading.data.metadata.trading_system_simulation_attributes import TradingSystemSimulationAttributes
from TETrading.position.position import Position
from TETrading.position.position_manager import PositionManager
from TETrading.trading_system.trading_session import TradingSession
from TETrading.signal_events.signal_handler import SignalHandler
from TETrading.utils.monte_carlo_functions import monte_carlo_simulate_returns, \
    monte_carlo_simulation_summary_data
from TETrading.metrics.metrics_summary_plot import returns_distribution_plot


class TradingSystem:
    """
    Data together with logic forms the trading system. Objects of this class
    use fields with data and logic to generate historic positions and new signals.

    Parameters
    ----------
    system_name : 'str'
        The name of the system. Will be used to identify it.
    data_dict : 'dict'
        A dict with key: symbol, value: Pandas DataFrame with data for the
        assets used in the system.
    entry_logic_function : 'function'
        The logic used for entering a position.
    exit_logic_function : 'function'
        The logic used to exit a position.
    """

    def __init__(
        self, system_name, data_dict: Dict[str, pd.DataFrame], 
        entry_logic_function: Callable, exit_logic_function: Callable
    ):
        self.__system_name = system_name
        assert isinstance(data_dict, dict), \
            "Parameter 'data_dict' must be a dict with format key 'Symbol' (str): value (Pandas.DataFrame)"
        self.__data_dict = data_dict
        assert isfunction(entry_logic_function), \
            "Parameter 'entry_logic_function' must be a function."
        self.__entry_logic_function = entry_logic_function
        assert isfunction(exit_logic_function), \
            "Parameter 'exit_logic_function' must be a function."
        self.__exit_logic_function = exit_logic_function

        self.__total_period_len = 0
        self.__full_pos_list: list[Position] = []
        self.__pos_lists: list[Position] = []
        self.__full_market_to_market_returns_list = np.array([])
        self.__full_mae_list = np.array([])
        self.__full_mfe_list = np.array([])

        self.__signal_handler = SignalHandler()

        self.__metrics_df: pd.DataFrame = pd.DataFrame(
            columns=TradingSystemMetrics.cls_attrs
        )
        self.__monte_carlo_simulations_df: pd.DataFrame = pd.DataFrame(
            columns=TradingSystemSimulationAttributes.cls_attrs
        )

    @property
    def total_period_len(self):
        return self.__total_period_len

    @property
    def full_pos_list(self):
        return self.__full_pos_list

    @property
    def pos_lists(self):
        return self.__pos_lists

    def _print_metrics_df(self):
        print('\nSystem performance summary: \n', self.__metrics_df.to_string())

    def _print_monte_carlo_sims_df(self, num_of_monte_carlo_sims):
        print(
            '\nMonte carlo simulation stats (' + str(num_of_monte_carlo_sims) + ' simulations):\n',
            self.__monte_carlo_simulations_df.to_string()
        )

    def __call__(
        self, *args, capital=10000, capital_fraction=None, avg_yearly_periods=251,
        system_evaluation_fields=TradingSystemMetrics.system_evaluation_fields,
        market_state_null_default=False,
        plot_performance_summary=False, save_summary_plot_to_path: str=None, 
        system_analysis_to_csv_path: str=None,
        plot_returns_distribution=False, save_returns_distribution_plot_to_path: str=None,
        run_monte_carlo_sims=False, num_of_monte_carlo_sims=2500, monte_carlo_data_amount=0.4,
        plot_monte_carlo=False, print_monte_carlo_df=False, 
        monte_carlo_analysis_to_csv_path: str=None, write_signals_to_file_path: str=None, 
        insert_data_to_db_bool=False, signal_handler_db_insert_funcs: Dict[str, Callable]=None,
        single_symbol_pos_list_db_insert_func: Callable=None, 
        json_format_single_symbol_pos_list_db_insert_func: Callable=None,
        full_pos_list_db_insert_func: Callable=None, 
        json_format_full_pos_list_db_insert_func: Callable=None,
        pos_list_slice_years_est=2, **kwargs
    ):
        """
        Iterates over data, creates a PositionManager instance and generates
        positions.

        Parameters
        ----------
        :param args:
            'tuple' : Args to pass along to PositionManager.generate_positions().
        :param capital:
            Keyword arg 'int/float' : The amount of capital to purchase assets with.
            Default value=10000
        :param capital_fraction:
            Keyword arg 'None/dict/float' : The fraction of the capital that will 
            be used to purchase assets with. Alternatively a dict where the keys 
            are the symbols of the assets in the trading systems and their values 
            are the fraction of the capital. Default value=None
        :param avg_yearly_periods:
            Keyword arg 'int' : The average number of trading periods in a year for 
            the asset class of the system. Default value=251
        :param system_evaluation_fields:
            Keyword arg 'tuple' : A tuple containing strings of metrics used to 
            evaluate a trading system. 
            Default value=TradingSystemMetrics.system_evaluation_fields property
        :param market_state_null_default:
            Keyword arg 'bool' : True/False decides whether the market_state property 
            should be assigned a null value by default or not. Default value=False
        :param plot_performance_summary:
            Keyword arg 'bool' : True/False decides whether to plot summary
            statistics or not. Default value=False
        :param save_summary_plot_to_path:
            Keyword arg 'None/str' : Provide a file path as a str to save the
            summary plot as a file. Default value=None
        :param system_analysis_to_csv_path:
            Keyword arg 'None/str' : Provide a file path as a str to save the
            system analysis Pandas DataFrame as a .csv file. Default value=None
        :param plot_returns_distribution:
            Keyword arg 'bool' : True/False decides whether to plot charts with
            returns, MAE and MFE distributions for the system. Default value=False
        :param save_returns_distribution_plot_to_path:
            Keyword arg 'None/str' : Provide a file path as a str to save the
            returns distribution plot as a file. Default value=None
        :param run_monte_carlo_sims:
            Keyword arg 'bool' : True/False decides whether to run Monte Carlo
            simulations on each assets return sequence. Default value=False
        :param num_of_monte_carlo_sims:
            Keyword arg 'int' : The number of Monte Carlo simulations to run.
            Default value=2500
        :param monte_carlo_data_amount:
            Keyword arg 'float' : The fraction of data to be used in the Monte Carlo
            simulations. Default value=0.4
        :param plot_monte_carlo:
            Keyword arg 'bool' : True/False decides whether to plot the results
            of the Monte Carlo simulations. Default value=False
        :param print_monte_carlo_df:
            Keyword arg 'bool' : True/False decides whether or not to print a Pandas
            DataFrame with stats generated from Monte Carlo simulations to the console.
            Default value=False
        :param monte_carlo_analysis_to_csv_path:
            Keyword arg 'None/str' : Provide a file path as a str to save the
            Monte Carlo simulations Pandas DataFrame as a CSV file. Default value=None
        :param write_signals_to_file_path:
            Keyword arg 'None/str' : Provide a file path as a str to save any signals
            generated by the system. Default value=None
        :param insert_data_to_db_bool:
            Keyword arg 'bool' : True/False decides whether or not data should be 
            inserted into database or not. Default value=False
        :param signal_handler_db_insert_funcs:
            Keyword arg 'None/dict' : A dict with functions as values. The functions 
            handles inserting data from an instance of the SignalHandler class to
            database. Default value=None
        :param single_symbol_pos_list_db_insert_func:
            Keyword arg 'None/function' : A function that handles inserting a list
            of Position objects for each instrument in the data_dict to database.
            Default value=None
        :param json_format_single_symbol_pos_list_db_insert_func:
            Keyword arg 'None/function' : A function that handles inserting a list
            of Positions objects in a json compatible format to database, with each
            instrument in the data_dict having its own separate list.
            Default value=None
        :param full_pos_list_db_insert_func: 
            Keyword arg 'None/function' : A function that handles inserting a list
            of Position objects generated by the trading system to database.
            Default value=None
        :param json_format_full_pos_list_db_insert_func:
            Keyword arg 'None/function' : A function that handles inserting a list
            of Position objects in a json compatible format to database that was 
            generated by the trading system. Default value=None
        :param pos_list_slice_years_est:
            Keyword arg 'int' : The number of years to estimate the amount of positions
            to slice the list of positions by. Default value=2
        :param kwargs:
            'dict' : Dictionary with keyword arguments to pass along to
            PositionManager.generate_positions().
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
                input('Enter to proceed')
                continue

            # if capital_fraction is a dict containing a key with the current value of
            # 'instrument', its value will be assigned to 'capital_f'
            if isinstance(capital_fraction, dict) and instrument in capital_fraction:
                capital_f = capital_fraction[instrument]
            # if capital_fraction is a float its value will be assigned to 'capital_f'
            elif isinstance(capital_fraction, float):
                capital_f = capital_fraction
            else:
                capital_f = 1.0

            pos_manager = PositionManager(
                instrument, len(data), capital, capital_f, 
                asset_price_series=asset_price_series
            )
            trading_session = TradingSession(
                self.__entry_logic_function, self.__exit_logic_function, data,
                signal_handler=self.__signal_handler, symbol=instrument
            )
            pos_manager.generate_positions(
                trading_session, *args,
                market_state_null_default=market_state_null_default, **kwargs
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
                    pos_manager.summarize_performance(
                        plot_fig=plot_performance_summary, 
                        save_fig_to_path=save_summary_plot_to_path
                    )
                except ValueError:
                    print('ValueError')

            # write trading system data and stats to DataFrame
            self.__metrics_df: pd.DataFrame = self.__metrics_df.append(
                pos_manager.metrics.summary_data_dict, ignore_index=True
            )

            # run Monte Carlo simulations, plot and write stats to DataFrame
            if run_monte_carlo_sims:
                print('\nRunning Monte Carlo simulations...')
                monte_carlo_sims_data_dicts_list = monte_carlo_simulate_returns(
                    pos_manager.metrics.positions, pos_manager.symbol, 
                    pos_manager.metrics.num_testing_periods,
                    start_capital=capital, capital_fraction=capital_f,
                    num_of_sims=num_of_monte_carlo_sims, data_amount_used=monte_carlo_data_amount,
                    print_dataframe=print_monte_carlo_df,
                    plot_fig=plot_monte_carlo, save_fig_to_path=save_summary_plot_to_path
                )
                monte_carlo_summary_data_dict = monte_carlo_simulation_summary_data(
                    monte_carlo_sims_data_dicts_list
                )
                self.__monte_carlo_simulations_df = self.__monte_carlo_simulations_df.append(
                    monte_carlo_summary_data_dict, ignore_index=True
                )

            # add system evaluation data to the SignalHandler
            if len(pos_manager.metrics.positions) > 0 and self.__signal_handler.entry_signal_given or \
                market_state_null_default:
                self.__signal_handler.add_system_evaluation_data(
                    pos_manager.metrics.summary_data_dict, system_evaluation_fields
                )

            #TODO: Implement protocol for function handling database communication.
            if insert_data_to_db_bool and single_symbol_pos_list_db_insert_func:
                single_symbol_pos_list_db_insert_func(
                    self.__system_name, instrument, 
                    pos_manager.metrics.positions[:], len(data)
                )
            #TODO: Implement protocol for function handling database communication.
            if insert_data_to_db_bool and json_format_single_symbol_pos_list_db_insert_func:
                json_format_single_symbol_pos_list_db_insert_func(
                    self.__system_name, instrument,
                    pos_manager.metrics.positions[:], len(data),
                    format='json'
                )

            self.__full_pos_list += pos_manager.metrics.positions[:]
            self.__pos_lists.append(pos_manager.metrics.positions[:])
            self.__total_period_len += len(data)

            if len(pos_manager.metrics.market_to_market_returns_list) > 0:
                self.__full_market_to_market_returns_list = np.concatenate(
                    (
                        self.__full_market_to_market_returns_list,
                        pos_manager.metrics.market_to_market_returns_list
                    ), axis=0
                )
                self.__full_mae_list = np.concatenate(
                    (self.__full_mae_list, pos_manager.metrics.w_mae_list), axis=0
                )
                self.__full_mfe_list = np.concatenate(
                    (self.__full_mfe_list, pos_manager.metrics.mfe_list), axis=0                    
                )

        self._print_metrics_df()

        if run_monte_carlo_sims:
            self._print_monte_carlo_sims_df(num_of_monte_carlo_sims)
            if monte_carlo_analysis_to_csv_path and monte_carlo_analysis_to_csv_path.endswith('.csv'):
                self.__monte_carlo_simulations_df.to_csv(monte_carlo_analysis_to_csv_path)

        if system_analysis_to_csv_path and system_analysis_to_csv_path.endswith('.csv'):
            self.__metrics_df.to_csv(system_analysis_to_csv_path)

        print(self.__signal_handler)
        if write_signals_to_file_path:
            self.__signal_handler.write_to_csv(
                write_signals_to_file_path, self.__system_name
            )

        #TODO: Implement protocol for function handling database communication.
        if insert_data_to_db_bool and signal_handler_db_insert_funcs:
            self.__signal_handler.insert_into_db(
                signal_handler_db_insert_funcs, self.__system_name
            )

        num_of_pos_insert_multiplier = pos_list_slice_years_est * 1.5
        sorted_pos_lists = sorted(self.__pos_lists, key=len, reverse=True)
        position_list_lengths = [len(i) for i in sorted_pos_lists[:int(len(self.__pos_lists) / 4 + 0.5)]] \
            if len(self.__pos_lists) > 1 \
            else [len(sorted_pos_lists[0])]
        data_periods = [len(v) for k, v in self.__data_dict.items()][:int(len(self.__data_dict) / 4 + 0.5)]
        avg_yearly_positions = int(np.mean(position_list_lengths) / (np.mean(data_periods) / avg_yearly_periods) + 0.5) \
            * num_of_pos_insert_multiplier
        full_pos_list_slice_param = int(avg_yearly_positions * (pos_list_slice_years_est * 1.5) + 0.5)
        sorted_full_pos_list: list[Position] = sorted(self.__full_pos_list, key=lambda x: x.entry_dt)
        sliced_pos_list: list[Position] = sorted_full_pos_list[-full_pos_list_slice_param:]
        num_of_periods = avg_yearly_periods * pos_list_slice_years_est * num_of_pos_insert_multiplier

        #TODO: Implement protocol for function handling database communication.
        if insert_data_to_db_bool and full_pos_list_db_insert_func:
            full_pos_list_db_insert_func(self.__system_name, sliced_pos_list, num_of_periods)

        #TODO: Implement protocol for function handling database communication.
        if insert_data_to_db_bool and json_format_full_pos_list_db_insert_func:
            json_format_full_pos_list_db_insert_func(
                self.__system_name, sliced_pos_list, num_of_periods, format='json'
            )

        returns_distribution_plot(
            self.__full_market_to_market_returns_list, self.__full_mae_list, self.__full_mfe_list,
            plot_fig=plot_returns_distribution, save_fig_to_path=save_returns_distribution_plot_to_path
        )
