from decimal import Decimal

import pandas as pd

from TETrading.position.position import Position
from TETrading.signal_events.signal_handler import SignalHandler
from TETrading.plots.candlestick_plots import candlestick_plot


class TradingSession:
    """
    A class that represents a trading session. By using the given
    dataframe together with the given entry and exit logic a
    TradingSession instance generates Position objects.

    Parameters
    ----------
    entry_logic_function: 'function'
        A function with logic for when to enter a market.
    exit_logic_function: 'function'
        A function with logic for when to exit a market.
    dataframe: 'Pandas.DataFrame'
        Data in the form of a Pandas DataFrame.
    signal_handler: Keyword arg 'None/SignalHandler'
        An instance of the SignalHandler class. Handles
        data from generated events/signals.
        Default value=None
    symbol: Keyword arg 'str'
        The ticker/symbol of the instrument to be traded
        in the current trading session. Default value=''
    """

    def __init__(
        self, entry_logic_function, exit_logic_function, dataframe: pd.DataFrame,
        signal_handler: SignalHandler=None, symbol=''
    ):
        self.__entry_logic_function = entry_logic_function
        self.__exit_logic_function = exit_logic_function
        self.__dataframe = dataframe
        self.__signal_handler = signal_handler
        self.__symbol = symbol
        self.__market_state_column = 'market_state'

    def __call__(
        self, *args, entry_args=None, exit_args=None, 
        max_req_periods_feature='req_period_iters', datetime_col_name='Date',
        close_price_col_name='Close', open_price_col_name='Open',
        fixed_position_size=True, capital=10000, commission_pct_cost=0.0,
        market_state_null_default=False,
        generate_signals=False, plot_positions=False, 
        save_position_figs_path=None, **kwargs
    ):
        """
        Generates positions using the __entry_logic_function and 
        __exit_logic_function members. If the given value for 
        generate_signals is True, signals will be generated from 
        the most recent data.

        Parameters
        ----------
        :param args:
            'tuple' : A tuple with arguments.
        :param entry_args:
            Keyword arg 'None/dict' : Key-value pairs with parameters used 
            with the entry logic. Default value=None
        :param exit_args:
            Keyword arg 'None/dict' : Key-value pairs with parameters used 
            with the exit logic. Default value=None
        :param max_req_periods_feature:
            Keyword arg 'str' : A key contained in the entry_args dict 
            that should have the value of the number of periods required 
            for all features to be calculated before being able to 
            generate signals from the data. Default value='req_period_iters'
        :param datetime_col_name:
            Keyword arg 'str' : The column of the objects __dataframe that
            contains time and date data. Default value='Date'
        :param close_price_col_name:
            Keyword arg 'str' : The column of the objects __dataframe that
            contains values for close prices. Default value='Close'
        :param open_price_col_name:
            Keyword arg 'str' : The column of the objects __dataframe that
            contains values for open prices. Default value='Open'
        :param fixed_position_size:
            Keyword arg 'bool' : True/False decides whether the capital
            used for positions generated should be at a fixed amount or not.
            Default value=True
        :param capital:
            Keyword arg 'int/float' : The capital given to purchase assets
            with. Default value=10000
        :param commission_pct_cost:
            Keyword arg 'float' : The transaction cost given as a percentage
            (a float from 0.0 to 1.0) of the total transaction.
            Default value=0.0
        :param generate_signals:
            Keyword arg 'bool' : True/False decides whether or not market
            events/signals should be generated from the most recent data.
            Default value=False
        :param plot_positions:
            Keyword arg 'bool' : True/False decides whether or not a plot of
            a candlestick chart visualizing the points of buying and selling a
            position should be displayed. Default value=False
        :param save_position_figs_path:
            Keyword arg 'None/str' : Provide a file path as a str to save a
            candlestick chart visualizing the points of buying and selling a
            position. Default value=None
        :param kwargs:
            'dict' : A dictionary with keyword arguments.
        """

        position = Position(-1)
        trailing_exit, trailing_exit_price = False, None 

        if isinstance(self.__dataframe.index, pd.DatetimeIndex):
            self.__dataframe.reset_index(level=0, inplace=True)

        for index, row in enumerate(self.__dataframe.itertuples()):
            # entry_args[max_req_periods_feature] is the parameter used 
            # with the longest period lookback required to calculate.
            if index <= entry_args[max_req_periods_feature]:
                continue

            if position and position.active_position is True:
                position.update(
                    Decimal(self.__dataframe[close_price_col_name].iloc[index-1])
                )
                # Call the exit logic function, passing required arguments.
                exit_condition, trailing_exit, trailing_exit_price = \
                    self.__exit_logic_function(
                        self.__dataframe.iloc[:index], trailing_exit, trailing_exit_price, 
                        position.entry_price, len(position.returns_list), exit_args=exit_args
                    )
                if exit_condition:
                    capital = position.exit_market(
                        self.__dataframe[open_price_col_name].iloc[index], 
                        self.__dataframe[datetime_col_name].iloc[index-1]
                    )
                    position.print_position_stats()
                    print(
                        f'Exit index {index}: '
                        f'{format(self.__dataframe[open_price_col_name].iloc[index], ".3f")}, '
                        f'{self.__dataframe[datetime_col_name].iloc[index]}\n'
                        f'Realised return: {position.position_return}'
                    )
                    if plot_positions:
                        if save_position_figs_path is not None:
                            position_figs_path = save_position_figs_path + (
                                fr'\{self.__dataframe.iloc[(index - len(position.returns_list))].Date.strftime("%Y-%m-%d")}.jpg'
                            )
                        else:
                            position_figs_path = save_position_figs_path
                        candlestick_plot(
                            self.__dataframe.iloc[(index-len(position.returns_list)-3):(index+3)],
                            position.entry_dt, position.entry_price, 
                            self.__dataframe[datetime_col_name].iloc[index], 
                            self.__dataframe[open_price_col_name].iloc[index], 
                            save_fig_to_path=position_figs_path
                        )
                    yield position
                continue

            # Instantiate a Position object if the position.active_position 
            # field is False and a call to entry_logic_func returns True.
            entry_signal, direction = self.__entry_logic_function(
                self.__dataframe.iloc[:index], entry_args=entry_args
            )
            if not position.active_position and entry_signal:
                position = Position(
                    capital, fixed_position_size=fixed_position_size, 
                    commission_pct_cost=commission_pct_cost
                )
                position.enter_market(
                    self.__dataframe[open_price_col_name].iloc[index], direction, 
                    self.__dataframe[datetime_col_name].iloc[index]
                )
                print(
                    f'\nEntry index {index}: '
                    f'{format(self.__dataframe[open_price_col_name].iloc[index], ".3f")}, '
                    f'{self.__dataframe[datetime_col_name].iloc[index]}'
                )

        # Handle the trading sessions current market state/events/signals.
        if market_state_null_default and generate_signals:
            self.__signal_handler.handle_entry_signal(
                self.__symbol, {
                    'signal_dt': self.__dataframe[datetime_col_name].iloc[-1],
                    self.__market_state_column: 'null'
                }
            )
            return
        if position.active_position and generate_signals:
            position.update(Decimal(self.__dataframe[close_price_col_name].iloc[-1]))
            position.print_position_status()
            self.__signal_handler.handle_active_position(
                self.__symbol, {
                    'signal_index': len(self.__dataframe), 
                    'signal_dt': self.__dataframe[datetime_col_name].iloc[-1], 
                    'symbol': self.__symbol, 
                    'direction': direction,
                    'periods_in_position': len(position.returns_list), 
                    'unrealised_return': position.unrealised_return,
                    self.__market_state_column: 'active'
                }
            )
            exit_condition, trailing_exit_price, trailing_exit = self.__exit_logic_function(
                self.__dataframe, trailing_exit, trailing_exit_price, 
                position.entry_price, len(position.returns_list), 
                position.unrealised_return, exit_args=exit_args
            )
            if exit_condition:
                self.__signal_handler.handle_exit_signal(
                    self.__symbol, {
                        'signal_index': len(self.__dataframe), 
                        'signal_dt': self.__dataframe[datetime_col_name].iloc[-1], 
                        'symbol': self.__symbol, 
                        'direction': direction,
                        'periods_in_position': len(position.returns_list),
                        'unrealised_return': position.unrealised_return,
                        self.__market_state_column: 'exit'
                    }
                )
                print(f'\nExit signal, exit next open\nIndex {len(self.__dataframe)}')
        elif not position.active_position and generate_signals:
            entry_signal, direction = self.__entry_logic_function(
                self.__dataframe, entry_args=entry_args
            )
            if entry_signal:
                self.__signal_handler.handle_entry_signal(
                    self.__symbol, {
                        'signal_index': len(self.__dataframe), 
                        'signal_dt': self.__dataframe[datetime_col_name].iloc[-1], 
                        'symbol': self.__symbol,
                        'direction': direction,
                        self.__market_state_column: 'entry'
                    }
                )
                print(f'\nEntry signal, buy next open\nIndex {len(self.__dataframe)}')
