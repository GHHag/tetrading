from TETrading.utils.metadata.market_state_enum import MarketState
from TETrading.signal_events.signals.system_signals import SystemSignals


class SignalHandler:
    """
    Handles signals given by the trading system.

    TODO: Implement methods _execute_signals and __call__.
    """

    def __init__(self):
        self.__entry_signals = SystemSignals()
        self.__exit_signals = SystemSignals()
        self.__active_positions = SystemSignals()
        self.__entry_signal_given = False

    @property
    def entry_signal_given(self):
        return self.__entry_signal_given

    def handle_entry_signal(self, symbol, data_dict):
        """
        Calls the __entry_signals members add_signal_data method,
        passing it the given symbol and data_dict.

        Parameters
        ----------
        :param symbol:
            'str' : The symbol/ticker of an asset.
        :param data_dict:
            'dict' : Data to be handled.
        """

        self.__entry_signal_given = True
        self.__entry_signals.add_data(symbol, data_dict)

    def handle_active_position(self, symbol, data_dict):
        """
        Calls the __active_positions members add_data method,
        passing it the given symbol and data_dict.

        Parameters
        ----------
        :param symbol:
            'str' : The symbol/ticker of an asset.
        :param data_dict:
            'dict' : Data to be handled.
        """

        self.__active_positions.add_data(symbol, data_dict)

    def handle_exit_signal(self, symbol, data_dict):
        """
        Calls the __exit_signals members add_signal_data method,
        passing it the given symbol and data_dict.

        Parameters
        ----------
        :param symbol:
            'str' : The symbol/ticker of an asset.
        :param data_dict:
            'dict' : Data to be handled.
        """

        self.__exit_signals.add_data(symbol, data_dict)

    def _execute_signals(self):
        # TODO: Implement functionality to be able to connect to brokers
        #  and execute orders with the use of an 'ExecutionHandler' class.
        pass

    def add_system_evaluation_data(self, evaluation_dict, evaluation_fields):
        """
        Adds the given evaluation data to the EntrySignals object member
        by calling its add_evaluation_data method.

        Parameters
        ----------
        :param evaluation_dict:
            'dict' : A dict with data generated by a TradingSession object.
        :param evaluation_fields:
            'tuple' : A tuple containing strings that should have corresponding
            fields in the given 'evaluation_dict'
        """

        self.__entry_signals.add_evaluation_data(
            {k: evaluation_dict[k] for k in evaluation_fields}
        )
        self.__entry_signal_given = False

    def write_to_csv(self, path, system_name):
        """
        Writes the dataframe field of the __entry_signals and __exit_signals
        members to a CSV file.

        Parameters
        ----------
        :param path:
            'str' : The path to where the CSV file will be written.
        :param system_name:
            'str' : The name of the system that generated the signals.
        """

        with open(path, 'a') as file:
            file.write("\n" + system_name + "\n")
            if self.__entry_signals.dataframe is not None:
                self.__entry_signals.dataframe.to_csv(path, mode='a')
            if self.__exit_signals.dataframe is not None:
                self.__exit_signals.dataframe.to_csv(path, mode='a')

    def insert_into_db(self, db_insert_funcs, system_name):
        """
        Insert data into database from the dataframes that holds data 
        and stats for signals and positions. If a system with the given
        name is not found in an attempt to query it from the database it
        will be inserted with a generated id of type 'int'.

        Parameters
        ----------
        :param db_insert_funcs:
            'dict' : A dict containing functions to handle inserting data to
            database as values. The keys are 'entry', 'exit' and 'active' and
            their corresponding data which the value are to handle the inserting 
            of is: 
            'entry': __entry_signals.dataframe
            'exit': __exit_signals.dataframe
            'active': __active_positions.dataframe
        :param system_name:
            'str' : The name of a system which it will be identified by in
            in the database.
        """

        if self.__entry_signals.dataframe is not None:
            insert_successful = db_insert_funcs[MarketState.ENTRY.value](
                system_name, self.__entry_signals.dataframe.to_json(orient='table')
            )

            if not insert_successful:
                raise Exception('DatabaseInsertException, failed to insert to database.')

        if self.__active_positions.dataframe is not None:
            insert_successful = db_insert_funcs[MarketState.ACTIVE.value](
                system_name, self.__active_positions.dataframe.to_json(orient='table')
            )

            if not insert_successful:
                raise Exception('DatabaseInsertException, failed to insert to database.')

        if self.__exit_signals.dataframe is not None:
            insert_successful = db_insert_funcs[MarketState.EXIT.value](
                system_name, self.__exit_signals.dataframe.to_json(orient='table')
            )

            if not insert_successful:
                raise Exception('DatabaseInsertException, failed to insert to database.')

    def get_position_sizing_dict(self, position_sizing_metric_str):
        return self.__entry_signals.get_pos_sizer_dict(position_sizing_metric_str)

    def __str__(self):
        return f'\n\
            Active positions\n{self.__active_positions}\n\n\
            Entry signals\n{self.__entry_signals}\n\n\
            Exit signals\n{self.__exit_signals}'

    def __call__(self):
        """
        Execute signals.

        TODO: Fully implement the methods functionality.
        """

        self._execute_signals()
