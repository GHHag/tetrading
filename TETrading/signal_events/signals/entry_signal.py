import pandas as pd


class EntrySignals:
    """
    Handles data for entry signals.
    """

    def __init__(self):
        self.__signal_data_list = []

    @property
    def dataframe(self):
        """
        Returns a Pandas DataFrame with data from the
        __signal_data_list member.

        :return:
            'Pandas DataFrame'
        """

        if self.__signal_data_list:
            return pd.DataFrame([instrument['data'] for instrument in self.__signal_data_list])
        else:
            return None

    def __str__(self):
        if self.__signal_data_list:
            return f'Entry signals\n{self.dataframe.sort_values(ascending=False, by="CAR25").to_string()}'
        else:
            return f'Entry signals\n{self.__signal_data_list}'

    def add_signal_data(self, symbol, data_dict):
        """
        Appends a dict with a symbol/ticker and given data to the
        __signal_data_list member.

        Parameters
        ----------
        :param symbol:
            'str' : The symbol/ticker of an asset.
        :param data_dict:
            'Dict' : A dict containing data of an entry signal.
        """

        self.__signal_data_list.append({'Ticker': symbol, 'data': data_dict})

    def add_evaluation_data(self, evaluation_data_dict):
        """
        Iterates over the __signal_data_list member and if a
        value of the 'Ticker' key matches a value in the given
        evaluation_data_dict that entry will be updated with
        the evaluation_data_dict.

        Parameters
        ----------
        :param evaluation_data_dict:
            'Dict' : A dict containing system evaluation data.
        """

        for instrument_dict in self.__signal_data_list:
            if instrument_dict['Ticker'] in evaluation_data_dict.values():
                instrument_dict['data'].update(evaluation_data_dict)
