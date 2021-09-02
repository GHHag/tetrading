import pandas as pd


class ExitSignals:
    """
    Handles data for exit signals.
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
            return f'Exit signals\n{self.dataframe.to_string()}'
        else:
            return f'Exit signals\n{self.__signal_data_list}'

    def add_signal_data(self, symbol, data_dict):
        """
        Appends a dict with a symbol/ticker and given data to the
        __signal_data_list member.

        Parameters
        ----------
        :param symbol:
            'str' : The symbol/ticker of an asset.
        :param data_dict:
            'Dict' : A dict containing data of an exit signal.
        """

        self.__signal_data_list.append({'Ticker': symbol, 'data': data_dict})
