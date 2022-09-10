from typing import Dict, List
import pandas as pd


class SystemSignals:
    """
    Handles data for signals.
    """

    def __init__(self):
        self.__data_list: List[Dict] = []

    @property
    def dataframe(self):
        """
        Returns a Pandas DataFrame with data from the __data_list
        member.

        :return:
            'Pandas.DataFrame'
        """

        if self.__data_list:
            return pd.DataFrame(instrument['data'] for instrument in self.__data_list)
        else:
            return None

    def __str__(self):
        if self.__data_list:
            return f'{self.dataframe.to_string()}'
        else:
            return f'{self.__data_list}'

    def add_data(self, symbol, data_dict):
        """
        Appends a dict with a symbol/ticker and given data to the
        __data_list member.

        Parameters
        ----------
        :param symbol:
            'str' : The symbol/ticker of an asset.
        :param data_dict:
            'dict' : A dict containing data of a position.
        """

        self.__data_list.append({'symbol': symbol, 'data': data_dict})

    def add_evaluation_data(self, evaluation_data_dict: Dict):
        """
        Iterates over the __data_list member and if a value
        of the 'symbol' key matches a value in the given
        evaluation_data_dict that entry will be updated with
        the evaluation_data_dict.

        Parameters
        ----------
        :param evaluation_data_dict:
            'dict' : A dict containing system evaluation data.
        """

        for instrument_dict in self.__data_list:
            if instrument_dict['symbol'] in evaluation_data_dict.values():
                assert 'data' in instrument_dict, \
                    "'instrument_dict' is missing required 'data' key is missing."
                assert isinstance(instrument_dict['data'], dict), \
                    "The value mapping to the 'data' key in 'instrument_dict' " \
                    "does not have the right format."
                instrument_dict['data'].update(evaluation_data_dict)
