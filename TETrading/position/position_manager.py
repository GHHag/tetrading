from TETrading.metrics.metrics import Metrics


class PositionManager:
    """
    Manages a collection of Position objects and their metrics.

    Parameters
    ----------
    symbol : 'str'
        The symbol/ticker of an asset.
    num_testing_periods : 'int'
        The number of periods in the data set.
    start_capital : 'int/float'
        The initial amount of capital to purchase assets with.
    capital_fraction : 'float'
        The fraction of the capital that will be used.
    asset_price_series : Keyword arg 'list'
        A price series of the asset. Default value=None
    """

    def __init__(
        self, symbol, num_testing_periods, start_capital, capital_fraction, 
        asset_price_series=None
    ):
        self.__symbol = symbol
        self.__asset_price_series = asset_price_series
        self.__num_testing_periods = num_testing_periods
        self.__start_capital = start_capital
        self.__capital_fraction = capital_fraction
        self.__safe_f_capital = self.__start_capital * self.__capital_fraction
        self.__uninvested_capital = self.__start_capital - self.__safe_f_capital

        self.__generated_positions = None
        self.__metrics = None

    @property
    def symbol(self):
        """
        The symbol/ticker of an asset.

        :return:
            'str'
        """

        return self.__symbol

    @property
    def metrics(self):
        """
        Returns the objects __metrics field if it's
        referable, otherwise returns None.

        :return:
            'Metrics'
        """

        if self.__metrics:
            return self.__metrics
        else:
            return None

    def generate_positions(self, trading_logic, *args, **kwargs):
        """
        Calls the trading_logic function to generate positions.
        Then calls _create_metrics to calculate metrics and
        statistics.

        Parameters
        ----------
        :param trading_logic:
            'function' : Logic to generate positions.
        :param args:
            'tuple' : A tuple with arguments to pass to the trade_logic
            function.
        :param kwargs:
            'dict' : A dict with keyword arguments to pass to the
            trade_logic function
        """

        self.__generated_positions = trading_logic(
            *args, capital=self.__safe_f_capital, **kwargs
        )
        self._create_metrics()

    def _create_metrics(self):
        """
        Creates an instance of the Metrics class, passing it the
        __generated_positions, __start_capital and
        __num_testing_periods.
        """

        if not self.__generated_positions:
            print('No positions generated.')
        else:
            self.__metrics = Metrics(
                self.__symbol, self.__start_capital, self.__num_testing_periods
            )
            self.__metrics(self.__generated_positions)

    def summarize_performance(self, plot_fig=False, save_fig_to_path=None):
        """
        Summarizes the performance of the managed positions,
        printing a summary of metrics and statistics, and plots
        a sheet with charts.

        Parameters
        ----------
        :param plot_fig:
            Keyword arg 'bool' : True/False decides if the figure
            will be plotted or not. Default value=False
        :param save_fig_to_path:
            Keyword arg 'None/str' : Provide a file path as a string
            to save the plot as a file. Default value=None
        """

        if len(self.__metrics.positions) < 1:
            print('No positions generated.')
        else:
            self.__metrics.print_metrics()
            if plot_fig or save_fig_to_path:
                self.__metrics.plot_performance_summary(
                    self.__asset_price_series, plot_fig=plot_fig, 
                    save_fig_to_path=save_fig_to_path
                )
