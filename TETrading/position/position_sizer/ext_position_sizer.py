from TETrading.position.position_sizer.position_sizer import IPositionSizer


class ExtPositionSizer(IPositionSizer):

    def __init__(self, objective_function_str, *args, **kwargs):
        self.__objective_function_str = objective_function_str
        self.__args = args
        self.__kwargs = kwargs

    @property
    def objective_function_str(self):
        return self.__objective_function_str

    def __call__(self, *args, symbol='', **kwargs):
        return {
            'symbol': symbol,
            'sharpe_ratio': kwargs['metrics_dict']['Sharpe ratio'],
            'profit_factor': kwargs['metrics_dict']['Profit factor'],
            'expectancy': kwargs['metrics_dict']['Expectancy']
        }
