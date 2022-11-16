from TETrading.data.metadata.trading_system_attributes import classproperty


class TradingSystemMetrics:

    __SYMBOL = 'symbol'
    __NUM_OF_POSITIONS = 'number_of_positions'
    __START_CAPITAL = 'start_capital'
    __FINAL_CAPITAL = 'final_capital'
    __TOTAL_GROSS_PROFIT = 'total_gross_profit'
    __AVG_POS_NET_PROFIT = 'avg_pos_net_profit'
    __PCT_WINS = '%_wins'
    __EXPECTANCY = 'expectancy'
    __PROFIT_FACTOR = 'profit_factor'
    __SHARPE_RATIO = 'sharpe_ratio'
    __RATE_OF_RETURN = 'rate_of_return'
    __MEAN_PROFIT_LOSS = 'mean_p/l'
    __MEDIAN_PROFIT_LOSS = 'median_p/l'
    __STD_OF_PROFIT_LOSS = 'std_of_p/l'
    __MEAN_RETURN = 'mean_return'
    __MEDIAN_RETURN = 'median_return'
    __STD_OF_RETURNS = 'std_of_returns'
    __AVG_MAE = 'avg_mae'
    __MIN_MAE = 'min_mae'
    __AVG_MFE = 'avg_mfe'
    __MAX_MFE = 'max_mfe'
    __MAX_DRAWDOWN = 'max_drawdown_(%)'
    __ROMAD = 'romad'
    __CAGR = 'cagr_(%)'

    @classproperty
    def MAX_DRAWDOWN(cls):
        return cls.__MAX_DRAWDOWN

    @classproperty
    def cls_attrs(cls):
        #return [v for v in cls.__dict__.values() if isinstance(v, str) and v != '__main__' and v != 'TETrading.utils.metadata.trading_system_metrics']
        return [
            v for v in cls.__dict__.values() if isinstance(v, str) and \
            v not in  ['__main__', 'TETrading.utils.metadata.trading_system_metrics']
        ]
