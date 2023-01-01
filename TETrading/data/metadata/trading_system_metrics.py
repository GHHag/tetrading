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
    def SYMBOL(cls):
        return cls.__SYMBOL

    @classproperty
    def TOTAL_GROSS_PROFIT(cls):
        return cls.__TOTAL_GROSS_PROFIT

    @classproperty
    def PROFIT_FACTOR(cls):
        return cls.__PROFIT_FACTOR

    @classproperty
    def EXPECTANCY(cls):
        return cls.__EXPECTANCY

    @classproperty
    def RATE_OF_RETURN(cls):
        return cls.__RATE_OF_RETURN

    @classproperty
    def MAX_DRAWDOWN(cls):
        return cls.__MAX_DRAWDOWN

    @classproperty
    def ROMAD(cls):
        return cls.__ROMAD

    @classproperty
    def CAGR(cls):
        return cls.__CAGR

    @classproperty
    def PCT_WINS(cls):
        return cls.__PCT_WINS

    @classproperty
    def cls_attrs(cls):
        return [
            v for v in cls.__dict__.values() if isinstance(v, str) and \
            v not in ['__main__', 'TETrading.data.metadata.trading_system_metrics']
        ]

    @classproperty
    def system_evaluation_fields(cls):
        return (
            cls.__SYMBOL, cls.__SHARPE_RATIO, cls.__EXPECTANCY,
            cls.__PROFIT_FACTOR, cls.__CAGR, cls.__PCT_WINS,
            cls.__MEAN_RETURN, cls.__MAX_DRAWDOWN, cls.__ROMAD
        )
