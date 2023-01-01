from TETrading.data.metadata.trading_system_attributes import classproperty


class TradingSystemSimulationAttributes:

    __SYMBOL = 'symbol'
    __START_CAPITAL = 'start_capital'
    __MEDIAN_GROSS_PROFIT = 'median_gross_profit'
    __MEAN_PCT_WINS = 'mean_%_wins'
    __MEDIAN_EXPECTANCY = 'median_expectancy'
    __MEDIAN_PROFIT_FACTOR = 'median_profit_factor'
    __AVG_RATE_OF_RETURN = 'avg_rate_of_return'
    __MEDIAN_RATE_OF_RETURN = 'median_rate_of_return'
    __MEDIAN_MAX_DRAWDOWN = 'median_max_drawdown_(%)'
    __AVG_ROMAD = 'avg_romad'
    __MEDIAN_ROMAD = 'median_romad'
    __AVG_CAGR = 'avg_cagr_(%)'
    __MEDIAN_CAGR = 'median_cagr_(%)'

    @classproperty
    def SYMBOL(cls):
        return cls.__SYMBOL

    @classproperty
    def START_CAPITAL(cls):
        return cls.__START_CAPITAL

    @classproperty
    def MEDIAN_GROSS_PROFIT(cls):
        return cls.__MEDIAN_GROSS_PROFIT

    @classproperty
    def MEAN_PCT_WINS(cls):
        return cls.__MEAN_PCT_WINS

    @classproperty
    def MEDIAN_EXPECTANCY(cls):
        return cls.__MEDIAN_EXPECTANCY

    @classproperty
    def MEDIAN_PROFIT_FACTOR(cls):
        return cls.__MEDIAN_PROFIT_FACTOR

    @classproperty
    def AVG_RATE_OF_RETURN(cls):
        return cls.__AVG_RATE_OF_RETURN

    @classproperty
    def MEDIAN_RATE_OF_RETURN(cls):
        return cls.__MEDIAN_RATE_OF_RETURN

    @classproperty
    def MEDIAN_MAX_DRAWDOWN(cls):
        return cls.__MEDIAN_MAX_DRAWDOWN

    @classproperty
    def AVG_ROMAD(cls):
        return cls.__AVG_ROMAD
    
    @classproperty
    def MEDIAN_ROMAD(cls):
        return cls.__MEDIAN_ROMAD

    @classproperty
    def AVG_CAGR(cls):
        return cls.__AVG_CAGR

    @classproperty
    def MEDIAN_CAGR(cls):
        return cls.__MEDIAN_CAGR

    @classproperty
    def cls_attrs(cls):
        return [
            v for v in cls.__dict__.values() if isinstance(v, str) and \
            v not in  ['__main__', 'TETrading.data.metadata.trading_system_simulation_metrics']
        ]
