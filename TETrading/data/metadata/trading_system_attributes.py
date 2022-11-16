class classproperty(property):
    def __get__(self, cls, owner):
        return classmethod(self.fget).__get__(None, owner)()


class TradingSystemAttributes:

    __MARKET_STATE = 'market_state'
    __SIGNAL_INDEX = 'signal_index'
    __SIGNAL_DT = 'signal_dt'
    __SYMBOL = 'symbol'
    __DIRECTION = 'direction'
    __PERIODS_IN_POSITION = 'periods_in_position'
    __UNREALISED_RETURN = 'unrealised_return'

    @classproperty
    def MARKET_STATE(cls):
        return cls.__MARKET_STATE

    @classproperty
    def SIGNAL_INDEX(cls):
        return cls.__SIGNAL_INDEX
    
    @classproperty
    def SIGNAL_DT(cls):
        return cls.__SIGNAL_DT

    @classproperty
    def SYMBOL(cls):
        return cls.__SYMBOL

    @classproperty
    def DIRECTION(cls):
        return cls.__DIRECTION

    @classproperty
    def PERIODS_IN_POSITION(cls):
        return cls.__PERIODS_IN_POSITION

    @classproperty
    def UNREALISED_RETURN(cls):
        return cls.__UNREALISED_RETURN
