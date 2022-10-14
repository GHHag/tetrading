from enum import Enum


class MarketState(Enum):

    ENTRY = 'entry'
    BUY = 'entry'
    ACTIVE = 'active'
    HOLD = 'active'
    EXIT = 'exit'
    SELL = 'exit'
