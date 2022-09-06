from decimal import Decimal


class Position:
    """
    Handles entering, exiting and data of a position.

    Parameters
    ----------
    capital : 'int/float/Decimal'
        The amount of capital to purchase assets with.
    fixed_position_size : Keyword arg 'bool'
        True/False decides if the position size should be the same
        fixed amount. The variable is used in the class' exit_market
        functions return statement control flow. Default value=True
    commission_pct_cost : Keyword arg 'float'
        The transaction cost given as a percentage 
        (a float from 0.0 to 1.0) of the total transaction. 
        Default value=0.0
    """

    def __init__(self, capital, fixed_position_size=True, commission_pct_cost=0.0):
        self.__entry_price = None
        self.__exit_price = None
        self.__position_size = None
        self.__direction = None
        self.__entry_dt, self.__exit_signal_dt = None, None
        self.__capital = Decimal(capital)
        self.__uninvested_capital = 0
        self.__fixed_position_size = fixed_position_size
        self.__commission_pct_cost = Decimal(commission_pct_cost)
        self.__commission = 0
        self.__active_position = False
        self.__last_price = None
        self.__unrealised_return = 0
        self.__unrealised_profit_loss = 0
        self.__returns_list = []
        self.__market_to_market_returns_list = []
        self.__position_profit_loss_list = []

    @property
    def entry_price(self):
        return self.__entry_price

    @property
    def position_size(self):
        return self.__position_size

    @property
    def commission(self):
        return self.__commission

    @property
    def active_position(self):
        return self.__active_position

    @property
    def entry_dt(self):
        return self.__entry_dt

    @property
    def exit_signal_dt(self):
        return self.__exit_signal_dt

    @property
    def capital(self):
        return self.__capital

    @property
    def fixed_position_size(self):
        return self.__fixed_position_size

    @property
    def returns_list(self):
        return self.__returns_list

    @property
    def unrealised_return(self):
        return self.__unrealised_return

    @property
    def unrealised_profit_loss(self):
        return self.__unrealised_profit_loss

    @property
    def position_return(self):
        """
        Calculates the positions return.

        :return:
            'Decimal'
        """

        if self.__direction == 'long':
            return Decimal(
                ((self.__exit_price - self.__entry_price) / self.__entry_price) * 100
            ).quantize(Decimal('0.02'))
        elif self.__direction == 'short':
            return Decimal(
                ((self.__entry_price - self.__exit_price) / self.__entry_price) * 100
            ).quantize(Decimal('0.02'))

    @property
    def net_result(self):
        """
        Calculates the positions net result.

        :return:
            'Decimal'
        """

        if self.__direction == 'long':
            return Decimal(
                (self.__position_size * self.__exit_price) - ((self.__position_size * self.__entry_price) + \
                                                                self.__commission)
            ).quantize(Decimal('0.02'))
        elif self.__direction == 'short':
            return Decimal(
                (self.__position_size * self.__entry_price) - ((self.__position_size * self.__exit_price) + \
                                                                self.__commission)
            ).quantize(Decimal('0.02'))

    @property
    def gross_result(self):
        """
        Calculates the positions gross result.

        :return:
            'Decimal'
        """

        if self.__direction == 'long':
            return Decimal(
                (self.__position_size * self.__exit_price) - (self.__position_size * self.__entry_price)
            ).quantize(Decimal('0.02'))
        elif self.__direction == 'short':
            return Decimal(
                (self.__position_size * self.__entry_price) - (self.__position_size * self.__exit_price)
            ).quantize(Decimal('0.02'))

    @property
    def profit_loss(self):
        """
        Calculates the positions P/L.

        :return:
            'Decimal'
        """

        if self.__direction == 'long':
            return Decimal(self.__exit_price - self.__entry_price).quantize(Decimal('0.02'))
        elif self.__direction == 'short':
            return Decimal(self.__entry_price - self.__exit_price).quantize(Decimal('0.02'))

    @property
    def mae(self):
        """
        Maximum adverse excursion, the minimum value from the
        list of unrealised returns.

        :return:
            'int/float/Decimal'
        """

        if min(self.__returns_list) >= 0:
            return 0
        else:
            return min(self.__returns_list)

    @property
    def mfe(self):
        """
        Maximum favorable excursion, the maximum value from the
        list of unrealised returns.

        :return:
            'int/float/Decimal'
        """
        if max(self.__returns_list) > 0:
            return max(self.__returns_list)
        else:
            return 0

    @property
    def market_to_market_returns_list(self):
        return self.__market_to_market_returns_list

    @property
    def to_dict(self): # dundermethod __dict__ om det finns?
        # vilka medlemmar vill jag ha med? de som behövs för att rita ut backtest grafer
        # typomvandlingar haer eller på annat staelle?
        return {
            'entry_dt': self.entry_dt,
            'exit_signal_dt': self.exit_signal_dt,
            'returns_list': [float(x) for x in self.returns_list],
            'position_return': float(self.position_return),
            'net_result': float(self.net_result),
            'gross_result': float(self.gross_result),
            'profit_loss': float(self.profit_loss),
            'mae': float(self.mae),
            'mfe': float(self.mfe),
       }

    def enter_market(self, entry_price, direction, entry_dt):
        """
        Enters market at the given price in the given direction.

        Parameters
        ----------
        :param entry_price:
            'int/float/Decimal' : The price of the asset when entering
            the market.
        :param direction:
            'str' : A string with 'long' or 'short', the direction that
            the position should be entered in.
        :param entry_dt:
            'Pandas Timestamp/Datetime' : Time and date when entering
            the market.
        """

        assert (self.__active_position is False), 'A position is already active'

        self.__entry_price = Decimal(entry_price)

        if direction not in ['long', 'short']:
            raise ValueError(
                'Direction of position specified incorrectly, make sure it is a string '
                f'with a value of either "long" or "short".\nGiven value: {direction}'
            )

        self.__direction = direction
        self.__position_size = int(self.__capital / self.__entry_price)
        self.__uninvested_capital = self.__capital - (self.__position_size * self.__entry_price)
        self.__commission = (self.__position_size * self.__entry_price) * self.__commission_pct_cost
        self.__entry_dt = entry_dt
        self.__active_position = True

    def exit_market(self, exit_price, exit_signal_dt):
        """
        Exits the market at the given price.

        Parameters
        ----------
        :param exit_price:
            'int/float/Decimal' : The price of the asset when exiting
            the market.
        :param exit_signal_dt:
            'Pandas Timestamp/Datetime' : Time and date when the signal
            to exit market was given.
        :return:
            'int/float/Decimal' : Returns the capital amount, which is
            the same as the Position was instantiated with if
            fixed_position_size was set to True. Otherwise it returns
            the position size multiplied with the exit price + the
            amount of capital that was left over when entering the
            market.
        """

        self.__exit_price = Decimal(exit_price)
        self.update(self.__exit_price)
        self.__exit_signal_dt = exit_signal_dt
        self.__active_position = False
        self.__commission += (self.__position_size * self.__exit_price) * self.__commission_pct_cost

        if not self.__fixed_position_size:
            self.__capital = Decimal(
                self.__position_size * self.__exit_price + self.__uninvested_capital
            ).quantize(Decimal('0.02'))
            return self.__capital
        else:
            return self.__capital

    def _unrealised_profit_loss(self, current_price):
        """
        Calculates and assigns the unrealised P/L, appends
        the value to __position_profit_loss_list.

        Parameters
        ----------
        :param current_price:
            'int/float/Decimal' : The assets most recent price.
        """

        if self.__direction == 'long':
            self.__unrealised_profit_loss = Decimal(
                current_price - self.__entry_price
            ).quantize(Decimal('0.02'))
            self.__position_profit_loss_list.append(self.__unrealised_profit_loss)
        elif self.__direction == 'short':
            self.__unrealised_profit_loss = Decimal(
                self.__entry_price - current_price
            ).quantize(Decimal('0.02'))
            self.__position_profit_loss_list.append(self.__unrealised_profit_loss)

    def _unrealised_return(self, current_price):
        """
        Calculates and assigns the unrealised return and the
        return from the two last recorded prices. Appends the
        values to __returns_list and __market_to_market_returns_list.

        Parameters
        ----------
        :param current_price:
            'int/float/Decimal' : The assets most recent price.
        """

        if self.__last_price is None:
            self.__last_price = self.__entry_price

        if self.__direction == 'long':
            unrealised_return = Decimal(
                ((current_price - self.__entry_price) / self.__entry_price) * 100
            ).quantize(Decimal('0.02'))
            self.__market_to_market_returns_list.append(
                Decimal(
                    (current_price - self.__last_price) / self.__last_price * 100
                ).quantize(Decimal('0.02'))
            )
            self.__returns_list.append(unrealised_return)
            self.__last_price = current_price
            self.__unrealised_return = unrealised_return
        elif self.__direction == 'short':
            unrealised_return = Decimal(
                ((self.__entry_price - current_price) / self.__entry_price) * 100
            ).quantize(Decimal('0.02'))
            self.__market_to_market_returns_list.append(
                Decimal(
                    (self.__last_price - current_price) / self.__last_price * 100
                ).quantize(Decimal('0.02'))
            )
            self.__returns_list.append(unrealised_return)
            self.__last_price = current_price
            self.__unrealised_return = unrealised_return

    def update(self, price):
        """
        Calls methods to update the unrealised return and
        unrealised profit and loss of the Position.

        Parameters
        ----------
        :param price:
            'float/Decimal' : The most recently updated price of the asset.
        """

        self._unrealised_return(price)
        self._unrealised_profit_loss(price)

    def print_position_status(self):
        """
        Prints the status of the Position.
        """

        if self.__active_position:
            print(
                f'Active position\n'
                f'Periods in position: {len(self.__returns_list)}\n'
                f'Unrealised return sequence: {list(map(float, self.__returns_list))}'
            )

    def print_position_stats(self):
        """
        Prints stats of the Position.
        """

        print(
            f'Unrealised P/L sequence: {list(map(float, self.__position_profit_loss_list))}\n'
            f'Market to market returns: {list(map(float, self.__market_to_market_returns_list))}\n'
            f'Unrealised return sequence: {list(map(float, self.__returns_list))}'
        )
