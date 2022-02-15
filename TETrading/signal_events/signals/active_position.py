from TETrading.signal_events.signals.system_signals import SystemSignals


class ActivePositions(SystemSignals):
    """
    Handles data for active positions.
    """

    def __init__(self):
        super().__init__()
