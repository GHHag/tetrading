import numpy as np
import mplfinance as mpf
import pandas as pd


def candlestick_plot(
    df_og, entry_date, entry_price, exit_date, exit_price, save_fig_to_path=None
):
    """
    Plots a candlestick chart of a price series, with marks
    where a market has been entered and exited.

    Parameters
    ----------
    :param df_og:
        'Pandas DataFrame' : A dataset with data to be plotted.
    :param entry_date:
        'Pandas Timestamp/Datetime' : Time and date of entry.
    :param entry_price:
        'float/Decimal' : The price when entering the market.
    :param exit_date:
        'Pandas Timestamp/Datetime' : Time and date of exit.
    :param exit_price:
        'float/Decimal' : The price when exiting the market.
    :param save_fig_to_path:
        Keyword arg 'None/str' : Provide a file path as a string
        to save the plot as a file. Default value=None
    """

    df = df_og.copy(deep=True)
    df.set_index(pd.to_datetime(df['Date']), inplace=True)
    df.drop(['Date'], axis=1, inplace=True)
    df['Entry_date'] = df.index == entry_date
    df['Exit_date'] = df.index == exit_date
    df['Entry_date'].replace({False: np.nan}, inplace=True)
    df['Exit_date'].replace({False: np.nan}, inplace=True)
    df['Entry_price'] = df['Entry_date'].replace({True: entry_price})
    df['Exit_price'] = df['Exit_date'].replace({True: exit_price})

    mc = mpf.make_marketcolors(
        up='#B9B9B9',
        down='black',
        edge={'up': '#B9B9B9', 'down': 'black'},
        wick={'up': '#B9B9B9', 'down': 'black'},
        volume={'up': 'green', 'down': 'red'},
        inherit=True
    )
    plot_style = mpf.make_mpf_style(
        base_mpf_style='yahoo', gridstyle='', facecolor='#4C5063', marketcolors=mc
    )
    ap0 = [
        mpf.make_addplot(
            df[['Entry_price']], scatter=True, marker='^', color='green', markersize=75
        ),
        mpf.make_addplot(
            df[['Exit_price']], scatter=True, marker='v', color='red', markersize=75
        )
    ]

    if save_fig_to_path:
        mpf.plot(
            df, type='candle', style=plot_style, volume=True, main_panel=0, volume_panel=1,
            addplot=ap0, savefig=dict(fname=save_fig_to_path, dpi=100, pad_inches=0.25)
        )
    else:
        mpf.plot(
            df, type='candle', style=plot_style, volume=True, main_panel=0, volume_panel=1,
            addplot=ap0
        )
