import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator


def strategy_metrics_summary_plot(returns_data, rolling_equity, underlying_price_series, mae_data,
                                  mfe_data, profitable_positions, position_period_lengths, symbol, summary_data_dict,
                                  plot_fig=False, save_fig_to_path=None):
    """
    Plots a summary of metrics and statistics.

    Parameters
    ----------
    :param returns_data:
        'List' : A collection of returns data.
    :param rolling_equity:
        'List' : A collection of equity.
    :param underlying_price_series:
        'List' : A collection of a price series.
    :param mae_data:
        'List' : A collection of maximum adverse excursion data.
    :param mfe_data:
        'List' : A collection of maximum favorable excursion data.
    :param profitable_positions:
        'List' : A collection of profitable positions.
    :param position_period_lengths:
        'List' : A collection of position period lengths.
    :param symbol:
        'str' : The symbol/ticker of an asset.
    :param summary_data_dict:
        'Dict' : A dict with data.
    :param plot_fig:
        Keyword arg 'Boolean' : True/False decides whether the plot
        should be shown during run time or not. Default value=False
    :param save_fig_to_path:
        Keyword arg 'None/str' : Provide a file path as a string to
        save the plot as a file. Default value=None
    """

    plt.style.use('seaborn')

    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(18, 8.5))
    fig.tight_layout()

    axs[0, 0].plot(rolling_equity, color='navy', linewidth=2)
    axs[0, 0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[0, 0].set_title('Equity')
    axs[0, 0].set_xlabel('Periods')
    axs[0, 0].set_ylabel('Equity')

    axs[0, 1].plot(underlying_price_series, color='black', linewidth=2)
    axs[0, 1].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[0, 1].set_title(f'{symbol} price chart')
    axs[0, 1].set_xlabel('Periods')
    axs[0, 1].set_ylabel('Price')

    axs[0, 2].hist(returns_data, edgecolor='black', linewidth=1.2, orientation='vertical',
                   bins=int(np.sqrt(len(returns_data))))
    axs[0, 2].yaxis.set_major_locator(MaxNLocator(integer=True))
    axs[0, 2].set_title('Returns distribution')
    axs[0, 2].set_xlabel('Return')
    axs[0, 2].set_ylabel('Frequency')

    axs[0, 3].hist(position_period_lengths, edgecolor='black', linewidth=1.2, orientation='horizontal',
                   bins=int(np.sqrt(len(position_period_lengths))))
    axs[0, 3].yaxis.set_major_locator(MaxNLocator(integer=True))
    axs[0, 3].set_title('Position period lengths')
    axs[0, 3].set_xlabel('Frequency')
    axs[0, 3].set_ylabel('Periods')

    axs[1, 0].stem([x for x in range(len(returns_data))], sorted(returns_data), markerfmt=' ', use_line_collection=True)
    axs[1, 0].set_title('Sorted returns distribution')
    axs[1, 0].set_xlabel('Periods')
    axs[1, 0].set_ylabel('Return')

    axs[1, 1].bar([x for x in range(len(returns_data))], returns_data)
    axs[1, 1].axhline(y=0, color='black', linewidth=0.5)
    axs[1, 1].set_title('Returns in historic order')
    axs[1, 1].set_xlabel('Periods')
    axs[1, 1].set_ylabel('Return')

    try:
        mae_mfe_dict = {
            'Positions in historic order': np.arange(0, len(profitable_positions)),
            'MFE': mfe_data,
            'Return': profitable_positions,
            'MAE': mae_data
        }
        ch_df = pd.DataFrame(mae_mfe_dict)

        ch_df.plot(x='Positions in historic order', y='MFE', kind='bar', ax=axs[1, 2], color='green')
        ch_df.plot(x='Positions in historic order', y='MAE', kind='bar', ax=axs[1, 2], color='red')
        ch_df.plot(x='Positions in historic order', y='Return', kind='bar', ax=axs[1, 2], color='black')
        axs[1, 2].set_title('MFE, MAE, Return')
        axs[1, 2].set_ylabel('Return')
        try:
            axs[1, 2].xaxis.set_ticks(np.arange(1, len(profitable_positions), len(profitable_positions) - 2))
        except ZeroDivisionError:
            pass
        axs[1, 2].xaxis.set_tick_params(rotation=0)
        axs[1, 2].xaxis.grid(False)
        axs[1, 2].yaxis.grid(True)
    except TypeError:
        print('\nMFE/MAE data missing')

    axs[1, 3].text(0.2, 0.3, f'Win rate:                {format(summary_data_dict["% wins"], ".2f")}%\n\n'
                             f'Gross profit               {summary_data_dict["Total gross profit"]}\n\n'
                             f'Profit factor:           {summary_data_dict["Profit factor"]}\n\n'
                             f'Sharpe ratio:            {summary_data_dict["Sharpe ratio"]}\n\n'
                             f'Expectancy:              {summary_data_dict["Expectancy"]}\n\n'
                             f'Max drawdown (%):        {format(summary_data_dict["Max drawdown (%)"], ".2f")}\n\n'
                             f'CAGR (%):                {summary_data_dict["CAGR (%)"]}', fontsize=12)
    axs[1, 3].axis('off')
    axs[1, 3].set_title('Performance summary')

    plt.tight_layout()

    if save_fig_to_path:
        plt.savefig(save_fig_to_path + '\\' + symbol + '.jpg')
    if plot_fig:
        plt.show()
    else:
        plt.close('all')


def returns_distribution_plot(market_to_market_returns, mae, mfe,
                              plot_fig=False, save_fig_to_path=None):
    """
    Plots histograms of given collections of returns, maximum adverse excursion
    and maximum favorable excursion.

    Parameters
    ----------
    :param market_to_market_returns:
        'List' : A collection of returns data.
    :param mae:
        'List' : A collection of maximum adverse excursion data.
    :param mfe:
        'List' : A collection of maximum favorable excursion data.
    :param plot_fig:
        Keyword arg 'Boolean' : True/False decides whether the plot
        should be shown during run time or not. Default value=False
    :param save_fig_to_path:
        Keyword arg 'None/str' : Provide a file path as a string to
        save the plot as a file. Default value=None
    """

    plt.style.use('seaborn')

    fig, ax = plt.subplots(3, 1, figsize=(9, 9))

    market_to_market_returns = np.array(list(map(float, market_to_market_returns)))
    market_to_market_returns = market_to_market_returns[
        (market_to_market_returns > -20) & (market_to_market_returns < 20)
    ]
    bins = max([int(np.sqrt(len(market_to_market_returns)) / 2), 8])
    ax[0].hist(market_to_market_returns, bins=bins, edgecolor='black')
    ax[0].set_title('Market to market returns distribution')
    ax[0].text(10.0, 10.0,
               f'Count: {len(market_to_market_returns)}\n'
               f'Mean: {round(np.mean(market_to_market_returns), 4)}\n'
               f'Median: {round(np.median(market_to_market_returns), 4)}\n'
               f'Std: {round(np.std(market_to_market_returns), 4)}\n'
               f'25%: {np.quantile(market_to_market_returns, 0.25)}\n'
               f'50%: {np.quantile(market_to_market_returns, 0.5)}\n'
               f'75%: {np.quantile(market_to_market_returns, 0.75)}\n')

    ax[1].hist(mae, bins=int(np.sqrt(len(mae))), edgecolor='black')
    ax[1].set_title('MAE distribution')

    ax[2].hist(mfe, bins=int(np.sqrt(len(mfe))), edgecolor='black')
    ax[2].set_title('MFE distribution')

    plt.tight_layout()

    if save_fig_to_path:
        plt.savefig(save_fig_to_path + '_returns_distribution' + '.jpg')
    if plot_fig:
        plt.show()
    else:
        plt.close('all')
