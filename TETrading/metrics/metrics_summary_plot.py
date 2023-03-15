import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

from TETrading.utils.metric_functions import calculate_max_drawdown, calculate_sharpe_ratio, \
    calculate_cagr 


def system_metrics_summary_plot(
    returns_data, rolling_equity, underlying_price_series, mae_data,
    mfe_data, position_returns_list, position_period_lengths, symbol, 
    summary_data_dict, plot_fig=False, save_fig_to_path=None
):
    """
    Plots a summary of metrics and statistics.

    Parameters
    ----------
    :param returns_data:
        'list' : A collection of returns data.
    :param rolling_equity:
        'list' : A collection of equity.
    :param underlying_price_series:
        'list' : A collection of a price series.
    :param mae_data:
        'list' : A collection of maximum adverse excursion data.
    :param mfe_data:
        'list' : A collection of maximum favorable excursion data.
    :param position_returns_list:
        'list' : A collection of position returns.
    :param position_period_lengths:
        'list' : A collection of position period lengths.
    :param symbol:
        'str' : The symbol/ticker of an asset.
    :param summary_data_dict:
        'dict' : A dict with data.
    :param plot_fig:
        Keyword arg 'bool' : True/False decides whether the plot
        the charts or not. Default value=False
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

    if underlying_price_series:
        underlying_returns = np.array(
            [
                (underlying_price_series[n] - underlying_price_series[n-1]) \
                / underlying_price_series[n-1] 
                for n in range(1, len(underlying_price_series))
            ]
        )
        underlying_sharpe = calculate_sharpe_ratio(underlying_returns)
        underlying_max_dd = calculate_max_drawdown(underlying_price_series)
        underlying_cagr = calculate_cagr(
            underlying_price_series[0], underlying_price_series[-1], 
            len(underlying_price_series)
        )

        axs[0, 1].plot(underlying_price_series, color='dodgerblue', linewidth=2)
        axs[0, 1].xaxis.set_major_locator(MaxNLocator(integer=True))
        axs[0, 1].set_title(f'{symbol} Price')
        axs[0, 1].set_xlabel('Periods')
        axs[0, 1].set_ylabel('Price')
        axs[0, 1].text(
            np.min(underlying_price_series), np.max(underlying_price_series)*0.9, 
            f'Sharpe ratio: {underlying_sharpe:3f}\n'
            f'Max drawdown (%): {underlying_max_dd:.3f}\n'
            f'CAGR (%): {underlying_cagr:.3f}',
            horizontalalignment='center'
        )

    returns = np.array(returns_data, dtype=float)
    pctl80_returns = returns[int(len(returns) * 0.1) : int(len(returns) * 0.90)]
    axs[0, 2].hist(
        pctl80_returns, orientation='vertical', color='navy',
        bins=int(np.sqrt(len(pctl80_returns)) * 2), alpha=0.5, 
        label='System returns'
    )
    distribution_stats_text = 'System: \n' \
            f'Mean: {np.mean(returns_data):.3f}\n' \
            f'Median: {np.median(returns_data):.3f}\n' \
            f'Std: {np.std(returns_data):.3f}'
    text_x_coord = np.min(pctl80_returns)

    if underlying_price_series:
        underlying_returns *= 100
        pctl80_underlying_returns = underlying_returns[
            int(len(underlying_returns) * 0.1) : int(len(underlying_returns) * 0.90)
        ]
        axs[0, 2].hist(
            pctl80_underlying_returns, orientation='vertical', color='dodgerblue',
            bins=int(np.sqrt(len(pctl80_underlying_returns)) * 2), alpha=0.5, 
            label='Underlying returns'
        )
        axs[0, 2].legend(loc='upper right')
        distribution_stats_text += '\n\nUnderlying:\n' \
            f'Mean: {np.mean(underlying_returns):.3f}\n' \
            f'Median: {np.median(underlying_returns):.3f}\n' \
            f'Std: {np.std(underlying_returns):.3f}'
        text_x_coord = np.min(pctl80_underlying_returns)

    axs[0, 2].yaxis.set_major_locator(MaxNLocator(integer=True))
    axs[0, 2].set_title('Returns (80th percentile)')
    axs[0, 2].set_xlabel('Return')
    axs[0, 2].set_ylabel('Frequency')
    vals, counts = np.unique(pctl80_returns, return_counts=True)
    axs[0, 2].text(
        text_x_coord, np.max(counts) * 0.5, distribution_stats_text
    )

    axs[0, 3].hist(
        position_period_lengths, edgecolor='black', linewidth=1.2, 
        orientation='horizontal', bins=len(position_period_lengths)
    )
    axs[0, 3].yaxis.set_major_locator(MaxNLocator(integer=True))
    axs[0, 3].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[0, 3].set_title('Position Period Lengths')
    axs[0, 3].set_xlabel('Frequency')
    axs[0, 3].set_ylabel('Periods')

    axs[1, 0].stem(
        [x for x in range(len(returns_data))], np.sort(returns_data), 
        markerfmt=' ', use_line_collection=True
    )
    axs[1, 0].set_title('Sorted Returns')
    axs[1, 0].set_xlabel('Periods')
    axs[1, 0].set_ylabel('Return')

    axs[1, 1].bar([x for x in range(len(returns_data))], returns_data)
    axs[1, 1].axhline(y=0, color='black', linewidth=0.5)
    axs[1, 1].set_title('Returns in Historic Order')
    axs[1, 1].set_xlabel('Periods')
    axs[1, 1].set_ylabel('Return')

    try:
        mae_mfe_dict = {
            'Positions in Historic Order': np.arange(0, len(position_returns_list)),
            'MFE': mfe_data,
            'Return': position_returns_list,
            'MAE': mae_data
        }
        ch_df = pd.DataFrame(mae_mfe_dict)

        ch_df.plot(
            x='Positions in Historic Order', y='MFE', kind='bar', 
            ax=axs[1, 2], color='lime'
        )
        ch_df.plot(
            x='Positions in Historic Order', y='MAE', kind='bar', 
            ax=axs[1, 2], color='red'
        )
        ch_df.plot(
            x='Positions in Historic Order', y='Return', kind='bar', 
            ax=axs[1, 2], color='black'
        )
        axs[1, 2].set_title('MFE, MAE, Return')
        axs[1, 2].set_ylabel('Return')
        try:
            axs[1, 2].xaxis.set_ticks(
                np.arange(0, len(position_returns_list), len(position_returns_list) - 1)
            )
        except ZeroDivisionError:
            pass
        axs[1, 2].xaxis.set_tick_params(rotation=0)
        axs[1, 2].xaxis.grid(False)
        axs[1, 2].yaxis.grid(True)
    except TypeError:
        print('\nMFE/MAE data missing')

    axs[1, 3].text(
        0.2, 0.3, 
        f'Win rate (%):            {summary_data_dict["%_wins"]:.2f}\n\n'
        f'Gross profit             {summary_data_dict["total_gross_profit"]}\n\n'
        f'Profit factor:           {summary_data_dict["profit_factor"]:.3f}\n\n'
        f'Sharpe ratio:            {summary_data_dict["sharpe_ratio"]:.3f}\n\n'
        f'Expectancy:              {summary_data_dict["expectancy"]:.3f}\n\n'
        f'Max drawdown (%):        {summary_data_dict["max_drawdown_(%)"]:.2f}\n\n'
        f'CAGR (%):                {summary_data_dict["cagr_(%)"]:.2f}', fontsize=12
    )
    axs[1, 3].axis('off')
    axs[1, 3].set_title('Performance Summary')

    plt.tight_layout()

    if save_fig_to_path:
        plt.savefig(save_fig_to_path + '\\' + symbol + '.jpg')
    if plot_fig:
        plt.show()
    else:
        plt.close('all')


def returns_distribution_plot(
    market_to_market_returns, mae, mfe, 
    plot_fig=False, save_fig_to_path=None
):
    """
    Plots histograms of given collections of returns, maximum adverse excursion
    and maximum favorable excursion.

    Parameters
    ----------
    :param market_to_market_returns:
        'list' : A collection of returns data.
    :param mae:
        'list' : A collection of maximum adverse excursion data.
    :param mfe:
        'list' : A collection of maximum favorable excursion data.
    :param plot_fig:
        Keyword arg 'bool' : True/False decides whether to plot
        the charts or not. Default value=False
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
    bins = np.max([int(np.sqrt(len(market_to_market_returns)) / 2), 8])
    ax[0].hist(market_to_market_returns, bins=bins, edgecolor='black')
    ax[0].set_title('Market to Market Returns Distribution')
    ax[0].text(
        10.0, 10.0,
        f'Count: {len(market_to_market_returns)}\n'
        f'Mean: {np.mean(market_to_market_returns):.3f}\n'
        f'Median: {np.median(market_to_market_returns):.3f}\n'
        f'Std: {np.std(market_to_market_returns):.3f}\n'
        f'25%: {np.quantile(market_to_market_returns, 0.25)}\n'
        f'50%: {np.quantile(market_to_market_returns, 0.5)}\n'
        f'75%: {np.quantile(market_to_market_returns, 0.75)}\n'
    )

    try:
        ax[1].hist(mae, bins=int(np.sqrt(len(mae))), edgecolor='black')
        ax[1].set_title('MAE distribution')
        ax[1].yaxis.set_major_locator(MaxNLocator(integer=True))
        ax[2].hist(mfe, bins=int(np.sqrt(len(mfe))), edgecolor='black')
        ax[2].yaxis.set_major_locator(MaxNLocator(integer=True))
        ax[2].set_title('MFE distribution')
    except ValueError:
        print('Failed to plot MAE and MFE')

    plt.tight_layout()

    if save_fig_to_path:
        plt.savefig(save_fig_to_path + '_returns_distribution' + '.jpg')
    if plot_fig:
        plt.show()
    else:
        plt.close('all')
