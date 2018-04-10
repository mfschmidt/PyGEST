import math
import logging

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pygest as ge


def mantel_correlogram(X, Y, by, bins=8, r_method='Pearson', save_as=None,
                       title='Mantel Correlogram', xlabel='distance bins', ylabel='correlation',
                       logger=None):
    """ Return a Mantel correlogram between vector_a and vector_b, over by
    
    :param X: For our purposes, usually an expression vector. Can be any vector of floats
    :param Y: For our purposes, usually a connectivity vector. Can be any vector of floats
    :param by: For our purposes, usually a distance vector. Can be any vector of floats
    :param bins: The number of bins can be specified
    :param r_method: The correlation can be calculated as 'Pearson', 'Spearman', or 'Kendall'
    :param save_as: A file name for saving out the correlogram
    :param title: The title of the plot
    :param xlabel: The x-axis (usually distance bins) label
    :param ylabel: The y-axis (X vs Y correlations) label
    :param logger: We can log notes to your logger or ours.
    :return: matplotlib (Figure, Axes) objects containing the correlogram plot
    """

    # Attach to the proper logger
    if logger is None:
        logger = logging.getLogger('pygest')

    # Figure out the boundaries of the distance bins
    dist_min = math.floor(min(by)) - (math.floor(min(by)) % bins)
    dist_max = math.ceil(max(by)) + bins - (math.ceil(max(by)) % bins)
    dist_x_axis = np.arange(dist_min, dist_max, dist_max / bins)
    logger.info("({:0.2f} - {:0.2f}) -> ({} - {}), {}".format(
        min(by), max(by), dist_min, dist_max, dist_x_axis
    ))

    # Calculate correlations for each distance bin separately.
    ds = []
    prs = []
    ns = []
    rlos = []
    rhis = []
    for a in dist_x_axis:
        # Create distance filters for this particular bin
        by_filter = np.logical_and(by >= a, by < a + dist_max / bins)
        logger.info("  {ts:,} of {all:,} distances are between {sm:0.1f} and {lg:0.1f}.".format(
            ts=np.count_nonzero(by_filter),
            all=len(by),
            sm=a, lg=a + dist_max / bins
        ))

        # Filter vectors for this bin.
        _Y = Y[by_filter]
        _X = X[by_filter]
        ns.append(len(_X))
        logger.info("  using {:,} (in distance range) of the {:,} original values.".format(
            len(_Y),
            len(Y)
        ))

        # Calculate the correlations for this distance bin
        _r = ge.corr(_X, _Y, method=r_method)
        prs.append(_r)
        ds.append(a + dist_max / bins / 2)

        # Since r values are not going to be normally distributed (except maybe right at zero)
        # we need to transform them to Fisher's normal z' and back.
        z_prime = 0.50 * math.log((1 + _r) / (1 - _r))
        z_se = 1 / math.sqrt(len(_X) - 3)
        z_lo = z_prime - z_se
        z_hi = z_prime + z_se
        r_lo = (math.exp(2 * z_lo) - 1) / (math.exp(2 * z_lo) + 1)
        r_hi = (math.exp(2 * z_hi) - 1) / (math.exp(2 * z_hi) + 1)
        rlos.append(r_lo)
        rhis.append(r_hi)

        logger.info("  r = {:0.4f} ({:0.3f} - {:0.3f}) for these {} sample-sample relationships.".format(
            _r, r_lo, r_hi, len(_X)))

    # Calculate an overall r for comparison
    r = ge.corr(X, Y, method=r_method)

    # Build the plot
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    ax.axis([dist_min, dist_max, -1.0, 1.0])
    ax.axhline(y=0, xmin=0, xmax=1, linestyle=':', color='gray')
    # ax.axhline(y=spearman, xmin=0, xmax=1, linestyle='--', color='green')
    # ax.axhline(y=kendall, xmin=0, xmax=1, linestyle='--', color='red')
    oline = ax.axhline(y=r, xmin=0, xmax=1, linestyle='--', color='black', linewidth=2)
    # sline, = ax.plot(ds, srs, linestyle='-', marker='o', color='green')
    # kline, = ax.plot(ds, krs, linestyle='-', marker='o', color='red')
    pline, = ax.plot(ds, prs, linestyle='-', marker='o', color='black', linewidth=2)
    ax.vlines(x=ds, ymin=rlos, ymax=rhis, linewidth=1, color='black')
    ax.hlines(y=rhis, xmin=[x - 1 for x in ds], xmax=[x + 1 for x in ds], linewidth=1, color='black')
    ax.hlines(y=rlos, xmin=[x - 1 for x in ds], xmax=[x + 1 for x in ds], linewidth=1, color='black')
    for i, n in enumerate(ns):
        ax.annotate('n=', (ds[i], -0.90), ha='center')
        ax.annotate(n, (ds[i], -0.97), ha='center')
    ax.legend((pline, oline), ('Pearson r', 'all distances'), loc='upper center')
    # ax.legend((pline, sline, kline, oline), ('Pearson r', 'Spearman r', 'Kendall tau', 'all distances'))
    ax.set_xticks(tuple(np.append(dist_x_axis, dist_max)))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if save_as is not None:
        fig.savefig(save_as)

    # fig.close()

    return fig, ax


def conn_vs_expr_scatter(X, Y, xd, yd, save_as=None,
                         title='Connectivity and Expression', xlabel='expression', ylabel='connectivity',
                         logger=None):
    """ Scatter the many values for Y vs X in background and yd vs xd in foreground (darker).

        This is helpful to visualize connectivity values and expression values juxtaposed.
        Overlaying xd and yd can show how a subset of X and Y may lie in a particular area
        of the plot or have a slightly different correlation.

    :param X: A vector of expression correlations
    :param Y: A vector of connectivity values
    :param xd: A vector of expression correlations, a subset of X to call out
    :param yd: A vector of connectivity values, a subset of Y to call out
    :param save_as: The file name if you'd like to save the plot generated
    :param title: Override the default title
    :param xlabel: Override the default x label
    :param ylabel: Override the default y label
    :param logger: Catch logging output and divert it wherever you like
    :return: matplotlib (Figure, Axes) objects containing the regression plot
    """

    # Attach to the proper logger
    if logger is None:
        logger = logging.getLogger('pygest')

    logger.info("Plotting {} foreground points over {} background points.".format(
        len(X), len(xd)
    ))

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    ax.axis([min(X), max(X), -1.0, 1.0])

    # Set the axes and plot the grid first
    ax.axis([0.6, 1.0, -0.4, 1.0])
    ax.axhline(y=0, xmin=0, xmax=1, linestyle=':', color='gray')
    # plt.axvline(x=0, ymin=0, ymax=1, linestyle=':', color='gray')
    ax.title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Plot the points next
    ax.plot(X, Y, '.', color='lightblue')
    ax.plot(xd, yd, '.', color='gray')

    # And put the fit lines over the top of everything.
    m, b = np.polyfit(X, Y, 1)
    md, bd = np.polyfit(xd, yd, 1)
    ax.plot(X, m * X + b, '-', color='blue')
    ax.plot(xd, md * xd + bd, '-', color='black')

    # Add annotations
    r = np.corrcoef(X, Y)[0, 1]
    ax.annotate("all:", (0.61, 0.97), ha="left", va="top", color="blue")
    ax.annotate("m = {:0.3f}".format(m), (0.62, 0.91), ha="left", va="top", color="blue")
    ax.annotate("r = {:0.3f}".format(r), (0.62, 0.85), ha="left", va="top", color="blue")
    rd = np.corrcoef(xd, yd)[0, 1]
    ax.annotate("dist:", (0.61, 0.78), ha="left", va="top", color="black")
    ax.annotate("m = {:0.3f}".format(md), (0.62, 0.72), ha="left", va="top", color="black")
    ax.annotate("r = {:0.3f}".format(rd), (0.62, 0.66), ha="left", va="top", color="black")

    if save_as is not None:
        logger.info("Saving plot to {}".format(save_as))
        fig.savefig(save_as)

    return fig, ax


def expr_heat_map(expression_df,
                  title="Expression Heat Map", fig_size=(5, 8), c_map="Reds",
                  save_as=None, logger=None):
    """ Build, save, and return a heat map plot.

    :param pandas.DataFrame expression_df: A pandas DataFrame containing data for the plot
    :param str title: Override the default plot title with one of your choosing
    :param tuple fig_size: Dimensions (mostly relative) of figure generated
    :param str c_map: A seaborn color scheme string
    :param str save_as: If provided, the plot will be saved to this filename
    :param logging.Logger logger: If provided, logging will be directed to this logger
    :return fig, ax: matplotlib figure and axes objects
    """

    # Attach to the proper logger
    if logger is None:
        logger = logging.getLogger('pygest')

    fig, ax = plt.subplots(figsize=fig_size)
    sns.set_style('whitegrid')
    sns.heatmap(expression_df, annot=False, ax=ax, cmap=c_map)
    ax.set_title(title)

    if save_as is not None:
        logger.info("Saving heat map to {}".format(save_as))
        fig.savefig(save_as)

    return fig, ax


def similarity_heat_map(similarity_matrix,
                        title="Heat Map", fig_size=(5, 5), c_map="Reds",
                        save_as=None, logger=None):
    """ Build, save, and return a heat map plot.

    :param pandas.DataFrame similarity_matrix: A pandas DataFrame containing data for the plot
    :param str title: Override the default plot title with one of your choosing
    :param tuple fig_size: Dimensions (mostly relative) of figure generated
    :param str c_map: A seaborn color scheme string
    :param str save_as: If provided, the plot will be saved to this filename
    :param logging.Logger logger: If provided, logging will be directed to this logger
    :return fig, ax: matplotlib figure and axes objects
    """

    # Attach to the proper logger
    if logger is None:
        logger = logging.getLogger('pygest')

    fig, ax = plt.subplots(figsize=fig_size)
    sns.set_style('whitegrid')
    sns.heatmap(similarity_matrix, annot=False, ax=ax, cmap=c_map, vmin=-1.0, vmax=1.0)
    ax.set_title(title)

    if save_as is not None:
        logger.info("Saving heat map to {}".format(save_as))
        fig.savefig(save_as)

    return fig, ax


def overlay_normal(ax, data, c="red"):
    """ Provide a normal distribution Axes for overlay onto existing plot, based on data's mean and sd
    :param matplotlib.Axes ax: The axes object to draw onto
    :param data: The original data for basing our normal distribution
    :param str c: A string referring to a seaborn color
    :return: The same axes passed as an argument, but with a normal curve drawn over it
    """

    norm_data = np.random.normal(loc=np.mean(data), scale=np.std(data), size=2048)
    sns.kdeplot(norm_data, color=c, ax=ax)
    # ax.vlines(x=np.mean(data), ymin=0.0, ymax=1.0, linewidth=0.5, color=c)
    ax.vlines(x=np.mean(data) - (2 * np.std(data)), ymin=0, ymax=5.0, linewidth=0.5, color=c)
    ax.vlines(x=np.mean(data), ymin=0, ymax=5.0, linewidth=0.5, color=c)
    ax.vlines(x=np.mean(data) + (2 * np.std(data)), ymin=0, ymax=5.0, linewidth=0.5, color=c)
    return ax


def distribution_plot(data,
                      title="Distribution", fig_size=(5, 5), c="red",
                      save_as=None, logger=None):
    """ Build, save, and return a heat map plot.

    :param pandas.DataFrame data: A pandas DataFrame containing data for the plot
    :param str title: Override the default plot title with one of your choosing
    :param tuple fig_size: Dimensions (mostly relative) of figure generated
    :param str c: A seaborn color string
    :param str save_as: If provided, the plot will be saved to this filename
    :param logging.Logger logger: If provided, logging will be directed to this logger
    :return fig, ax: matplotlib figure and axes objects
    """

    # Density plots can take a long time to build with big samples; subsample if necessary
    max_density_length = 4096

    # Attach to the proper logger
    if logger is None:
        logger = logging.getLogger('pygest')

    opp_c = "blue" if c == "red" else "red"

    fig, ax = plt.subplots(figsize=fig_size)
    sns.set_style('whitegrid')
    sub_data = data if len(data) <= max_density_length else np.random.choice(data, max_density_length)
    ax = overlay_normal(sns.distplot(sub_data, hist=True, rug=True, color=c), sub_data, c=opp_c)
    ax.set_title(title)

    if save_as is not None:
        logger.info("Saving distribution plot to {}".format(save_as))
        fig.savefig(save_as)

    return fig, ax
