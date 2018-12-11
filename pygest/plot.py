import math
import logging
import re
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pygest as ge
from pygest.convenience import bids_val
from pygest import algorithms


def mantel_correlogram(X, Y, by, bins=8, r_method='Pearson', fig_size=(8, 5), save_as=None,
                       title='Mantel Correlogram', xlabel='distance bins', ylabel='correlation',
                       logger=None):
    """ Return a Mantel correlogram between vector_a and vector_b, over by
    
    :param X: For our purposes, usually an expression vector. Can be any vector of floats
    :param Y: For our purposes, usually a connectivity vector. Can be any vector of floats
    :param by: For our purposes, usually a distance vector. Can be any vector of floats
    :param bins: The number of bins can be specified
    :param r_method: The correlation can be calculated as 'Pearson', 'Spearman', or 'Kendall'
    :param tuple fig_size: size of desired plot, in inches (width, height)
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
    fig = plt.figure(figsize=fig_size)
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
    sns.set_style('white')
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
    sns.set_style('white')
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
    max_density_length = 1024

    # Attach to the proper logger
    if logger is None:
        logger = logging.getLogger('pygest')

    opp_c = "blue" if c == "red" else "red"

    fig, ax = plt.subplots(figsize=fig_size)
    sns.set_style('white')
    sub_data = data if len(data) <= max_density_length else np.random.choice(data, max_density_length)
    ax = overlay_normal(sns.distplot(sub_data, hist=True, rug=True, color=c), sub_data, c=opp_c)
    ax.set_title(title)

    if save_as is not None:
        logger.info("Saving distribution plot to {}".format(save_as))
        fig.savefig(save_as)

    return fig, ax


def heat_and_density_plot(value_matrix, density_position='top',
                          title="Heat Map", fig_size=(6, 4), ratio=3, c_map="Reds",
                          save_as=None, logger=None):
    """ Build, save, and return a heat map plot.

    :param value_matrix: A DataFrame or matrix containing data for the plot
    :param str density_position: Which edge gets the density_plot?
    :param str title: Override the default plot title with one of your choosing
    :param tuple fig_size: Dimensions (mostly relative) of figure generated
    :param integer ratio: This number-to-one heat map to density plot size
    :param str c_map: A seaborn color scheme string
    :param str save_as: If provided, the plot will be saved to this filename
    :param logging.Logger logger: If provided, logging will be directed to this logger
    :return fig: matplotlib figure object
    """

    # Attach to the proper logger
    if logger is None:
        logger = logging.getLogger('pygest')

    fig = plt.figure(figsize=fig_size)
    if density_position == 'left':
        gs = plt.GridSpec(ratio, ratio + 1)
        ax_main = fig.add_subplot(gs[:, 1:])
        ax_dens = fig.add_subplot(gs[:, 0])
        go_vertical = True
    elif density_position == 'right':
        gs = plt.GridSpec(ratio, ratio + 1)
        ax_main = fig.add_subplot(gs[:, :-1])
        ax_dens = fig.add_subplot(gs[:, -1])
        go_vertical = True
    elif density_position == 'bottom':
        gs = plt.GridSpec(ratio + 1, ratio)
        ax_main = fig.add_subplot(gs[:-1, :])
        ax_dens = fig.add_subplot(gs[-1, :])
        go_vertical = False
    else:  # density_position == 'top' or some invalid setting triggering 'top' default
        # GridSpec is set with nrows, ncols
        gs = plt.GridSpec(ratio + 1, ratio)
        # For a top-density, use [all rows after the 0th x all columns] for main
        ax_main = fig.add_subplot(gs[1:, :])
        # For a top-density, use [0th row x all columns] for density plot
        ax_dens = fig.add_subplot(gs[0, :])
        go_vertical = False

    # Density plots can take a long time to build with big samples; subsample if necessary
    max_density_length = 1024
    if isinstance(value_matrix, pd.DataFrame):
        value_matrix = value_matrix.as_matrix()
    if value_matrix.shape[0] == value_matrix.shape[1]:
        value_vector = value_matrix[np.tril_indices(n=value_matrix.shape[0], k=-1)]
    else:
        value_vector = value_matrix.flatten()
    if len(value_vector) <= max_density_length:
        sub_vector = value_vector
    else:
        sub_vector = np.random.choice(value_vector, max_density_length)

    sns.set_style('white')
    c = c_map.lower()[:-1]
    sns.heatmap(value_matrix, annot=False, ax=ax_main, cmap=c_map,
                xticklabels=False, yticklabels=False)
    sns.set_style('white')
    ax_dens = overlay_normal(sns.distplot(
        sub_vector, hist=True, rug=True, color=c, ax=ax_dens, vertical=go_vertical
    ), sub_vector, c=c)
    ax_dens.set_title(title)

    if save_as is not None:
        logger.info("Saving heat map to {}".format(save_as))
        fig.savefig(save_as)

    return fig


def whack_a_probe_plot(donor, hemisphere, samples, conns, conss=None, nulls=None, fig_size=(16, 9),
                       save_as=None, logger=None):
    """ Plot increasing correlations by different whack-a-probe algorithms.

    :param donor: The donor of interest
    :param hemisphere: The donor's hemisphere of interest
    :param samples: The subset (cor, sub, all) of donor's samples to represent
    :param conns: A list of tuples, each tuple (name, DataFrame), each DataFrame representing rising correlations
    :param conss: A list of tuples, each tuple (name, DataFrame), each DataFrame representing rising correlations
    :param nulls: A list of tuples, each tuple (name, DataFrame), each DataFrame representing rising correlations
    :param fig_size: The size, in inches, of the figure (width, height)
    :param str save_as: If provided, the plot will be saved to this filename
    :param logging.Logger logger: If provided, logging will be directed to this logger
    :return: matplotlib figure object
    """

    if logger is None:
        logger = logging.getLogger("pygest")

    fig, ax = plt.subplots(figsize=fig_size)
    # Plot a single horizontal line at y=0
    ax.axhline(0, 0, 17000, color='gray')

    # Plot the nulls first, so they are in the background
    print("nulls = ".format(nulls))
    if (nulls is not None) and len(nulls) > 0:
        for a_null in nulls:
            col = 'r' if 'r' in a_null[1].columns else 'b'
            if 'smrt' in a_null[0]:
                lc = 'lightgray'
            elif 'once' in a_null[0]:
                lc = 'lightgray'
            else:
                lc = 'yellow'
            if 'Unnamed: 0' in a_null[1].columns:
                ax.plot(list(a_null[1]['Unnamed: 0']), list(a_null[1][col]), linestyle=':', color=lc)
            else:
                print("{}: {}".format(a_null[0], a_null[1].columns))

        # Also, plot the averaged null, our expected tortured r-value if we are only begging noise to confess
        max_filter = ['max' in x[0] for x in nulls]
        if sum(max_filter) > 0:
            max_nulls = [i for (i, v) in zip(nulls, max_filter) if v]
            mean_max_nulls = np.mean([x[1]['r' if 'r' in x[1].columns else 'b'] for x in max_nulls], axis=0)
            leg_label = "{}, mean max r={:0.3f}".format('shuffled', max(mean_max_nulls))
            ax.plot(list(nulls[0][1]['Unnamed: 0']), mean_max_nulls,
                    linestyle=':', color='darkgray', label=leg_label)
        min_filter = ['min' in x[0] for x in nulls]
        if sum(min_filter) > 0:
            min_nulls = [i for (i, v) in zip(nulls, min_filter) if v]
            mean_min_nulls = np.mean([x[1]['r' if 'r' in x[1].columns else 'b'] for x in min_nulls], axis=0)
            leg_label = "{}, mean min r={:0.3f}".format('shuffled', min(mean_min_nulls))
            ax.plot(list(nulls[0][1]['Unnamed: 0']), mean_min_nulls,
                    linestyle=':', color='darkgray', label=leg_label)

    # Finally, plot the real curves
    for a_real in conns:
        if 'smrt' in a_real[0]:
            ls = '-'
            lc = 'black'
        elif 'once' in a_real[0]:
            ls = '--'
            lc = 'black'
        else:
            ls = '-'
            lc = 'yellow'
        if 'Unnamed: 0' in a_real[1].columns:
            if 'max' in a_real[0]:
                leg_label = "{}, max r={:0.3f}".format(
                    a_real[0][6:], max(list(a_real[1]['r' if 'r' in a_real[1].columns else 'b']))
                )
            elif 'min' in a_real[0]:
                leg_label = "{}, min r={:0.3f}".format(
                    a_real[0][6:], min(list(a_real[1]['r' if 'r' in a_real[1].columns else 'b']))
                )
            else:
                leg_label = a_real[0][6:]
            ax.plot(list(a_real[1]['Unnamed: 0']), list(a_real[1]['r' if 'r' in a_real[1].columns else 'b']),
                    label=leg_label, linestyle=ls, color=lc)

        else:
            print("{}: {}".format(a_real[0], a_real[1].columns))

    if conss is not None and len(conss) > 0:
        for a_real in conss:
            if 'smrt' in a_real[0]:
                ls = '-'
                lc = 'green'
            elif 'once' in a_real[0]:
                ls = '--'
                lc = 'green'
            else:
                ls = '-'
                lc = 'yellow'
            if 'Unnamed: 0' in a_real[1].columns:
                if 'max' in a_real[0]:
                    leg_label = "{}, max r={:0.3f}".format(
                        a_real[0][6:], max(list(a_real[1]['r' if 'r' in a_real[1].columns else 'b']))
                    )
                elif 'min' in a_real[0]:
                    leg_label = "{}, min r={:0.3f}".format(
                        a_real[0][6:], min(list(a_real[1]['r' if 'r' in a_real[1].columns else 'b']))
                    )
                else:
                    leg_label = a_real[0][6:]
                ax.plot(list(a_real[1]['Unnamed: 0']), list(a_real[1]['r' if 'r' in a_real[1].columns else 'b']),
                        label=leg_label, linestyle=ls, color=lc)

            else:
                print("{}: {}".format(a_real[0], a_real[1].columns))

    # Tweak the legend, then add it to the axes, too
    def leg_sort(t):
        """ Sort the legend in a way that maps to peaks of lines visually. """
        score = 0
        if 'smrt' in t[0]:
            score += 3
        elif 'once' in t[0]:
            score += 2
        else:
            score += 1
        if 'max' in t[0]:
            score *= -1
        elif 'min' in t[0]:
            score *= 1
        return score

    handles, labels = ax.get_legend_handles_labels()
    # sort both labels and handles by labels
    labels, handles = zip(*sorted(zip(labels, handles), key=leg_sort))
    ax.legend(handles, labels, loc=2)

    # Finish it off with a title
    ax.set_title("{}, {} hemisphere, {}".format(donor, hemisphere, samples))

    if save_as is not None:
        logger.info("Saving whack-a-probe plot to {}".format(save_as))
        fig.savefig(save_as)

    return fig


def push_plot_via_dict(data, d):
    """ Use settings from a json file to specify a push plot.

    :param data: and instance of the pygest.Data object
    :param d: a dictionary specifying configuration for plot
    :return: 0 for success, integer error code for failure
    """

    print("Pretending to make a plot from {}".format(d))

    colormap = ['black', 'blue', 'red', 'green', 'gray', 'orange', 'violet']

    plottables = []
    for subgroup_spec in d['intra'].keys():
        for i, subgroup in enumerate(d['intra'][subgroup_spec]):
            bids_filter = d['controls'].copy()
            bids_filter[subgroup_spec] = subgroup
            plottables.append({
                'files': data.derivatives(bids_filter),
                'color': colormap[i % len(colormap)],
                'linestyle': ':',
                'label_keys': [subgroup_spec]
            })

    for curve in plottables:
        print("Curves in {}".format(os.path.join(d['outdir'], d['filename'])))
        print(curve)

    fig, ax = push_plot(plottables, title=d['title'])
    fig.savefig(os.path.join(d['outdir'], d['filename']))

    return 0


def push_vs_null_plot(data, donor, hem, ctx, alg='smrt', cmp='conn', msk='none',
                      label_keys=None):
    """ Use reasonable defaults to generate a push_plot for a particular dataset.
        This function does all of the gathering of files and generation of lists
        for sending to push_plot.

    :param data: an instance of the pygest.Data object
    :param donor: a string representing the donor of interest
    :param hem: a single character representing left or right hemisphere
    :param ctx: 'cor' or 'sub' to indicate which sample set to use
    :param alg: 'smrt' for the smart efficient algorithm, 'once' or 'evry' for alternatives
    :param cmp: 'conn' for connectivity, or 'cons' for connectivity similarity comparisons
    :param msk: 'none' for full data, or 'fine', 'coarse' or an integer for masked data
    :param label_keys: A list of keys can limit the size of the legend
    :return figure: axes of the plot
    """

    the_filters = {'sub': donor, 'hem': hem, 'ctx': ctx, 'alg': alg, 'cmp': cmp,
                   'msk': msk, 'adj': 'none', 'exclusions': ['test', 'NULL', ], }

    # Get results for actual values and three types of shuffles
    the_results = {}
    for result_type in ['none', 'raw', 'dist', 'edge', ]:
        # Ask Data for a list of files that match our filters
        the_results[result_type] = data.derivatives(the_filters, shuffle=result_type)
        print("Retrieved {} results for {} shuffles.".format(len(the_results[result_type]), result_type))

    # Set up several sets of curves to be plotted
    plottables = [
        {'files': the_results['raw'], 'color': 'gray', 'linestyle': ':',
         'label': 'shuffle (n={})'.format(len(the_results['raw']))},
        {'files': the_results['dist'], 'color': 'red', 'linestyle': ':',
         'label': 'weighted (n={})'.format(len(the_results['dist']))},
        {'files': the_results['edge'], 'color': 'blue', 'linestyle': ':',
         'label': 'edges (n={})'.format(len(the_results['edge']))},
        {'files': the_results['none'], 'color': 'black', 'linestyle': '-',
         'label_keys': ['cmp', ]},
    ]
    the_title = "{}_{}_{}_{}_{} actual vs shuffling".format(donor, hem, ctx, cmp, msk)
    return push_plot(plottables, the_title, label_keys=label_keys, fig_size=(8, 5))


def push_plot(push_sets, title="Push Plot", label_keys=None, fig_size=(16, 12), save_as=None):
    """ Draw a plot with multiple push results overlaid for comparison.

    :param push_sets: a list of dicts, each dict contains ('files', optional 'color', optional 'linestyle')
    :param title: override the default "Push Plot" title with something more descriptive
    :param label_keys: if specified, labels will be generated from these keys and the files in push_sets
    :param fig_size: override the default (16, 9) fig_size
    :param save_as: if specified, the plot will be drawn into the file provided
    :return: figure, axes of the plot
    """

    fig, ax = plt.subplots(figsize=fig_size)
    # Plot a single horizontal line at y=0
    ax.axhline(0, 0, 17000, color='gray')

    # Plot each push_set
    ls = '-'
    lc = 'black'
    label = ''
    for push_set in push_sets:
        if 'linestyle' in push_set:
            ls = push_set['linestyle']
        if 'color' in push_set:
            lc = push_set['color']
        if 'label' in push_set:
            label = push_set['label']
        if 'label_keys' in push_set:
            label_keys = push_set['label_keys']
        if len(push_set) > 0:
            ax = plot_pushes(push_set['files'], linestyle=ls, color=lc, label=label, label_keys=label_keys, axes=ax)

    # Tweak the legend, then add it to the axes, too
    def legend_sort_val(t):
        """ Sort the legend in a way that maps to peaks of lines visually. """
        val = re.compile(r"^.*=(.*)$").search(t[0]).groups()[0]
        # Return the negative so high values are first in the vertical legend.
        return float(val) * -1.0

    max_handles = []
    min_handles = []
    max_labels = []
    min_labels = []
    null_handles = []
    null_labels = []
    handles, labels = ax.get_legend_handles_labels()

    # sort both labels and handles by labels
    if len(labels) > 0 and len(handles) > 0:
        labels, handles = zip(*sorted(zip(labels, handles), key=legend_sort_val))
        for i, l in enumerate(labels):
            if "max" in l:
                max_handles.append(handles[i])
                max_labels.append(labels[i])
            elif "min" in l:
                min_handles.append(handles[i])
                min_labels.append(labels[i])
            else:
                null_handles.append(handles[i])
                null_labels.append(labels[i])

    # Add a full legend (that will be emptied) and three separate legends with appropriate labels in each.
    ax.legend(handles, labels, loc=7)
    if len(max_labels) > 0:
        ax.add_artist(ax.legend(max_handles, max_labels, loc=2))
    if len(null_labels) > 0:
        ax.add_artist(ax.legend(null_handles, null_labels, loc=6))
    if len(min_labels) > 0:
        ax.add_artist(ax.legend(min_handles, min_labels, loc=3))

    # Finish it off with a title
    ax.set_title(title)

    if save_as is not None:
        fig.savefig(save_as)

    return fig, ax


def plot_pushes(files, axes=None, label='', label_keys=None, linestyle='-', color='black'):
    """ Plot the push results from a list of tsv files onto axes. This plots as many curves
        as are in files, and is called repeatedly (each time with different curves) on a
        single figure by push_plot.

    :param list files: A list of full file paths to tsv data holding push results
    :param axes: matplotlib axes for plotting to
    :param label: if supplied, override the calculated label for use in the legend for this set of results
    :param label_keys: if supplied, calculated the label from these fields
    :param linestyle: this linestyle will be used to plot these results
    :param color: this color will be used to plot these results
    :returns: the axes containing the representations of results in files
    """

    if axes is None:
        fig, axes = plt.subplots()

    # Remember values for duplicate labels so we can average them at the end if necessary.
    label_values = {}

    for f in files:
        df = pd.read_csv(f, sep='\t')
        column = 'r' if 'r' in df.columns else 'b'

        # If a label is not provided, create it, and in a way we can modify it later.
        # These labels are not tied to the axes, yet, and exist only in the local function
        if label == '':
            if label_keys is None:
                # default values, if they aren't specified
                label_keys = ['tgt', 'alg', 'msk', ]
            label_group = "_".join([bids_val(k, f) for k in label_keys])
            if bids_val('tgt', f) == 'max':
                label_group = label_group + ", max r"
                try:
                    label_values[label_group].append(df[column].max())
                except KeyError:
                    label_values[label_group] = [df[column].max(), ]
            elif bids_val('tgt', f) == 'min':
                label_group = label_group + ", min r"
                try:
                    label_values[label_group].append(df[column].min())
                except KeyError:
                    label_values[label_group] = [df[column].min(), ]
        else:
            label_group = label

        # TODO: provide some summary statistics or horizontal leader lines to the right for each group
        #  with quantitative data to supplement the visual.
        # If this axes' label already exists in the figure, plot this curve without a label.
        real_handles, axes_labels = axes.get_legend_handles_labels()
        if label_group in [x.split("=")[0] for x in axes_labels]:
            axes.plot(list(df['Unnamed: 0']), list(df[column]), linestyle=linestyle, color=color)
        else:
            if label == '':
                use_label = "{}={:0.3f}".format(label_group, np.mean(label_values[label_group]))
            else:
                use_label = label_group
            axes.plot(list(df['Unnamed: 0']), list(df[column]), linestyle=linestyle, color=color, label=use_label)
    return axes
