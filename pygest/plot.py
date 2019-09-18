import math
import logging
import re
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pygest as ge
from pygest.convenience import bids_val, dict_from_bids, short_cmp, p_string
from pygest.algorithms import pct_similarity
from scipy.stats import ttest_ind


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


def heat_map(expression_df,
             title="Heat Map", fig_size=(5, 8), c_map="Reds",
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

    # Finally, plot the real curves
    def plot_curves(the_curve, ls, lc):
        if 'max' in the_curve[0]:
            legend_label = "{}, max r={:0.3f}".format(
                the_curve[0][6:], max(list(the_curve[1]['r' if 'r' in the_curve[1].columns else 'b']))
            )
        elif 'min' in the_curve[0]:
            legend_label = "{}, min r={:0.3f}".format(
                the_curve[0][6:], min(list(the_curve[1]['r' if 'r' in the_curve[1].columns else 'b']))
            )
        else:
            legend_label = the_curve[0][6:]
        ax.plot(list(the_curve[1].index), list(the_curve[1]['r' if 'r' in the_curve[1].columns else 'b']),
                label=legend_label, linestyle=ls, color=lc)

    # Plot the nulls first, so they are in the background
    print("nulls = ".format(nulls))
    if nulls is not None and len(nulls) > 0:
        for a_null in nulls:
            if 'smrt' in a_null[0]:
                plot_curves(a_null, ls=':', lc='lightgray')
            elif 'once' in a_null[0]:
                plot_curves(a_null, ls=':', lc='lightgray')
            else:
                plot_curves(a_null, ls=':', lc='yellow')

        # Also, plot the averaged null, our expected tortured r-value if we are only begging noise to confess
        def plot_mean_curves(mm, f, the_nulls):
            the_filter = [mm in x[0] for x in the_nulls]
            if sum(the_filter) > 0:
                the_nulls = [i for (i, v) in zip(the_nulls, the_filter) if v]
                mean_the_nulls = np.mean([x[1]['r' if 'r' in x[1].columns else 'b'] for x in the_nulls], axis=0)
                ll = "{}, mean {} r={:0.3f}".format('shuffled', mm, f(mean_the_nulls))
                ax.plot(list(the_nulls[0][1].index), mean_the_nulls, linestyle=':', color='darkgray', label=ll)

        plot_mean_curves("max", max, nulls)
        plot_mean_curves("min", min, nulls)

    if conns is not None and len(conns) > 0:
        for a_real in conns:
            if 'smrt' in a_real[0]:
                plot_curves(a_real, ls='-', lc='black')
            elif 'once' in a_real[0]:
                plot_curves(a_real, ls='--', lc='black')
            else:
                plot_curves(a_real, ls='-', lc='yellow')

    if conss is not None and len(conss) > 0:
        for a_real in conss:
            if 'smrt' in a_real[0]:
                plot_curves(a_real, ls='-', lc='green')
            elif 'once' in a_real[0]:
                plot_curves(a_real, ls='--', lc='green')
            else:
                plot_curves(a_real, ls='-', lc='yellow')

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
    ax.set_title("{}, {} hemisphere, {} set".format(donor, hemisphere, samples))

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


def push_vs_null_plot(data, donor, hem, samp, algo='smrt', comp='conn', mask='none',
                      label_keys=None):
    """ Use reasonable defaults to generate a push_plot for a particular dataset.
        This function does all of the gathering of files and generation of lists
        for sending to push_plot.

    :param data: an instance of the pygest.Data object
    :param donor: a string representing the donor of interest
    :param hem: a single character representing left or right hemisphere
    :param samp: 'cor' or 'sub' to indicate which sample set to use
    :param algo: 'smrt' for the smart efficient algorithm, 'once' or 'evry' for alternatives
    :param comp: 'conn' for connectivity, or 'cons' for connectivity similarity comparisons
    :param mask: 'none' for full data, or 'fine', 'coarse' or an integer for masked data
    :param label_keys: A list of keys can limit the size of the legend
    :return figure: axes of the plot
    """

    the_filters = {'sub': donor, 'hem': hem, 'samp': samp, 'algo': algo, 'comp': comp,
                   'mask': mask, 'adj': 'none', 'exclusions': ['test', 'NULL', ], }

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
         'label_keys': ['comp', ]},
    ]
    the_title = "{}_{}_{}_{}_{} actual vs shuffling".format(donor, hem, samp, comp, mask)
    return push_plot(plottables, the_title, label_keys=label_keys, fig_size=(8, 5))


def push_plot(push_sets, title="Push Plot", label_keys=None, plot_overlaps=False, fig_size=(16, 12), save_as=None):
    """ Draw a plot with multiple push results overlaid for comparison.

    :param push_sets: a list of dicts, each dict contains ('files', optional 'color', optional 'linestyle')
    :param title: override the default "Push Plot" title with something more descriptive
    :param label_keys: if specified, labels will be generated from these keys and the files in push_sets
    :param plot_overlaps: If true, calculate pct_overlap for each group and annotate the plot with them
    :param fig_size: override the default (16, 9) fig_size
    :param save_as: if specified, the plot will be drawn into the file provided
    :return: figure, axes of the plot
    """

    fig, ax = plt.subplots(figsize=fig_size)
    fig.tight_layout()

    # Plot a single horizontal line at y=0
    ax.axhline(0, 0, 17000, color='gray')
    if len(push_sets) == 0:
        return fig, ax

    # Plot each push_set
    ls = '-'
    lc = 'black'
    label = ''
    curve_list = []
    for i, push_set in enumerate(push_sets):
        if 'linestyle' in push_set:
            ls = push_set['linestyle']
        if 'color' in push_set:
            lc = push_set['color']
        if 'label' in push_set:
            label = push_set['label']
        if 'label_keys' in push_set:
            label_keys = push_set['label_keys']
        if len(push_set) > 0:
            ax, df = plot_pushes(push_set['files'], axes=ax, label=label, label_keys=label_keys,
                                 linestyle=ls, color=lc)
            df['push_set'] = i
            if len(df) > 0:
                curve_list.append(df)
    all_curves = pd.concat(curve_list, axis=0, sort=True)

    # Append summary statistics to a label
    def label_add_summary(x, d):
        if (len(d) == 0) or (len(d['best_score']) == 0):
            return "{} empty".format(x)
        if plot_overlaps and (len(d['f']) > 1):
            return "{}={:0.2f} with {:0.1%} overlap (n={})".format(
                x, np.mean(d['best_score']), pct_similarity(d['f']), len(d.index)
            )
        else:
            return "{}={:0.2f} (n={})".format(
                x, np.mean(d['best_score']), len(d.index)
            )

    # Tweak the legend, then add it to the axes, too
    def legend_sort_val(t):
        """ Sort the legend in a way that maps to peaks of lines visually. """
        val = re.compile(r"^.*r=(\S+) .*$").search(t[0]).groups()[0]
        # Return the negative so high values are first in the vertical legend.
        return float(val) * -1.0

    max_handles = []
    min_handles = []
    max_labels = []
    min_labels = []
    null_handles = []
    null_labels = []

    handles, labels = ax.get_legend_handles_labels()
    # Add summary statistics to labels
    labels = [label_add_summary(x, all_curves[all_curves['group'] == x]) for x in labels]
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
    :returns axes, pd.DataFrame: the axes containing the representations of results in files
    """

    if axes is None:
        fig, axes = plt.subplots()
    if len(files) == 0:
        return axes, pd.DataFrame()

    # Remember values for duplicate labels so we can average them at the end if necessary.
    summary_list = []
    # label_values = {}
    # label_files = {}

    for f in files:
        df = pd.read_csv(f, sep='\t', index_col=0)
        measure = 'r' if 'r' in df.columns else 'b'
        summary = {'f': f, 'measure': measure, 'tgt': bids_val('tgt', f)}

        if summary['tgt'] == 'max':
            the_best_score = df[measure].max()
            the_best_index = df[measure][5:].idxmax()
        elif summary['tgt'] == 'min':
            the_best_score = df[measure].min()
            the_best_index = df[measure][5:].idxmin()
        else:
            the_best_score = 0.0
            the_best_index = 0
        summary['best_score'] = the_best_score
        summary['best_index'] = the_best_index

        # If a label is not provided, create it, and in a way we can modify it later.
        # These labels are not tied to the axes, yet, and exist only in the local function
        if label == '':
            if label_keys is None:
                # default values, if they aren't specified
                label_keys = ['tgt', 'alg', 'msk', ]
            label_group = "_".join([short_cmp(bids_val(k, f)) for k in label_keys])
            label_group = label_group + ", {} {}".format(bids_val('tgt', f), measure)
            # try:
            #     label_values[label_group].append(best_score)
            # except KeyError:
            #     # If the label_group does not yet appear in label_values, start the list
            #     label_values[label_group] = [best_score, ]
            # try:
            #     label_files[label_group].append(f)
            # except KeyError:
            #     # If the label_group does not yet appear in label_files, start the list
            #     label_files[label_group] = [f, ]
        else:
            label_group = label
        summary['group'] = label_group

        # Plot the curve on the axes
        real_handles, axes_labels = axes.get_legend_handles_labels()
        if label_group in [x.split("=")[0] for x in axes_labels]:
            # If a label already exists, just plot the line without a label.
            axes.plot(list(df.index), list(df[measure]), linestyle=linestyle, color=color)
        else:
            # If there's no label, make one and plot the line with it.
            axes.plot(list(df.index), list(df[measure]), linestyle=linestyle, color=color, label=label_group)

        summary_list.append(summary)

    summaries = pd.DataFrame(summary_list)

    # Plot a center-point for the average 2D index,score in this group.
    if len(summaries.index) > 0:
        for grp in list(set(summaries['group'])):
            grp_df = summaries[summaries['group'] == grp]
            x_pos = np.mean(grp_df['best_index'])
            y_pos = np.mean(grp_df['best_score'])
            axes.plot(x_pos, y_pos, marker="D", markeredgecolor="white", markeredgewidth=2.0,
                      markersize=6.0, markerfacecolor=color)

    return axes, summaries


def plot_a_vs_b(data, label, a_value, b_value, base_set):
    """ Plot a in black solid lines and b in red dotted lines
    """
    # Compare old richiardi cortical samples to new Schmidt cortical samples.
    a = data.derivatives({**base_set, label: a_value}, shuffle='none', as_df=False)
    b = data.derivatives({**base_set, label: b_value}, shuffle='none', as_df=False)
    fig, ax = push_plot(
        [{'files': b, 'linestyle': ':', 'color': 'red'},
         {'files': a, 'linestyle': '-', 'color': 'black'}],
        title="{} vs {} {}s".format(a_value, b_value, label),
        label_keys=[label, ],
        fig_size=(10, 8),
        plot_overlaps=True,
    )
    return fig, ax


def plot_a_vs_null(data, label, a_value, base_set):
    """ Plot a in black solid lines and null distributions in red and blue dotted lines
    """
    # Compare old richiardi cortical samples to new Schmidt cortical samples.
    a = data.derivatives({**base_set, label: a_value}, shuffle='none', as_df=False)
    b = data.derivatives({**base_set, label: a_value}, shuffle='dist', as_df=False)
    c = data.derivatives({**base_set, label: a_value}, shuffle='raw', as_df=False)
    fig, ax = push_plot([
        {'files': c, 'linestyle': ':', 'color': 'red'},
        {'files': b, 'linestyle': ':', 'color': 'green'},
        {'files': a, 'linestyle': '-', 'color': 'black'}, ],
        title="{} vs null {}s".format(a_value, label),
        label_keys=[label, 'shuffle'],
        fig_size=(10, 8),
        plot_overlaps=True,
    )
    return fig, ax


def plot_a_vs_null_and_test(pygest_data, df, fig_size=(12, 8), addon=None):
    """ Plot a in black solid lines and null distributions in red and blue dotted lines
    """
    # Extract some characteristics from the data.
    main_traits = dict_from_bids(df['train_file'].unique()[0])
    factor = [f for f in df['factor'].unique() if len(f) > 0][0]
    # train_value = d[factor]
    train_value = df[df['phase'] == 'train']['value'].values[0]
    test_value = df[df['phase'] == 'test']['value'].values[0]
    descriptor = '_'.join([main_traits['sub'], main_traits['hem'], main_traits['ctx']])
    if factor not in ['sub', 'hem', 'ctx']:
        descriptor = descriptor + ' (' + factor + '=' + train_value + ')'

    a = [df['train_file'].unique()[0], ]
    b = pygest_data.derivatives(main_traits, shuffle='edge', as_df=False)
    c = pygest_data.derivatives(main_traits, shuffle='dist', as_df=False)
    d = pygest_data.derivatives(main_traits, shuffle='raw', as_df=False)
    if addon is None:
        fig, ax = push_plot([
            {'files': d, 'linestyle': ':', 'color': 'green'},
            {'files': c, 'linestyle': ':', 'color': 'red'},
            {'files': b, 'linestyle': ':', 'color': 'orchid'},
            {'files': a, 'linestyle': '-', 'color': 'black'}, ],
            title=descriptor,
            label_keys=[factor, 'shuffle'],
            fig_size=fig_size,
            plot_overlaps=False,
        )
    else:
        aa = [df['train_file'].unique()[0].replace('smrt', addon), ]
        bb = pygest_data.derivatives({**main_traits, 'alg': addon}, shuffle='edge', as_df=False)
        cc = pygest_data.derivatives({**main_traits, 'alg': addon}, shuffle='dist', as_df=False)
        dd = pygest_data.derivatives({**main_traits, 'alg': addon}, shuffle='raw', as_df=False)
        fig, ax = push_plot([
            {'files': dd, 'linestyle': ':', 'color': 'burlywood'},
            {'files': cc, 'linestyle': ':', 'color': 'gray'},
            {'files': bb, 'linestyle': ':', 'color': 'gray'},
            {'files': d, 'linestyle': ':', 'color': 'green'},
            {'files': c, 'linestyle': ':', 'color': 'red'},
            {'files': b, 'linestyle': ':', 'color': 'orchid'},
            {'files': aa, 'linestyle': '-', 'color': 'black'},
            {'files': a, 'linestyle': '-', 'color': 'black'}, ],
            title=descriptor,
            label_keys=[factor, 'shuffle', 'alg'],
            fig_size=fig_size,
            plot_overlaps=False,
        )

    # Move and resize rising plot of training data to make room for new box plots
    ax.set_position([0.04, 0.12, 0.48, 0.80])
    ax.set_yticklabels([])
    ax.set_label('rise')
    ax.set_xlabel('Training')
    ax.set_ylabel('Mantel Correlation')
    ax.yaxis.tick_right()

    # Create two box plots, one with training data, one with test data
    # (the_ax = ax_train, the_order = train_order, the_palette = train_color)
    the_order = {
        "train": ['train', 'edge', 'dist', 'agno'],
        "grays": ['train', 'edge', 'dist', 'agno'],
        "test": ['test', 'r_edge', 'r_dist', 'r_agno', 'random'],
    }
    the_palette = {
        "train": sns.color_palette(['black', 'orchid', 'red', 'green']),
        "test": sns.color_palette(['black', 'orchid', 'red', 'green', 'cyan']),
        "grays": sns.color_palette(['black', 'burlywood', 'gray', 'gray']),
    }

    def the_plots(the_data, tt, the_ax):
        """ Repeatable set of boxplot and swarmplot axes; just pass in data. """
        sns.boxplot(x='phase', y='score', data=the_data, ax=the_ax, order=the_order[tt], palette=the_palette[tt])
        sns.swarmplot(x='phase', y='score', data=the_data, ax=the_ax, order=the_order[tt], palette=the_palette[tt])
        the_ax.set_ylabel(None)
        the_ax.set_xlabel(tt)
        the_ax.set_ylim(ax.get_ylim())

    # Train pane
    ax_train = fig.add_axes([0.54, 0.12, 0.17, 0.80], label='train')
    if addon is None:
        the_plots(the_data=df[df['value'] == train_value], tt="train", the_ax=ax_train)
    else:
        the_plots(the_data=df[(df['algo'] == 'smrt') & (df['value'] == train_value)], tt="train", the_ax=ax_train)
        the_plots(the_data=df[(df['algo'] == addon) & (df['value'] == train_value)], tt="grays", the_ax=ax_train)
    ax_train.set_yticklabels([])
    ax_train.yaxis.tick_right()
    ax_train.set_title("train ({})".format("=".join([factor, train_value])))

    # Test pane
    ax_test = fig.add_axes([0.75, 0.12, 0.23, 0.80], label='test')
    if addon is None:
        the_plots(the_data=df[df['value'] == test_value], tt="test", the_ax=ax_test)
    else:
        the_plots(the_data=df[(df['algo'] == 'smrt') & (df['value'] == test_value)], tt="test", the_ax=ax_test)
    ax_test.set_title("test ({})".format("=".join([factor, test_value])))

    # With 'addon', we get 'once' or 'evry' rows, but we ignore those for these calculations.
    if addon is not None:
        df = df[df['algo'] == 'smrt']

    # Calculate overlaps for each column in the test boxplot, and annotate accordingly
    # These have moved from the rising plot axes legend
    overlap_columns = [
        {'phase': 'test', 'x': 0.0},
        {'phase': 'r_edge', 'x': 1.0},
        {'phase': 'r_dist', 'x': 2.0},
        {'phase': 'r_agno', 'x': 3.0},
        {'phase': 'random', 'x': 4.0},
    ]
    for col in overlap_columns:
        overlaps = df[df['phase'] == col['phase']]['overlap'].values
        max_y = max(df[df['phase'] == 'train']['score'].values)
        y_overlap = max(df[df['phase'] == col['phase']]['score'].values)
        try:
            overlap_annotation = "{:0.1%}\nsimilar".format(np.nanmean(overlaps))
        except TypeError:
            overlap_annotation = "similarity\nN/A"
        if y_overlap > max_y:
            ax_test.text(col['x'], y_overlap - 0.02, overlap_annotation, ha='center', va='top')
        else:
            ax_test.text(col['x'], y_overlap + 0.02, overlap_annotation, ha='center', va='bottom')

    return fig, (ax, ax_train, ax_test)


def plot_train_vs_test(df, mask_results=False, title="Title", fig_size=(12, 8), ymin=None, ymax=None):
    """ Plot train in black solid lines and test in red and blue dotted lines
    """

    # We can calculate train and test results by pure data or masked data.
    if mask_results:
        s_train = 'masked_train_score'
        s_test = 'masked_test_score'
        s_axis_label_mod = " (masked)"
    else:
        s_train = 'train_score'
        s_test = 'test_score'
        s_axis_label_mod = ""

    if s_train not in df.columns or s_test not in df.columns:
        print("Plotting train-vs-test, but don't have {} and {} columns!".format(s_train, s_test))
        return None

    # Calculate (or blindly accept) the range of the y-axis, which must be the same for all four axes.
    if (ymax is None) and (len(df.index) > 0):
        highest_possible_score = max(max(df['top_score']), max(df[s_train]), max(df[s_test]))
    else:
        highest_possible_score = ymax
    if (ymin is None) and (len(df.index) > 0):
        lowest_possible_score = min(min(df['top_score']), min(df[s_train]), min(df[s_test]))
    else:
        lowest_possible_score = ymin
    y_limits = (lowest_possible_score, highest_possible_score)

    # Plot the first pane, rising lines representing rising Mantel correlations as probes are dropped.
    a = df.loc[df['shuffle'] == 'none', 'path']
    b = df.loc[df['shuffle'] == 'edge', 'path']
    c = df.loc[df['shuffle'] == 'dist', 'path']
    d = df.loc[df['shuffle'] == 'agno', 'path']
    fig, ax = push_plot([
        {'files': list(d), 'linestyle': ':', 'color': 'green'},
        {'files': list(c), 'linestyle': ':', 'color': 'red'},
        {'files': list(b), 'linestyle': ':', 'color': 'orchid'},
        {'files': list(a), 'linestyle': '-', 'color': 'black'}, ],
        # title="Split-half train vs test results",
        label_keys=['shuffle'],
        fig_size=fig_size,
        title="",
        plot_overlaps=False,
    )
    fig.suptitle(title, fontsize=10)
    # Move and resize rising plot of training data to make room for new box plots
    ax.set_position([0.04, 0.12, 0.39, 0.80])
    ax.set_yticklabels([])
    ax.set_label('rise')
    ax.set_xlabel('Training')
    ax.set_ylabel('Mantel Correlation')
    ax.yaxis.tick_right()
    ax.set_ylim(bottom=y_limits[0], top=y_limits[1])

    # Create two box plots, one with training data, one with test data
    shuffle_order = ['none', 'edge', 'dist', 'agno']
    shuffle_color_boxes = sns.color_palette(['gray', 'orchid', 'red', 'green'])
    shuffle_color_points = sns.color_palette(['black', 'orchid', 'red', 'green'])

    """ Training box and swarm plots """
    ax_top = fig.add_axes([0.43, 0.12, 0.15, 0.80], label='top')
    sns.boxplot(x='shuffle', y='top_score', data=df, ax=ax_top,
                order=shuffle_order, palette=shuffle_color_boxes)
    sns.swarmplot(x='shuffle', y='top_score', data=df, ax=ax_top,
                  order=shuffle_order, palette=shuffle_color_points)
    ax_top.set_yticklabels([])
    ax_top.yaxis.tick_right()
    ax_top.set_ylabel(None)
    ax_top.set_xlabel('Training')
    # ax_train.set_title("train")
    ax_top.set_ylim(ax.get_ylim())

    """ Train box and swarm plots """
    ax_train = fig.add_axes([0.62, 0.12, 0.15, 0.80], label='train')
    sns.boxplot(x='shuffle', y=s_train, data=df, ax=ax_train,
                order=shuffle_order, palette=shuffle_color_boxes)
    sns.swarmplot(x='shuffle', y=s_train, data=df, ax=ax_train,
                  order=shuffle_order, palette=shuffle_color_points)
    ax_train.set_ylabel(None)
    ax_train.set_xlabel('Train' + s_axis_label_mod)
    # ax_test.set_title("test")
    ax_train.set_ylim(ax.get_ylim())

    """ Test box and swarm plots """
    ax_test = fig.add_axes([0.81, 0.12, 0.15, 0.80], label='test')
    sns.boxplot(x='shuffle', y=s_test, data=df, ax=ax_test,
                order=shuffle_order, palette=shuffle_color_boxes)
    sns.swarmplot(x='shuffle', y=s_test, data=df, ax=ax_test,
                  order=shuffle_order, palette=shuffle_color_points)
    ax_test.set_ylabel(None)
    ax_test.set_xlabel('Test' + s_axis_label_mod)
    # ax_test.set_title("test")
    ax_test.set_ylim(ax.get_ylim())

    """ Calculate overlaps and p-values for each column in the test boxplot, and annotate accordingly. """
    overlap_columns = [
        {'shuffle': 'none', 'xo': 0.0, 'xp': 0.0},
        {'shuffle': 'edge', 'xo': 1.0, 'xp': 0.5},
        {'shuffle': 'dist', 'xo': 2.0, 'xp': 1.0},
        {'shuffle': 'agno', 'xo': 3.0, 'xp': 1.5},
    ]
    actual_results = df[df['shuffle'] == 'none']['test_score'].values
    for i, col in enumerate(overlap_columns):
        overlaps = df[df['shuffle'] == col['shuffle']]['test_overlap'].values
        test_scores = df[df['shuffle'] == col['shuffle']]['test_score'].values
        try:
            max_y = max(df[df['phase'] == 'train']['test_score'].values)
        except ValueError:
            max_y = highest_possible_score
        try:
            y_overlap = max(test_scores)
            y_pval = max(max(test_scores), max(actual_results))
        except ValueError:
            y_overlap = highest_possible_score
            y_pval = highest_possible_score
        try:
            overlap_annotation = "{:0.1%}\nsimilar".format(np.nanmean(overlaps))
            t, p = ttest_ind(actual_results, test_scores)
            print("    plotting, full p = {}".format(p))
            p_annotation = p_string(p)
        except TypeError:
            overlap_annotation = "similarity\nN/A"
            p_annotation = "p N/A"

        if y_overlap > max_y:
            y_overlap = y_overlap - 0.02
            ax_test.text(col['xo'], y_overlap, overlap_annotation, ha='center', va='top')
        else:
            # expected path, annotate just above the swarm's top-most point.
            y_overlap = y_overlap + 0.02
            ax_test.text(col['xo'], y_overlap, overlap_annotation, ha='center', va='bottom')

        if i > 0:
            gap = 0.04
            y_pval = y_pval + 0.06
            y_pline = y_pval + 0.01 + (gap * i)
            ax_test.hlines(y_pline, 0.0, col['xo'], colors='k', linewidth=1)
            ax_test.vlines(0.0, y_pval, y_pline, colors='k', linewidth=1)
            ax_test.vlines(col['xo'], y_overlap + 0.06, y_pline, colors='k', linewidth=1)
            ax_test.text(col['xp'], y_pline, p_annotation, ha='center', va='bottom')

    return fig, (ax, ax_top, ax_train, ax_test)
