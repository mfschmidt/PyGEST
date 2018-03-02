import math
import logging

import numpy as np
import matplotlib.pyplot as plt

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
    :return: a matplotlib object containing the correlogram plot
    """
    
    # Attach to the proper logger
    if logger is None:
        logger = logging.getLogger('pygest')
    
    dist_min = math.floor(min(by)) - (math.floor(min(by)) % bins)
    dist_max = math.ceil(max(by)) + bins - (math.ceil(max(by)) % bins)
    dist_x_axis = np.arange(dist_min, dist_max, dist_max / bins)
    logger.info(
        "({:0.2f} - {:0.2f}) -> ({} - {}), {}".format(min(by), max(by), dist_min, dist_max, dist_x_axis))

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
    plt.axis([dist_min, dist_max, -1.0, 1.0])
    plt.axhline(y=0, xmin=0, xmax=1, linestyle=':', color='gray')
    # plt.axhline(y=spearman, xmin=0, xmax=1, linestyle='--', color='green')
    # plt.axhline(y=kendall, xmin=0, xmax=1, linestyle='--', color='red')
    oline = plt.axhline(y=r, xmin=0, xmax=1, linestyle='--', color='black', linewidth=2)
    # sline, = plt.plot(ds, srs, linestyle='-', marker='o', color='green')
    # kline, = plt.plot(ds, krs, linestyle='-', marker='o', color='red')
    pline, = plt.plot(ds, prs, linestyle='-', marker='o', color='black', linewidth=2)
    plt.vlines(x=ds, ymin=rlos, ymax=rhis, linewidth=1, color='black')
    plt.hlines(y=rhis, xmin=[x - 1 for x in ds], xmax=[x + 1 for x in ds], linewidth=1, color='black')
    plt.hlines(y=rlos, xmin=[x - 1 for x in ds], xmax=[x + 1 for x in ds], linewidth=1, color='black')
    for i, n in enumerate(ns):
        plt.annotate('n=', (ds[i], -0.90), ha='center')
        plt.annotate(n, (ds[i], -0.97), ha='center')
    plt.legend((pline, oline), ('Pearson r', 'all distances'), loc='upper center')
    # plt.legend((pline, sline, kline, oline), ('Pearson r', 'Spearman r', 'Kendall tau', 'all distances'))
    plt.xticks(tuple(np.append(dist_x_axis, dist_max)))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if save_as is not None:
        plt.savefig(save_as)

    return plt
