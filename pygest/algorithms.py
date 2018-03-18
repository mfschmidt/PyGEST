import os
import numpy as np
import pandas as pd
from scipy import stats
import time
import multiprocessing
import logging

from pygest import workers


""" This file has some duplicate function going on.

    1. Step two of determining maximally influential genes:
    maximize_correlations is the preferred function to calculate most influential
        genes by removing genes one-at-a-time summatively until the maximal r value
        is achieved. It is single-threaded and uses scipy's pearsonr function.
    deprecated_ordered_genes does the same thing, but allows for bunches of settings that just
        make it confusing. On top of that, the map-reduce part does not work. But the
        code is kept around in case we need to play with multi-processing later.
    
    2. Step one (and repeated periodically) of determining maximally influential genes:
    order_by_correlation is the preferred function to do the one-probe-at-a-time
        whack-a-gene to calculate the approximate list of influential genes. It 
        uses scipy.stats.pearsonr, single threaded, and returns a pandas.DataFrame.
    deprecated_whack_a_gene is the deprecated original function to do the same thing. It has
        numerous options that can be confusing. And it returns a dictionary that
        must be converted to a DataFrame before sorting by correlation.
"""


def corr_expr_conn(expr, conn, method='', logger=None):
    """ Perform a correlation on the two matrices or vectors provided.

        Calling this repeatedly can be slow because it checks type on expr and conn
        each time. If either is not a vector, this function will convert it to a vector.
        To speed up many repeated correlations, pre-vectorize the values first, and perhaps
        consider using numpy.corrcoef() or scipy.pearsonr() independently.

    :param expr: gene expression matrix
    :param conn: functional connectivity matrix
    :param method: default numpy Pearson r, or specify 'Pearson' or 'Spearman' to use scipy
    :param logger: Any logging output can be handled by the caller's logger if desired.
    :returns: scalar Pearson correlation
    """

    # Deal with logging
    if not isinstance(logger, logging.Logger):
        if logger is None:
            logger = logging.getLogger('pygest')
        else:
            raise TypeError("A logger must be a logging.Logger object, not a {}".format(type(logger)))

    # Initialize some arrays
    expr_mat = None
    conn_mat = None
    final_expr_vector = None
    final_conn_vector = None

    # Handle DataFrames
    if isinstance(expr, pd.DataFrame) and isinstance(conn, pd.DataFrame):
        logger.debug("CEC: correlating two DataFrames, filtering and converting to arrays")
        overlap = [well_id for well_id in conn.index if well_id in expr.columns]
        expr_mat = np.corrcoef(expr.loc[:, overlap], rowvar=False)
        conn_mat = conn.loc[overlap, overlap].as_matrix()
        # Both matrices MUST now both be identically-shaped square matrices.
        logger.debug("CEC: expression & connectivity filtered down to {} matrices".format(expr_mat.shape))
    elif isinstance(expr, pd.DataFrame):
        if expr.shape[1] == conn.shape[0] == conn.shape[1]:
            expr_mat = np.corrcoef(expr, rowvar=False)
            conn_mat = conn.as_matrix()
        else:
            raise TypeError("If expr is a DataFrame, conn must be a DataFrame or expr-column-matched array.")
    elif isinstance(conn, pd.DataFrame):
        if conn.shape[1] == expr.shape[0] == expr.shape[1]:
            expr_mat = expr.as_matrix()
            conn_mat = conn.as_matrix()
        else:
            raise TypeError("If conn is a DataFrame, expr must be a DataFrame or conn-column-matched array.")
    else:
        # Validate passed variables as numpy arrays (vectors are 1-D arrays)
        if isinstance(expr, np.ndarray):
            # Make sure the expression matrix is a correlation matrix, building it if necessary
            try:
                if expr.shape[0] == expr.shape[1]:
                    logger.debug("expr is {}, assuming it's already a correlation matrix.".format(expr.shape))
                    final_expr_vector = expr[np.tril_indices(n=expr.shape[0], k=-1)]
                else:
                    logger.debug("expr is {}, building a correlation matrix from it.".format(expr.shape))
                    expr_mat = np.corrcoef(expr, rowvar=False)
                    final_expr_vector = expr_mat[np.tril_indices(n=expr_mat.shape[0], k=-1)]
            except IndexError:
                logger.debug("expr is {}, assuming it's already triangular vector.".format(expr.shape))
                final_expr_vector = expr
        else:
            raise TypeError("CEC requires input as DataFrames or numpy arrays. expr is a {}".format(
                type(expr)
            ))
        if isinstance(conn, np.ndarray):
            # Make sure the connectivity matrix is a square matrix. We can't build it on our own
            try:
                if conn.shape[0] == conn.shape[1]:
                    logger.debug("conn is {}, assuming it's a connectivity matrix.".format(conn.shape))
                    final_conn_vector = conn[np.tril_indices(n=conn.shape[0], k=-1)]
                else:
                    raise ValueError("CEC expects a square 2D array or a 1D vector, not {}".format(conn.shape))
            except IndexError:
                logger.debug("conn is {}, assuming it's already a triangular vector.".format(conn.shape))
                final_conn_vector = conn
        else:
            raise TypeError("CEC requires input as DataFrames or numpy arrays. conn is a {}".format(
                type(conn)
            ))

    # If we haven't yet built vectors, do so
    if final_expr_vector is None:
        final_expr_vector = expr_mat[np.tril_indices(n=expr_mat.shape[0], k=-1)]
    if final_conn_vector is None:
        final_conn_vector = conn_mat[np.tril_indices(n=conn_mat.shape[0], k=-1)]

    # Final check that our correlation can be run
    if len(final_expr_vector) != len(final_conn_vector):
        raise ValueError("Vector lengths differ {} vs {}, preventing a correlation from being run.".format(
            len(final_expr_vector), len(final_conn_vector)
        ))

    # And we should be good to do the correlation, now!
    logger.debug("CEC: {} {}-length expression vector with {}-length connectivity vector.".format(
        method, len(final_expr_vector), len(final_conn_vector)
    ))
    if method == 'Spearman':
        r, p = stats.spearmanr(final_expr_vector, final_conn_vector)
        return r
    elif method == 'Pearson':
        r, p = stats.pearsonr(final_expr_vector, final_conn_vector)
        return r
    elif method == 'Kendall':
        r, p = stats.kendalltau(final_expr_vector, final_conn_vector)
        return r
    else:
        return np.corrcoef(final_expr_vector, final_conn_vector)[0, 1]


def deprecated_whack_a_gene(expr, conn, method='', corr='', cores=0, chunk_size=1, logger=None):
    """ Perform repeated correlations between versions of expr's correlation matrix and conn.

    :param pd.DataFrame expr: gene expression DataFrame [probes x samples]
    :param pd.DataFrame conn: functional connectivity DataFrame [samples x samples]
    :param str method: '' default, just run serially, dependent on underlying numpy and BLAS/MKL
                       'multi-threaded' spawn {cores} threads to do correlations
                       'map-reduce' map the correlation set and reduce the results
    :param str corr: '' default uses numpy Pearson r,
                     'pearson' specifies scipy's stats.pearsonr,
                     'spearman' specifies scipy's stats.spearmanr
    :param int cores: use this many cores to run correlations
    :param int chunk_size: specify how many probe_ids to dump into each process
    :param logging.Logger logger: Any logging output can be handled by the caller's logger if desired.
    :return: dictionary with probe_id keys and Pearson correlation values for each removed probe
    """

    # Check propriety of arguments
    if not isinstance(expr, pd.DataFrame):
        raise TypeError("whack_a_gene expects 'expr' to be a pandas.DataFrame, not {}.".format(
            type(expr)
        ))
    if not isinstance(conn, pd.DataFrame):
        raise TypeError("whack_a_gene expects 'conn' to be a pandas.DataFrame, not {}.".format(
            type(conn)
        ))
    if logger is None:
        logger = logging.getLogger('pygest')

    # Determine overlap and log incoming numbers.
    overlapping_ids = [well_id for well_id in conn.index if well_id in expr.columns]
    logger.info("whack_a_gene starting with a [{} x {}] expression matrix and a [{} x {}] connectivity matrix.".format(
        expr.shape[0], expr.shape[1], conn.shape[0], conn.shape[1]
    ))
    logger.info("whack_a_gene found {} samples overlapping in both expression and connectivity.".format(
        len(overlapping_ids)
    ))

    full_start = time.time()

    # Convert DataFrames to matrices, then vectors, for coming correlations
    X = conn.loc[overlapping_ids, overlapping_ids].as_matrix()
    X = X[np.tril_indices(X.shape[0], k=-1)]
    # Pre-prune the expr DataFrame to avoid having to repeat it in the loop below.
    expr = expr.loc[:, overlapping_ids]
    Y = np.corrcoef(expr, rowvar=False)
    Y = Y[np.tril_indices(Y.shape[0], k=-1)]
    logger.debug("             created {}-len expression and {}-len connectivity vectors.".format(
        len(X), len(Y)
    ))

    # Run the repeated correlations, saving each one keyed to the missing gene when it was generated.
    # The key is probe_id, allowing lookup of probe_name or gene_name information later.
    correls = {0: np.corrcoef(Y, X)[0, 1]}

    # The function that does all the work (one implementation, see workers.py also).
    # This should be as tight and fast and optimized as possible.
    def whack_and_numpy(probe_id):
        y = np.corrcoef(expr.drop(labels=probe_id, axis=0), rowvar=False)
        y = y[np.tril_indices(n=y.shape[0], k=-1)]
        return {probe_id: np.corrcoef(y, X)[0, 1]}

    def whack_and_pearson(probe_id):
        y = np.corrcoef(expr.drop(labels=probe_id, axis=0), rowvar=False)
        y = y[np.tril_indices(n=y.shape[0], k=-1)]
        return {probe_id: stats.pearsonr(y, X)[0]}

    def whack_and_spearman(probe_id):
        y = np.corrcoef(expr.drop(labels=probe_id, axis=0), rowvar=False)
        y = y[np.tril_indices(n=y.shape[0], k=-1)]
        return {probe_id: stats.spearmanr(y, X)[0]}

    if cores > 0 or method == 'multi-threaded':
        if cores < 1:
            cores = multiprocessing.cpu_count() - 1
        # Evenly distribute the genes to whack across {cores} processes.
        logger.info("whack_a_gene asked to use multi-threading and {} with {} processes. {} possible.".format(
            corr, cores, multiprocessing.cpu_count()
        ))
        # We desperately want to share memory space. Copying the enormous expr_working
        # dataframe for each probe, then copying it again to remove the probe would be
        # a huge waste of time and resources. This leads us to either use lightweight
        # threads, or explicitly share the memory with heavier processes.
        tasks = multiprocessing.JoinableQueue()
        mgr = multiprocessing.Manager()
        d = mgr.dict()
        ns = mgr.Namespace()
        ns.expr = expr
        ns.conn = X
        ns.corr = corr
        logger.debug("Creating {} consumers.".format(cores))
        consumers = [workers.Consumer(tasks, d) for i in range(cores)]
        logger.debug("Starting consumers.")
        for w in consumers:
            w.start()

        logger.debug("Queuing up probe_ids, and tacking on poison pills for workers")
        for p in expr.index:
            tasks.put(workers.DropOneTask(ns, p))
        for i in range(cores):
            tasks.put(None)

        # Wait for all tasks to finish before returning
        tasks.join()
        logger.debug("Consumers finished!")

        correls.update(d)

    elif method == 'map-reduce':
        if cores < 1:
            cores = multiprocessing.cpu_count() - 1
        # Evenly distribute the genes to whack across {cores} processes.
        logger.info("whack_a_gene asked to use map-reduce with {}, {} processes. {} possible.".format(
            corr, cores, multiprocessing.cpu_count()
        ))
        print("Map-reduce does not work. Running single-process.")

        # mgr = multiprocessing.Manager()
        # d = mgr.dict()
        # ns = mgr.Namespace()
        # ns.expr = expr
        # ns.conn = X
        # mapper = workers.SimpleMapReduce(workers.probe_to_r, workers.dict_update, ns, cores)
        # mapper(expr.index, chunk_size)
        if corr == 'Pearson':
            f = whack_and_pearson
        elif corr == 'Spearman':
            f = whack_and_spearman
        else:
            f = whack_and_numpy
        for p in expr.index:
            correls.update(f(p))
    else:
        # If nobody bothered to request multi-processing, just run it and accept whatever BLAS or MKL
        # configuration their system is using. No bother.
        logger.info("whack_a_gene running a single {} thread, asked to use {} processes. {} possible.".format(
            corr, cores, multiprocessing.cpu_count()
        ))
        if corr.lower() == 'pearson':
            f = whack_and_pearson
        elif corr.lower() == 'spearman':
            f = whack_and_spearman
        else:
            f = whack_and_numpy
        for p in expr.index:
            correls.update(f(p))

    elapsed = time.time() - full_start

    # Log results
    logger.info("whack_a_gene ran {} {} correlations with {} processes and OPENBLAS_NUM_THREADS={} in {:0.2f}s.".format(
        len(expr.index), corr, cores, os.environ['OPENBLAS_NUM_THREADS'], elapsed
    ))

    # Return the list of correlations
    return correls


def order_by_correlation(expr, conn, logger=None):
    """ Perform repeated correlations between versions of expr's correlation matrix and conn.

    :param pd.DataFrame expr: gene expression DataFrame [probes x samples]
    :param pd.DataFrame conn: functional connectivity DataFrame [samples x samples]
    :param logging.Logger logger: Any logging output can be handled by the caller's logger if desired.
    :return: dictionary with probe_id keys and Pearson correlation values for each removed probe
    """

    f_name = 'order_by_correlation'
    # Check propriety of arguments
    if not isinstance(expr, pd.DataFrame):
        raise TypeError("{} expects 'expr' to be a pandas.DataFrame, not {}.".format(
            f_name, type(expr)
        ))
    if not isinstance(conn, pd.DataFrame):
        raise TypeError("{} expects 'conn' to be a pandas.DataFrame, not {}.".format(
            f_name, type(conn)
        ))
    if logger is None:
        logger = logging.getLogger('pygest')

    # Determine overlap and log incoming numbers.
    overlapping_ids = [well_id for well_id in conn.index if well_id in expr.columns]
    logger.info("{} starting with expr [{} x {}] and conn [{} x {}] - {} overlap.".format(
        f_name, expr.shape[0], expr.shape[1], conn.shape[0], conn.shape[1], len(overlapping_ids)
    ))

    full_start = time.time()

    # Convert DataFrames to matrices, then vectors, for coming correlations
    conn = conn.loc[overlapping_ids, overlapping_ids]
    conn_mat = conn.as_matrix()
    conn_vec = conn_mat[np.tril_indices(conn_mat.shape[0], k=-1)]
    # Pre-prune the expr DataFrame to avoid having to repeat it in the loop below.
    expr = expr.loc[:, overlapping_ids]
    expr_mat = np.corrcoef(expr, rowvar=False)
    expr_vec = expr_mat[np.tril_indices(expr_mat.shape[0], k=-1)]
    logger.debug("             created {}-len expression and {}-len connectivity vectors.".format(
        len(conn_vec), len(expr_vec)
    ))

    # Run the repeated correlations, saving each one keyed to the missing gene when it was generated.
    # The key is probe_id, allowing lookup of probe_name or gene_name information later.
    correls = {0: stats.pearsonr(expr_vec, conn_vec)[0]}

    for p in expr.index:
        expr_mat = np.corrcoef(expr.drop(labels=p, axis=0), rowvar=False)
        expr_vec = expr_mat[np.tril_indices(n=expr_mat.shape[0], k=-1)]
        correls.update({p: stats.pearsonr(expr_vec, conn_vec)[0]})

    elapsed = time.time() - full_start

    # Log results
    logger.info("{} ran {} pearsonr correlations w/o extra processes and OPENBLAS_NUM_THREADS={} in {:0.2f}s.".format(
        f_name, len(expr.index), os.environ['OPENBLAS_NUM_THREADS'], elapsed
    ))

    # Return the DataFrame of correlations, based on the dictionary we just built
    corr_df = pd.Series(correls, name='r')
    corr_df.index.name = 'probe_id'
    corr_df = pd.DataFrame(corr_df)
    corr_df['delta'] = corr_df['r'] - corr_df.loc[0, 'r']
    return corr_df


def deprecated_ordered_genes(expr, conn, ranks, method='', corr='', cores=0, chunk_size=1, logger=None):
    """ Remove each probe (additionally) from the original expression matrix, in order
        of least positive impact. After each removal, re-correlate with connectivity.

    :param pd.DataFrame expr: gene expression DataFrame [probes x samples]
    :param pd.DataFrame conn: functional connectivity DataFrame [samples x samples]
    :param list ranks: an list of probe_ids, in order of desired removal
    :param str method: '' default, just run serially, dependent on underlying numpy and BLAS/MKL
                       'multi-threaded' spawn {cores} threads to do correlations
                       'map-reduce' map the correlation set and reduce the results
    :param str corr: '' default uses numpy Pearson r,
                     'pearson' specifies scipy's stats.pearsonr,
                     'spearman' specifies scipy's stats.spearmanr
    :param int cores: use this many cores to run correlations
    :param int chunk_size: specify how many probe_ids to dump into each process
    :param logging.Logger logger: Any logging output can be handled by the caller's logger if desired.
    :return: dictionary with probe_id keys and Pearson correlation values for each removed probe
    """

    # Check propriety of arguments
    if not isinstance(expr, pd.DataFrame):
        raise TypeError("ordered_genes expects 'expr' to be a pandas.DataFrame, not {}.".format(
            type(expr)
        ))
    if not isinstance(conn, pd.DataFrame):
        raise TypeError("ordered_genes expects 'conn' to be a pandas.DataFrame, not {}.".format(
            type(conn)
        ))
    if not isinstance(ranks, list):
        raise TypeError("ordered_genes expects 'ranks' to be an ordered list, not {}.".format(
            type(ranks)
        ))
    if logger is None:
        logger = logging.getLogger('pygest')

    # Determine overlap and log incoming numbers.
    overlapping_ids = [well_id for well_id in conn.index if well_id in expr.columns]
    logger.info("ordered_genes starting with a [{} x {}] expression matrix and a [{} x {}] connectivity matrix.".format(
        expr.shape[0], expr.shape[1], conn.shape[0], conn.shape[1]
    ))
    logger.info("ordered_genes found {} samples overlapping in both expression and connectivity.".format(
        len(overlapping_ids)
    ))

    full_start = time.time()

    # Convert DataFrames to matrices, then vectors, for coming correlations
    X = conn.loc[overlapping_ids, overlapping_ids].as_matrix()
    X = X[np.tril_indices(X.shape[0], k=-1)]
    # Pre-prune the expr DataFrame to avoid having to repeat it in the loop below.
    expr = expr.loc[:, overlapping_ids]
    Y = np.corrcoef(expr, rowvar=False)
    Y = Y[np.tril_indices(Y.shape[0], k=-1)]
    logger.debug("             created {}-len expression and {}-len connectivity vectors.".format(
        len(X), len(Y)
    ))

    # Run the repeated correlations, saving each one keyed to the missing gene when it was generated.
    # The key is probe_id, allowing lookup of probe_name or gene_name information later.
    correls = {0: np.corrcoef(Y, X)[0, 1]}
    ns = {0: expr.shape[0]}

    # The function that does all the work (one implementation, see workers.py also).
    # This should be as tight and fast and optimized as possible.
    def whack_and_numpy(probe_id):
        y = np.corrcoef(expr.drop(labels=probe_id, axis=0), rowvar=False)
        y = y[np.tril_indices(n=y.shape[0], k=-1)]
        return {len(probe_id): np.corrcoef(y, X)[0, 1]}

    def whack_and_pearson(probe_id):
        y = np.corrcoef(expr.drop(labels=probe_id, axis=0), rowvar=False)
        y = y[np.tril_indices(n=y.shape[0], k=-1)]
        return {len(probe_id): stats.pearsonr(y, X)[0]}

    def whack_and_spearman(probe_id):
        y = np.corrcoef(expr.drop(labels=probe_id, axis=0), rowvar=False)
        y = y[np.tril_indices(n=y.shape[0], k=-1)]
        return {len(probe_id): stats.spearmanr(y, X)[0]}

    if cores > 0 or method == 'multi-threaded':
        if cores < 1:
            cores = multiprocessing.cpu_count() - 1
        # Evenly distribute the genes to whack across {cores} processes.
        logger.info("ordered_genes asked to use multi-threading and {} with {} processes. {} possible.".format(
            corr, cores, multiprocessing.cpu_count()
        ))
        # We desperately want to share memory space. Copying the enormous expr_working
        # dataframe for each probe, then copying it again to remove the probe would be
        # a huge waste of time and resources. This leads us to either use lightweight
        # threads, or explicitly share the memory with heavier processes.
        tasks = multiprocessing.JoinableQueue()
        mgr = multiprocessing.Manager()
        d = mgr.dict()
        ns = mgr.Namespace()
        ns.expr = expr
        ns.conn = X
        ns.corr = corr
        logger.debug("Creating {} consumers.".format(cores))
        consumers = [workers.Consumer(tasks, d) for i in range(cores)]
        logger.debug("Starting consumers.")
        for w in consumers:
            w.start()

        logger.debug("Queuing up rank lists, and tacking on poison pills for workers")
        for i, p in enumerate(ranks):
            if 0 < i < len(ranks) - 1:
                tasks.put(workers.DropToTask(ns, ranks[:i]))
        for i in range(cores):
            tasks.put(None)

        # Wait for all tasks to finish before returning
        tasks.join()
        logger.debug("Consumers finished!")

        correls.update(d)

    elif method == 'map-reduce':
        if cores < 1:
            cores = multiprocessing.cpu_count() - 1
        # Evenly distribute the genes to whack across {cores} processes.
        logger.info("ordered_genes asked to use map-reduce with {}, {} processes. {} possible.".format(
            corr, cores, multiprocessing.cpu_count()
        ))
        print("Map-reduce does not work. Running single-process.")

        # mgr = multiprocessing.Manager()
        # d = mgr.dict()
        # ns = mgr.Namespace()
        # ns.expr = expr
        # ns.conn = X
        # mapper = workers.SimpleMapReduce(workers.probe_to_r, workers.dict_update, ns, cores)
        # mapper(expr.index, chunk_size)
        if corr == 'Pearson':
            f = whack_and_pearson
        elif corr == 'Spearman':
            f = whack_and_spearman
        else:
            f = whack_and_numpy
        for p in expr.index:
            correls.update(f(p))
    else:
        # If nobody bothered to request multi-processing, just run it and accept whatever BLAS or MKL
        # configuration their system is using. No bother.
        logger.info("ordered_genes running a single {} thread, asked to use {} processes. {} possible.".format(
            corr, cores, multiprocessing.cpu_count()
        ))

        if corr.lower() == 'pearson':
            f = whack_and_pearson
        elif corr.lower() == 'spearman':
            f = whack_and_spearman
        else:
            f = whack_and_numpy
        for i, p in enumerate(ranks):
            d_entry = f(ranks[:(i + 1)])
            print("{:>5}. {}: {}".format(i, p, d_entry[len(ranks[:(i + 1)])]))
            correls.update(d_entry)
            # If we are more than half-way through, or if we aren't getting the highest correlation thus far
            # re-do the whack-a-gene on the remaining probes
            if i > len(ranks) / 2 or d_entry[len(ranks[:(i + 1)])] < max(correls.values()):
                print("    re-ordering remaining {} probes. (i={}, p={})".format(len(ranks[i + 1:]), i, p))
                new_corrs = deprecated_whack_a_gene(expr.loc[ranks[i + 1:], :], conn, method, corr, cores, chunk_size, logger)
                new_ranks = pd.Series(new_corrs, name='r')
                new_ranks.index.name = 'probe_id'
                new_ranks = pd.DataFrame(new_ranks)
                # new_ranks['gene'] = new_ranks.index.to_series().map(data.map('probe_id', 'gene_symbol'))
                new_ranks['delta'] = new_ranks['r'] - new_ranks.loc[0, 'r']
                ordered_probes = list(new_ranks.sort_values(by='delta', ascending=False).index)
                if 0 in ordered_probes:
                    ordered_probes.remove(0)

    elapsed = time.time() - full_start

    # Log results
    logger.info("ordered_genes ran {} {} correlations with {} processes and OPENBLAS_NUM_THREADS={} in {:0.2f}s.".format(
        len(expr.index), corr, cores, os.environ['OPENBLAS_NUM_THREADS'], elapsed
    ))

    # Return the list of correlations
    return correls


def maximize_correlation(expr, conn, ranks, logger=None):
    """ Remove each probe (additionally) from the original expression matrix, in order
        of least positive impact. After each removal, re-correlate with connectivity.

    :param pd.DataFrame expr: gene expression DataFrame [probes x samples]
    :param pd.DataFrame conn: functional connectivity DataFrame [samples x samples]
    :param list ranks: an list of probe_ids, in order of desired removal
    :param logging.Logger logger: Any logging output can be handled by the caller's logger if desired.
    :return: dictionary with probe_id keys and Pearson correlation values for each removed probe
    """

    f_name = 'maximize_correlations'

    # Check propriety of arguments
    if not isinstance(expr, pd.DataFrame):
        raise TypeError("{} expects 'expr' to be a pandas.DataFrame, not {}.".format(
            f_name, type(expr)
        ))
    if not isinstance(conn, pd.DataFrame):
        raise TypeError("{} expects 'conn' to be a pandas.DataFrame, not {}.".format(
            f_name, type(conn)
        ))
    if not isinstance(ranks, list):
        raise TypeError("{} expects 'ranks' to be an ordered list, not {}.".format(
            f_name, type(ranks)
        ))
    if logger is None:
        logger = logging.getLogger('pygest')

    # Determine overlap and log incoming numbers.
    overlapping_ids = [well_id for well_id in conn.index if well_id in expr.columns]
    logger.info("{} starting with expr [{} x {}] & corr [{} x {}] - {} overlapping.".format(
        f_name, expr.shape[0], expr.shape[1], conn.shape[0], conn.shape[1], len(overlapping_ids)
    ))

    full_start = time.time()

    # Convert DataFrames to matrices, then vectors, for coming correlations
    conn = conn.loc[overlapping_ids, overlapping_ids]
    conn_mat = conn.as_matrix()
    conn_vec = conn_mat[np.tril_indices(conn_mat.shape[0], k=-1)]
    # Pre-prune the expr DataFrame to avoid having to repeat it in the loop below.
    expr = expr.loc[:, overlapping_ids]
    expr_mat = np.corrcoef(expr, rowvar=False)
    expr_vec = expr_mat[np.tril_indices(expr_mat.shape[0], k=-1)]

    # Run the repeated correlations, saving each one keyed to the missing gene when it was generated.
    # The key is probe_id, allowing lookup of probe_name or gene_name information later.
    correls = {0: stats.pearsonr(expr_vec, conn_vec)[0]}

    i = 0
    j = 0
    probes_removed = [0]
    last_p = 0
    # Any fewer than 3 probes left will result in a 2-item matrix and failed correlations.
    while len(ranks) > 3:
        i += 1
        p = ranks.pop(0)
        removed_probe = expr.loc[[p, ], :]
        expr = expr.drop(labels=p, axis=0)
        expr_mat = np.corrcoef(expr, rowvar=False)
        expr_vec = expr_mat[np.tril_indices(n=expr_mat.shape[0], k=-1)]
        r = stats.pearsonr(expr_vec, conn_vec)[0]
        print("{:>5}. {}: {}".format(i-j, p, r))

        # Only use the correlation if it's the best thus far, or we've already re-ordered this one.
        if r > max(correls.values()):
            probes_removed.append(p)
            correls.update({i-j: r})
        elif last_p == p:
            logging.info("    r({})=={:0.5f} < {:0.5f}, but we re-ordered it & it's still lowest.".format(
                p, r, max(correls.values())
            ))
            probes_removed.append(p)
            correls.update({i-j: r})
        else:
            # If we aren't getting the highest correlation thus far, re-order the remaining probes
            print("    re-ordering remaining {} probes. (i={}, j={}, p={})".format(len(ranks), i, j, p))
            # Replace the removed probe, include it in the re-ordering
            j += 1
            expr = pd.concat([expr, removed_probe], axis=0)
            new_ranks = order_by_correlation(expr, conn, logger=logger)
            ranks = list(new_ranks.sort_values(by='delta', ascending=False).index)
            if 0 in ranks:
                ranks.remove(0)
        last_p = p

    elapsed = time.time() - full_start

    # Log results
    logger.info("{} ran {} correlations, {} re-orders with scipy and OPENBLAS_NUM_THREADS={} in {:0.2f}s.".format(
        f_name, i, j, os.environ['OPENBLAS_NUM_THREADS'], elapsed
    ))

    # Return the list of correlations
    gene_list = pd.Series(correls, name='r')
    gene_list.index.name = 'rank'
    gene_list = pd.DataFrame(gene_list)
    logger.info("{}: gene_list is {} long. Inserting {}-len probe_id list, leaving {}-len.".format(
        f_name, len(gene_list.index), len(probes_removed), len(ranks)
    ))
    logger.info("end of probes_removed = {}, all of ranks = {}.".format(
        probes_removed[-2:], ranks
    ))

    gene_list['probe_id'] = probes_removed

    # Finish the list with final 3 uncorrelatable top probes, filled with 0.0 correlations.
    remainder = pd.DataFrame(
        data={'r': [0.0, 0.0, 0.0], 'probe_id': ranks},
        index=[max(gene_list.index) + 1, max(gene_list.index) + 2, max(gene_list.index) + 3]
    )

    return pd.concat([gene_list, remainder], axis=0)
