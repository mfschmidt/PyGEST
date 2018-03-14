import os
import numpy as np
import pandas as pd
from scipy import stats
import time
import multiprocessing
import logging

from pygest import workers


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


def whack_a_gene(expr, conn, method='', corr='', cores=0, chunk_size=1, logger=None):
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


def ordered_genes(expr, conn, ranks, method='', corr='', cores=0, chunk_size=1, logger=None):
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
            correls.update(f(ranks[:(i + 1)]))

    elapsed = time.time() - full_start

    # Log results
    logger.info("ordered_genes ran {} {} correlations with {} processes and OPENBLAS_NUM_THREADS={} in {:0.2f}s.".format(
        len(expr.index), corr, cores, os.environ['OPENBLAS_NUM_THREADS'], elapsed
    ))

    # Return the list of correlations
    return correls
