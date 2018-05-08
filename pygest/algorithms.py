import os
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.genmod.families import links
import time
import multiprocessing
import logging
import pickle
from pygest import workers

# Safely extract and remember how many threads the underlying matrix libraries are set to use.
BLAS_THREADS = os.environ['OPENBLAS_NUM_THREADS'] if 'OPENBLAS_NUM_THREADS' in os.environ else '0'
# Also, this cannot be changed dynamically. The state of the environment variable at the time
# numpy is imported is the state that is used by numpy permanently (until it is re-loaded, anyway).


# Algorithms can go by different names and must all refer to the same actual behavior.
# This mapping allows us to interpret all of them, and only have to match one canonical string for each.
algorithms = {
    'smart': 'smrt',
    'smrt': 'smrt',
    'one': 'once',
    'once': 'once',
    '1': 'once',
    'simp': 'once',
    'sing': 'once',
    'simple': 'once',
    'single': 'once',
    'e': 'evry',
    'ev': 'evry',
    'evry': 'evry',
    'every': 'evry',
    'ex': 'evry',
    'exh': 'evry',
    'exhaustive': 'evry',
    'exhaust': 'evry',
    'complete': 'evry',
    'comp': 'evry',
    'agg': 'evry',
    'aggressive': 'evry',
}


def correlate(expr, conn, method='', logger=None):
    """ Perform a correlation on the two matrices or vectors provided.

        The strength of this function is that you can pass it DataFrames, matrices, vectors,
        etc. and it will attempt to "do the right thing" to perform a correlation between
        expr and conn. It's largely a convenience function to wrap other correlations.
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


def get_beta(y, x, adj, adjust='linear'):
    """ Run a generalized linear model on y and x, including 'adj', and return the 'coef' beta coefficient

    :param y: endogenous, or dependent, variable - as a Series or DataFrame
    :param x: exogenous, or independent, variable along with any adjusters
    :param adj: the covariate, always distance in this case?, to include in the model
    :param adjust: 'linear' or 'log' model
    :return: A scalar float64 representing the beta coefficient, 'coef'
    """

    if adjust == 'log':
        link = links.log
    else:
        link = links.identity

    endog = pd.DataFrame({'y': y})
    exog = sm.add_constant(pd.DataFrame({'x': x, 'adj': adj}))
    result = sm.GLM(endog, exog, family=sm.families.Gaussian(link)).fit()
    # print(result.summary())
    return result.params['x']


def reorder_probes(expr, conn, dist=None, ascending=True,
                   mask=None, adjust='none', include_full=False, procs=0, logger=None):
    """ For each probe, knock it out and re-calculate relation between versions of expr's correlation matrix and conn.
        If adjustments are specified, a GLM will be used. If not, a Pearson correlation will.

    :param pd.DataFrame expr: gene expression DataFrame [probes x samples]
    :param pd.DataFrame conn: functional connectivity DataFrame [samples x samples]
    :param pd.DataFrame dist: distance DataFrame [samples x samples]
    :param boolean ascending: True to order with positive impact probes first, False to reverse order
    :param str mask: Mask out True edges
    :param str adjust: Include distance in a model
    :param boolean include_full: True to include the full correlation as probe_id==0, default is False
    :param int procs: How many processes to spread out over
    :param logging.Logger logger: Any logging output can be handled by the caller's logger if desired.
    :return: dictionary with probe_id keys and Pearson correlation values for each removed probe
    """

    # There are several ways to access the function name. I think all are ugly, so this works.
    f_name = 'reorder_probes'

    # Check propriety of arguments
    if dist is None:
        dist = pd.DataFrame()
    if logger is None:
        logger = logging.getLogger('pygest')
    for df in [('expr', expr), ('conn', conn), ('dist', dist)]:
        if not isinstance(df[1], pd.DataFrame):
            raise TypeError("{} expects '{}' to be a pandas.DataFrame, not {}.".format(
                f_name, df[0], type(df[1])
            ))

    # Determine overlap and log incoming numbers.
    overlapping_ids = [well_id for well_id in conn.index if well_id in expr.columns]
    logger.info("{} starting...".format(f_name))
    logger.debug("    with expr [{} x {}] and conn [{} x {}] - {} samples overlap.".format(
        expr.shape[0], expr.shape[1], conn.shape[0], conn.shape[1], len(overlapping_ids)
    ))

    full_start = time.time()

    # Convert DataFrames to matrices, then vectors, for coming correlations
    conn_mat = conn.loc[overlapping_ids, overlapping_ids].as_matrix()
    conn_vec = conn_mat[np.tril_indices(conn_mat.shape[0], k=-1)]
    # Pre-prune the expr DataFrame to avoid having to repeat it in the loop below.
    expr = expr.loc[:, overlapping_ids]
    expr_mat = np.corrcoef(expr, rowvar=False)
    expr_vec = expr_mat[np.tril_indices(expr_mat.shape[0], k=-1)]
    # Same for the distance matrix
    dist_mat = dist.loc[overlapping_ids, overlapping_ids].as_matrix()
    dist_vec = dist_mat[np.tril_indices(dist_mat.shape[0], k=-1)]
    # If we didn't get a real mask, make one that won't change anything.
    if mask is None:
        mask = np.ones(conn_vec.shape, dtype=bool)
        # Run the repeated correlations, saving each one keyed to the missing gene when it was generated.
        # The key is probe_id, allowing lookup of probe_name or gene_name information later.
    if adjust == 'none':
        score_name = 'r'
        score_method = 'pearson r'
        scores = {0: stats.pearsonr(expr_vec[mask], conn_vec[mask])[0]}
    else:
        score_name = 'b'
        score_method = 'glm'
        scores = {0: get_beta(conn_vec[mask], expr_vec[mask], dist_vec[mask], adjust)}

    # Set things up appropriately for multi-processing
    if procs == 0:
        # It's most likely best to spread out over all cores available, leaving one for overhead.
        procs = max(1, multiprocessing.cpu_count() - 1)
        logger.debug("    No core count specified, deciding to use {}, based on {} CPUs.".format(
            procs, procs + 1
        ))

    if procs == 1:
        # Run all correlations serially within the current process.
        logger.info("    One core requested; proceeding within existing process.")
        for p in expr.index:
            expr_mat = np.corrcoef(expr.drop(labels=p, axis=0), rowvar=False)
            expr_vec = expr_mat[np.tril_indices(n=expr_mat.shape[0], k=-1)]
            if adjust in ['linear', 'log']:
                scores[p] = get_beta(conn_vec, expr_vec, dist_vec, adjust)
            else:
                scores[p] = stats.pearsonr(expr_vec[mask], conn_vec[mask])[0]
    else:
        # Spawn {procs} extra processes, each running correlations in parallel
        logger.info("    {n} cores requested; spawning {n} new process{s}.".format(
            n=procs, s='es' if procs > 1 else ''
        ))
        print("Re-ordering {} probes ... (can take a few minutes)".format(len(expr.index)))
        queue = multiprocessing.JoinableQueue()
        mgr = multiprocessing.Manager()
        score_dict = mgr.dict()

        if adjust in ['linear', 'log']:
            # Create a worker process to use GLMs on each core/proc available.
            modelers = []
            for i in range(procs):
                # Let each process have its own copy of expr, rather than try to copy it with each task later
                # This results in a handful of copies rather than tens of thousands
                modelers.append(workers.LinearModeler(queue, expr, conn_vec, dist_vec, mask))
            for c in modelers:
                c.start()

            # Split the probes into slices, two per process (although 1 or 3 or 20 may be just as good).
            probe_slices = np.array_split(expr.index, procs * 2)
            for probe_slice in probe_slices:
                # Instantiate the task, and put it where it will be picked up and executed.
                queue.put(workers.LinearModelingTask(list(probe_slice), score_dict, adjust))
        else:
            # Create a worker process to perform correlations on each core/proc available.
            correlators = []
            for i in range(procs):
                # Let each process have its own copy of expr, rather than try to copy it with each task later
                # This results in a handful of copies rather than tens of thousands
                correlators.append(workers.Correlator(queue, expr, conn_vec, mask))
            for c in correlators:
                c.start()

            # Split the probes into slices for each task.
            probe_slices = np.array_split(expr.index, procs * 2)
            for probe_slice in probe_slices:
                queue.put(workers.CorrelationTask(list(probe_slice), score_dict, 'Pearson'))

        # At the end of the queue, place a message to quit and clean up (a poison pill) for each worker
        for i in range(procs):
            queue.put(None)

        # Then simply wait for them to finish.
        queue.join()

        # And record their results.
        scores.update(score_dict)

    elapsed = time.time() - full_start

    # Log results
    logger.debug("    ran {} {} times in {} process (OPENBLAS_NUM_THREADS={}).".format(
        score_method, len(expr.index), procs, BLAS_THREADS
    ))
    logger.info("    {} {}s in {:0.2f}s.".format(len(expr.index), score_name, elapsed))

    # Return the DataFrame of correlations, based on the dictionary we just built
    score_df = pd.Series(scores, name=score_name)
    score_df.index.name = 'probe_id'
    score_df = pd.DataFrame(score_df)
    score_df['delta'] = score_df[score_name] - score_df.loc[0, score_name]

    if include_full:
        return score_df.sort_values(by='delta', ascending=ascending)
    else:
        return score_df.sort_values(by='delta', ascending=ascending).drop(labels=0, axis=0)


def retrieve_progress(progress_file):
    if os.path.isfile(progress_file):
        try:
            with open(progress_file, 'rb') as f:
                df = pickle.load(f)
            ps = list(df['probe_id'])
            rs = df['r'].to_dict()
            re_orders = list(df['refresh'])
            return ps, rs, re_orders
        except FileNotFoundError:
            print("Cannot open {}; starting over.".format(progress_file))
        except pickle.UnpicklingError:
            print("Opened {}; cannot understand its contents; starting over.".format(progress_file))
        except KeyError:
            print("Found a DataFrame in {}, but could not get progress from it; starting over.".format(progress_file))
    return [], {}, []


def save_progress(progress_file, rs, ps, re_orders):
    # Return the list of correlations
    df = pd.Series(rs, name='r')
    df.index.name = 'rank'
    df = pd.DataFrame(df)
    df['probe_id'] = ps
    df['refresh'] = re_orders
    with open(progress_file, 'wb') as f:
        pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)


def push_score(expr, conn, dist,
               algo=algorithms['smrt'], ascending=True,
               mask=None, adjust='none', progress_file=None, cores=0, logger=None):
    """ Remove each probe (additionally) from the original expression matrix, in order
        of least positive impact. After each removal, re-correlate with connectivity.

    :param pd.DataFrame expr: gene expression DataFrame [probes x samples]
    :param pd.DataFrame conn: functional connectivity DataFrame [samples x samples]
    :param pd.DataFrame dist: distance DataFrame [samples x samples]
    :param str algo: 'once' orders probes only once, and sticks to that order throughout sequential probe removal
                     'smrt' orders probes 'once', then re-orders it each time the correlation drops
                     'evry' re-runs whack-a-gene every single iteration
    :param bool ascending: True to maximize positive correlation, False to pursue most negative correlation
    :param np.array mask: A boolean mask to filter out unwanted edges in triangle vectors
    :param str adjust: String indicating adjustment style, 'log' or anything else is treated as linear 'identity'
    :param str progress_file: An intermediate file to save progress and resume if necessary
    :param int cores: Spreading out to {cores} multiple processors can be specified
    :param logging.Logger logger: Any logging output can be handled by the caller's logger if desired.
    :return: DataFrame with probe_id keys and Pearson correlation values for each removed probe
    """

    f_name = 'push_score'
    total_probes = len(expr.index)

    # Check propriety of arguments
    if dist is None:
        dist = pd.DataFrame()
    if logger is None:
        logger = logging.getLogger('pygest')
    for df in [('expr', expr), ('conn', conn), ('dist', dist)]:
        if not isinstance(df[1], pd.DataFrame):
            raise TypeError("{} expects '{}' to be a pandas.DataFrame, not {}.".format(
                f_name, df[0], type(df[1])
            ))

    if adjust == 'none':
        score_name = 'r'
    else:
        score_name = 'b'

    # Determine overlap and log incoming numbers.
    overlapping_ids = [well_id for well_id in conn.index if well_id in expr.columns]
    logger.info("{} starting...".format(f_name))
    logger.info("    with expr [{} x {}] & corr [{} x {}] - {} overlapping.".format(
        expr.shape[0], expr.shape[1], conn.shape[0], conn.shape[1], len(overlapping_ids)
    ))

    # If, for any reason, we don't have enough samples to be reasonable, don't waste the resources.
    if len(overlapping_ids) < 4:
        logger.info("No point maximizing score of only {} samples. Returning empty DataFrame.".format(
            len(overlapping_ids)
        ))
        return pd.DataFrame(data={score_name: [], 'probe_id': []}, index=[])

    full_start = time.time()

    # Convert DataFrames to matrices, then vectors, for coming correlations
    conn = conn.loc[overlapping_ids, overlapping_ids]
    conn_mat = conn.as_matrix()
    conn_vec = conn_mat[np.tril_indices(conn_mat.shape[0], k=-1)]
    # Same for the distance matrix
    dist_mat = dist.loc[overlapping_ids, overlapping_ids].as_matrix()
    dist_vec = dist_mat[np.tril_indices(dist_mat.shape[0], k=-1)]

    # If we didn't get a real mask, make one that won't change anything.
    if mask is None:
        mask = np.ones(conn_vec.shape, dtype=bool)
    else:
        overlap_mask = np.array([well_id in overlapping_ids for well_id in expr.columns], dtype=bool)
        overlap_mat = overlap_mask[:, None] & overlap_mask
        overlap_vec = overlap_mat[np.tril_indices(overlap_mat.shape[0], k=-1)]
        logger.debug("    mask going from {}/{} to {}/{}.".format(
            sum(mask), len(mask), sum(mask[overlap_vec]), len(mask[overlap_vec])
        ))
        mask = mask[overlap_vec]

    # Pre-prune the expr DataFrame to avoid having to repeat it in the loop below.
    expr = expr.loc[:, overlapping_ids]
    # But there's no need to create a matrix and vector, that will be repeated later

    # Run the repeated correlations, saving each one keyed to the missing gene when it was generated.
    # The key is probe_id, allowing lookup of probe_name or gene_name information later.
    i = 0
    j = 0
    last_p = 0
    # peaked_already = False
    probes_removed = []
    re_orders = []
    re_ordered = True
    scores = {}
    # Initially, we need a first probe_id order, regardless of method
    if progress_file is not None and os.path.isfile(progress_file):
        probes_removed, scores, re_orders = retrieve_progress(progress_file)
        expr = expr.drop(labels=probes_removed, axis=0)
        i = len(probes_removed)  # May not be the same as the prior i, with j's included.
        last_p = probes_removed[-1]
        logger.info("  picked up progress file, starting at probe {}.".format(i))
    new_ranks = reorder_probes(
        expr, conn, dist, ascending=ascending, mask=mask, adjust=adjust, procs=cores, logger=logger
    )
    ranks = list(new_ranks.index)
    # Any fewer than 4 probes left will likely result in a quick  1.0 correlation and repeated re-ordering.
    # Any fewer than 3 will result in a 2-item matrix and failed correlations.
    # I don't know the threshold for worthwhile calculation, but it's certainly above 4
    while len(ranks) > 4:
        i += 1
        p = ranks.pop(-1)
        removed_probe = expr.loc[[p, ], :]
        probes_removed.append(p)
        re_orders.append(re_ordered)
        expr = expr.drop(labels=p, axis=0)
        expr_mat = np.corrcoef(expr, rowvar=False)
        expr_vec = expr_mat[np.tril_indices(n=expr_mat.shape[0], k=-1)]
        if adjust in ['linear', 'log']:
            score = get_beta(conn_vec, expr_vec, dist_vec, adjust)
        else:
            score = stats.pearsonr(expr_vec[mask], conn_vec[mask])[0]
        logger.debug("{:>5} of {:>5}. {}: {}".format(i - j, total_probes, p, score))
        print("{:>6} down, {} to go : {:0.0%}       ".format(
            i - j, len(expr.index), ((i - j) / total_probes)
        ), end='\r')

        # Evry means make for damn sure we knock out the worst gene each time, so re-order them every time.
        # Use the 'algorithms' lookup map to ensure any changes in spellings don't break us.
        if algorithms[algo] == 'evry':
            # re-order every time, no matter what
            new_ranks = reorder_probes(
                expr, conn, dist, ascending=ascending, mask=mask, adjust=adjust, procs=cores, logger=logger
            )
            ranks = list(new_ranks.index)
            re_ordered = True
        # If this correlation isn't the best so far, don't use it. Unless, of course, it's really the best we have left.
        elif algorithms[algo] == 'smrt' and len(scores) > 0 and last_p != p:  # and not peaked_already:
            # re-order the remaining probes only if we aren't getting better correlations thus far.
            if (ascending and score < max(scores.values())) or ((not ascending) and score > min(scores.values())):
                print("    re-ordering remaining {} probes. (i={}, j={}, p={})".format(len(ranks), i, j, p))
                # Replace the removed probe, include it in the re-ordering
                j += 1
                expr = pd.concat([expr, removed_probe], axis=0)
                probes_removed = probes_removed[:-1]
                re_orders = re_orders[:-1]
                new_ranks = reorder_probes(
                    expr, conn, dist, ascending=ascending, mask=mask, adjust=adjust, procs=cores, logger=logger
                )
                ranks = list(new_ranks.index)
                re_ordered = True
            else:
                re_ordered = False
        else:
            re_ordered = False
        if last_p == p:
            # peaked_already = True
            logger.info("    r({})=={:0.5f} < {:0.5f}, but we re-ordered probes & it's still lower.".format(
                p, score, max(scores.values())
            ))

        scores.update({i - j: score})
        last_p = p

        # Save an intermediate if we just re-ordered and have a progress_file to use.
        if re_orders[-1] and progress_file is not None:
            try:
                logger.debug("    {}s[{}] == {:0.3f}; ps[{}] == {}; re_orders[{}] == {}".format(
                    score_name,
                    len(scores) - 1, scores[i - j - 1],
                    len(probes_removed) - 1, probes_removed[-2],
                    len(re_orders) - 1, re_orders[-2]
                ))
            except KeyError:
                logger.debug("KEY {}s, ps, re_orders {} long: {}: {}, {}".format(
                    score_name, len(scores), p, re_orders[-2:], p in scores
                ))
            except IndexError:
                logger.debug("IDX {}s only {} long; ps {}; re_orders {}".format(
                    score_name, len(scores), len(probes_removed), len(re_orders)
                ))
            try:
                logger.debug("    {}s[{}] == {:0.3f}; ps[{}] == {}; re_orders[{}] == {}".format(
                    score_name,
                    len(scores), scores[i - j],
                    len(probes_removed), probes_removed[-1],
                    len(re_orders), re_orders[-1]
                ))
            except KeyError:
                logger.debug("KEY {}s, ps, re_orders {} long: {}: {}, {}".format(
                    score_name, len(scores), p, re_orders[-2:], p in scores
                ))
            except IndexError:
                logger.debug("IDX {}s only {} long; ps {}; re_orders {}".format(
                    score_name, len(scores), len(probes_removed), len(re_orders)
                ))
            save_progress(progress_file, scores, probes_removed, re_orders)

    elapsed = time.time() - full_start

    # Log results
    logger.info("{} ran {} {} scores ({} were drop-and-re-orders, OPENBLAS_NUM_THREADS={}) in {:0.2f}s.".format(
        f_name, i, score_name, j, BLAS_THREADS, elapsed
    ))

    # Return the list of correlations
    gene_list = pd.Series(scores, name=score_name)
    gene_list.index.name = 'rank'
    gene_list = pd.DataFrame(gene_list)
    logger.info("{}: gene_list is {} long. Inserting {}-len probe_id list, leaving {}-len.".format(
        f_name, len(gene_list.index), len(probes_removed), len(ranks)
    ))
    logger.info("end of probes_removed = {}, all of ranks = {}.".format(
        probes_removed[-2:], ranks
    ))

    gene_list['probe_id'] = probes_removed
    gene_list['re_ordered'] = re_orders

    # Finish the list with final 4 unscorable top probes, filled with 0.0 correlations.
    ii = max(gene_list.index)
    remainder = pd.DataFrame(
        data={
            score_name: [0.0, 0.0, 0.0, 0.0],
            'probe_id': ranks,
            're_ordered': [False, False, False, False]},
        index=[ii + 1, ii + 2, ii + 3, ii + 4]
    )

    return pd.concat([gene_list, remainder], axis=0)


def shuffled(df, cols=True, seed=0):
    """ Return a copy of the dataframe with either columns or rows shuffled.

    :param pandas.DataFrame df: the dataframe to copy and shuffle
    :param boolean cols: default to shuffle columns, if set to False, rows will shuffle instead.
    :param int seed: set numpy's random seed if desired
    :returns: A copy of the original (unaltered) DataFrame with either columns (default) or rows shuffled.
    """

    np.random.seed(seed)
    shuffled_df = df.copy(deep=True)
    if cols:
        shuffled_df.columns = np.random.permutation(df.columns)
    else:
        shuffled_df.index = np.random.permutation(df.index)
    return shuffled_df


def save_df_as_csv(path, out_file=None, sep=','):
    """ Convert a pickled DataFrame into a csv file for easier viewing.

    :param str path: The path to the picked DataFrame file
    :param str out_file: An alternative csv file path if just changing .df to .csv isn't desired.
    :param str sep: The character, a comma by defafult, to separate fields in the text file
    """

    if out_file is None:
        out_file = path[:path.rfind('.')] + '.csv'

    if os.path.isfile(path):
        with open(path, 'rb') as f:
            df = pickle.load(f)
        df.to_csv(out_file, sep=sep)
    else:
        print("{} is not a file.".format(path))


def save_df_as_tsv(path, out_file=None, sep='\t'):
    """ Convert a pickled DataFrame into a csv file for easier viewing.

    :param str path: The path to the picked DataFrame file
    :param str out_file: An alternative csv file path if just changing .df to .tsv isn't desired.
    :param str sep: The character, a tab by defafult, to separate fields in the text file
    """

    if out_file is None:
        out_file = path[:path.rfind('.')] + '.tsv'

    if os.path.isfile(path):
        with open(path, 'rb') as f:
            df = pickle.load(f)
        df.to_csv(out_file, sep=sep)
    else:
        print("{} is not a file.".format(path))


def top_probes(tsv_file, n=0):
    """ Return the top probes from the tsv_file specified.

    :param tsv_file: The file containing pushr output
    :param int n: How many probes would you like returned? Zero to get all genes past the peak.
    :return list: A list of probes still in the mix after maxxing or minning whack_a_probe.
    """

    if os.path.isfile(tsv_file):
        df = pd.read_csv(tsv_file, sep='\t')
        score_name = 'b' if 'b' in df.columns else 'r'
        if n == 0 and len(df.index) > 6:
            # The final value, [-1], is first, followed by each in sequence.
            # If the third value is greater than the first, this is a 'max' run
            if df[score_name].values[-3] > df[score_name].values[-1]:
                n = df[score_name][5:].idxmax() + 1
            else:
                n = df[score_name][5:].idxmin() + 1
        return list(df['probe_id'][:n])
    else:
        # Or if the file doesn't exist...
        return []
