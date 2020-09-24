import os
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.genmod.families import links
from scipy.stats import kendalltau
import time
import multiprocessing
import logging
import pickle
import filecmp
from brainsmash.mapgen.base import Base

from pygest.convenience import map_pid_to_eid, json_lookup, get_ranks_from_file


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


def file_is_equivalent(a, b, verbose):
    """ Compare files, returning one word describing their relationship. """

    def dataframes_differ(dfa, dfb):
        """ Do the actual comparison, if we have actual dataframe types to compare. """

        dfs_differ = False
        comment_list = []
        similarity_list = []

        if dfa.shape == dfb.shape:
            comment_list.append("Both dataframes are [{}x{}]".format(dfa.shape[0], dfa.shape[1]))
            if dfa.index.equals(dfb.index):
                similarity_list.append("indices match")
            else:
                dfs_differ = True
                similarity_list.append("{} indices overlap; {} differ.".format(
                    len(set(dfa.index).intersection(set(dfb.index))),
                    len(set(dfa.index).difference(set(dfb.index))) + len(set(dfb.index).difference(set(dfa.index))),
                ))

            if dfa.columns.equals(dfb.columns):
                similarity_list.append("columns match")
            else:
                simis = len(set(dfa.columns).intersection(set(dfb.columns)))
                diffs = len(set(dfa.columns).difference(set(dfb.columns))) + \
                        len(set(dfb.columns).difference(set(dfa.columns)))
                similarity_list.append("{} columns overlap; {} differ.".format(simis, diffs))
                if diffs > 0:
                    dfs_differ = True

            if 'probe_id' in dfa.columns and 'probe_id' in dfb.columns:
                if dfa['probe_id'].equals(dfb['probe_id']):
                    similarity_list.append("probe order matches")
                else:
                    similarity_list.append("different probe order")
                    dfs_differ = True

            if dfa.equals(dfb):
                similarity_list.append("All elements in the dataframe are identical.")
            else:
                similarity_list.append("Dataframe elements differ.")
                # But we're not this picky about equality. Slight differences in floats can wreck this.

        else:
            comment_list.append("The shapes differ: [{}x{}] vs [{}x{}]".format(
                dfa.shape[0], dfa.shape[1], dfb.shape[0], dfb.shape[1]
            ))
            dfs_differ = True

        return dfs_differ, similarity_list, comment_list

    def load_dataframe(path):
        """ Determine how dataframe is stored and load it appropriately. """
        if path.endswith(".df"):
            with open(path, "rb") as f:
                return pickle.load(f)
        elif path.endswith(".csv"):
            return pd.read_csv(path, index_col=0, header=0, sep=",")
        elif path.endswith(".tsv"):
            return pd.read_csv(path, index_col=0, header=0, sep="\t")
        else:
            return None

    if filecmp.cmp(a, b, shallow=False):
        verbose and print("{} identical: {} == {}".format(a[a.rfind("."):], a[a.rfind("/"):], b[b.rfind("/"):]))
        return True

    if a.endswith(".json") and b.endswith(".json"):
        a_version = json_lookup("pygest version", a)
        b_version = json_lookup("pygest version", b)
        a_datetime = json_lookup("began", a)
        b_datetime = json_lookup("began", b)
        if a_version == b_version and a_datetime == b_datetime:
            verbose and print("json equivalent: {} on {} -v- {} on {}".format(
                a_version, a_datetime, b_version, b_datetime
            ))
            return True
        else:
            verbose and print("json different: {} on {} -v- {} on {}".format(
                a_version, a_datetime, b_version, b_datetime
            ))
            return False
    elif a[-3:] in ['csv', 'tsv', '.df'] and b[-3:] in ['csv', 'tsv', '.df']:
        files_differ, similarities, comments = dataframes_differ(load_dataframe(a), load_dataframe(b))
        if verbose:
            if files_differ:
                print("different:")
            else:
                print("equivalent:")
            for comment in comments:
                print("  " + comment)
            print("  " + "; ".join(similarities))
            return not files_differ
    else:
        # Log files aren't considered under any other criteria than the first filecmp check.
        return False


def one_mask(df, mask_type, sample_type, data, logger=None):
    """ return a vector of booleans from the lower triangle of a matching-matrix based on 'mask_type'

    :param df: pandas.DataFrame with samples as columns
    :param str mask_type: A list of strings to specify matching masks, or a minimum distance to mask out
    :param str sample_type: Samples can be 'wellid' or 'parcelid'
    :param data: PyGEST.Data object with access to AHBA data
    :param logger: A logger object to receive debug information
    :return: Boolean 1-D vector to remove items (False values in mask) from any sample x sample triangle vector
    """

    def handle_log(maybe_logger, severity, message):
        if maybe_logger is None:
            print(message)
        else:
            if severity == "info":
                maybe_logger.info(message)
            if severity == "warn":
                maybe_logger.warn(message)

    # If mask is a number, use it as a distance filter
    try:
        # Too-short values to mask out are False, keepers are True.
        min_dist = float(mask_type)
        distance_vector = data.distance_vector(df.columns, sample_type=sample_type)
        if len(distance_vector) != (len(df.columns) * (len(df.columns) - 1)) / 2:
            handle_log(logger, "warn", "        MISMATCH in expr and dist!!! Some sample IDs probably not found.")
        mask_vector = np.array(distance_vector > min_dist, dtype=bool)
        handle_log(logger, "info", "        masking out {:,} of {:,} edges closer than {}mm apart.".format(
            np.count_nonzero(np.invert(mask_vector)), len(mask_vector), min_dist
        ))
        handle_log(logger, "info", "        mean dist of masked edges  : {:0.2f} [{:0.2f} to {:0.2f}].".format(
            np.mean(distance_vector[~mask_vector]),
            np.min(distance_vector[~mask_vector]),
            np.max(distance_vector[~mask_vector]),
        ))
        handle_log(logger, "info", "        mean dist of unmasked edges: {:0.2f} [{:0.2f} to {:0.2f}].".format(
            np.mean(distance_vector[mask_vector]),
            np.min(distance_vector[mask_vector]),
            np.max(distance_vector[mask_vector]),
        ))
        return mask_vector
    except TypeError:
        pass
    except ValueError:
        pass

    # Mask is not a number, see if it's a pickled dataframe
    if os.path.isfile(mask_type):
        with open(mask_type, 'rb') as f:
            mask_df = pickle.load(f)
        if isinstance(mask_df, pd.DataFrame):
            # Note what we started with so we can report after we tweak the dataframe.
            # Too-variant values to mask out are False, keepers are True.
            orig_vector = mask_df.values[np.tril_indices(n=mask_df.shape[0], k=-1)]
            orig_falses = np.count_nonzero(~orig_vector)
            orig_length = len(orig_vector)
            handle_log(logger, "info", "Found {} containing {:,} x {:,} mask".format(
                mask_type, mask_df.shape[0], mask_df.shape[1]
            ))
            handle_log(logger, "info", "    generating {:,}-len vector with {:,} False values to mask.".format(
                orig_length, orig_falses
            ))

            # We can only use well_ids found in BOTH df and our new mask, make shapes match.
            unmasked_ids = [well_id for well_id in df.columns if well_id not in mask_df.columns]
            usable_ids = [well_id for well_id in df.columns if well_id in mask_df.columns]
            usable_df = mask_df.reindex(index=usable_ids, columns=usable_ids)
            usable_vector = usable_df.values[np.tril_indices(n=len(usable_ids), k=-1)]
            usable_falses = np.count_nonzero(~usable_vector)
            usable_length = len(usable_vector)
            handle_log(logger, "info", "    {:,} well_ids not found in the mask; padding with Falses.".format(
                len(unmasked_ids)
            ))
            pad_rows = pd.DataFrame(np.zeros((len(unmasked_ids), len(mask_df.columns)), dtype=bool),
                                    columns=mask_df.columns, index=unmasked_ids)
            mask_df = pd.concat([mask_df, pad_rows], axis=0)
            pad_cols = pd.DataFrame(np.zeros((len(mask_df.index), len(unmasked_ids)), dtype=bool),
                                    columns=unmasked_ids, index=mask_df.index)
            mask_df = pd.concat([mask_df, pad_cols], axis=1)
            mask_vector = mask_df.values[np.tril_indices(n=mask_df.shape[0], k=-1)]
            mask_falses = np.count_nonzero(~mask_vector)
            mask_trues = np.count_nonzero(mask_vector)
            handle_log(logger, "info", "    padded mask matrix out to {:,} x {:,}".format(
                mask_df.shape[0], mask_df.shape[1]
            ))
            handle_log(logger, "info", "      with {:,} True, {:,} False, {:,} NaNs in triangle.".format(
                mask_trues, mask_falses, np.count_nonzero(np.isnan(mask_vector))
            ))

            shaped_mask_df = mask_df.reindex(index=df.columns, columns=df.columns)
            shaped_vector = shaped_mask_df.values[np.tril_indices(n=len(df.columns), k=-1)]
            handle_log(logger, "info", "    masking out {:,} (orig {:,}, {:,} usable) hi-var".format(
                np.count_nonzero(~shaped_vector), orig_falses, usable_falses,
            ))
            handle_log(logger, "info", "      of {:,} (orig {:,}, {:,} usable) edges.".format(
                len(shaped_vector), orig_length, usable_length
            ))
            return shaped_vector
        else:
            handle_log(logger, "warn", "{} is a file, but not a pickled dataframe. Skipping mask.".format(mask_type))
            do_nothing_mask = np.ones((len(df.columns), len(df.columns)), dtype=bool)
            return do_nothing_mask[np.tril_indices(n=len(df.columns), k=-1)]

    # Mask is not a number, so treat it as a matching filter
    if mask_type[:4] == 'none':
        items = list(df.columns)
    elif mask_type[:4] == 'fine':
        items = data.samples(samples=df.columns)['fine_name']
    elif mask_type[:6] == 'coarse':
        items = data.samples(samples=df.columns)['coarse_name']
    else:
        items = data.samples(samples=df.columns)['structure_name']
    mask_array = np.ndarray((len(items), len(items)), dtype=bool)

    # There is, potentially, a nice vectorized way to mark matching values as True, but I can't find it.
    # So, looping works and is easy to read, although it might cost us a few extra ms.
    for i, y in enumerate(items):
        for j, x in enumerate(items):
            # Generate one edge of the match matrix
            mask_array[i][j] = True if mask_type == 'none' else (x != y)
    mask_vector = mask_array[np.tril_indices(n=mask_array.shape[0], k=-1)]

    handle_log(logger, "info", "        masking out {:,} of {:,} '{}' edges.".format(
        sum(np.invert(mask_vector)), len(mask_vector), mask_type
    ))

    # if len(mask_vector) == 0:
    #     mask_vector = np.ones(int(len(df.columns) * (len(df.columns) - 1) / 2), dtype=bool)

    return mask_vector


def create_edge_shuffle_map(dist_vec, edge_tuple, logger):
    """
    One randomized null comparison creates distance bins, and shuffles edges within each bin. This function creates
    the map so that each shuffling swaps the same edges each time.

    We currently hard-code the bin boundaries within this function. If desired, we can rewrite this to allow them
    to be defined elsewhere and passed in.

    :param dist_vec: A vector of distances between samples, from the lower triangle of a distance matrix
    :param edge_tuple: The seed used to generate the initial random shuffling order, and the bin size
    :param logger: A logger object to receive debug information
    :return: A python dictionary mapping the original edge order to a new shuffled edge order
    """
    if edge_tuple[0] is None:
        return None
    else:
        np.random.seed(edge_tuple[0])
        shuffle_map = {}
        # A dataframe is an easy way to pair an index with the actual well_id values in column 0
        # bin distances (min is 1.0, max is around 168.0)
        edge_df = pd.DataFrame(dist_vec)

        # Generate the ranges where shuffling can only occur within each bin.
        # Default is logarithmic
        bin_boundaries = [(0, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 128), (128, 999)]
        if edge_tuple[1] > 0:
            # Or fixed-width bins can be specified.
            bin_boundaries = [(x, x+edge_tuple[1]) for x in range(0, int(max(dist_vec) + 1), edge_tuple[1])]

        for bin_limits in bin_boundaries:
            bin_values = edge_df[(edge_df[0] > bin_limits[0]) & (edge_df[0] <= bin_limits[1])]
            logger.debug("    {} / {:,} edges fall between {} and {}, shuffled".format(
                len(bin_values), len(edge_df), bin_limits[0], bin_limits[1]
            ))
            # Map shuffled indices as values onto the original indices as keys
            bin_map = dict(zip(list(bin_values.index), np.random.permutation(list(bin_values.index))))
            shuffle_map.update(bin_map)

        logger.debug("    shuffle map has {:,} edges".format(len(shuffle_map)))

    return shuffle_map


def correlate_and_vectorize_expression(expr, shuffle_map=None):
    """
    Create an expression similarity matrix and lower triangle vector from the expr dataframe.

    This is one of two workhorses of the entire system, called thousands of times for each optimization

    :param expr: A dataframe holding gene expression level values
    :param shuffle_map: If necessary, a pre-defined map (dict) to re-arrange edges for edge-shuffling
    :return: An expression similarity vector
    """

    expr_mat = np.corrcoef(expr.values, rowvar=False)
    expr_vec = expr_mat[np.tril_indices_from(expr_mat, k=-1)]
    if shuffle_map is not None:
        expr_vec = np.array([expr_vec[shuffle_map[i]] for (i, x) in enumerate(list(expr_vec))])
    return expr_vec


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
        # Possibly critically important:
        # If the connectivity matrix only has lower-triangle values, its index or columns MUST
        # come first in the list comprehension below. Changing the order of wellids can dilute
        # connectivity data with a bunch of zeros from the upper triangle.
        overlap = [well_id for well_id in conn.index if well_id in expr.columns]
        expr_mat = np.corrcoef(expr.loc[:, overlap], rowvar=False)
        conn_mat = conn.loc[overlap, overlap].values
        # Both matrices MUST now both be identically-shaped square matrices.
        logger.debug("CEC: expression & connectivity filtered down to {} matrices".format(expr_mat.shape))
    elif isinstance(expr, pd.DataFrame):
        if expr.shape[1] == conn.shape[0] == conn.shape[1]:
            expr_mat = np.corrcoef(expr, rowvar=False)
            conn_mat = conn
        else:
            raise TypeError("If expr is a DataFrame, conn must be a DataFrame or expr-column-matched array.")
    elif isinstance(conn, pd.DataFrame):
        if conn.shape[1] == expr.shape[0] == expr.shape[1]:
            expr_mat = expr
            conn_mat = conn.values
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


def make_similarity(df):
    """
    Convert any square numeric matrix to a similarity matrix

    :param df: Original matrix, usually connectivity
    :return: Similarity matrix, usually connectivity similarity
    """

    conn_mat = df.to_numpy(dtype=np.float64)
    if conn_mat.shape[0] == conn_mat.shape[1]:
        n = conn_mat.shape[0]
    else:
        return None

    # Generate a zero-filled n x n matrix, then populate each edge with a similarity value between nodes.
    similarity_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            # Only bother calculating the lower left triangle (where i>j). Then copy that value across the diagonal.
            if i >= j:
                # Correlate each row by each column, but with identity edges filtered out.
                exclusion_filter = [(x != i) and (x != j) for x in range(n)]
                vi = conn_mat[:, i][exclusion_filter]
                vj = conn_mat[:, j][exclusion_filter]
                # TODO: See if I can just generate a single matrix of post-filtered vectors (vi, vj) then do
                #       a single correlation step instead of ((n * (n-1)) / 2) steps
                similarity_mat[i, j] = np.corrcoef(vi, vj)[0, 1]
                similarity_mat[j, i] = similarity_mat[i, j]

    return pd.DataFrame(similarity_mat, columns=df.columns, index=df.columns)


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
    if adj is None:
        exog = sm.add_constant(pd.DataFrame({'x': x}))
    else:
        exog = sm.add_constant(pd.DataFrame({'x': x, 'adj': adj}))

    result = sm.GLM(endog, exog, family=sm.families.Gaussian(link)).fit()
    # print(result.summary())
    return result.params['x']


def mask_bad_values(vec, desc, logger):
    """ Return a mask with False aligning with any bad values.

    :param vec: A vector to be checked for bad values.
    :param desc: A string describing the vector.
    :param logger: where to send messages.
    """
    logger.info(" : {} vector has {:,} Infs and {:,} NaNs, out of {:,}. Masking them out.".format(
        desc, np.count_nonzero(np.isinf(vec)), np.count_nonzero(np.isnan(vec)), len(vec)
    ))
    return ~(np.isinf(vec) | np.isnan(vec))


def combine_masks(explicit_mask, value_masks, distance_vector, logger):
    """ Return a combined mask from the masks list.

    :param explicit_mask: Usually a distance mask, pre-defined to mask out values.
    :param value_masks: A list of masks, generated to remove bad values.
    :param distance_vector: A distance vector for reporting before- and after-mask average distances.
    :param logger: where to send messages.
    """

    value_mask = None
    for mask in value_masks:
        if value_mask is None:
            value_mask = mask
        else:
            value_mask = value_mask * mask

    logger.info(" : {:,} explicitly masked edges, {:,} masked for bad values.".format(
        np.count_nonzero(np.invert(explicit_mask)), np.count_nonzero(np.invert(value_mask))
    ))
    final_mask = explicit_mask & value_mask
    logger.info(" : Using {:,}, removing {:,} total edges.".format(
        np.count_nonzero(final_mask), np.count_nonzero(~final_mask)
    ))
    logger.info("     mean distance {:0.2f} (of {:,} finite values being masked)".format(
        np.mean(distance_vector[~explicit_mask & value_mask]), np.count_nonzero(~explicit_mask & value_mask)
    ))
    logger.info("     mean distance {:0.2f} (of {:,} finite values being kept)".format(
        np.mean(distance_vector[final_mask]), np.count_nonzero(final_mask)
    ))

    return final_mask


def reorder_probes(expr, conn_vec, dist_vec=None, shuffle_map=None, ascending=True,
                   mask=None, adjust='none', include_full=False, procs=0, logger=None):
    """ For each probe, knock it out and re-calculate relation between versions of expr's correlation matrix and conn.
        If adjustments are specified, a GLM will be used. If not, a Pearson correlation will.

        This function gets called a lot, and is optimized for speed over usability. Dataframes
        must be the same size or this will not correct it for the caller.

    :param pd.DataFrame expr: gene expression DataFrame [probes x samples]
    :param np.array conn_vec: functional connectivity DataFrame [samples x samples]
    :param np.array dist_vec: distance triangle vector
    :param shuffle_map: shuffle map dict for consistent shuffling of each new expr vector
    :param boolean ascending: True to order with positive impact probes first, False to reverse order
    :param mask: Mask out True edges
    :param str adjust: Include distance in a model
    :param boolean include_full: True to include the full correlation as probe_id==0, default is False
    :param int procs: How many processes to spread out over
    :param logging.Logger logger: Any logging output can be handled by the caller's logger if desired.
    :return: dictionary with probe_id keys and Pearson correlation values for each removed probe
    """

    # There are several ways to access the function name. I think all are ugly, so this works.
    f_name = 'reorder_probes'
    from pygest import workers

    # Check propriety of arguments
    if dist_vec is None:
        dist_vec = np.ndarray((0, 0))
    if logger is None:
        logger = logging.getLogger('pygest')
    # Assume expr, conn, and dist are compatible. We aren't calculating overlap or fixing it.
    if not isinstance(expr, pd.DataFrame):
        raise TypeError("{} expects expr to be a pandas.DataFrame, not {}.".format(f_name, type(expr)))
    if not isinstance(conn_vec, np.ndarray):
        raise TypeError("{} expects conn to be a numpy array, not {}.".format(f_name, type(conn_vec)))
    if not isinstance(dist_vec, np.ndarray):
        raise TypeError("{} expects dist to be a numpy array, not {}.".format(f_name, type(dist_vec)))

    full_start = time.time()

    # Pre-convert the expr DataFrame to avoid having to repeat it in the loop below.
    expr_vec = correlate_and_vectorize_expression(expr, shuffle_map)

    # If we didn't get a real mask, make one that won't change anything.
    if mask is None:
        mask = np.ones(conn_vec.shape, dtype=bool)
        # Run the repeated correlations, saving each one keyed to the missing gene when it was generated.
        # The key is probe_id, allowing lookup of probe_name or gene_name information later.

    # No matter the mask provided, or not, we need to remove NaNs and Infs or we'll error out when we hit them.
    valid_expr_mask = mask_bad_values(expr_vec, "Expression", logger)
    valid_dist_mask = mask_bad_values(dist_vec, "Distance", logger)
    valid_conn_mask = mask_bad_values(conn_vec, "Comparator", logger)
    mask = combine_masks(mask, [valid_expr_mask, valid_dist_mask, valid_conn_mask, ], dist_vec, logger)

    if adjust in ['linear', 'log']:
        score_name = 'b'
        score_method = 'glm'
        scores = {0: get_beta(conn_vec[mask], expr_vec[mask], dist_vec[mask], adjust)}
    elif adjust in ['slope']:
        score_name = 'm'
        score_method = 'glm'
        scores = {0: get_beta(conn_vec[mask], expr_vec[mask], None, adjust)}
    else:
        score_name = 'r'
        score_method = 'pearson r'
        scores = {0: stats.pearsonr(expr_vec[mask], conn_vec[mask])[0]}

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
            expr_vec = correlate_and_vectorize_expression(expr.drop(labels=p, axis=0), shuffle_map)
            if adjust in ['linear', 'log']:
                scores[p] = get_beta(conn_vec[mask], expr_vec[mask], dist_vec[mask], adjust)
            elif adjust in ['slope']:
                scores[p] = get_beta(conn_vec[mask], expr_vec[mask], None, adjust)
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

        if adjust in ['linear', 'log', 'slope']:
            # Create a worker process to use GLMs on each core/proc available.
            modelers = []
            for i in range(procs):
                # Let each process have its own copy of expr, rather than try to copy it with each task later
                # This results in a handful of copies rather than tens of thousands
                if adjust in ['linear', 'log']:
                    modelers.append(workers.LinearModeler(queue, expr, conn_vec, dist_vec, mask, shuffle_map))
                else:
                    modelers.append(workers.LinearModeler(queue, expr, conn_vec, None, mask, shuffle_map))
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
                correlators.append(workers.Correlator(queue, expr, conn_vec, mask, shuffle_map))
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


def report_backup(score_name, records, logger):
    """ Log the last two records to ensure they're written correctly after backing up. """

    n = len(records)

    def log_one_rec(recs, idx):
        try:
            this_rec = recs[idx]
            logger.debug("    rec[{}]: {}: {:0.6f}; probe_id: {:<8}; re_order: {}".format(
                this_rec['seq'], score_name, this_rec[score_name], this_rec['probe_id'], this_rec['re_ordered'],
            ))
        except KeyError:
            logger.debug("ERR:KEY {}s only {} long, tried key {}".format(score_name, n, idx))
        except IndexError:
            logger.debug("ERR:IDX {}s only {} long, tried idx {}".format(score_name, n, idx))

    log_one_rec(records, -2)
    log_one_rec(records, -1)


def push_score(expr, conn, dist,
               algo=algorithms['smrt'], ascending=True, dump_intermediates=None,
               mask=None, adjust='none', edge_tuple=(None, None), progress_file=None, cores=0, logger=None):
    """ Remove each probe (additionally) from the original expression matrix, in order
        of least positive impact. After each removal, re-correlate with connectivity.

    :param pd.DataFrame expr: gene expression DataFrame [probes x samples]
    :param pd.DataFrame conn: functional connectivity DataFrame [samples x samples]
    :param pd.DataFrame dist: distance DataFrame [samples x samples]
    :param str algo: 'once' orders probes only once, and sticks to that order throughout sequential probe removal
                     'smrt' orders probes 'once', then re-orders it each time the correlation drops
                     'evry' re-runs whack-a-gene every single iteration
    :param bool ascending: True to maximize positive correlation, False to pursue most negative correlation
    :param str dump_intermediates: A path for saving out intermediate edge vertices for later analysis
    :param np.array mask: A boolean mask to filter out unwanted edges in triangle vectors
    :param edge_tuple: A PRNG seed to control replicability of null distributions and a bin size
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

    if adjust in ['linear', 'log', 'slope']:
        score_name = 'b'
    else:
        score_name = 'r'

    # The distance-mask is already made, aligned with existing expression. DO NOT change the column orders.
    if np.sum(expr.columns != dist.columns) + np.sum(expr.columns != dist.columns) > 0:
        logger.info("Expression, Comparator, and Distance samples do not match completely. But they MUST.")
        logger.info("Returning empty DataFrame.")
        return pd.DataFrame(data={score_name: [], 'probe_id': []}, index=[])

    # If, for any reason, we don't have enough samples to be reasonable, don't waste the resources.
    if len(expr.columns) < 4:
        logger.info("No point maximizing score of only {} samples. Returning empty DataFrame.".format(
            len(expr.columns)
        ))
        return pd.DataFrame(data={score_name: [], 'probe_id': []}, index=[])

    full_start = time.time()

    # The expr_mat and expr_vec will be re-correlated later, but are needed here to mask out Infs and NaNs.
    expr_mat = np.corrcoef(expr, rowvar=False)

    # Convert dataframes (with duplicated upper and lower triangles) to efficient one-dimensional vector each.
    conn_vec = conn.values[np.tril_indices(conn.shape[0], k=-1)]
    dist_vec = dist.values[np.tril_indices(dist.shape[0], k=-1)]
    expr_vec = expr_mat[np.tril_indices(expr_mat.shape[0], k=-1)]

    # If we didn't get a real mask, make one that won't change anything.
    if mask is None or len(mask) == 0:
        mask = np.ones(conn_vec.shape, dtype=bool)

    # No matter the mask provided, or not, we need to remove NaNs and Infs or we'll error out when we hit them.
    valid_expr_mask = mask_bad_values(expr_vec, "Expression", logger)
    valid_dist_mask = mask_bad_values(dist_vec, "Distance", logger)
    valid_conn_mask = mask_bad_values(conn_vec, "Comparator", logger)

    mask = combine_masks(mask, [valid_expr_mask, valid_dist_mask, valid_conn_mask, ], dist_vec, logger)

    # Generate a shuffle that can be used to identically shuffle new expr edges each iteration
    shuffle_map = create_edge_shuffle_map(dist_vec, edge_tuple, logger)
    if shuffle_map is not None:
        pickle.dump(shuffle_map, open(progress_file.replace(".partial.", ".shuffle_map."), "wb"))

    logger.info("    with expr [{} x {}] & corr [{} x {}] & dist [{} x {}] - {}-len mask.".format(
        expr.shape[0], expr.shape[1],
        conn.shape[0], conn.shape[1],
        dist.shape[0], dist.shape[1], len(mask)
    ))

    # Run the repeated correlations, saving each one keyed to the missing gene when it was generated.
    # The key is probe_id, allowing lookup of probe_name or gene_name information later.
    i = 0
    j = 0
    last_p = 0
    alt_records = []
    hi_score = -1.0
    lo_score = 1.0

    # Pick up where we left off, if we are re-starting an interrupted optimization.
    if progress_file is not None and os.path.isfile(progress_file):
        i, j, alt_records = pickle.load(open(progress_file, 'rb'))
        expr = expr.drop(labels=[d['probe_id'] for d in alt_records], axis=0)
        last_p = alt_records[-1]['probe_id']
        logger.info("  picked up progress file, starting at probe {}.".format(i - j))
    else:
        logger.info("Mantel correlation for {:,} genes is r = {:0.5f}".format(
            expr.shape[0], stats.pearsonr(expr_vec[mask], conn_vec[mask])[0]
        ))

    # Initial probe-re-order, whether from nothing, or from prior loaded file.
    ranks = list(expr.index)  # Not yet ranked, but gotta start with something. Rank it later.
    do_reranking = True

    finished = False
    while not finished:

        # Re-rank probes if needed.
        if do_reranking:
            new_ranks = reorder_probes(
                expr, conn_vec, dist_vec, shuffle_map,
                ascending=ascending, mask=mask, adjust=adjust, procs=cores, logger=logger
            )
            ranks = list(new_ranks.index)
            do_reranking = False
            re_ordered = True
        else:
            re_ordered = False

        # Any fewer than 4 probes left will likely result in a quick  1.0 correlation and repeated re-ordering.
        # Any fewer than 3 will result in a 2-item matrix and failed correlations.
        # I don't know the threshold for worthwhile calculation, but it's certainly above 4
        if len(ranks) <= 5:
            finished = True

        # Set counters and context, per-iteration.
        keep_record = True  # Each iteration, expect to keep the record created (unless explicitly set to False)
        i += 1  # Count total iterations; combine with j (re-ranks) to determine which record is actually current.
        p = ranks.pop(-1)
        removed_probe = expr.loc[[p, ], :]
        current_record = {'seq': i - j, 'probe_id': p, score_name: 0.000, 're_ordered': re_ordered, }
        expr = expr.drop(labels=p, axis=0)
        expr_vec = correlate_and_vectorize_expression(expr, shuffle_map)

        # In rare cases, we might want to examine the process rather than just the end state.
        if dump_intermediates is not None:
            if os.path.isdir(dump_intermediates):
                pd.DataFrame({'expr': expr_vec, 'conn': conn_vec}).to_pickle(
                    os.path.join(dump_intermediates, 'probe-{:0>6}.df').format(len(ranks))
                )

        # Calculate the new score.
        try:
            if adjust in ['linear', 'log']:
                score = get_beta(conn_vec[mask], expr_vec[mask], dist_vec[mask], adjust)
            elif adjust in ['slope']:
                score = get_beta(conn_vec[mask], expr_vec[mask], None, adjust)
            else:
                score = stats.pearsonr(expr_vec[mask], conn_vec[mask])[0]
        except ValueError:
            score = 0.00
            finished = True

        current_record[score_name] = score
        hi_score = max(hi_score, score)
        lo_score = min(lo_score, score)
        logger.debug("{:>5} of {:>5}. {}: {}  -  i={}, j={}".format(i - j, total_probes, p, score, i, j))
        print("{:>6} down, {} to go : {:0.0%}       ".format(
            i - j, len(expr.index), ((i - j) / total_probes)
        ), end='\r')

        # Determine whether to re-rank before calculating the next one.
        if algorithms[algo] == 'evry':
            do_reranking = True
        # If this correlation isn't the best so far, don't use it. Unless, of course, it's really the best we have left.
        elif algorithms[algo] == 'smrt':
            plateau = (ascending and score < hi_score) or ((not ascending) and score > lo_score)
            if i > 1 and plateau:  # and not peaked_already:
                if last_p != p:
                    if not finished:
                        # When a plateau is detected, ignore the current failed record. Back up, re-rank, and do better.
                        logger.info("    re-ordering remaining {} probes. (i={},j={},p={})".format(len(ranks), i, j, p))
                        # Replace the removed probe, include it in the re-ordering
                        logger.info("    removing weak r={:0.6f} at {} from dropping probe {}".format(score, i - j, p))
                        expr = pd.concat([expr, removed_probe], axis=0)
                        do_reranking = True
                        j += 1  # Count deleted records to subtract from overall iteration counter.
                        # Do not append the current record. We will try to do better. Unless we're on the last one.
                        keep_record = False
                else:
                    # We've plateaued, and we got the same best probe anyway. We've peaked. Continue along.
                    logger.info("    r({})=={:0.6f} < {:0.6f}, but we re-ordered probes & it's still lower.".format(
                        p, score, hi_score
                    ))
                    logger.info("    ** Peaked!  i={}, j={}".format(i, j, ))
                    score_existing = "n/a"
                    if (i - j) in [d['seq'] for d in alt_records]:
                        score_existing = "{:0.6f}".format(alt_records[i - j][score_name])
                    logger.info("    this will write score[{} of {}] of {} with {:0.6f}.".format(
                        i - j, len(alt_records), score_existing, score,
                    ))

        if keep_record:
            alt_records.append(current_record)

        # Save an intermediate if we just re-ordered and have a progress_file to use.
        if re_ordered and progress_file is not None:
            if i > 1:
                report_backup(score_name, alt_records, logger)
            pickle.dump((i, j, alt_records), open(progress_file, "wb"), protocol=pickle.HIGHEST_PROTOCOL)

        last_p = p  # Avoid re-ranking forever after the peak; save and continue if the best probe still drops the score

    elapsed = time.time() - full_start

    # Log results
    logger.info("{} ran {} {} scores ({} were drop-and-re-orders, OPENBLAS_NUM_THREADS={}) in {:0.2f}s.".format(
        f_name, i, score_name, j, BLAS_THREADS, elapsed
    ))

    # Return the list of correlations
    alt_gene_list = pd.DataFrame(data=alt_records)
    logger.info("{}: gene_list is {:,}-long, leaving {}-len.".format(
        f_name, len(alt_gene_list), len(ranks)
    ))
    logger.info("end of probes_removed = {}, all of ranks = {}.".format(
        [d['probe_id'] for d in alt_records[-2:]], ranks
    ))

    # Finish the list with final few un-scorable top probes, filled with 0.0 correlations.
    next_seq = max([d['seq'] for d in alt_records]) + 1
    remainder = pd.DataFrame(
        data={
            'seq': list(range(next_seq, next_seq + len(ranks))),
            score_name: [0.0, ] * len(ranks),
            'probe_id': ranks,
            're_ordered': [False, ] * len(ranks)
        },
    )

    return pd.concat([alt_gene_list, remainder], sort=False, axis=0).set_index('seq')


def brainsmash_shuffled(expr_df, dist_df, seed=None, logger=None):
    """ Return a copy of the expr_df DataFrame with edges shuffled, then adjusted to original spatial autocorrelation.

    :param pandas.DataFrame expr_df: the DataFrame to copy and shuffle
    :param pandas.DataFrame dist_df: the distance DataFrame
    :param int seed: set the randomizer seed for reproducibility
    :param logger: The stream to send log information
    :returns: A copy of the expr_df DataFrame, shuffled and adjusted to preserve original spatial autocorrelation
    """

    if logger:
        logger.info("Starting brainsmash shuffle of [{} x {}] expression dataframe.".format(*expr_df.shape))

    shared_ids = [wid for wid in expr_df.columns if wid in dist_df.columns]
    adj_ge_values = {}
    for i, probe in enumerate(expr_df.index):
        base = Base(x=expr_df.loc[probe, shared_ids].values, D=dist_df.loc[shared_ids, shared_ids].values, seed=seed)
        adj_ge_values[probe] = base(n=1)

    if logger:
        logger.info("Finished brainsmash shuffle, returning [{} x {}] surrogate expression dataframe.".format(
            len(expr_df.index), len(shared_ids)
        ))

    return pd.DataFrame.from_dict(adj_ge_values, orient='index', columns=shared_ids)


def cols_shuffled(expr_df, dist_df=None, algo="agno", seed=0):
    """ Return a copy of the expr_df DataFrame with columns shuffled randomly.

    :param pandas.DataFrame expr_df: the DataFrame to copy and shuffle
    :param pandas.DataFrame dist_df: the distance DataFrame to inform us about distances between columns
    :param str algo: Agnostic to distance ('agno') or distance aware ('dist')?
    :param int seed: set numpy's random seed if desired
    :returns: A copy of the expr_df DataFrame with columns shuffled.
    """

    shuffled_df = expr_df.copy(deep=True)
    np.random.seed(seed)

    if algo == "agno":
        shuffled_df.columns = np.random.permutation(expr_df.columns)
    elif algo == "dist":
        # Make a distance-similarity matrix, allowing us to characterize one well_id's distance-similarity to another.
        diss = pd.DataFrame(data=np.corrcoef(dist_df.values), columns=dist_df.columns, index=dist_df.index)

        # Old and new well_id indices
        available_ids = list(expr_df.columns)
        shuffled_well_ids = []

        # For each well_id in the original list, replace it with another one as distance-similar as possible.
        for well_id in list(expr_df.columns):
            # Do we want to avoid same tissue-class?
            # This algo allows for keeping the same well_id and doesn't even look at tissue-class.
            # sort the distance-similarity by THIS well_id's column, but use corresponding index of well_ids
            candidates = diss.sort_values(by=well_id, ascending=False).index
            candidates = [x for x in candidates if x in available_ids]
            if len(candidates) == 1:
                candidate = candidates[0]
            elif len(candidates) < 20:
                candidate = np.random.permutation(candidates)[0]
            else:
                n_candidates = min(20, int(len(candidates) / 5.0))
                candidate = np.random.permutation(candidates[:n_candidates])[0]

            # We have our winner, save it to our new list and remove it from what's available.
            shuffled_well_ids.append(candidate)
            available_ids.remove(candidate)

        shuffled_df.columns = shuffled_well_ids
    else:
        shuffled_df = pd.DataFrame()

    # Column labels have been shuffled; return a dataframe with identically ordered labels and moved data.
    return shuffled_df.loc[:, expr_df.columns], dict(zip(expr_df.columns, shuffled_df.columns))


def run_results(tsv_file, top=None):
    """ Read through the tsv file provided and return a dictionary with relevant results.

    :param tsv_file: A tsv file containing ordered probe information from pygest
    :param top: How to determine top probes: None thresholds at top score. <1 takes a percentage. >1 takes a quantity
    :return: a dictionary containing summarized results.
    """

    results = {}
    n = 0

    # print("reading results from {}".format(tsv_file))
    df = pd.read_csv(tsv_file, sep='\t', index_col=0).sort_index()
    # Most results are correlations with an 'r' column. But some are GLMs with a 'b' column instead.
    score_name = 'b' if 'b' in df.columns else 'r'

    results['initial'] = df[score_name][1]
    df_meat = df.loc[1:len(df) - 5, :]
    if len(df) > 6:
        if df.loc[3, score_name] > df.loc[1, score_name]:
            # The third value is greater than the first, so this is a 'max' run.
            # The final four values are all reported as 0.00, but are the strongest probes.
            results['tgt'] = 'max'
            n = len(df) - df_meat[score_name].idxmax() + 1  # +1 to ensure the probe at max is included in the list
            results['best'] = df_meat[score_name].max()
        else:
            # The third value is not greater than the first, so this is a 'min' run.
            # The final four values are all reported as 0.00, but are the strongest probes.
            results['tgt'] = 'min'
            n = len(df) - df_meat[score_name].idxmin() + 1  # +1 to ensure the probe at min is included in the list
            results['best'] = df_meat[score_name].min()

    # The results are in reverse order of their 'discovery' so we need to invert this to report a high peak.
    results['peak'] = int(df_meat[score_name].idxmax())

    # If a top threshold is specified, override the discovered peak, n. But don't change results['peak']
    try:
        if 0.0 < float(top) < 1.0:
            n = int(len(df) * top)
        elif 1 <= int(top) <= len(df):
            n = int(top)
    except TypeError:
        # No problem, None is the default and will take us here rather than over-write the top results.
        pass

    results['score_type'] = score_name
    results['top_probes'] = list(df['probe_id'])[-n:]
    results['n'] = len(df)

    return results


def kendall_tau(results, truncate_on_mismatch=True):
    """ Read each file in a list and return the kendall tau between each.

    :param results: a list of tsv-formatted result files
    :param truncate_on_mismatch: Normally, truncate results to match before comparison, if False, lists must match.
    :returns: a float value representing the average kendall tau rank-correlation between the files
    """

    m = kendall_tau_matrix(results, truncate_on_mismatch)
    return np.mean(m[np.tril_indices_from(m, k=-1)])


def kendall_tau_list(results, truncate_on_mismatch=True):
    """ Read each file in a list and return the average kendall tau of each file with all others.
        See kendall_tau_matrix for calculation details

    :param results: a list of paths to tsv-formatted result files
    :param truncate_on_mismatch: Normally, truncate results to match before comparison, if False, lists must match.
    :returns: a list representing the average kendall tau for each file vs all others
    """

    m = kendall_tau_matrix(results, truncate_on_mismatch)
    # We want the mean of each row or column (same thing in a similarity matrix),
    # but we must exclude the identity diagonal (all 1.0's), so we calculate the mean rather than just np.mean(...)
    # This is the mean of similarity for each individual file vs all others.
    return list((np.sum(m, axis=0) - 1.0) / (len(m) - 1)) if len(m) > 1 else [0, ] * len(m)


def kendall_tau_dataframe(results, idx=None, truncate_on_mismatch=True):
    """ Read a list of files, return all tau correlations in a dataframe.

    :param results: a list of tsv-formatted result files
    :param idx: optional index with a label for each file in the list
    :param truncate_on_mismatch: Normally, truncate results to match before comparison, if False, lists must match.
    :returns: A dataframe containing a matrix of tau correlations as edges between results as nodes
    """

    if idx is None:
        idx = list(range(0, len(results), 1))

    return pd.DataFrame(kendall_tau_matrix(results, truncate_on_mismatch), columns=idx, index=idx)


def kendall_tau_matrix(result_files, truncate_on_mismatch=True):
    """ Read a list of files, perform all comparisons, and return tau correlations in a matrix.
        It is assumed that each file in result_files contains results of the same length, corresponding
        to the same indices.

    :param result_files: a list of tsv-formatted result files
    :param truncate_on_mismatch: Normally, truncate results to match before comparison, if False, lists must match.
    :returns: A dataframe containing a matrix of tau correlations as edges between results as nodes
    """

    # Loading files is relatively expensive; load them all once and save their contents to memory.
    result_values = []
    for i, f in enumerate(result_files):
        result_values.append(get_ranks_from_file(f).rename(columns={"rank": "rank{:>04}".format(i)}))

    all_results = pd.concat(result_values, axis=1)

    if truncate_on_mismatch:
        all_results = all_results.dropna(axis=0, how='any')
        result_values = [list(all_results[col]) for col in all_results.columns]
    elif all_results.isnull().values.any:
        raise ValueError("NaN values exist in data intended for Kendall tau calculation.")

    # Calculate the Kendall tau for each edge in the matrix; save time by duplicating across the diagonal & filling 1's
    taus = np.zeros((len(result_values), len(result_values)), dtype=np.float64)
    for row, y in enumerate(result_values):
        for col, x in enumerate(result_values):
            if col < row:
                taus[row, col], p = kendalltau(result_values[row], result_values[col])
                taus[col, row] = taus[row, col]
            elif col == row:
                taus[row, col] = 1.0

    return taus


def pct_similarity(results, map_probes_to_genes_first=True, top=None):
    """ Read each file in a list and return the percent overlap of their top genes.
        See pct_similarity_matrix_raw for calculation details

    :param results: a list of paths to tsv-formatted result files or a list of lists of probes for comparison
    :param map_probes_to_genes_first: if True, map each probe to its corresponding gene, then compare overlap of genes
    :param top: How many probes would you like? None for all genes past the peak. <1 for pctage, >1 for quantity
    :returns: a float value representing the percentage overlap of top genes from a list of files
    """

    m = pct_similarity_matrix(results, map_probes_to_genes_first, top)
    return np.mean(m[np.tril_indices_from(m, k=-1)])


def pct_similarity_list(results, map_probes_to_genes_first=True, top=None):
    """ Read each file in a list and return the percent overlap of each file with all others.
        See pct_similarity_matrix_raw for calculation details

    :param results: a list of paths to tsv-formatted result files or a list of lists to compare
    :param map_probes_to_genes_first: if True, map each probe to its corresponding gene, then compare overlap of genes
    :param top: How many probes would you like? None for all genes past the peak. <1 for pctage, >1 for quantity
    :returns: a list representing the percentage overlap of top genes for each file vs all others
    """

    m = pct_similarity_matrix(results, map_probes_to_genes_first, top)
    # We want the mean of each row or column (same thing in a similarity matrix),
    # but we must exclude the identity diagonal (all 1.0's), so we calculate the mean rather than just np.mean(...)
    # This is the mean of similarity for each individual file vs all others.
    return list((np.sum(m, axis=0) - 1.0) / (len(m) - 1)) if len(m) > 1 else [0, ] * len(m)


def pct_similarity_matrix(probe_lists, map_probes_to_genes_first=True, top=None):
    """ Read each file in a list and return the percent overlap of their top genes.

    For our purposes, the percent overlap is twice the length of the intersection of the two sets
    divided by the length of both sets, end to end. This is the Dice's coefficient, and is the cleanest way to
    allow the pct_similarity measure to be any value from 0.00 to 1.00.

    :param probe_lists: a list of lists of probes, or a list of files to be parsed
    :param map_probes_to_genes_first: if True, map each probe to its corresponding gene, then compare overlap of genes
    :param top: How many probes would you like? None for all genes past the peak. <1 for pctage, >1 for quantity
    :returns: a numpy array representing the percentage overlap of top genes from a list of files
    """

    if len(probe_lists) > 0 and type(probe_lists[0]) == str:
        # If we are passed files, read the probe lists from them.
        probe_lists = [run_results(f, top)['top_probes'] for f in probe_lists]

    m = np.zeros((len(probe_lists), len(probe_lists)))
    for row, y_probes in enumerate(probe_lists):
        y_genes = [map_pid_to_eid(y) for y in y_probes]
        for col, x_probes in enumerate(probe_lists):
            if row < col:
                if map_probes_to_genes_first:
                    x_genes = [map_pid_to_eid(x) for x in x_probes]
                    # We cannot use set intersections because we may need to count duplicate genes multiple times.
                    intersection = sum([1 for x in y_genes if x in x_genes])
                else:
                    intersection = sum([1 for x in y_probes if x in x_probes])

                # Pct Similarity is Dice's coefficient, duplicated across the diagonal.
                m[row][col] = float(2.0 * intersection / (len(y_probes) + len(x_probes)))
                m[col][row] = m[row][col]
            elif row == col:
                m[row][col] = 1.0
    return m


def save_df_as_csv(path, out_file=None, sep=','):
    """ Convert a pickled DataFrame into a csv file for easier viewing.

    :param str path: The path to the picked DataFrame file
    :param str out_file: An alternative csv file path if just changing .df to .csv isn't desired.
    :param str sep: The character, a comma by default, to separate fields in the text file
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
    :param str sep: The character, a tab by default, to separate fields in the text file
    """

    if out_file is None:
        out_file = path[:path.rfind('.')] + '.tsv'

    if os.path.isfile(path):
        with open(path, 'rb') as f:
            df = pickle.load(f)
        df.to_csv(out_file, sep=sep)
    else:
        print("{} is not a file.".format(path))


def top_probes(tsv_file, top=None):
    """ Return the top probes from the tsv_file specified.

    :param tsv_file: The file containing pushr output
    :param top: How many probes would you like? None for all genes past the peak. <1 for pctage, >1 for quantity
    :return list: A list of probes still in the mix after maxxing or minning whack_a_probe.
    """

    return run_results(tsv_file, top)['top_probes']


def best_score(tsv_file):
    """ Return the best score from the tsv_file specified.

    :param tsv_file: The file containing push output
    :return float: The highest score in a max run, or the lowest in a min run.
    """

    return run_results(tsv_file)['best']


def initial_score(tsv_file):
    """ Return the initial score from the tsv_file specified.

    :param tsv_file: The file containing push output
    :return float: The first score, before any probes were dropped.
    """

    return run_results(tsv_file)['initial']
