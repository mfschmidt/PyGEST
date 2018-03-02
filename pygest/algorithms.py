import numpy as np
import pandas as pd
from scipy import stats

import logging


def corr_expr_conn(expr, conn, method='', logger=None):
    """ Perform a correlation on the two matrices or vectors provided.

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
