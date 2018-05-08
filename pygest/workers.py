import multiprocessing
# import time
import os
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.genmod.families import links


# These patterns are modified from the excellent multiprocessing information
# at https://pymotw.com/3/multiprocessing/. It's the best resource I've ever found
# on multiprocessing with python.


class Correlator(multiprocessing.Process):
    """ Checks a queue of correlation jobs, running each in turn.
    """

    def __init__(self, task_queue, expr, conn_vec, mask):
        """ Create process with a copy of expression data and a connectivivty vector.
            Each task will have its own probe list and correlation dictionary.
        """
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self.expr = expr
        self.conn_vec = conn_vec
        self.mask = mask
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        # print("Correlator::__init__ {}: initializing ({})...".format(self.name, __name__))

    def run(self):
        """ As long as there are more tasks, keep running them.
        """
        while True:
            next_task = self.task_queue.get()
            if next_task is None:
                # print("  {}::run done. ate the poison pill.".format(self.name))
                self.task_queue.task_done()
                break
            # print("  {}::run correlating {}".format(self.name, next_task))
            # time_a = time.time()
            next_task(self.expr, self.conn_vec, self.mask)
            # time_b = time.time()
            # print("  {}::run finished in {:0.2f}".format(self.name, time_b - time_a))
            self.task_queue.task_done()
        # print("    ... {} done and out.".format(self.name))


class CorrelationTask:

    def __init__(self, probes, r_dict, algorithm):
        """ Create a task that will perform correlations on each probe in probes, storing them in r_dict.
            The task needs to be passed expression and connectivity data by the process running it.
        """
        self.probes = probes
        self.correlations = r_dict
        self.algorithm = algorithm.lower()
        # print("      CorrelationTask::__init__ {}: initializing ({})".format(self.__str__(), __name__))

    def __call__(self, expr, conn_vec, mask):
        # print("      CorrelationTask::__call__ <{}> OT {}".format(self.__str__(), os.environ['OPENBLAS_NUM_THREADS']))
        for p in self.probes:
            expr_mat = np.corrcoef(expr.drop(labels=p, axis=0), rowvar=False)
            expr_vec = expr_mat[np.tril_indices(n=expr_mat.shape[0], k=-1)]
            if self.algorithm == 'pearson':
                self.correlations[p] = stats.pearsonr(expr_vec[mask], conn_vec[mask])[0]
            elif self.algorithm == 'spearman':
                self.correlations[p] = stats.spearmanr(expr_vec[mask], conn_vec[mask])[0]
            else:
                self.correlations[p] = np.corrcoef(expr_vec[mask], conn_vec[mask])[0, 1]
        # No return; correlations are stored in dictionary as they are calculated.

    def __str__(self):
        return "running {} on {} probes ({})".format(
            self.algorithm, len(self.probes), __name__
        )


class LinearModeler(multiprocessing.Process):
    """ Checks a queue of correlation jobs, running each in turn.
    """

    def __init__(self, task_queue, expr, conn_vec, dist_vec, mask):
        """ Create process with a copy of expression data and a connectivivty vector.
            Each task will have its own probe list and correlation dictionary.
        """
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self.expr = expr
        self.conn_vec = conn_vec[mask]
        self.dist_vec = dist_vec[mask]
        self.mask = mask
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        # print("Correlator::__init__ {}: initializing ({})...".format(self.name, __name__))

    def run(self):
        """ As long as there are more tasks, keep running them.
        """
        while True:
            next_task = self.task_queue.get()
            if next_task is None:
                # print("  {}::run done. ate the poison pill.".format(self.name))
                self.task_queue.task_done()
                break
            # print("  {}::run correlating {}".format(self.name, next_task))
            # time_a = time.time()
            next_task(self.expr, self.conn_vec, self.dist_vec, self.mask)
            # time_b = time.time()
            # print("  {}::run finished in {:0.2f}".format(self.name, time_b - time_a))
            self.task_queue.task_done()
        # print("    ... {} done and out.".format(self.name))


class LinearModelingTask:

    def __init__(self, probes, b_dict, link_function):
        """ Create a task that will perform linear models on each probe in probes, storing them in r_dict.
            The task needs to be passed expression and connectivity data by the process running it.
        """
        self.probes = probes
        self.betas = b_dict
        self.link_function = link_function
        if self.link_function == 'log':
            self.link = links.log
        else:
            self.link = links.identity
        # print("      CorrelationTask::__init__ {}: initializing ({})".format(self.__str__(), __name__))

    def __call__(self, expr, conn_vec, dist_vec, mask):
        # print("      CorrelationTask::__call__ <{}> OT {}".format(self.__str__(), os.environ['OPENBLAS_NUM_THREADS']))
        for p in self.probes:
            # Generate a new expression similarity matrix without this one probe.
            expr_mat = np.corrcoef(expr.drop(labels=p, axis=0), rowvar=False)
            expr_vec = expr_mat[np.tril_indices(n=expr_mat.shape[0], k=-1)][mask]
            # Use a GLM to calculate the relationship between conn and expr, adjusted for distance
            endog = pd.DataFrame({'y': conn_vec})
            exog = sm.add_constant(pd.DataFrame({'x': expr_vec, 'dist': dist_vec}))
            result = sm.GLM(endog, exog, family=sm.families.Gaussian(self.link)).fit()
            self.betas[p] = result.params['x']

        # No return; correlations are stored in dictionary as they are calculated.

    def __str__(self):
        return "running {} ({}) on {} probes ({})".format(
            "GLM", self.link_function, len(self.probes), __name__
        )
