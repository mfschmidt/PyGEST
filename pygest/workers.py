import multiprocessing
# import time
import os
import numpy as np
from scipy import stats


""" These patterns are modified from the excellent multiprocessing information
    at https://pymotw.com/3/multiprocessing/. It's the best resource I've ever found
    on multiprocessing with python.
"""


class Correlator(multiprocessing.Process):
    """ Checks a queue of correlation jobs, running each in turn.
    """

    def __init__(self, task_queue, expr, conn_vec):
        """ Create process with a copy of expression data and a connectivivty vector.
            Each task will have its own probe list and correlation dictionary.
        """
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self.expr = expr
        self.conn_vec = conn_vec
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
            next_task(self.expr, self.conn_vec)
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

    def __call__(self, expr, conn_vec):
        # print("      CorrelationTask::__call__ <{}> OT {}".format(self.__str__(), os.environ['OPENBLAS_NUM_THREADS']))
        for p in self.probes:
            expr_mat = np.corrcoef(expr.drop(labels=p, axis=0), rowvar=False)
            expr_vec = expr_mat[np.tril_indices(n=expr_mat.shape[0], k=-1)]
            if self.algorithm == 'pearson':
                self.correlations[p] = stats.pearsonr(expr_vec, conn_vec)[0]
            elif self.algorithm == 'spearman':
                self.correlations[p] = stats.spearmanr(expr_vec, conn_vec)[0]
            else:
                self.correlations[p] = np.corrcoef(expr_vec, conn_vec)[0, 1]
        # No return; correlations are stored in dictionary as they are calculated.

    def __str__(self):
        return "running {} on {} probes ({})".format(
            self.algorithm, len(self.probes), __name__
        )
