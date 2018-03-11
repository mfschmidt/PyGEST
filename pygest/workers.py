import multiprocessing
import itertools
import collections
import numpy as np
from scipy import stats


"""
    These patterns are modified from the excellent multiprocessing information
    at pymotw.com/3/multiprocessing/. It's the best resource I've ever found
    on multiprocessing with python.
"""


class Consumer(multiprocessing.Process):
    """ Consumes probe_ids and produces correlation dictionary entries from them.
    """

    def __init__(self, task_queue, d):
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self.d = d

    def run(self):
        """ As long as there are more tasks, keep running them.
        """

        while True:
            next_task = self.task_queue.get()
            if next_task is None:
                # print("{} ate the poison pill.".format(self.name))
                self.task_queue.task_done()
                break
            # logging.debug("{}: {}".format(self.name, next_task))
            answer = next_task()
            # logging.debug("          : {}".format(next_task))
            self.task_queue.task_done()
            self.d[answer[0]] = answer[1]
        # print("    {} done and out.".format(self.name))


class Task:

    def __init__(self, ns, p):
        self.expr = ns.expr
        self.conn = ns.conn
        self.corr = ns.corr
        self.p = p
        self.r = 0.0

    def __call__(self):
        y = np.corrcoef(self.expr.drop(labels=self.p, axis=0), rowvar=False)
        y = y[np.tril_indices(n=y.shape[0], k=-1)]
        if self.corr == 'pearson':
            self.r = stats.pearsonr(y, self.conn)[0]
        elif self.corr == 'spearman':
            self.r = stats.spearmanr(y, self.conn)[0]
        else:
            self.r = np.corrcoef(y, self.conn)[0, 1]
        return self.p, self.r

    def __str__(self):
        return "r({self.p}) = {self.r:0.16f}".format(self=self)


# TODO: This does not work. The MapReducer can only map probe_ids, not pass actual data sets.
def probe_to_r(probe_id):
    """ Whack each probe in my_tuple[2] from my_tuple[0] and correlate with my_tuple[1]
        These threading libraries and their single arguments make this ugly to document.

    :param probe_id: Contains an expression DataFrame at position 0,
                     a connectivity vector at position 1,
                     and a list of probes to knock out at position 2.
    """

    expr = None  # an inaccessible DataFrame of original expression values
    conn = None  # an inaccessible vector of connectivity values
    d = {}
    for probe in probe_id:
        y = np.corrcoef(expr.drop(labels=probe, axis=0), rowvar=False)
        y = y[np.tril_indices(n=y.shape[0], k=-1)]
        d[probe] = stats.pearsonr(y, conn[1])[0]
    return d


# TODO: This does not work. The MapReducer can only map probe_ids, not pass actual data sets.
def dict_update(new_dict):
    """ Update the main dictionary with new values.

    :param new_dict: 0 is the main dictionary, 1 is the new dictionary
    """

    d = {}  # an inaccessible master dictionary to update with new key-value pairs
    d.update(new_dict)


class SimpleMapReduce:

    def __init__(self, map_func, reduce_func, ns, num_workers=None):
        """
        map_func

          Function to map inputs to intermediate data. Takes as
          argument one input value and returns a tuple with the
          key and a value to be reduced.

        reduce_func

          Function to reduce partitioned version of intermediate
          data to final output. Takes as argument a key as
          produced by map_func and a sequence of the values
          associated with that key.

        num_workers

          The number of workers to create in the pool. Defaults
          to the number of CPUs available on the current host.
        """
        self.map_func = map_func
        self.reduce_func = reduce_func
        self.ns = ns
        self.pool = multiprocessing.Pool(num_workers)

    def partition(self, mapped_values):
        """Organize the mapped values by their key.
        Returns an unsorted sequence of tuples with a key
        and a sequence of values.
        """
        partitioned_data = collections.defaultdict(list)
        for key, value in mapped_values:
            partitioned_data[key].append(value)
        return partitioned_data.items()

    def __call__(self, inputs, chunk_size=1):
        """Process the inputs through the map and reduce functions
        given.

        inputs
          An iterable containing the input data to be processed.

        chunksize=1
          The portion of the input data to hand to each worker.
          This can be used to tune performance during the mapping
          phase.
        """
        map_responses = self.pool.map(
            self.map_func,
            inputs,
            chunksize=chunk_size,
        )
        partitioned_data = self.partition(
            itertools.chain(*map_responses)
        )
        reduced_values = self.pool.map(
            self.reduce_func,
            partitioned_data,
        )
        return reduced_values
