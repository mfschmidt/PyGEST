import multiprocessing
import numpy as np


class Consumer(multiprocessing.Process):
    """ Cnsumes probe_ids and produces correlation dictionary entries from them.
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
        self.p = p
        self.r = 0.0

    def __call__(self):
        y = np.corrcoef(self.expr.drop(labels=self.p, axis=0), rowvar=False)
        y = y[np.tril_indices(n=y.shape[0], k=-1)]
        self.r = np.corrcoef(y, self.conn)[0, 1]
        return self.p, self.r

    def __str__(self):
        return "r({self.p}) = {self.r:0.16f}".format(self=self)
