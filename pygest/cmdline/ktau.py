import os
import pandas as pd
import argparse

from scipy.stats import kendalltau
from pygest.convenience import get_ranks_from_file


class Ktau:
    """ A command to convert csv or tsv files to dataframes """
    # Normally, all commands would inherit from Command, but this is a simple command that needs none of the
    # overhead supplied by Command, so no need for it.

    def __init__(self, args):
        self._arguments = args
        self._parser = argparse.ArgumentParser(description="ktau calculator")
        self._add_arguments()
        self._args = self._parser.parse_args(self._arguments)

    def _add_arguments(self):
        """ Add command-specific arguments, then Command's generic arguments. """
        self._parser.add_argument("a",
                                  help="A csv or tsv file to compare with another.")
        self._parser.add_argument("b",
                                  help="A csv or tsv file to compare with another.")

    def _post_process_arguments(self):
        """ After arguments are processed, tweak what we need to. """
        if self._args.log == '':
            # We really don't need to log ktau results to a log file by default.
            self._args.log = 'null'

    def run(self):
        """ Read data from tsv files and calculate the order similarity of their ordered probes. """

        # Read each file provided
        a_ranks = get_ranks_from_file(self._args.a)
        b_ranks = get_ranks_from_file(self._args.b)

        if a_ranks is not None and b_ranks is not None:
            tau, p = kendalltau(a_ranks, b_ranks)
            print("tau={:0.3f}; p={:0.4}".format(tau, p))

        return 0
